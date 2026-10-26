#include "vision.h"
#include "core/axial_patch_vision.h"
#include "mfq_paged_prefix_cache.h"

namespace mfq::cuda::deepseek_v41_runtime {
namespace {

struct HostLoading {
    bool& value;bool previous;
    explicit HostLoading(CudaExecutionContext& execution):value(execution.loading_cpu_layer),previous(value){value=true;}
    ~HostLoading(){value=previous;}
};

Tensor vector_weight(CudaExecutionContext& execution,const mfq::ModelSource& source,
                     const std::string& name,int64_t width) {
    auto value=load_dense_gpu(execution,source,name);
    MFQ_RUNTIME_CHECK(value.dim()==1 && value.size(0)==width,"DeepSeek-V4.1 vision vector shape mismatch: "+name);
    return value;
}

struct Span {int64_t begin,end,height,width,rows,columns;};
std::vector<Span> validate_media(const CommonConfig& config,const std::vector<int64_t>& tokens,
                                 const MfqMultimodalInput& media) {
    const auto count=media.image_bounds.size()/4;
    const auto patch_width=3*config.vision.patch_size*config.vision.patch_size;
    if(media.processor!=MfqMultimodalProcessor::deepseek_v41 || media.processor_name!="deepseek_v41" ||
            !count || media.image_bounds.size()%4 || media.vision_grid.size()!=count*4 ||
            media.pixel_shape.size()!=3 || media.pixel_shape[0]!=int64_t(count) ||
            media.pixel_shape[1]<=0 || media.pixel_shape[2]!=patch_width ||
            media.vision_grid_shape!=std::vector<int64_t>{int64_t(count),4} ||
            !media.video_grid.empty() || tokens.empty())
        throw std::invalid_argument("DeepSeek-V4.1 vision payload geometry is invalid");
    const auto maximum=uint64_t(media.pixel_shape[1]);
    if(maximum>SIZE_MAX/uint64_t(patch_width)/count ||
            media.pixel_values.size()!=count*maximum*uint64_t(patch_width))
        throw std::invalid_argument("DeepSeek-V4.1 vision patch storage is invalid");
    for(const auto token:tokens)if(token<0 || token>=config.vocab)
        throw std::invalid_argument("DeepSeek-V4.1 vision prompt token is outside the vocabulary");
    std::vector<Span> spans;spans.reserve(count);int64_t previous=0;
    const auto ratio=config.vision.downsample_ratio;
    for(std::size_t i=0;i<count;++i) {
        const auto* bound=media.image_bounds.data()+4*i;
        const auto* grid=media.vision_grid.data()+4*i;
        const int64_t h=grid[0],w=grid[1],rows=grid[2],columns=grid[3];
        if(bound[0]!=0 || bound[1]!=int64_t(i) || h<=0 || w<=0 || h>int64_t(maximum)/w ||
                rows!=(h+ratio-1)/ratio || columns!=(w+ratio-1)/ratio ||
                bound[2]<previous || bound[3]>int64_t(tokens.size()) ||
                bound[3]-bound[2]!=rows*(columns+1)+2 || rows*columns>config.vision.max_image_tokens)
            throw std::invalid_argument("DeepSeek-V4.1 image span is invalid");
        for(auto position=bound[2];position<bound[3];++position)if(tokens[position]!=config.image_token_id)
            throw std::invalid_argument("DeepSeek-V4.1 image span contains a text token");
        spans.push_back({bound[2],bound[3],h,w,rows,columns});previous=bound[3];
    }
    return spans;
}
} // namespace

Vision Vision::load(std::shared_ptr<const mfq::ModelSource> source,const CommonConfig& config) {
    MFQ_RUNTIME_CHECK(source && config.has_vision() && config.image_token_id>=0 &&
        config.image_token_id<config.vocab && config.vision.hidden%config.vision.n_heads==0 &&
        (config.vision.hidden/config.vision.n_heads)%4==0,"invalid DeepSeek-V4.1 CUDA vision config");
    for(const auto* name:{"vision.patch_embedding.weight","vision.output_norm.weight",
            "vision.aligner.input.weight","vision.aligner.output.weight","vision.special_token.start",
            "vision.special_token.newline","vision.special_token.end"})
        MFQ_RUNTIME_CHECK(has_tensor(*source,name),std::string("DeepSeek-V4.1 vision tensor missing: ")+name);
    return Vision(std::move(source),config);
}

mfq::StepSequence<Tensor> Vision::encode(CudaExecutionContext& execution,Tensor patches,int height,int width) const {
    const auto& v=config_.vision;
    const mfq::models::axial_patch_vision::Config config{
        v.hidden,v.n_layers,v.n_heads,v.intermediate,v.patch_size,
        v.downsample_ratio,config_.hidden,v.rope_theta,1e-6};
    auto tower=mfq::cuda::axial_patch_vision::CudaAxialPatchVisionTower::load(execution,*source_,config);
    auto steps=tower.encode_steps(execution,std::move(patches),height,width);
    while(auto step=steps.next()) {
        if(step.value)co_yield std::move(*step.value);
        else co_yield step.state;
    }
}

mfq::StepSequence<CudaPreparedPrompt> Vision::prepare(DeepseekV41CausalLm& language,
        const std::vector<int64_t>& tokens,const MfqMultimodalInput& media) const {
    auto spans=validate_media(config_,tokens,media);
    auto& execution=*language.execution;
    auto options=mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
    auto ids=mfq_tensor_backend::tensor(tokens,options).reshape({1,-1});
    auto embeddings=language.embed_forward(ids).contiguous();
    std::vector<uint8_t> mask(tokens.size(),0);
    std::string request_key="deepseek_v41|";
    for(std::size_t i=0;i<spans.size();++i) {
        const auto& span=spans[i];
        const auto patch_width=media.pixel_shape[2];
        const auto* first=media.pixel_values.data()+int64_t(i)*media.pixel_shape[1]*patch_width;
        auto key=mfq::cache::block_hash_hex(mfq::cache::sha256(first,
            std::size_t(span.height*span.width*patch_width)*sizeof(float)))+"|"+
            std::to_string(span.height)+"|"+std::to_string(span.width);
        Tensor visual;
        if(cached_key_==key && cached_image_.defined())visual=cached_image_;
        else {
            auto patches=mfq_tensor_backend::from_blob(const_cast<float*>(first),{span.height*span.width,patch_width},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU))
                .to(mfq_tensor_backend::kCUDA).to(mfq_tensor_backend::kFloat16);
            auto encoding=encode(execution,std::move(patches),int(span.height),int(span.width));
            while(auto step=encoding.next()){if(step.value)visual=std::move(*step.value);else co_yield step.state;}
            cached_key_=key;cached_image_=visual;
        }
        MFQ_RUNTIME_CHECK(visual.dim()==2 && visual.size(0)==span.rows*span.columns && visual.size(1)==config_.hidden,
                          "DeepSeek-V4.1 aligned image shape mismatch");
        Tensor start,newline,end;
        {HostLoading loading(execution);
            start=vector_weight(execution,*source_,"vision.special_token.start",config_.hidden).to(mfq_tensor_backend::kCUDA);
            newline=vector_weight(execution,*source_,"vision.special_token.newline",config_.hidden).to(mfq_tensor_backend::kCUDA);
            end=vector_weight(execution,*source_,"vision.special_token.end",config_.hidden).to(mfq_tensor_backend::kCUDA);}
        std::vector<Tensor> rows{start.reshape({1,-1}).to(visual.scalar_type())};
        for(int64_t row=0;row<span.rows;++row) {
            rows.push_back(visual.narrow(0,row*span.columns,span.columns));
            rows.push_back(newline.reshape({1,-1}).to(visual.scalar_type()));
        }
        rows.push_back(end.reshape({1,-1}).to(visual.scalar_type()));
        embeddings.narrow(1,span.begin,span.end-span.begin).copy_(
            mfq_tensor_backend::cat(rows,0).unsqueeze(0).to(embeddings.scalar_type()));
        std::fill(mask.begin()+span.begin,mask.begin()+span.end,1);
        request_key+=key+"|"+std::to_string(span.begin)+"|"+std::to_string(span.end)+";";
        co_yield mfq::StepState::advanced;
    }
    CudaPreparedPrompt prepared{tokens,std::move(embeddings),{},0,std::move(request_key)};
    prepared.image_mask=mfq_tensor_backend::tensor(mask,mfq_tensor_backend::TensorOptions()
        .dtype(mfq_tensor_backend::kUInt8).device(mfq_tensor_backend::kCPU)).reshape({1,-1});
    static std::atomic_flag reported=ATOMIC_FLAG_INIT;
    if(!reported.test_and_set())std::fprintf(stderr,"deepseek_v41_vision_dispatch enabled=1 images=%zu weights=streamed engram=text_only router=modality_bias\n",spans.size());
    co_yield std::move(prepared);
}
} // namespace mfq::cuda::deepseek_v41_runtime
