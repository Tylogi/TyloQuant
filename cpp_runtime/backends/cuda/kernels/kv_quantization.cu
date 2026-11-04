#include <math_constants.h>
#include "mfq_cuda_kv_quantization.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
namespace {
int rotation_width(int dimension) {
    MFQ_RUNTIME_CHECK(dimension > 0 && dimension <= 1024, "unsupported TurboQuant KV head dimension");
    int width = 1;
    while (width < dimension) width *= 2;
    return width;
}
bool valid_bits(double b) { return b == 0 || b == 2 || b == 2.5 || b == 3 || b == 3.5 || b == 4 || b == 6 || b == 8; }
struct Tables { tb::Tensor centers, signs; };
Tables tables(int width, int bits, bool value, const tb::Device& device) {
    static std::mutex mutex;
    static std::map<std::tuple<int,int,bool,int>,Tables> cached;
    std::lock_guard<std::mutex> guard(mutex);
    int ordinal = 0; MFQ_CUDA_CHECK(cudaGetDevice(&ordinal));
    const auto key = std::make_tuple(width,bits,value,ordinal);
    if (auto found = cached.find(key); found != cached.end()) return found->second;
    // Same deterministic sphere codebook and signed Hadamard rotation as Metal v1.
    constexpr int grid = 32768;
    const int levels = 1 << bits;
    std::vector<double> points(grid),weights(grid),cumulative(grid);
    double total = 0;
    for(int i=0;i<grid;++i) {
        const double x=-1.0+1e-6+(2.0-2e-6)*i/(grid-1);
        points[i]=x;weights[i]=std::pow(1.0-x*x,(width-3)/2.0);cumulative[i]=total+=weights[i];
    }
    std::vector<float> centers(levels),signs(width);
    for(int i=0;i<levels;++i) {
        const auto at=std::lower_bound(cumulative.begin(),cumulative.end(),total*(i+0.5)/levels);
        centers[i]=float(points[std::min(grid-1,int(at-cumulative.begin()))]);
    }
    for(int iteration=0;iteration<100;++iteration) {
        std::vector<double> sums(levels,0),masses(levels,0);
        int cell=0;
        for(int i=0;i<grid;++i) {
            while(cell+1<levels && points[i]>(centers[cell]+centers[cell+1])*0.5)++cell;
            sums[cell]+=points[i]*weights[i];masses[cell]+=weights[i];
        }
        double change=0;
        for(int i=0;i<levels;++i)if(masses[i]>0) {
            const float next=float(sums[i]/masses[i]);
            change=std::max(change,std::abs(double(next-centers[i])));centers[i]=next;
        }
        if(change<1e-6)break;
    }
    uint32_t state=0x6d2b79f5u^uint32_t(width*7919+(value?1:0));
    for(auto& sign:signs){state^=state<<13;state^=state>>17;state^=state<<5;sign=state&1?1.f:-1.f;}
    const auto copy=[&](const std::vector<float>& source) {
        auto cpu=tb::empty({int64_t(source.size())},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        std::copy(source.begin(),source.end(),cpu.data_ptr<float>());
        return cpu.to(device);
    };
    Tables result{copy(centers),copy(signs)};
    // A cached table can be reused by a different execution stream.
    MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
    cached.emplace(key,result);return result;
}
__global__ void encode_kernel(const float* input,uint32_t* out,const float* centers,const float* signs,
        int d,int w,int bits,int packed) {
    const int lane=threadIdx.x,row=blockIdx.x;
    __shared__ float work[1024],norms[1024];
    __shared__ uint32_t codes[1024];
    const float x=lane<d?input[int64_t(row)*d+lane]:0.f;
    work[lane]=x*signs[lane];norms[lane]=x*x;__syncthreads();
    for(int stride=w/2;stride;stride/=2){if(lane<stride)norms[lane]+=norms[lane+stride];__syncthreads();}
    const float norm=sqrtf(norms[0]);
    for(int stride=1;stride<w;stride*=2) {
        const float a=work[lane&~stride],b=work[lane|stride];__syncthreads();
        work[lane]=(lane&stride)?a-b:a+b;__syncthreads();
    }
    const float rotated=norm>0?work[lane]/(norm*sqrtf(float(w))):0.f;
    int lo=0,hi=(1<<bits)-1;
    while(lo<hi){int mid=(lo+hi)/2;if(rotated>(centers[mid]+centers[mid+1])*0.5f)lo=mid+1;else hi=mid;}
    codes[lane]=lo;__syncthreads();
    auto* dest=out+int64_t(row)*packed;
    if(lane==0)dest[0]=__float_as_uint(norm);
    for(int word=lane;word<packed-1;word+=w) {
        uint32_t result=0;
        for(int bit=0;bit<32;++bit){const int offset=word*32+bit;if(offset<w*bits)result|=((codes[offset/bits]>>(offset%bits))&1u)<<bit;}
        dest[1+word]=result;
    }
}
__device__ float code(const uint32_t* row,int d,int bits,const float* centers) {
    const int offset=d*bits,shift=offset%32;
    uint32_t value=row[1+offset/32]>>shift;
    if(shift+bits>32)value|=row[2+offset/32]<<(32-shift);
    return centers[value&((1u<<bits)-1)];
}
__global__ void decode_kernel(const uint32_t* input,float* out,const float* centers,const float* signs,
        int d,int w,int bits,int packed) {
    const int lane=threadIdx.x,row=blockIdx.x;
    __shared__ float work[1024];
    const auto* source=input+int64_t(row)*packed;
    work[lane]=code(source,lane,bits,centers);__syncthreads();
    for(int stride=1;stride<w;stride*=2) {
        const float a=work[lane&~stride],b=work[lane|stride];__syncthreads();
        work[lane]=(lane&stride)?a-b:a+b;__syncthreads();
    }
    if(lane<d)out[int64_t(row)*d+lane]=work[lane]*signs[lane]*__uint_as_float(source[0])/sqrtf(float(w));
}
struct CacheView {
    const uint32_t* data;
    int64_t batch_stride,head_stride,token_stride;
    const int64_t* chunks=nullptr;
    const int32_t* table=nullptr;
    int table_width=0,page_size=0,chunk_pages=0,heads=0,packed=0;
    const uint32_t* hot_data=nullptr;
    const int32_t* hot_mapping=nullptr;
    int64_t hot_cells=0,cold_cells=0;
    const int32_t* cold_mapping=nullptr;
    IndirectKvView indirect{};
    int indirect_offset = 0;
    // Bind once per CTA; packed dimensions share the same request allocations.
    __device__ int64_t bind_indirect_row(int batch) {
        if (!indirect.active(batch) || !indirect.address(batch, indirect_offset))
            return 0;
        const auto logical = indirect.capacity(batch, indirect_offset);
        auto extent = logical;
        data = static_cast<const uint32_t*>(indirect.address(batch, indirect_offset));
        hot_mapping = nullptr;
        cold_mapping = nullptr;
        if (indirect.exclusive(batch)) {
            const int hot = indirect_offset ? HotKvStateFields::Values : HotKvStateFields::Keys;
            const int cold = indirect_offset ? HotKvStateFields::ColdValues : HotKvStateFields::ColdKeys;
            data = static_cast<const uint32_t*>(indirect.tier_address(batch, cold));
            hot_data = static_cast<const uint32_t*>(indirect.tier_address(batch, hot));
            hot_mapping = static_cast<const int32_t*>(indirect.tier_address(batch, HotKvStateFields::Mapping));
            cold_mapping = static_cast<const int32_t*>(indirect.tier_address(batch, HotKvStateFields::ColdMapping));
            hot_cells = indirect.tier_capacity(batch, hot);
            extent = cold_cells = indirect.tier_capacity(batch, cold);
        }
        batch_stride = 0;
        head_stride = extent * packed;
        token_stride = packed;
        indirect = {};
        return logical;
    }
    template<bool use_hot=false> __device__ const uint32_t* row(int batch,int head,int token) const {
        if (indirect.states) {
            return indirect.resolve<const uint32_t>(batch, token, head, packed, indirect_offset);
        }
        if constexpr(use_hot) {
            if(hot_mapping) {
                const int slot=hot_mapping[token];
                if(slot>=0)return hot_data+(int64_t(head)*hot_cells+slot)*token_stride;
                if(cold_mapping) {
                    const int cold=cold_mapping[token];
                    return cold>=0 && cold<cold_cells?data+(int64_t(head)*cold_cells+cold)*token_stride:nullptr;
                }
            }
        }
        if(!page_size)return data+int64_t(batch)*batch_stride+int64_t(head)*head_stride+int64_t(token)*token_stride;
        const int physical=table[int64_t(batch)*table_width+token/page_size];
        if(physical<0)return nullptr;
        auto* base=reinterpret_cast<const uint32_t*>(chunks[physical/chunk_pages]);
        return base+((int64_t(physical%chunk_pages)*heads+head)*page_size+token%page_size)*packed;
    }
};
__global__ void paged_write_kernel(CacheView view,const uint32_t* source,const int64_t* positions,
        int tokens,int position_batches) {
    const int row=blockIdx.x,token=row%tokens,head=(row/tokens)%view.heads,batch=row/(tokens*view.heads);
    const int64_t pos=positions[(position_batches?int64_t(batch)*tokens:0)+token];
    if(pos<0 || pos>=int64_t(view.table_width)*view.page_size)return;
    auto* dest=const_cast<uint32_t*>(view.row(batch,head,int(pos)));
    if(dest)for(int i=threadIdx.x;i<view.packed;i+=blockDim.x)dest[i]=source[int64_t(row)*view.packed+i];
}
__device__ float warp_sum(float x){for(int i=16;i;i/=2)x+=__shfl_xor_sync(0xffffffff,x,i);return x;}
__device__ float warp_max(float x){for(int i=16;i;i/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,i));return x;}
__host__ __device__ constexpr int attention_tile(int width) {
    return width<=256?8:width<=512?4:2;
}
template<int W,bool use_hot=false> __global__ void attention_kernel(const float* query,CacheView keys,CacheView values,
        const float* kc,const float* vc,const float* signs,const int32_t* selected,const int64_t* positions,
        const int64_t* lengths,const float* mask,int mask_width,float* partial,float* stats,
        int heads,int kv_heads,int tokens,int d,int capacity,int slots,int parts,int span,
        int offset,int position_batches,int bk,int bv,float scale) {
    constexpr int SG=4,TILE=attention_tile(W),N=(W+31)/32;
    const int lane=threadIdx.x%32,sg=threadIdx.x/32,tid=threadIdx.x;
    const int part=blockIdx.x,group=blockIdx.y,bt=blockIdx.z,batch=bt/tokens,token=bt%tokens;
    const int gqa=heads/kv_heads,groups=(gqa+SG-1)/SG,kh=group/groups,local=(group%groups)*SG+sg,head=kh*gqa+local;
    const bool active=local<gqa;
    const int boundary=positions?int(positions[(position_batches?int64_t(batch)*tokens:0)+token]):offset+token;
    int visible=min(capacity,min(boundary+1,lengths?int(lengths[batch]):capacity));
    if(keys.indirect.states) {
        const auto key_capacity=keys.bind_indirect_row(batch);
        const auto value_capacity=values.bind_indirect_row(batch);
        visible=int(min(int64_t(visible),min(key_capacity,value_capacity)));
    }
    const int64_t qrow=(int64_t(batch)*heads+head)*tokens+token;
    __shared__ float kt[TILE*W],vt[TILE*W];
    __shared__ int ids[TILE];
    float q[N],acc[N];
    #pragma unroll
    for(int i=0;i<N;++i){int dim=lane+i*32;q[i]=active&&dim<d?query[qrow*d+dim]*signs[dim]:0.f;acc[i]=0.f;}
    for(int stride=1;stride<min(W,32);stride*=2)
        for(int i=0;i<N;++i){float other=__shfl_xor_sync(0xffffffff,q[i],stride);q[i]=lane&stride?other-q[i]:q[i]+other;}
    for(int stride=1;stride<N;stride*=2){float next[N];for(int i=0;i<N;++i){float a=q[i&~stride],b=q[i|stride];next[i]=i&stride?a-b:a+b;}for(int i=0;i<N;++i)q[i]=next[i];}
    for(int i=0;i<N;++i)q[i]*=rsqrtf(float(W));
    float maximum=-CUDART_INF_F,mass=0;
    const int end=selected?slots: ::min(slots,::max(0,visible));
    for(int start=part*span;start<min((part+1)*span,end);start+=TILE) {
        for(int at=tid;at<TILE*W;at+=SG*32) {
            const int slot=start+at/W,dim=at%W;
            int id=slot<slots&&slot<(part+1)*span?(selected?selected[int64_t(bt)*slots+slot]:slot):-1;
            bool valid=id>=0&&id<visible;
            if(valid&&mask)valid=mask[int64_t(batch)*mask_width+id]!=0;
            const auto* kr=valid?keys.row<use_hot>(batch,kh,id):nullptr;
            const auto* vr=valid?values.row<use_hot>(batch,kh,id):nullptr;
            valid=valid&&kr&&vr;
            if(dim==0)ids[at/W]=valid?id:-1;
            kt[at]=valid?__uint_as_float(kr[0])*code(kr,dim,bk,kc):0.f;
            vt[at]=valid?__uint_as_float(vr[0])*code(vr,dim,bv,vc):0.f;
        }
        __syncthreads();
        float scores[TILE],weights[TILE],top=maximum;
        for(int j=0;j<TILE;++j) {
            float score=0;
            for(int i=0;i<N;++i)if(lane+i*32<W)score+=q[i]*kt[j*W+lane+i*32];
            score=warp_sum(score);scores[j]=ids[j]>=0?score*scale:-CUDART_INF_F;top=fmaxf(top,scores[j]);
        }
        const float factor=::isfinite(maximum)?expf(maximum-top):0.f;
        mass*=factor;
        for(int j=0;j<TILE;++j){weights[j]=::isfinite(scores[j])?expf(scores[j]-top):0.f;mass+=weights[j];}
        maximum=top;
        for(int i=0;i<N;++i){acc[i]*=factor;if(lane+i*32<W)for(int j=0;j<TILE;++j)acc[i]+=weights[j]*vt[j*W+lane+i*32];}
        __syncthreads();
    }
    if(active) {
        const int64_t row=qrow*parts+part;
        for(int i=0;i<N;++i)if(lane+i*32<W)partial[row*W+lane+i*32]=acc[i];
        if(lane==0){stats[row*2]=maximum;stats[row*2+1]=mass;}
    }
}
__global__ void attention_output(const float* partial,const float* stats,const float* signs,float* out,
        int w,int d,int parts,int heads,int tokens) {
    const int lane=threadIdx.x,row=blockIdx.x;
    __shared__ float factors[32],norm,work[1024];
    if(lane<32) {
        float top=lane<parts?stats[(int64_t(row)*parts+lane)*2]:-CUDART_INF_F;top=warp_max(top);
        float mass=0;
        if(lane<parts){int64_t at=(int64_t(row)*parts+lane)*2;float f=::isfinite(stats[at])?expf(stats[at]-top):0.f;factors[lane]=f;mass=f*stats[at+1];}
        mass=warp_sum(mass);if(lane==0)norm=mass;
    }
    __syncthreads();
    float x=0;
    if(lane<w){for(int i=0;i<parts;++i)x+=factors[i]*partial[(int64_t(row)*parts+i)*w+lane];work[lane]=norm>0?x/norm:0.f;}
    __syncthreads();
    for(int stride=1;stride<w;stride*=2){float a=lane<w?work[lane&~stride]:0.f,b=lane<w?work[lane|stride]:0.f;__syncthreads();if(lane<w)work[lane]=lane&stride?a-b:a+b;__syncthreads();}
    const int token=row%tokens,head=(row/tokens)%heads,batch=row/(tokens*heads);
    if(lane<d)out[((int64_t(batch)*tokens+token)*heads+head)*d+lane]=work[lane]*signs[lane]*rsqrtf(float(w));
}
CacheView view(const tb::Tensor& data,const tb::Tensor& chunks,const QuantizedKvPages& pages,int packed) {
    CacheView result{};result.packed=packed;
    if(pages.enabled()) {
        result.chunks=chunks.data_ptr<int64_t>();result.table=pages.table.data_ptr<int32_t>();
        result.table_width=int(pages.table.size(1));result.page_size=pages.page_size;
        result.chunk_pages=pages.pages_per_chunk;result.heads=pages.heads;
    } else {
        MFQ_RUNTIME_CHECK(data.is_cuda()&&data.dim()==4&&data.scalar_type()==tb::kInt32&&data.size(3)==packed&&data.stride(3)==1,"invalid packed KV layout");
        result.data=reinterpret_cast<const uint32_t*>(data.data_ptr());result.batch_stride=data.stride(0);result.head_stride=data.stride(1);result.token_stride=data.stride(2);
    }
    return result;
}
}

KvQuantization cuda_kv_quantization() {
    const char* text=std::getenv("MFQ_KV_TURBOQUANT_BITS");if(!text||!*text)return {};
    size_t consumed=0;double bits=std::stod(text,&consumed);
    MFQ_RUNTIME_CHECK(consumed==std::string(text).size()&&valid_bits(bits),"invalid TurboQuant KV bit width");return {bits};
}
int cuda_kv_packed_width(int d,int bits){MFQ_RUNTIME_CHECK(bits>=2&&bits<=8,"invalid TurboQuant component bits");return 1+(rotation_width(d)*bits+31)/32;}
std::string cuda_kv_quantization_tag(){auto q=cuda_kv_quantization();return q.enabled()?"-tq-v1-k"+std::to_string(q.key_bits())+"v"+std::to_string(q.value_bits()):"";}
tb::Tensor cuda_kv_encode(const tb::Tensor& source,int bits,bool value) {
    MFQ_RUNTIME_CHECK(source.is_cuda()&&source.dim()>0,"TurboQuant requires a CUDA tensor");
    MfqCudaGuard guard(source.get_device());
    const int d=int(source.size(-1)),w=rotation_width(d),p=cuda_kv_packed_width(d,bits);
    auto input=source.to(tb::kFloat32).contiguous();auto shape=source.sizes().vec();shape.back()=p;
    auto output=tb::empty(shape,source.options().dtype(tb::kInt32));const auto rows=input.numel()/d;
    MFQ_RUNTIME_CHECK(rows<=std::numeric_limits<int>::max(),"TurboQuant encode grid overflow");
    auto table=tables(w,bits,value,source.device());
    if(rows)encode_kernel<<<unsigned(rows),w,0,mfq_get_current_cuda_stream()>>>(input.data_ptr<float>(),reinterpret_cast<uint32_t*>(output.data_ptr()),table.centers.data_ptr<float>(),table.signs.data_ptr<float>(),d,w,bits,p);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
tb::Tensor cuda_kv_decode(const tb::Tensor& source,int d,int bits,bool value,tb::ScalarType dtype) {
    const int w=rotation_width(d),p=cuda_kv_packed_width(d,bits);
    MFQ_RUNTIME_CHECK(source.is_cuda()&&source.dim()>0&&source.scalar_type()==tb::kInt32&&source.size(-1)==p,"invalid packed TurboQuant tensor");
    MfqCudaGuard guard(source.get_device());auto input=source.contiguous();auto shape=source.sizes().vec();shape.back()=d;
    auto output=tb::empty(shape,source.options().dtype(tb::kFloat32));const auto rows=source.numel()/p;
    MFQ_RUNTIME_CHECK(rows<=std::numeric_limits<int>::max(),"TurboQuant decode grid overflow");auto table=tables(w,bits,value,source.device());
    if(rows)decode_kernel<<<unsigned(rows),w,0,mfq_get_current_cuda_stream()>>>(reinterpret_cast<const uint32_t*>(input.data_ptr()),output.data_ptr<float>(),table.centers.data_ptr<float>(),table.signs.data_ptr<float>(),d,w,bits,p);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return output.to(dtype);
}
void cuda_kv_write_paged(const QuantizedKvPages& pages,const tb::Tensor& keys,const tb::Tensor& values,const tb::Tensor& positions) {
    MFQ_RUNTIME_CHECK(pages.enabled()&&keys.dim()==4&&values.dim()==4&&keys.scalar_type()==tb::kInt32&&values.scalar_type()==tb::kInt32,"invalid paged TurboQuant write");
    auto pos=positions.to(tb::kInt64).contiguous();MfqCudaGuard guard(keys.get_device());
    for(auto pair:{std::pair{keys,pages.keys},std::pair{values,pages.values}}) {
        auto source=pair.first.contiguous();auto cache=view({},pair.second,pages,int(source.size(-1)));
        const auto rows=source.numel()/source.size(-1);
        paged_write_kernel<<<unsigned(rows),128,0,mfq_get_current_cuda_stream()>>>(cache,reinterpret_cast<const uint32_t*>(source.data_ptr()),pos.data_ptr<int64_t>(),int(source.size(2)),pos.dim()==2);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}
tb::Tensor cuda_kv_attention(const tb::Tensor& query,const tb::Tensor& keys,const tb::Tensor& values,
        KvQuantization quant,const tb::Tensor& selected,const tb::Tensor& positions,int64_t offset,
        const QuantizedKvPages& pages,int64_t capacity,const tb::Tensor& lengths,const tb::Tensor& mask,
        double scale,HotKvView hot,IndirectKvView indirect) {
    MFQ_RUNTIME_CHECK(quant.enabled()&&valid_bits(quant.bits)&&query.is_cuda()&&query.dim()==4,"invalid TurboQuant attention");
    MfqCudaGuard guard(query.get_device());
    const int b=int(query.size(0)),h=int(query.size(1)),t=int(query.size(2)),d=int(query.size(3)),w=rotation_width(d);
    if(pages.enabled()) {
        MFQ_RUNTIME_CHECK(pages.table.is_cuda()&&pages.table.scalar_type()==tb::kInt32&&pages.table.dim()==2&&pages.table.size(0)>=b&&pages.table.is_contiguous()&&pages.pages_per_chunk>0,"invalid TurboQuant page table");
        MFQ_RUNTIME_CHECK(pages.keys.is_cuda()&&pages.values.is_cuda()&&pages.keys.scalar_type()==tb::kInt64&&pages.values.scalar_type()==tb::kInt64&&pages.keys.is_contiguous()&&pages.values.is_contiguous()&&pages.keys.numel()==pages.values.numel(),"invalid TurboQuant page pointers");
    } else if (indirect.states) {
        MFQ_RUNTIME_CHECK(indirect.rows == b && t == 1 && keys.dim() == 4 && values.dim() == 4 &&
            keys.size(0) == 1 && values.size(0) == 1 && keys.size(1) == values.size(1),
            "invalid indirect TurboQuant cache geometry");
    } else {
        MFQ_RUNTIME_CHECK(keys.dim()==4&&values.dim()==4&&keys.size(0)==b&&values.size(0)==b&&keys.size(1)==values.size(1)&&keys.size(2)==values.size(2),"invalid TurboQuant cache geometry");
    }
    const int kh=pages.enabled()?pages.heads:int(keys.size(1));
    const int64_t allocated=pages.enabled()?pages.table.size(1)*pages.page_size:keys.size(2);
    if(!capacity)capacity=allocated;
    MFQ_RUNTIME_CHECK((indirect.states || capacity<=allocated)&&capacity<=std::numeric_limits<int>::max(),
        "TurboQuant attention capacity exceeds allocation");
    const int slots=selected.defined()?int(selected.size(-1)):int(capacity);
    MFQ_RUNTIME_CHECK(b>0&&h>0&&kh>0&&h%kh==0&&t>0&&b*t<=65535&&capacity>0&&slots>0,"invalid TurboQuant attention dimensions");
    if(selected.defined())MFQ_RUNTIME_CHECK(selected.dim()==3&&selected.size(0)==b&&selected.size(1)==t,"invalid TurboQuant selection");
    if(positions.defined())MFQ_RUNTIME_CHECK((positions.dim()==1&&positions.numel()==t)||(positions.dim()==2&&positions.size(0)==b&&positions.size(1)==t),"invalid TurboQuant positions");
    if(lengths.defined())MFQ_RUNTIME_CHECK(lengths.numel()==b,"invalid TurboQuant sequence lengths");
    if(mask.defined())MFQ_RUNTIME_CHECK(mask.dim()==2&&mask.size(0)==b&&mask.size(1)>=capacity,"invalid TurboQuant attention mask");
    auto q=query.to(tb::kFloat32).contiguous();
    auto ids=selected.defined()?selected.to(tb::kInt32).contiguous():tb::Tensor{};
    auto pos=positions.defined()?positions.to(tb::kInt64).contiguous():tb::Tensor{};
    auto len=lengths.defined()?lengths.to(tb::kInt64).contiguous():tb::Tensor{};
    auto valid=mask.defined()?mask.to(tb::kFloat32).contiguous():tb::Tensor{};
    const auto kt=tables(w,quant.key_bits(),false,query.device()),vt=tables(w,quant.value_bits(),true,query.device());
    auto kv=view(keys,pages.keys,pages,cuda_kv_packed_width(d,quant.key_bits()));
    auto vv=view(values,pages.values,pages,cuda_kv_packed_width(d,quant.value_bits()));
    kv.indirect = vv.indirect = indirect;
    vv.indirect_offset = 1;
    if(hot.mapping){kv.hot_data=static_cast<const uint32_t*>(hot.keys);vv.hot_data=static_cast<const uint32_t*>(hot.values);kv.hot_mapping=vv.hot_mapping=hot.mapping;kv.hot_cells=vv.hot_cells=hot.cells;kv.cold_mapping=vv.cold_mapping=hot.cold_mapping;kv.cold_cells=vv.cold_cells=hot.cold_cells;}
    // Prefill already has enough query rows to fill the GPU. Splitting those rows
    // multiplies scratch storage without adding useful parallelism (GiB at 8K).
    const int groups=kh*((h/kh+3)/4);
    int parts=b*h*t>=1024?1:std::min(16,(slots+255)/256);
    if(indirect.states) {
        cudaDeviceProp properties{};
        MFQ_CUDA_CHECK(cudaGetDeviceProperties(&properties,query.get_device()));
        const int64_t blocks_per_part=int64_t(b)*t*groups;
        const int parallel_parts=int((properties.multiProcessorCount+blocks_per_part-1)/blocks_per_part);
        const int tiles=(slots+attention_tile(w)-1)/attention_tile(w);
        // Split independent tiles across SMs before serializing mapped-RAM reads.
        parts=std::min(16,std::min(tiles,std::max(parts,parallel_parts)));
    }
    const int span=((slots+parts*8-1)/(parts*8))*8;
    auto partial=tb::empty({b,h,t,parts,w},q.options()),stats=tb::empty({b,h,t,parts,2},q.options());
    auto output=tb::empty({b,t,h,d},q.options());
    const dim3 grid(parts,groups,b*t);
    #define TQ_LAUNCH_H(W,H) attention_kernel<W,H><<<grid,128,0,mfq_get_current_cuda_stream()>>>(q.data_ptr<float>(),kv,vv,kt.centers.data_ptr<float>(),vt.centers.data_ptr<float>(),kt.signs.data_ptr<float>(),ids.defined()?ids.data_ptr<int32_t>():nullptr,pos.defined()?pos.data_ptr<int64_t>():nullptr,len.defined()?len.data_ptr<int64_t>():nullptr,valid.defined()?valid.data_ptr<float>():nullptr,valid.defined()?int(valid.size(1)):0,partial.data_ptr<float>(),stats.data_ptr<float>(),h,kh,t,d,int(capacity),slots,parts,span,int(offset),pos.defined()&&pos.dim()==2,quant.key_bits(),quant.value_bits(),float(scale>0?scale:1/std::sqrt(double(d))))
    #define TQ_LAUNCH(W) if(hot.mapping || indirect.states){TQ_LAUNCH_H(W,true);}else{TQ_LAUNCH_H(W,false);}
    switch(w){case 1:TQ_LAUNCH(1);break;case 2:TQ_LAUNCH(2);break;case 4:TQ_LAUNCH(4);break;case 8:TQ_LAUNCH(8);break;case 16:TQ_LAUNCH(16);break;case 32:TQ_LAUNCH(32);break;case 64:TQ_LAUNCH(64);break;case 128:TQ_LAUNCH(128);break;case 256:TQ_LAUNCH(256);break;case 512:TQ_LAUNCH(512);break;case 1024:TQ_LAUNCH(1024);break;}
    #undef TQ_LAUNCH
    #undef TQ_LAUNCH_H
    attention_output<<<b*h*t,std::max(32,w),0,mfq_get_current_cuda_stream()>>>(partial.data_ptr<float>(),stats.data_ptr<float>(),vt.signs.data_ptr<float>(),output.data_ptr<float>(),w,d,parts,h,t);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
}
