#pragma once
#include "mfq_flash_next_runtime.h"
#include <array>
#include <memory>
#include <numeric>

namespace mfq::flash_next {

// QSA uses the same rotate-half mapping for query/key/indexer and pooled keys.
// Positions describe logical token order independently of sparse pool IDs.
class Rotary {
public:
    Rotary(int64_t dimension,int64_t maximum,double base,std::vector<int64_t> sections={},bool interleaved=false)
        : dimension_(dimension),maximum_(maximum),base_(base),sections_(std::move(sections)),interleaved_(interleaved) {
        MFQ_RUNTIME_CHECK(dimension>0 && dimension%2==0 && maximum>0 && std::isfinite(base) && base>0,
            "invalid Qwen4 rotary configuration");
        MFQ_RUNTIME_CHECK((sections_.empty() && !interleaved_) ||
            (sections_.size()==3 && std::all_of(sections_.begin(),sections_.end(),[](auto n){return n>=0;}) &&
            std::accumulate(sections_.begin(),sections_.end(),int64_t(0))==dimension/2),
            "Qwen4 MRoPE sections disagree with rotary width");
    }
    Tensor forward(const Tensor& value,const Tensor& positions) const {
        MFQ_RUNTIME_CHECK(value.is_cuda() && value.dim()==4 && value.size(-1)>=dimension_ &&
            positions.is_cuda() && positions.device()==value.device() && positions.dim()>=1 && positions.dim()<=3 &&
            positions.size(-1)==value.size(2),"Qwen4 RoPE requires [B,H,T,D] with compatible positions");
        const auto b=value.size(0),t=value.size(2),pairs=dimension_/2;
        const auto axes=positions.dim()==1?1:positions.size(0);
        const auto batches=positions.dim()<3?1:positions.size(1);
        MFQ_RUNTIME_CHECK(axes>0 && (batches==1 || batches==b),"Qwen4 RoPE position batch mismatch");
        auto input=value.scalar_type()==tb::kFloat32?value:value.to(tb::kFloat16);
        auto f=input.to(tb::kFloat32);
        auto pair=tb::arange(pairs,value.options().dtype(tb::kInt64));
        auto frequencies=tb::pow(tb::full({pairs},base_,f.options()),-(pair.to(tb::kFloat32)*2)/double(dimension_));
        auto axis_ids=tb::zeros({pairs},pair.options());
        if (!sections_.empty()) {
            if (interleaved_) {
                auto remainder=pair.remainder(3);
                auto a1=(remainder==1) & (pair<sections_[1]*3);
                auto a2=(remainder==2) & (pair<sections_[2]*3);
                axis_ids=tb::where(a1,tb::full_like(pair,1),tb::where(a2,tb::full_like(pair,2),axis_ids));
            } else {
                axis_ids=tb::where(pair<sections_[0],axis_ids,
                    tb::where(pair<sections_[0]+sections_[1],tb::full_like(pair,1),tb::full_like(pair,2)));
            }
            axis_ids=tb::where(axis_ids<axes,axis_ids,tb::zeros_like(axis_ids));
        }
        Tensor cosine,sine;
        for (int64_t axis=0;axis<std::min<int64_t>(axes,3);++axis) {
            auto pos=(positions.dim()==1?positions:positions.select(0,axis)).reshape({batches,t})
                .to(tb::kInt32).clamp(0,maximum_-1).to(tb::kFloat32);
            auto angles=pos.unsqueeze(-1)*frequencies;
            auto ca=tb::cos(angles),sa=tb::sin(angles);
            if (axis==0) { cosine=ca; sine=sa; }
            else { cosine=tb::where(axis_ids==axis,ca,cosine); sine=tb::where(axis_ids==axis,sa,sine); }
        }
        cosine=cosine.unsqueeze(1);sine=sine.unsqueeze(1);
        auto first=f.narrow(-1,0,pairs),second=f.narrow(-1,pairs,pairs);
        return tb::cat({first*cosine-second*sine,second*cosine+first*sine,
            f.narrow(-1,dimension_,f.size(-1)-dimension_)},-1).to(input.scalar_type());
    }
private:
    int64_t dimension_,maximum_;
    double base_;
    std::vector<int64_t> sections_;
    bool interleaved_;
};

struct QsaWeights {
    Linear query,key,value,output,index_query_key;
    Tensor query_norm,key_norm,index_query_norm,index_key_norm;
};
struct QsaConfig {
    int64_t heads,kv_heads,width,index_heads,index_width,pool,budget,maximum;
    double eps;
};

class Qsa {
public:
    Qsa(QsaWeights weights,QsaConfig config,std::shared_ptr<Rotary> rotary)
        : w_(std::move(weights)),c_(config),rotary_(std::move(rotary)),
          keys_(config.maximum,config.kv_heads*config.width),
          values_(config.maximum,config.kv_heads*config.width),index_(config.maximum,config.index_width) {
        MFQ_RUNTIME_CHECK(c_.heads>0 && c_.kv_heads>0 && c_.heads%c_.kv_heads==0 &&
            c_.width>0 && c_.index_heads>0 && c_.pool>0 && c_.budget>0 && c_.budget%c_.pool==0 && rotary_,
            "invalid Qwen4 QSA configuration");
    }
    int64_t position() const {return keys_.position();}
    void reset() {keys_.reset();values_.reset();index_.reset();}
    void truncate(int64_t keep) {
        MFQ_RUNTIME_CHECK(keep>=0 && keep<=keys_.position() && keep<=values_.position() && keep<=index_.position(),
            "invalid Qwen4 QSA cache truncation");
        keys_.truncate(keep);values_.truncate(keep);index_.truncate(keep);
    }
    Tensor forward(const Tensor& hidden,const Tensor& current_positions,const Tensor& full_positions,bool use_cache,
        std::vector<Tensor>* selection_trace=nullptr) {
        MFQ_RUNTIME_CHECK(hidden.is_cuda() && hidden.dim()==3 && hidden.size(0)>0 && hidden.size(1)>0,
            "Qwen4 QSA requires nonempty [B,T,H] input");
        const auto b=hidden.size(0),t=hidden.size(1),offset=use_cache?position():0;
        MFQ_RUNTIME_CHECK(t<=c_.maximum-offset && full_positions.size(-1)==offset+t &&
            (!use_cache || (offset==values_.position() && offset==index_.position())),
            "Qwen4 QSA cache/position geometry mismatch");
        auto pair=w_.query(hidden).reshape({b,t,c_.heads,2*c_.width});
        auto gate=pair.narrow(-1,c_.width,c_.width);
        auto query=rms_norm(pair.narrow(-1,0,c_.width),w_.query_norm.to(tb::kFloat32)+1,c_.eps).permute({0,2,1,3});
        auto key=rms_norm(w_.key(hidden).reshape({b,t,c_.kv_heads,c_.width}),
            w_.key_norm.to(tb::kFloat32)+1,c_.eps).permute({0,2,1,3});
        auto value=w_.value(hidden).reshape({b,t,c_.kv_heads,c_.width});
        query=rotary_->forward(query,current_positions);
        key=rotary_->forward(key,current_positions).permute({0,2,1,3});
        auto iqk=w_.index_query_key(hidden);
        auto iq=rms_norm(iqk.narrow(-1,0,c_.index_heads*c_.index_width).reshape({b,t,c_.index_heads,c_.index_width}),
            w_.index_query_norm.to(tb::kFloat32)+1,c_.eps).permute({0,2,1,3});
        iq=rotary_->forward(iq,current_positions).permute({0,2,1,3});
        auto raw=iqk.narrow(-1,c_.index_heads*c_.index_width,c_.index_width);
        try {
            if (use_cache) {
                key=keys_.append(key.reshape({b,t,-1})).reshape({b,offset+t,c_.kv_heads,c_.width});
                value=values_.append(value.reshape({b,t,-1})).reshape({b,offset+t,c_.kv_heads,c_.width});
                raw=index_.append(raw);
            }
            key=key.permute({0,2,1,3});value=value.permute({0,2,1,3});
            Tensor attended;
            if (offset+t<=c_.budget) attended=mfq_flash_next::qwen4_dense_gqa_attention(query,key,value,offset);
            else {
                const auto pools=(offset+t)/c_.pool;
                auto pooled=raw.narrow(1,0,pools*c_.pool).reshape({b,pools,c_.pool,c_.index_width})
                    .to(tb::kFloat32).mean(2).to(raw.scalar_type());
                pooled=rms_norm(pooled,w_.index_key_norm.to(tb::kFloat32)+1,c_.eps);
                auto starts=tb::arange(pools,raw.options().dtype(tb::kInt64))*c_.pool;
                auto positions=full_positions.index_select(-1,starts);
                pooled=rotary_->forward(pooled.unsqueeze(1),positions).squeeze(1);
                auto scores=mfq_flash_next::qsa_block_scores(iq,pooled);
                auto ids=select_pooled_blocks(scores,offset,offset+t,c_.pool,c_.budget,true);
                if (selection_trace) *selection_trace={scores,ids,iq,pooled};
                attended=mfq_flash_next::qwen4_sparse_gqa_attention(query,key,value,ids);
            }
            auto gated=attended.to(tb::kFloat32)*tb::sigmoid(gate.to(tb::kFloat32));
            return w_.output(gated.reshape({b,t,c_.heads*c_.width}).to(hidden.scalar_type()));
        } catch (...) { if (use_cache) truncate(offset); throw; }
    }
private:
    QsaWeights w_;
    QsaConfig c_;
    std::shared_ptr<Rotary> rotary_;
    SequenceCache keys_,values_,index_;
};
} // namespace mfq::flash_next
