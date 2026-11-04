#include "pooled_cache.h"
#include "runtime/execution_options.h"

#include <algorithm>

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;

std::size_t PooledCache::growth_bytes(int64_t tokens,int64_t pool,int64_t width,std::size_t element_size) const {
    const auto required=(tokens+pool-1)/pool;
    if(values_.defined() && values_.size(1)>=required)return 0;
    auto capacity=values_.defined()?values_.size(1):int64_t(4);
    while(capacity<required)capacity*=2;
    return std::size_t(capacity)*width*element_size;
}

std::size_t PooledCache::position_bytes(int64_t capacity) const {
    if(positions_.defined() && positions_.size(-1)==capacity)return 0;
    const auto axes=positions_.defined()?positions_.numel()/positions_.size(-1):3;
    return std::size_t(axes)*capacity*sizeof(int64_t);
}

void PooledCache::reset() {
    // Keep positions_: retained decode graphs captured its address.
    values_={};
    processed_rows_={};
    completed_=0;
    processed_=0;
}

void PooledCache::truncate(int64_t tokens,int64_t pool) {
    MFQ_RUNTIME_CHECK(tokens>=0 && pool>0,"invalid pooled cache truncation");
    completed_=std::min(completed_,tokens/pool);
}

void PooledCache::bind(const Tensor& values) {
    MFQ_RUNTIME_CHECK(values.defined() && values.dim()==3 && values.size(1)>=completed_,
        "invalid pooled cache storage");
    if(values_.defined() && values_.data_ptr()==values.data_ptr())return;
    if(completed_)values.narrow(1,0,completed_).copy_(values_.narrow(1,0,completed_));
    values_=values;
}

void PooledCache::reserve(const Tensor& raw,int64_t capacity) {
    if(values_.defined() && values_.size(1)>=capacity)return;
    auto grown=values_.defined()?values_.size(1):int64_t(4);
    while(grown<capacity)grown*=2;
    const auto dtype=raw.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16;
    auto next=tb::zeros({raw.size(0),grown,raw.size(2)},raw.options().dtype(dtype));
    bind(next);
}

PooledCache::Tensor PooledCache::update_range(const RotaryEmbedding& rotary,const Tensor& raw,
        const Tensor& weight,const Tensor& full_positions,int64_t first,int64_t count,int64_t pool,double eps) {
    MFQ_RUNTIME_CHECK(pool>0 && first>=0 && count>0 && raw.dim()==3 && raw.size(1)==count*pool &&
        full_positions.size(-1)>=(first+count)*pool,"invalid pooled source range");
    auto starts=(tb::arange(count,full_positions.options().dtype(tb::kInt64))+first)*pool;
    auto positions=full_positions.index_select(-1,starts);
    const auto dtype=raw.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16;
    auto output=tb::empty({raw.size(0),count,raw.size(2)},raw.options().dtype(dtype));
    rotary.pool_groups_into(raw,weight,positions,output,pool,eps);
    return output;
}

PooledCache::Tensor PooledCache::update(const RotaryEmbedding& rotary,const Tensor& raw,
        const Tensor& weight,const Tensor& full_positions,int64_t tokens,int64_t pool,double eps,bool cache) {
    MFQ_RUNTIME_CHECK(pool>0 && tokens>=0 && raw.dim()==3 && tokens<=raw.size(1) &&
        tokens<=full_positions.size(-1),"invalid pooled cache update");
    const auto groups=tokens/pool;
    const auto first=cache && incremental_ && runtime_options::qsa_pool_incremental()?completed_:0;
    MFQ_RUNTIME_CHECK(first<=groups,"pooled watermark exceeds context");
    if(!cache) {
        if(!groups)return tb::empty({raw.size(0),0,raw.size(2)},raw.options());
        return update_range(rotary,raw.narrow(1,0,groups*pool),weight,full_positions,0,groups,pool,eps);
    }
    reserve(raw,(raw.size(1)+pool-1)/pool);
    if(groups>first) {
        auto starts=tb::arange(first,groups,full_positions.options().dtype(tb::kInt64))*pool;
        rotary.pool_groups_into(raw.narrow(1,first*pool,(groups-first)*pool),weight,
            full_positions.index_select(-1,starts),values_.narrow(1,first,groups-first),pool,eps,{},{},processed_rows_);
        completed_=groups;
        processed_+=groups-first;
    }
    return values_.narrow(1,0,groups);
}

void PooledCache::prepare_graph(const Tensor& full_positions,int64_t capacity) {
    MFQ_RUNTIME_CHECK(capacity>=full_positions.size(-1),"pooled position cache exceeds capacity");
    auto shape=full_positions.sizes().vec();
    shape.back()=capacity;
    // Graphs retained across requests read and write this buffer by address,
    // so a later request must refill it rather than allocate a new one.
    if(positions_.defined() && positions_.sizes().vec()==shape && positions_.device()==full_positions.device())
        positions_.zero_();
    else positions_=tb::zeros(shape,full_positions.options().dtype(tb::kInt64));
    positions_.narrow(-1,0,full_positions.size(-1)).copy_(full_positions);
}

void PooledCache::update_graph(const RotaryEmbedding& rotary,const Tensor& raw,const Tensor& weight,
        const Tensor& positions,const Tensor& cache_positions,int64_t pool,double eps) {
    positions_.index_copy_(-1,cache_positions,positions.to(tb::kInt64));
    rotary.pool_groups_into(raw,weight,positions.to(tb::kInt64),values_,pool,eps,
        cache_positions,positions_,processed_rows_);
}

void PooledCache::advance_graph(int64_t tokens,int64_t pool) {
    const auto groups=tokens/pool;
    processed_+=std::max<int64_t>(0,groups-completed_);
    completed_=groups;
}

PooledCacheState PooledCache::capture(int64_t tokens,int64_t pool) const {
    PooledCacheState state;
    state.completed=std::min(completed_,tokens/pool);
    if(state.completed)state.values=values_.narrow(1,0,state.completed).contiguous().clone();
    return state;
}

void PooledCache::restore(const PooledCacheState& state,int64_t tokens,int64_t pool) {
    MFQ_RUNTIME_CHECK(state.completed>=0 && state.completed<=tokens/pool &&
        (!state.completed || (state.values.defined() && state.values.dim()==3 &&
            state.values.size(1)==state.completed)),"invalid pooled session watermark");
    if(state.completed) {
        if(values_.defined())values_.narrow(1,0,state.completed).copy_(state.values);
        else values_=state.values.clone();
    }
    completed_=state.completed;
}

} // namespace mfq::cuda
