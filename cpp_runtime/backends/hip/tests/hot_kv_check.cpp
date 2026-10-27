#include "hot_kv_check.h"
#include "mfq_hip_kv_hot_cache.h"
#include "mfq_hip_kv_quantization.h"
#include "mfq_hip_context.h"
#include <hip/hip_runtime.h>
#include <map>
#include <memory>
#include <tuple>

namespace mfq::cuda::tests {
namespace tb=mfq_tensor_backend;
std::vector<tb::Tensor> hot_kv_check(const std::string& operation,
        const std::vector<tb::Tensor>& a,const nlohmann::json& p) {
    const bool exclusive=operation=="hot_kv_exclusive";
    MFQ_RUNTIME_CHECK((exclusive || operation=="hot_kv_cached") && a.size()>=4,"invalid hot KV test operation");
    const auto source=[&](const tb::Tensor& value) {
        const int64_t tail=p.value("source_tail",0);
        if(!tail)return value;
        auto padded=tb::zeros({value.size(0),value.size(2),value.size(1),value.size(3)+tail},value.options());
        auto view=padded.narrow(3,0,value.size(3)).permute({0,2,1,3});view.copy_(value);return view;
    };
    const auto k=source(a.at(0)),v=source(a.at(1));const auto& selected=a.at(2);const auto& query=a.at(3);
    const int64_t heads=k.size(1),maximum=k.size(2),kw=k.size(3),vw=v.size(3),cells=p.at("cells");
    const int64_t chunk=p.value("chunk",maximum),index_width=p.value("index_width",0);
    MFQ_RUNTIME_CHECK(chunk>0 && maximum>cells && cells>0,"invalid exclusive KV test capacity");
    const auto type=p.value("cast_half",false)?tb::kFloat16:k.scalar_type();
    const bool mapped=p.value("mapped",false);
    struct Buffers {HotKvCache hot;tb::Tensor logical_k,logical_v,positions,index;std::vector<tb::Tensor> hosts;};
    using Key=std::tuple<int,int64_t,int64_t,int64_t,int64_t,int64_t,int,int64_t,bool,bool>;
    static std::map<Key,std::shared_ptr<Buffers>> buffers;
    const Key key{k.get_device(),heads,maximum,cells,kw,vw,int(type),index_width,mapped,exclusive};
    auto& holder=buffers[key];
    if(!holder) {
        auto prepared=std::make_shared<Buffers>();auto& state=*prepared;
        const auto options=k.options().dtype(type);
        state.hot=HotKvCache::allocate(heads,maximum,cells,kw,vw,options);
        const int64_t physical=exclusive?maximum-cells:maximum;
        const auto cold=[&](int64_t width) {
            if(!mapped)return tb::empty({1,heads,physical,width},options);
            auto host=tb::empty({1,heads,physical,width},options.device(tb::kCPU).pinned_memory(true));
            void* alias=nullptr;MFQ_NATIVE_CUDA_CHECK(hipHostGetDevicePointer(&alias,host.data_ptr(),0));
            auto output=tb::from_blob(alias,{1,heads,physical,width},options);
            state.hosts.push_back(std::move(host));return output;
        };
        if(exclusive) {
            state.hot.set_cold_storage(cold(kw),cold(vw));
            // Only tier-aware kernels read these logical views. Physical cold
            // allocations and mapped-host owners stay alive in this buffer set.
            state.logical_k=tb::from_blob(state.hot.cold_keys.data_ptr(),{1,heads,maximum,kw},options);
            state.logical_v=tb::from_blob(state.hot.cold_values.data_ptr(),{1,heads,maximum,vw},options);
        } else {state.logical_k=cold(kw);state.logical_v=cold(vw);}
        state.positions=tb::arange(maximum,options.dtype(tb::kInt64));
        if(index_width)state.index=tb::empty({1,maximum,index_width},options.dtype(tb::kFloat16));
        holder=std::move(prepared);
    }
    auto& state=*holder;auto& hot=state.hot;hot.reset();
    if(index_width) {
        MFQ_RUNTIME_CHECK(a.size()>4 && a[4].dim()==3 && a[4].size(1)==maximum && a[4].size(2)==index_width,
            "invalid hot KV index fixture");state.index.zero_();
    }
    const auto append=[&](int64_t source,int64_t destination,int64_t count) {
        auto keys=k.narrow(2,source,count),values=v.narrow(2,source,count);
        auto positions=state.positions.narrow(0,destination,count);
        auto index=index_width?a[4].narrow(1,source,count):tb::Tensor{};
        if(!hot.append_decode(keys,values,positions,index,state.index)) {
            hot.append(keys.to(type),values.to(type),positions);
            if(index_width)state.index.index_copy_(1,positions,index.to(state.index.scalar_type()));
        }
    };
    if(exclusive)for(int64_t start=0;start<maximum;start+=chunk)append(start,start,std::min(chunk,maximum-start));
    else {
        state.logical_k.copy_(k.to(type));state.logical_v.copy_(v.to(type));
        if(index_width)state.index.copy_(a[4].to(state.index.scalar_type()));
        hot.publish(state.logical_k,state.logical_v,state.positions);
    }
    const int rounds=p.value("prepare_rounds",2);
    for(int i=0;i<rounds;++i)hot.prepare(state.logical_k,state.logical_v,selected);
    if(p.contains("rewrites"))for(const auto& rewrite:p.at("rewrites")) {
        const int64_t from=rewrite.at(1).get<int64_t>(),to=rewrite.at(0).get<int64_t>();
        if(exclusive)append(from,to,1);
        else {
            state.logical_k.narrow(2,to,1).copy_(k.narrow(2,from,1).to(type));
            state.logical_v.narrow(2,to,1).copy_(v.narrow(2,from,1).to(type));
            if(index_width)state.index.narrow(1,to,1).copy_(a[4].narrow(1,from,1).to(state.index.scalar_type()));
            hot.publish(state.logical_k,state.logical_v,state.positions.narrow(0,to,1));
        }
        hot.prepare(state.logical_k,state.logical_v,selected);
    }
    const int64_t invalidated=p.value("invalidate_prefix",0);
    if(invalidated)hot.invalidate(state.positions.narrow(0,0,invalidated));
    const auto gather_cached=[&](const tb::Tensor& canonical,const tb::Tensor& resident) {
        auto slots=tb::clamp(hot.mapping,0,cells-1).to(tb::kInt64);
        auto active=(hot.mapping>=0).reshape({1,1,maximum,1});
        return tb::where(active,resident.index_select(2,slots),canonical);
    };
    auto gathered=exclusive?hot.gather(maximum):std::pair{gather_cached(state.logical_k,hot.keys),gather_cached(state.logical_v,hot.values)};
    const double bits=p.value("bits",0.);
    auto attended=bits?cuda_kv_attention(query,state.logical_k,state.logical_v,KvQuantization{bits},selected,{},
        maximum-query.size(2),{},0,{},{},0,hot.view()):tb::zeros({1},query.options());
    const auto exact=[](tb::Tensor value){return value.scalar_type()==tb::kInt32?value.to(tb::kInt64):value;};
    auto accounting=tb::tensor(std::vector<int64_t>{int64_t(hot.bytes()),int64_t(exclusive?hot.cold_keys.nbytes()+hot.cold_values.nbytes():state.logical_k.nbytes()+state.logical_v.nbytes())});
    return {exact(gathered.first),exact(gathered.second),attended,hot.mapping.to(tb::kInt64),
        (exclusive?hot.cold_mapping:tb::full_like(hot.mapping,-1)).to(tb::kInt64),hot.slot_ids.to(tb::kInt64),hot.control,
        index_width?state.index:tb::zeros({1},query.options()),accounting};
}
}
