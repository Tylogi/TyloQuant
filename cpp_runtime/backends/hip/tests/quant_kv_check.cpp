#include "quant_kv_check.h"
#include "mfq_hip_kv_quantization.h"
#include "mfq_hip_context.h"
#include <map>
#include <tuple>
#include <stdexcept>
namespace mfq::cuda::tests {
std::vector<mfq_tensor_backend::Tensor> quant_kv_check(const std::string& op,
        const std::vector<mfq_tensor_backend::Tensor>& a,const nlohmann::json& p) {
    using namespace mfq_tensor_backend;
    if(op=="turbo_kv_encode") {
        auto input=a.at(0);
        if(p.value("bf16",false))input=input.to(kBFloat16);
        if(p.value("legacy_cast",false))input=input.to(kFloat32);
        auto packed=mfq::cuda::cuda_kv_encode(input,p.at("bits"),p.value("value",false));
        return {p.value("benchmark_only",false)?packed:packed.to(kInt64)};
    }
    if(op=="turbo_kv_decode") {
        const auto dtype=p.value("bf16",false)?kBFloat16:p.value("half",false)?kFloat16:kFloat32;
        auto output=mfq::cuda::cuda_kv_decode(a.at(0),p.at("dimension"),p.at("bits"),
            p.value("value",false),p.value("legacy_cast",false)?kFloat32:dtype);
        return {output.to(dtype)};
    }
    if(op=="turbo_kv_attention") {
        const auto tensor=[&](size_t index){return index<a.size()?a[index]:Tensor{};};
        const auto tailed=[&](const Tensor& input) {
            const auto tail=p.value("storage_tail",int64_t(0));
            if(!tail)return input;
            auto shape=input.sizes().vec();shape[2]+=tail;
            auto backing=empty(shape,input.options());auto view=backing.narrow(2,0,input.size(2));
            view.copy_(input);return view;
        };
        auto keys=tailed(a.at(1)),values=tailed(a.at(2));
        return {mfq::cuda::cuda_kv_attention(a.at(0),keys,values,
            mfq::cuda::KvQuantization{p.at("bits").get<double>()},tensor(3),tensor(4),
            p.value("offset",int64_t(0)),{},p.value("capacity",int64_t(0)),tensor(5),tensor(6),p.value("scale",0.))};
    }
    if(op=="turbo_kv_paged") {
        const auto tensor=[&](size_t index){return index<a.size()?a[index]:Tensor{};};
        const int chunks=p.at("chunks"),page_size=p.at("page_size"),chunk_pages=p.at("chunk_pages");
        MFQ_RUNTIME_CHECK(chunks>0 && page_size>0 && chunk_pages>0,"invalid paged KV test geometry");
        const int heads=int(a.at(1).size(1));
        struct Buffers {std::vector<Tensor> keys,values;Tensor kp,vp;};
        using Key=std::tuple<int,int,int,int,int,int64_t,int64_t>;
        static std::map<Key,Buffers> buffers;
        auto& cell=buffers[{a.at(0).get_device(),chunks,page_size,chunk_pages,heads,a.at(1).size(-1),a.at(2).size(-1)}];
        if(cell.keys.empty()) {
            std::vector<int64_t> kp,vp;
            for(int i=0;i<chunks;++i) {
                cell.keys.push_back(zeros({chunk_pages,heads,page_size,a.at(1).size(-1)},a.at(1).options()));
                cell.values.push_back(zeros({chunk_pages,heads,page_size,a.at(2).size(-1)},a.at(2).options()));
                kp.push_back(reinterpret_cast<int64_t>(cell.keys.back().data_ptr()));
                vp.push_back(reinterpret_cast<int64_t>(cell.values.back().data_ptr()));
            }
            cell.kp=mfq_tensor_backend::tensor(kp).to(a.at(0).device());
            cell.vp=mfq_tensor_backend::tensor(vp).to(a.at(0).device());
        }
        for(auto& value:cell.keys)value.zero_();for(auto& value:cell.values)value.zero_();
        mfq::cuda::QuantizedKvPages pages{cell.kp,cell.vp,a.at(7),page_size,chunk_pages,heads};
        auto positions=tensor(8);
        if(!positions.defined())positions=arange(a.at(1).size(2),a.at(0).options().dtype(kInt64));
        mfq::cuda::cuda_kv_write_paged(pages,a.at(1),a.at(2),positions);
        return {mfq::cuda::cuda_kv_attention(a.at(0),{},{},mfq::cuda::KvQuantization{p.at("bits").get<double>()},
            tensor(3),tensor(4),p.value("offset",int64_t(0)),pages,p.value("capacity",int64_t(0)),
            tensor(5),tensor(6),p.value("scale",0.))};
    }
    throw std::runtime_error("Unknown compressed KV test operation: "+op);
}
}
