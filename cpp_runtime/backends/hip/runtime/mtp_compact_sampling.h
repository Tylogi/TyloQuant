#pragma once
#include "mfq_hip_sampling_ops.h"
#include "mtp_policy.h"
#include "mfq/runtime.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <vector>

namespace mfq::cuda::mtp_sampling {
using Tensor=mfq_tensor_backend::Tensor;
using Distribution=mfq::engine::mtp::CompactDistribution;

inline bool use_packets(const Tensor& logits) {
    const auto* option=std::getenv("MFQ_MTP_RADIX_TOPK");
    if(option)return option[0]=='1' && logits.size(0)<=32;
    // Host-transfer savings dominate small verification windows. Larger
    // windows retain the measured faster generic selection implementation.
    return logits.size(0)<=8 || (logits.size(0)<=16 && logits.size(1)<=131072);
}

inline std::vector<Distribution> rows(Tensor logits,const MfqSamplingParams& parameters,bool packets) {
    MFQ_RUNTIME_CHECK(logits.dim()==2 && parameters.top_k>0 && parameters.top_k<=64 &&
        parameters.top_k<=logits.size(1) && parameters.temperature>0,
        "HIP compact MTP sampling geometry is invalid");
    logits=logits.contiguous();std::vector<Distribution> result;
    result.reserve(size_t(logits.size(0)));
    if(packets && logits.size(0)<=32) {
        const auto host=sample_top_k_compact_cuda(logits,parameters.top_k).cpu().contiguous();
        const auto* data=host.data_ptr<int64_t>();std::array<float,64> values;std::array<int64_t,64> indices;
        for(int64_t row=0;row<host.size(0);++row) {
            for(int index=0;index<parameters.top_k;++index) {
                const auto packet=uint64_t(data[row*host.size(1)+index]);
                const auto bits=uint32_t(packet>>32);std::memcpy(&values[index],&bits,sizeof(bits));
                indices[index]=uint32_t(packet);
            }
            result.push_back(mfq::engine::mtp::compact_distribution_from_topk(values.data(),indices.data(),
                int(parameters.top_k),parameters.temperature,parameters.top_p));
        }
    } else {
        auto selected=mfq_tensor_backend::topk(logits,parameters.top_k,-1,true,true);
        auto values=std::get<0>(selected).to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kFloat32).contiguous();
        auto indices=std::get<1>(selected).to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kInt64).contiguous();
        for(int64_t row=0;row<values.size(0);++row)
            result.push_back(mfq::engine::mtp::compact_distribution_from_topk(
                values.data_ptr<float>()+row*values.size(1),indices.data_ptr<int64_t>()+row*indices.size(1),
                int(values.size(1)),parameters.temperature,parameters.top_p));
    }
    return result;
}
} // namespace mfq::cuda::mtp_sampling
