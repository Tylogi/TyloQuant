#pragma once
#include "mfq_tensor_backend.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
namespace mfq::cuda::tests {
std::vector<mfq_tensor_backend::Tensor> quant_kv_check(const std::string&,
    const std::vector<mfq_tensor_backend::Tensor>&,const nlohmann::json&);
}
