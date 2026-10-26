#pragma once
#include "mfq_tensor_backend.h"
#include <nlohmann/json.hpp>
#include <vector>
namespace mfq::cuda::tests {
std::vector<Tensor> qwen4_predictor_graph_check(const std::vector<Tensor>&,const nlohmann::json&);
}
