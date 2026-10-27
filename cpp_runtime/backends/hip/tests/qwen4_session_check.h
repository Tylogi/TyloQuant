#pragma once
#include "mfq_tensor_backend.h"
#include <nlohmann/json.hpp>
#include <vector>
namespace mfq::cuda::tests {
std::vector<Tensor> qwen4_session_check(const nlohmann::json&);
}
