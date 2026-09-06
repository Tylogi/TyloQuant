#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq {

struct TensorParallelSlice {
    int device = 0;
    int64_t begin = 0;
    int64_t end = 0;

    int64_t size() const {
        return end - begin;
    }
};

inline std::vector<TensorParallelSlice> plan_tensor_parallel_slices(
        int64_t extent,
        int64_t granularity,
        const std::vector<int> & devices,
        const std::vector<double> & weights) {
    if (extent <= 0) {
        throw std::runtime_error("tensor-parallel extent must be positive");
    }
    if (granularity <= 0) {
        throw std::runtime_error(
            "tensor-parallel granularity must be positive");
    }
    if (devices.empty()) {
        throw std::runtime_error(
            "tensor-parallel device list must not be empty");
    }
    if (!weights.empty() && weights.size() != devices.size()) {
        throw std::runtime_error(
            "tensor-parallel split count must match device count");
    }
    if (extent < static_cast<int64_t>(devices.size()) * granularity) {
        throw std::runtime_error(
            "tensor-parallel extent is too small for the device count");
    }

    std::vector<double> normalized(
        devices.size(), 1.0 / static_cast<double>(devices.size()));
    if (!weights.empty()) {
        double sum = 0.0;
        for (double value : weights) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::runtime_error(
                    "tensor-parallel split weights must be finite and positive");
            }
            sum += value;
        }
        for (size_t index = 0; index < weights.size(); ++index) {
            normalized[index] = weights[index] / sum;
        }
    }

    const size_t count = devices.size();
    std::vector<int64_t> boundaries(count + 1, 0);
    boundaries.back() = extent;
    double cumulative = 0.0;
    for (size_t index = 1; index < count; ++index) {
        cumulative += normalized[index - 1];
        const double target = cumulative * static_cast<double>(extent);
        int64_t boundary =
            static_cast<int64_t>(std::llround(
                target / static_cast<double>(granularity))) *
            granularity;
        const int64_t minimum =
            boundaries[index - 1] + granularity;
        const int64_t maximum =
            extent -
            static_cast<int64_t>(count - index) * granularity;
        boundaries[index] = std::clamp(boundary, minimum, maximum);
    }

    std::vector<TensorParallelSlice> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        result.push_back(
            {devices[index], boundaries[index], boundaries[index + 1]});
    }
    return result;
}

inline void validate_tensor_parallel_slices(
        const std::vector<TensorParallelSlice> & slices,
        int64_t extent,
        int64_t granularity) {
    if (slices.empty() || slices.front().begin != 0 ||
        slices.back().end != extent) {
        throw std::runtime_error(
            "tensor-parallel slices do not cover the full extent");
    }
    for (size_t index = 0; index < slices.size(); ++index) {
        const auto & slice = slices[index];
        if (slice.begin >= slice.end) {
            throw std::runtime_error(
                "tensor-parallel slice must not be empty");
        }
        if (index > 0 && slices[index - 1].end != slice.begin) {
            throw std::runtime_error(
                "tensor-parallel slices must be contiguous");
        }
        if (index + 1 < slices.size() &&
            (slice.begin % granularity != 0 ||
             slice.end % granularity != 0)) {
            throw std::runtime_error(
                "interior tensor-parallel boundaries are misaligned");
        }
    }
}

}  // namespace mfq
