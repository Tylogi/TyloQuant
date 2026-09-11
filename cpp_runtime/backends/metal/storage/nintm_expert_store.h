#pragma once

#include "hf_safetensors_store.h"
#include "mfq_container.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::metal {

class MlxNintMxfp4Unsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Architecture-neutral exact-range view over three canonical routed MoE
// projections. It accepts any NIM2 layout whose experts are stored as native
// MXFP4 cohorts, including virtual views over untouched HF checkpoints.
class MlxNintMxfp4ExpertStore {
public:
    static constexpr std::size_t kParts = 6;

    MlxNintMxfp4ExpertStore(
        const MfqContainer& model,
        std::vector<std::string> layer_prefixes,
        std::vector<std::size_t> experts_per_layer,
        std::size_t hidden_size,
        std::size_t intermediate_size);

    std::size_t num_layers() const noexcept;
    std::size_t num_experts(std::size_t layer) const;
    std::size_t max_num_experts() const noexcept;
    std::size_t slot_bytes() const noexcept;

    MlxNativeMxfp4ExpertLoadStats load(
        std::size_t layer,
        std::size_t expert,
        std::span<std::byte> slot) const;
    MlxNativeMxfp4ExpertLoadStats load_scatter(
        std::size_t layer,
        std::size_t expert,
        const MlxNativeMxfp4ExpertDestination& destination) const;
    MlxNativeMxfp4ExpertLoadStats load_gate_up_scatter(
        std::size_t layer,
        std::size_t expert,
        const MlxNativeMxfp4ExpertDestination& destination) const;
    MlxNativeMxfp4ExpertLoadStats load_scales_scatter(
        std::size_t layer,
        std::size_t expert,
        const MlxNativeMxfp4ExpertDestination& destination) const;
    MlxNativeMxfp4ExpertLoadStats load_gate_scatter(
        std::size_t layer,
        std::size_t expert,
        const MlxNativeMxfp4ExpertDestination& destination) const;
    MlxNativeMxfp4ExpertLoadStats load_up_scatter(
        std::size_t layer,
        std::size_t expert,
        const MlxNativeMxfp4ExpertDestination& destination) const;
    MlxNativeMxfp4ExpertLoadStats load_down_scatter(
        std::size_t layer,
        std::size_t expert,
        const MlxNativeMxfp4ExpertDestination& destination) const;

    MlxNativeMxfp4ExpertView view(
        std::span<const std::byte> slot) const;

private:
    struct Part {
        std::string record;
        std::uint64_t offset = 0;
        std::uint64_t nbytes = 0;
    };
    struct ExpertRecord {
        std::array<Part, kParts> parts;
    };

    const ExpertRecord& expert_record(
        std::size_t layer,
        std::size_t expert) const;
    MlxNativeMxfp4ExpertLoadStats load_parts(
        const ExpertRecord& record,
        std::span<const std::size_t> parts,
        const MlxNativeMxfp4ExpertDestination& destination) const;

    MfqContainer model_;
    std::size_t num_layers_ = 0;
    std::size_t max_num_experts_ = 0;
    std::size_t slot_bytes_ = 0;
    std::array<std::size_t, kParts + 1> slot_offsets_{};
    std::vector<std::size_t> experts_per_layer_;
    std::vector<std::size_t> expert_offsets_;
    std::vector<ExpertRecord> experts_;
};

} // namespace mfq::metal
