#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <mlx/mlx.h>

namespace mfq::metal {
class MfqContainer;
class MlxTqExpertCache;
class MlxMappedTqRows;
struct MlxTqRoutePlan {
    mlx::core::array block_meta, block_count;
    int max_blocks, block_rows, route_count, experts;
};
std::optional<mlx::core::array> make_tq_reuse_plan(std::span<const int> ids,
    int tokens, int routes, int experts);
std::optional<mlx::core::array> make_tq_reuse_plan(const mlx::core::array& ids, int experts);

class MlxTqWeight {
public:
    static MlxTqWeight from_blob(std::span<const std::uint8_t> blob,
        int input_columns = 0, int experts = 1);
    static MlxTqWeight from_pool(std::span<const std::uint8_t> blob,
        const mlx::core::array& words, std::span<const std::uint64_t> expert_offsets,
        int input_columns = 0);
    static MlxTqWeight load(const MfqContainer& model, const std::string& name);
    bool can_concatenate_rows(const MlxTqWeight& other) const;
    static MlxTqWeight concatenate_projections(const std::vector<MlxTqWeight>& weights);
    static MlxTqWeight concatenate_experts(const std::vector<MlxTqWeight>& weights);
    static std::vector<mlx::core::array> matmul_many(
        const std::vector<MlxTqWeight>& weights, const mlx::core::array& input);
    MlxTqWeight expert_view(int expert) const;
    mlx::core::array matmul(const mlx::core::array& input) const;
    mlx::core::array rms_matmul(const mlx::core::array& input, float eps) const;
    mlx::core::array grouped_row_matmul(const mlx::core::array& input, int groups) const;
    bool can_fuse_gated_mix(const mlx::core::array& input) const;
    mlx::core::array gated_mix(const mlx::core::array& input,
        const mlx::core::array& streams, int groups) const;
    mlx::core::array routed_matmul(const mlx::core::array& input,
        const mlx::core::array& ids, const mlx::core::array* reuse = nullptr) const;
    mlx::core::array routed_matmul_sorted(const mlx::core::array& input,
        const mlx::core::array& ids, const mlx::core::array& order,
        const MlxTqRoutePlan& plan) const;
    mlx::core::array routed_matmul_reduce(const mlx::core::array& input,
        const mlx::core::array& ids, const mlx::core::array& weights,
        const mlx::core::array* reuse = nullptr) const;
    mlx::core::array swiglu(const MlxTqWeight& up, const mlx::core::array& input,
        const mlx::core::array& ids, float limit = 0.f, bool gelu = false,
        const mlx::core::array* reuse = nullptr) const;
    mlx::core::array swiglu_split(const mlx::core::array& input,
        const mlx::core::array& ids, float limit = 0.f, bool gelu = false) const;
    mlx::core::array embedding(const mlx::core::array& ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;
    int input_size() const noexcept { return input_columns_; }
    int output_size() const noexcept { return rows_ / experts_; }
    int experts() const noexcept { return experts_; }
    std::size_t packed_nbytes() const noexcept { return nbytes_; }

private:
    friend class MlxTqExpertCache;
    static MlxTqWeight from_blob_impl(std::span<const std::uint8_t> blob,
        int input_columns, int experts, const mlx::core::array* words,
        std::span<const std::uint64_t> expert_offsets);
    static MlxTqWeight from_expert_blob(std::span<const std::uint8_t> blob,
        int input_columns, int experts, const mlx::core::array* words,
        std::span<const std::uint64_t> expert_offsets);
    static MlxTqWeight from_expert_rows(const MlxMappedTqRows& rows, int experts,
        const mlx::core::array* words, std::span<const std::uint64_t> expert_offsets);
    void set_expert_word_offset(int expert, std::uint64_t word_offset);
    static std::vector<mlx::core::array> matmul_many_impl(
        std::span<const MlxTqWeight> weights, const mlx::core::array& input);
    bool use_column_inputs(const mlx::core::array& input, int tokens, int routes) const;
    mlx::core::array prepare_column_inputs(const mlx::core::array& input,
        const mlx::core::array& ids) const;
    std::vector<mlx::core::array> prepare_column_pair_inputs(const MlxTqWeight& up,
        const mlx::core::array& input, const mlx::core::array& ids) const;
    mlx::core::array project_column_inputs(const mlx::core::array& input,
        const mlx::core::array& ids, const MlxTqWeight* up = nullptr,
        float limit = 0.f, bool gelu = false) const;
    MlxTqWeight(mlx::core::array words, mlx::core::array descriptors,
        mlx::core::array scales, int rows, int width, int input_columns,
        int experts, int group_size, int columns, std::size_t bytes, bool pooled = false);
    mlx::core::array project(const mlx::core::array& input,
        const mlx::core::array& ids, const MlxTqWeight* up,
        float limit, bool gelu, int split = 0, const mlx::core::array* reuse = nullptr) const;
    static MlxTqWeight concatenate(const std::vector<MlxTqWeight>& weights, bool projections);
    mlx::core::array words_, descriptors_, scales_;
    int rows_, width_, input_columns_, experts_, group_size_, columns_;
    bool pooled_;
    std::size_t nbytes_;
};
} // namespace mfq::metal
