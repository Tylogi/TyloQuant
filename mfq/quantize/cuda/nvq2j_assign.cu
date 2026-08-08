#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

constexpr int kStates = 16;
constexpr int kBanks = 4;
constexpr int kStatesPerBank = kStates / kBanks;
constexpr int kVector = 8;
constexpr int kVectorsPerGroup = 3;
constexpr int kGroup = 24;
constexpr int kAssignThreads = 96;
constexpr int kBankSearchThreads = kBanks * kVectorsPerGroup * 32;
constexpr int kSearchStepsPerBlock = 5;
constexpr int kParallelSearchThreads =
    kSearchStepsPerBlock * kVectorsPerGroup * 32;
constexpr int kSearchReduceThreads = 256;
constexpr int kRefitThreads = 256;
constexpr int kRefitWarps = kRefitThreads / 32;

__device__ __forceinline__ bool better(
    float error,
    int index,
    float best_error,
    int best_index) {
    return error < best_error || (error == best_error && index < best_index);
}

template <int Entries>
__device__ __forceinline__ int code_offset(
    int bank,
    int coordinate,
    int entry) {
    // [bank, coordinate, entry] makes all lanes read adjacent entries.
    return (bank * kVector + coordinate) * Entries + entry;
}

template <int Entries, typename IndexType>
__global__ void nvq2j_assign_kernel(
    const float* __restrict__ value,
    const float* __restrict__ objective_weight,
    const float* __restrict__ anchor,
    const float* __restrict__ scale_lut,
    const uint8_t* __restrict__ bank_for_state,
    const int8_t* __restrict__ codebooks,
    uint8_t* __restrict__ out_state,
    IndexType* __restrict__ out_indices,
    int padded_width,
    int valid_width,
    int groups_per_row) {
    const int flat_group = blockIdx.x;
    const int row = flat_group / groups_per_row;
    const int group = flat_group - row * groups_per_row;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int first_position = group * kGroup;
    const int valid_group = min(kGroup, valid_width - first_position);

    __shared__ float shared_value[kGroup];
    __shared__ float shared_weight[kGroup];
    __shared__ float vector_error[kStates][kVectorsPerGroup];
    __shared__ int vector_index[kStates][kVectorsPerGroup];
    __shared__ float best_error;
    __shared__ int best_state;
    __shared__ int best_indices[kVectorsPerGroup];

    if (tid < kGroup) {
        if (tid < valid_group) {
            const int64_t offset =
                static_cast<int64_t>(row) * padded_width
                + first_position
                + tid;
            shared_value[tid] = value[offset];
            shared_weight[tid] = objective_weight[offset];
        } else {
            shared_value[tid] = 0.0f;
            shared_weight[tid] = 0.0f;
        }
    }
    if (tid == 0) {
        best_error = FLT_MAX;
        best_state = 0;
#pragma unroll
        for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
            best_indices[vector] = 0;
        }
    }
    __syncthreads();

    const float row_anchor = anchor[row];
    for (int bank = 0; bank < kBanks; ++bank) {
        int states_for_bank[kStates / kBanks];
        int state_count = 0;
#pragma unroll
        for (int state = 0; state < kStates; ++state) {
            if (bank_for_state[state] == bank) {
                if (state_count < kStates / kBanks) {
                    states_for_bank[state_count] = state;
                }
                ++state_count;
            }
        }
        int valid_vector = valid_group - warp * kVector;
        valid_vector = max(0, min(kVector, valid_vector));
        float local_error[kStates / kBanks];
        int local_index[kStates / kBanks];
#pragma unroll
        for (int rank = 0; rank < kStates / kBanks; ++rank) {
            local_error[rank] = FLT_MAX;
            local_index[rank] = 0;
        }

        for (int entry = lane; entry < Entries; entry += 32) {
            float signal = 0.0f;
            float dot = 0.0f;
            float norm = 0.0f;
#pragma unroll
            for (int coordinate = 0; coordinate < kVector; ++coordinate) {
                if (coordinate < valid_vector) {
                    const float code = static_cast<float>(
                        codebooks[code_offset<Entries>(
                            bank, coordinate, entry)]);
                    const float source =
                        shared_value[warp * kVector + coordinate];
                    const float weight =
                        shared_weight[warp * kVector + coordinate];
                    signal = fmaf(weight * source, source, signal);
                    dot = fmaf(weight * source, code, dot);
                    norm = fmaf(weight * code, code, norm);
                }
            }
#pragma unroll
            for (int rank = 0; rank < kStates / kBanks; ++rank) {
                const int state = states_for_bank[rank];
                const float scale = row_anchor * scale_lut[state];
                const float error = fmaf(
                    scale * scale,
                    norm,
                    fmaf(-2.0f * scale, dot, signal));
                if (better(
                        error,
                        entry,
                        local_error[rank],
                        local_index[rank])) {
                    local_error[rank] = error;
                    local_index[rank] = entry;
                }
            }
        }
#pragma unroll
        for (int rank = 0; rank < kStates / kBanks; ++rank) {
            for (int shift = 16; shift > 0; shift >>= 1) {
                const float other_error =
                    __shfl_down_sync(
                        0xffffffff, local_error[rank], shift);
                const int other_index =
                    __shfl_down_sync(
                        0xffffffff, local_index[rank], shift);
                if (lane < shift
                        && better(
                            other_error,
                            other_index,
                            local_error[rank],
                            local_index[rank])) {
                    local_error[rank] = other_error;
                    local_index[rank] = other_index;
                }
            }
            if (lane == 0) {
                const int state = states_for_bank[rank];
                vector_error[state][warp] = local_error[rank];
                vector_index[state][warp] = local_index[rank];
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
#pragma unroll
        for (int state = 0; state < kStates; ++state) {
            const float group_error =
                vector_error[state][0]
                + vector_error[state][1]
                + vector_error[state][2];
            if (better(group_error, state, best_error, best_state)) {
                best_error = group_error;
                best_state = state;
#pragma unroll
                for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
                    best_indices[vector] = vector_index[state][vector];
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        out_state[flat_group] = static_cast<uint8_t>(best_state);
#pragma unroll
        for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
            out_indices[
                static_cast<int64_t>(flat_group) * kVectorsPerGroup + vector] =
                static_cast<IndexType>(best_indices[vector]);
        }
    }
}

template <int Entries, typename IndexType>
__global__ void nvq2j_refit_anchor_kernel(
    const float* __restrict__ value,
    const float* __restrict__ objective_weight,
    const float* __restrict__ previous_anchor,
    const float* __restrict__ scale_lut,
    const uint8_t* __restrict__ bank_for_state,
    const int8_t* __restrict__ codebooks,
    const uint8_t* __restrict__ states,
    const IndexType* __restrict__ indices,
    float* __restrict__ out_anchor,
    int padded_width,
    int valid_width,
    int groups_per_row) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float numerator = 0.0f;
    float denominator = 0.0f;

    for (int position = tid; position < valid_width; position += blockDim.x) {
        const int group = position / kGroup;
        const int in_group = position - group * kGroup;
        const int vector = in_group / kVector;
        const int coordinate = in_group - vector * kVector;
        const int flat_group = row * groups_per_row + group;
        const int state = states[flat_group];
        const int bank = bank_for_state[state];
        const int entry =
            indices[
                static_cast<int64_t>(flat_group) * kVectorsPerGroup + vector];
        const float code = static_cast<float>(
            codebooks[code_offset<Entries>(bank, coordinate, entry)]);
        const float basis = scale_lut[state] * code;
        const float x =
            value[static_cast<int64_t>(row) * padded_width + position];
        const float weight =
            objective_weight[
                static_cast<int64_t>(row) * padded_width + position];
        numerator = fmaf(weight * x, basis, numerator);
        denominator = fmaf(weight * basis, basis, denominator);
    }

    for (int shift = 16; shift > 0; shift >>= 1) {
        numerator += __shfl_down_sync(0xffffffff, numerator, shift);
        denominator += __shfl_down_sync(0xffffffff, denominator, shift);
    }
    __shared__ float warp_numerator[kRefitWarps];
    __shared__ float warp_denominator[kRefitWarps];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_numerator[warp] = numerator;
        warp_denominator[warp] = denominator;
    }
    __syncthreads();
    if (warp == 0) {
        numerator = lane < kRefitWarps ? warp_numerator[lane] : 0.0f;
        denominator = lane < kRefitWarps ? warp_denominator[lane] : 0.0f;
        for (int shift = 16; shift > 0; shift >>= 1) {
            numerator += __shfl_down_sync(0xffffffff, numerator, shift);
            denominator += __shfl_down_sync(0xffffffff, denominator, shift);
        }
        if (lane == 0) {
            float fitted = denominator > 0.0f
                ? fmaxf(numerator / denominator, 0.0f)
                : previous_anchor[row];
            out_anchor[row] = __half2float(__float2half_rn(fitted));
        }
    }
}

template <int Entries>
__global__ void nvq2j_search_banks_kernel(
    const float* __restrict__ xgroup,
    const float* __restrict__ wgroup,
    const int8_t* __restrict__ codebooks,
    const float* __restrict__ bank_qmax,
    float* __restrict__ out_scale,
    int32_t* __restrict__ out_indices,
    int groups_per_row,
    int valid_last,
    int search_steps) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int bank = warp / kVectorsPerGroup;
    const int vector = warp - bank * kVectorsPerGroup;
    int valid_group = kGroup;
    if (group % groups_per_row == groups_per_row - 1) {
        valid_group = valid_last;
    }
    int valid_vector = valid_group - vector * kVector;
    valid_vector = max(0, min(kVector, valid_vector));
    const float* values =
        xgroup + static_cast<int64_t>(group) * kGroup;
    const float* weights =
        wgroup + static_cast<int64_t>(group) * kGroup;

    __shared__ int selected[kBanks][kVectorsPerGroup];
    __shared__ int best_indices[kBanks][kVectorsPerGroup];
    __shared__ float current_scale[kBanks];
    __shared__ float best_scale[kBanks];
    __shared__ float best_error[kBanks];
    __shared__ float max_abs;

    if (tid == 0) {
        float value = 0.0f;
        for (int position = 0; position < valid_group; ++position) {
            value = fmaxf(value, fabsf(values[position]));
        }
        max_abs = value;
    }
    if (vector == 0 && lane == 0) {
        best_scale[bank] = 0.0f;
        best_error[bank] = FLT_MAX;
    }
    __syncthreads();

    for (int step = 0; step < search_steps; ++step) {
        if (vector == 0 && lane == 0) {
            const float qmax = bank_qmax[bank];
            const float offset = search_steps == 1
                ? -0.12f * qmax
                : -0.12f * qmax
                    + static_cast<float>(step)
                        * (0.24f * qmax / (search_steps - 1));
            current_scale[bank] =
                max_abs > 0.0f ? max_abs / (qmax + offset) : 0.0f;
        }
        __syncthreads();

        float local_error = FLT_MAX;
        int local_index = 0;
        const float first_scale = current_scale[bank];
        for (int entry = lane; entry < Entries; entry += 32) {
            float dot = 0.0f;
            float norm = 0.0f;
#pragma unroll
            for (int coordinate = 0; coordinate < kVector; ++coordinate) {
                if (coordinate < valid_vector) {
                    const float code = static_cast<float>(
                        codebooks[
                            (bank * Entries + entry) * kVector
                            + coordinate]);
                    const float source =
                        values[vector * kVector + coordinate];
                    const float weight =
                        weights[vector * kVector + coordinate];
                    dot = fmaf(weight * source, code, dot);
                    norm = fmaf(weight * code, code, norm);
                }
            }
            const float error = fmaf(
                first_scale * first_scale,
                norm,
                -2.0f * first_scale * dot);
            if (better(error, entry, local_error, local_index)) {
                local_error = error;
                local_index = entry;
            }
        }
        for (int shift = 16; shift > 0; shift >>= 1) {
            const float other_error =
                __shfl_down_sync(0xffffffff, local_error, shift);
            const int other_index =
                __shfl_down_sync(0xffffffff, local_index, shift);
            if (lane < shift
                    && better(
                        other_error,
                        other_index,
                        local_error,
                        local_index)) {
                local_error = other_error;
                local_index = other_index;
            }
        }
        if (lane == 0) {
            selected[bank][vector] = local_index;
        }
        __syncthreads();

        if (vector == 0 && lane == 0) {
            float numerator = 0.0f;
            float denominator = 0.0f;
            for (int position = 0; position < valid_group; ++position) {
                const int selected_vector = position / kVector;
                const int coordinate = position % kVector;
                const int entry = selected[bank][selected_vector];
                const float code = static_cast<float>(
                    codebooks[
                        (bank * Entries + entry) * kVector
                        + coordinate]);
                const float weight = weights[position];
                numerator = fmaf(
                    weight * values[position], code, numerator);
                denominator = fmaf(
                    weight * code, code, denominator);
            }
            current_scale[bank] = denominator > 0.0f
                ? fmaxf(numerator / denominator, 0.0f)
                : 0.0f;
        }
        __syncthreads();

        local_error = FLT_MAX;
        local_index = 0;
        const float second_scale = current_scale[bank];
        for (int entry = lane; entry < Entries; entry += 32) {
            float dot = 0.0f;
            float norm = 0.0f;
#pragma unroll
            for (int coordinate = 0; coordinate < kVector; ++coordinate) {
                if (coordinate < valid_vector) {
                    const float code = static_cast<float>(
                        codebooks[
                            (bank * Entries + entry) * kVector
                            + coordinate]);
                    const float source =
                        values[vector * kVector + coordinate];
                    const float weight =
                        weights[vector * kVector + coordinate];
                    dot = fmaf(weight * source, code, dot);
                    norm = fmaf(weight * code, code, norm);
                }
            }
            const float error = fmaf(
                second_scale * second_scale,
                norm,
                -2.0f * second_scale * dot);
            if (better(error, entry, local_error, local_index)) {
                local_error = error;
                local_index = entry;
            }
        }
        for (int shift = 16; shift > 0; shift >>= 1) {
            const float other_error =
                __shfl_down_sync(0xffffffff, local_error, shift);
            const int other_index =
                __shfl_down_sync(0xffffffff, local_index, shift);
            if (lane < shift
                    && better(
                        other_error,
                        other_index,
                        local_error,
                        local_index)) {
                local_error = other_error;
                local_index = other_index;
            }
        }
        if (lane == 0) {
            selected[bank][vector] = local_index;
        }
        __syncthreads();

        if (vector == 0 && lane == 0) {
            float numerator = 0.0f;
            float denominator = 0.0f;
            for (int position = 0; position < valid_group; ++position) {
                const int selected_vector = position / kVector;
                const int coordinate = position % kVector;
                const int entry = selected[bank][selected_vector];
                const float code = static_cast<float>(
                    codebooks[
                        (bank * Entries + entry) * kVector
                        + coordinate]);
                const float weight = weights[position];
                numerator = fmaf(
                    weight * values[position], code, numerator);
                denominator = fmaf(
                    weight * code, code, denominator);
            }
            const float fitted = denominator > 0.0f
                ? fmaxf(numerator / denominator, 0.0f)
                : 0.0f;
            current_scale[bank] = fitted;
            float error = 0.0f;
            for (int position = 0; position < valid_group; ++position) {
                const int selected_vector = position / kVector;
                const int coordinate = position % kVector;
                const int entry = selected[bank][selected_vector];
                const float code = static_cast<float>(
                    codebooks[
                        (bank * Entries + entry) * kVector
                        + coordinate]);
                const float residual =
                    fitted * code - values[position];
                error = fmaf(
                    weights[position] * residual, residual, error);
            }
            if (error < best_error[bank]) {
                best_error[bank] = error;
                best_scale[bank] = fitted;
#pragma unroll
                for (
                    int selected_vector = 0;
                    selected_vector < kVectorsPerGroup;
                    ++selected_vector) {
                    best_indices[bank][selected_vector] =
                        selected[bank][selected_vector];
                }
            }
        }
        __syncthreads();
    }

    if (vector == 0 && lane == 0) {
        out_scale[
            static_cast<int64_t>(group) * kBanks + bank] =
            best_scale[bank];
#pragma unroll
        for (
            int selected_vector = 0;
            selected_vector < kVectorsPerGroup;
            ++selected_vector) {
            out_indices[
                (static_cast<int64_t>(group) * kBanks + bank)
                    * kVectorsPerGroup
                + selected_vector] =
                best_indices[bank][selected_vector];
        }
    }
}

template <int Entries>
__global__ void nvq2j_reassign_banks_kernel(
    const float* __restrict__ xgroup,
    const float* __restrict__ wgroup,
    const float* __restrict__ scales,
    const int8_t* __restrict__ codebooks,
    int32_t* __restrict__ out_indices,
    int groups_per_row,
    int valid_last) {
    const int group = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int bank = warp / kVectorsPerGroup;
    const int vector = warp - bank * kVectorsPerGroup;
    int valid_group = kGroup;
    if (group % groups_per_row == groups_per_row - 1) {
        valid_group = valid_last;
    }
    int valid_vector = valid_group - vector * kVector;
    valid_vector = max(0, min(kVector, valid_vector));
    const float scale =
        scales[static_cast<int64_t>(group) * kBanks + bank];
    const float* values =
        xgroup + static_cast<int64_t>(group) * kGroup
        + vector * kVector;
    const float* weights =
        wgroup + static_cast<int64_t>(group) * kGroup
        + vector * kVector;
    float local_error = FLT_MAX;
    int local_index = 0;
    for (int entry = lane; entry < Entries; entry += 32) {
        float dot = 0.0f;
        float norm = 0.0f;
#pragma unroll
        for (int coordinate = 0; coordinate < kVector; ++coordinate) {
            if (coordinate < valid_vector) {
                const float code = static_cast<float>(
                    codebooks[
                        (bank * Entries + entry) * kVector
                        + coordinate]);
                const float source = values[coordinate];
                const float weight = weights[coordinate];
                dot = fmaf(weight * source, code, dot);
                norm = fmaf(weight * code, code, norm);
            }
        }
        const float error = fmaf(
            scale * scale, norm, -2.0f * scale * dot);
        if (better(error, entry, local_error, local_index)) {
            local_error = error;
            local_index = entry;
        }
    }
    for (int shift = 16; shift > 0; shift >>= 1) {
        const float other_error =
            __shfl_down_sync(0xffffffff, local_error, shift);
        const int other_index =
            __shfl_down_sync(0xffffffff, local_index, shift);
        if (lane < shift
                && better(
                    other_error,
                    other_index,
                    local_error,
                    local_index)) {
            local_error = other_error;
            local_index = other_index;
        }
    }
    if (lane == 0) {
        out_indices[
            (static_cast<int64_t>(group) * kBanks + bank)
                * kVectorsPerGroup
            + vector] = local_index;
    }
}

template <int Entries>
__device__ __forceinline__ int nvq2j_search_one_vector(
    const float* __restrict__ values,
    const float* __restrict__ weights,
    const int8_t* __restrict__ codebook,
    float scale,
    int vector,
    int valid_vector,
    int lane) {
    float local_error = FLT_MAX;
    int local_index = 0;
    for (int entry = lane; entry < Entries; entry += 32) {
        float dot = 0.0f;
        float norm = 0.0f;
#pragma unroll
        for (int coordinate = 0; coordinate < kVector; ++coordinate) {
            if (coordinate < valid_vector) {
                const float code = static_cast<float>(
                    codebook[entry * kVector + coordinate]);
                const float source =
                    values[vector * kVector + coordinate];
                const float weight =
                    weights[vector * kVector + coordinate];
                dot = fmaf(weight * source, code, dot);
                norm = fmaf(weight * code, code, norm);
            }
        }
        const float error = fmaf(
            scale * scale, norm, -2.0f * scale * dot);
        if (better(error, entry, local_error, local_index)) {
            local_error = error;
            local_index = entry;
        }
    }
    for (int shift = 16; shift > 0; shift >>= 1) {
        const float other_error =
            __shfl_down_sync(0xffffffff, local_error, shift);
        const int other_index =
            __shfl_down_sync(0xffffffff, local_index, shift);
        if (lane < shift
                && better(
                    other_error,
                    other_index,
                    local_error,
                    local_index)) {
            local_error = other_error;
            local_index = other_index;
        }
    }
    return local_index;
}

template <int Entries>
__global__ void nvq2j_search_bank_chunks_kernel(
    const float* __restrict__ xgroup,
    const float* __restrict__ wgroup,
    const int8_t* __restrict__ codebooks,
    const float* __restrict__ bank_qmax,
    float* __restrict__ chunk_error,
    float* __restrict__ chunk_scale,
    int32_t* __restrict__ chunk_indices,
    int groups_per_row,
    int valid_last,
    int search_steps,
    int chunks_per_bank) {
    const int64_t flat_chunk = blockIdx.x;
    const int chunk = flat_chunk % chunks_per_bank;
    const int64_t group_bank = flat_chunk / chunks_per_bank;
    const int bank = group_bank % kBanks;
    const int group = group_bank / kBanks;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int local_step = warp / kVectorsPerGroup;
    const int vector = warp - local_step * kVectorsPerGroup;
    const int search_step =
        chunk * kSearchStepsPerBlock + local_step;
    const bool active = search_step < search_steps;
    int valid_group = kGroup;
    if (group % groups_per_row == groups_per_row - 1) {
        valid_group = valid_last;
    }
    int valid_vector = valid_group - vector * kVector;
    valid_vector = max(0, min(kVector, valid_vector));
    const float* values =
        xgroup + static_cast<int64_t>(group) * kGroup;
    const float* weights =
        wgroup + static_cast<int64_t>(group) * kGroup;

    __shared__ int8_t shared_codebook[Entries * kVector];
    __shared__ int selected[kSearchStepsPerBlock][kVectorsPerGroup];
    __shared__ float current_scale[kSearchStepsPerBlock];
    __shared__ float candidate_error[kSearchStepsPerBlock];
    __shared__ float candidate_scale[kSearchStepsPerBlock];
    __shared__ int candidate_indices[
        kSearchStepsPerBlock][kVectorsPerGroup];
    __shared__ float max_abs;

    const int8_t* source_codebook =
        codebooks + static_cast<int64_t>(bank) * Entries * kVector;
    for (
        int offset = tid;
        offset < Entries * kVector;
        offset += blockDim.x) {
        shared_codebook[offset] = source_codebook[offset];
    }
    if (tid == 0) {
        float value = 0.0f;
        for (int position = 0; position < valid_group; ++position) {
            value = fmaxf(value, fabsf(values[position]));
        }
        max_abs = value;
    }
    __syncthreads();

    if (vector == 0 && lane == 0 && active) {
        const float qmax = bank_qmax[bank];
        const float offset = search_steps == 1
            ? -0.12f * qmax
            : -0.12f * qmax
                + static_cast<float>(search_step)
                    * (0.24f * qmax / (search_steps - 1));
        current_scale[local_step] =
            max_abs > 0.0f ? max_abs / (qmax + offset) : 0.0f;
    }
    __syncthreads();

    if (active) {
        const int index = nvq2j_search_one_vector<Entries>(
            values,
            weights,
            shared_codebook,
            current_scale[local_step],
            vector,
            valid_vector,
            lane);
        if (lane == 0) {
            selected[local_step][vector] = index;
        }
    }
    __syncthreads();

    if (vector == 0 && lane == 0 && active) {
        float numerator = 0.0f;
        float denominator = 0.0f;
        for (int position = 0; position < valid_group; ++position) {
            const int selected_vector = position / kVector;
            const int coordinate = position % kVector;
            const int entry =
                selected[local_step][selected_vector];
            const float code = static_cast<float>(
                shared_codebook[entry * kVector + coordinate]);
            const float weight = weights[position];
            numerator = fmaf(
                weight * values[position], code, numerator);
            denominator = fmaf(
                weight * code, code, denominator);
        }
        current_scale[local_step] = denominator > 0.0f
            ? fmaxf(numerator / denominator, 0.0f)
            : 0.0f;
    }
    __syncthreads();

    if (active) {
        const int index = nvq2j_search_one_vector<Entries>(
            values,
            weights,
            shared_codebook,
            current_scale[local_step],
            vector,
            valid_vector,
            lane);
        if (lane == 0) {
            selected[local_step][vector] = index;
        }
    }
    __syncthreads();

    if (vector == 0 && lane == 0 && active) {
        float numerator = 0.0f;
        float denominator = 0.0f;
        for (int position = 0; position < valid_group; ++position) {
            const int selected_vector = position / kVector;
            const int coordinate = position % kVector;
            const int entry =
                selected[local_step][selected_vector];
            const float code = static_cast<float>(
                shared_codebook[entry * kVector + coordinate]);
            const float weight = weights[position];
            numerator = fmaf(
                weight * values[position], code, numerator);
            denominator = fmaf(
                weight * code, code, denominator);
        }
        const float fitted = denominator > 0.0f
            ? fmaxf(numerator / denominator, 0.0f)
            : 0.0f;
        float error = 0.0f;
        for (int position = 0; position < valid_group; ++position) {
            const int selected_vector = position / kVector;
            const int coordinate = position % kVector;
            const int entry =
                selected[local_step][selected_vector];
            const float code = static_cast<float>(
                shared_codebook[entry * kVector + coordinate]);
            const float residual =
                fitted * code - values[position];
            error = fmaf(
                weights[position] * residual, residual, error);
        }
        candidate_error[local_step] = error;
        candidate_scale[local_step] = fitted;
#pragma unroll
        for (
            int selected_vector = 0;
            selected_vector < kVectorsPerGroup;
            ++selected_vector) {
            candidate_indices[local_step][selected_vector] =
                selected[local_step][selected_vector];
        }
    }
    __syncthreads();

    if (tid == 0) {
        float best_chunk_error = FLT_MAX;
        float best_chunk_scale = 0.0f;
        int best_chunk_indices[kVectorsPerGroup] = {0, 0, 0};
#pragma unroll
        for (
            int candidate = 0;
            candidate < kSearchStepsPerBlock;
            ++candidate) {
            const int global_step =
                chunk * kSearchStepsPerBlock + candidate;
            if (global_step < search_steps
                    && candidate_error[candidate] < best_chunk_error) {
                best_chunk_error = candidate_error[candidate];
                best_chunk_scale = candidate_scale[candidate];
#pragma unroll
                for (
                    int selected_vector = 0;
                    selected_vector < kVectorsPerGroup;
                    ++selected_vector) {
                    best_chunk_indices[selected_vector] =
                        candidate_indices[candidate][selected_vector];
                }
            }
        }
        chunk_error[flat_chunk] = best_chunk_error;
        chunk_scale[flat_chunk] = best_chunk_scale;
#pragma unroll
        for (
            int selected_vector = 0;
            selected_vector < kVectorsPerGroup;
            ++selected_vector) {
            chunk_indices[
                flat_chunk * kVectorsPerGroup + selected_vector] =
                best_chunk_indices[selected_vector];
        }
    }
}

__global__ void nvq2j_reduce_bank_chunks_kernel(
    const float* __restrict__ chunk_error,
    const float* __restrict__ chunk_scale,
    const int32_t* __restrict__ chunk_indices,
    float* __restrict__ out_scale,
    int32_t* __restrict__ out_indices,
    int64_t group_banks,
    int chunks_per_bank) {
    const int64_t group_bank =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group_bank >= group_banks) {
        return;
    }
    const int64_t first_chunk = group_bank * chunks_per_bank;
    float best_error = FLT_MAX;
    float best_scale = 0.0f;
    int best_indices[kVectorsPerGroup] = {0, 0, 0};
    for (int chunk = 0; chunk < chunks_per_bank; ++chunk) {
        const int64_t candidate = first_chunk + chunk;
        const float error = chunk_error[candidate];
        if (error < best_error) {
            best_error = error;
            best_scale = chunk_scale[candidate];
#pragma unroll
            for (
                int vector = 0;
                vector < kVectorsPerGroup;
                ++vector) {
                best_indices[vector] =
                    chunk_indices[
                        candidate * kVectorsPerGroup + vector];
            }
        }
    }
    out_scale[group_bank] = best_scale;
#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
        out_indices[
            group_bank * kVectorsPerGroup + vector] =
            best_indices[vector];
    }
}

template <int Entries>
__global__ void nvq2j_reassign_bank_blocks_kernel(
    const float* __restrict__ xgroup,
    const float* __restrict__ wgroup,
    const float* __restrict__ scales,
    const int8_t* __restrict__ codebooks,
    int32_t* __restrict__ out_indices,
    int groups_per_row,
    int valid_last) {
    const int64_t group_bank = blockIdx.x;
    const int bank = group_bank % kBanks;
    const int group = group_bank / kBanks;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    int valid_group = kGroup;
    if (group % groups_per_row == groups_per_row - 1) {
        valid_group = valid_last;
    }
    int valid_vector = valid_group - warp * kVector;
    valid_vector = max(0, min(kVector, valid_vector));
    const float* values =
        xgroup + static_cast<int64_t>(group) * kGroup;
    const float* weights =
        wgroup + static_cast<int64_t>(group) * kGroup;
    const int8_t* source_codebook =
        codebooks + static_cast<int64_t>(bank) * Entries * kVector;
    __shared__ int8_t shared_codebook[Entries * kVector];
    for (
        int offset = tid;
        offset < Entries * kVector;
        offset += blockDim.x) {
        shared_codebook[offset] = source_codebook[offset];
    }
    __syncthreads();
    const int index = nvq2j_search_one_vector<Entries>(
        values,
        weights,
        shared_codebook,
        scales[group_bank],
        warp,
        valid_vector,
        lane);
    if (lane == 0) {
        out_indices[
            group_bank * kVectorsPerGroup + warp] = index;
    }
}

void check_nvq2j_inputs(
    const torch::Tensor& value,
    const torch::Tensor& objective_weight,
    const torch::Tensor& initial_anchor,
    const torch::Tensor& scale_lut,
    const torch::Tensor& bank_for_state,
    const torch::Tensor& codebooks,
    int64_t valid_width,
    int64_t refine_steps) {
    TORCH_CHECK(
        value.is_cuda() && value.is_contiguous()
            && value.scalar_type() == torch::kFloat32
            && value.dim() == 2 && value.size(0) > 0
            && value.size(1) % kGroup == 0,
        "nvq2j_assign: value must be CUDA contiguous float32 [rows,padded_K]");
    TORCH_CHECK(
        objective_weight.is_cuda() && objective_weight.is_contiguous()
            && objective_weight.scalar_type() == torch::kFloat32
            && objective_weight.sizes() == value.sizes(),
        "nvq2j_assign: objective_weight must be CUDA contiguous float32 "
        "with the same shape as value");
    TORCH_CHECK(
        valid_width > 0 && valid_width <= value.size(1)
            && value.size(1) - valid_width < kGroup,
        "nvq2j_assign: invalid unpadded width");
    TORCH_CHECK(
        initial_anchor.is_cuda() && initial_anchor.is_contiguous()
            && initial_anchor.scalar_type() == torch::kFloat32
            && initial_anchor.dim() == 1
            && initial_anchor.size(0) == value.size(0),
        "nvq2j_assign: initial_anchor must be CUDA float32 [rows]");
    TORCH_CHECK(
        scale_lut.is_cuda() && scale_lut.is_contiguous()
            && scale_lut.scalar_type() == torch::kFloat32
            && scale_lut.sizes() == torch::IntArrayRef({kStates}),
        "nvq2j_assign: scale_lut must be CUDA float32 [16]");
    TORCH_CHECK(
        bank_for_state.is_cuda() && bank_for_state.is_contiguous()
            && bank_for_state.scalar_type() == torch::kUInt8
            && bank_for_state.sizes() == torch::IntArrayRef({kStates}),
        "nvq2j_assign: bank_for_state must be CUDA uint8 [16]");
    TORCH_CHECK(
        codebooks.is_cuda() && codebooks.is_contiguous()
            && codebooks.scalar_type() == torch::kInt8
            && codebooks.dim() == 3
            && codebooks.size(0) == kBanks
            && codebooks.size(1) == kVector
            && (codebooks.size(2) == 256
                || codebooks.size(2) == 1024
                || codebooks.size(2) == 4096),
        "nvq2j_assign: codebooks must be CUDA int8 [4,8,256|1024|4096]");
    TORCH_CHECK(
        refine_steps >= 0 && refine_steps <= 4,
        "nvq2j_assign: refine_steps must be in [0,4]");
    const int device = value.get_device();
    TORCH_CHECK(
        initial_anchor.get_device() == device
            && objective_weight.get_device() == device
            && scale_lut.get_device() == device
            && bank_for_state.get_device() == device
            && codebooks.get_device() == device,
        "nvq2j_assign: tensors must share one CUDA device");
    auto bank_host = bank_for_state.cpu().contiguous();
    const auto* bank_values = bank_host.data_ptr<uint8_t>();
    int bank_counts[kBanks] = {};
    for (int state = 0; state < kStates; ++state) {
        const int bank = bank_values[state];
        TORCH_CHECK(bank < kBanks,
                    "nvq2j_assign: bank_for_state contains an invalid bank");
        ++bank_counts[bank];
    }
    for (int bank = 0; bank < kBanks; ++bank) {
        TORCH_CHECK(
            bank_counts[bank] == kStatesPerBank,
            "nvq2j_assign: every bank must own exactly four states");
    }
}

}  // namespace

std::vector<torch::Tensor> nvq2j_assign_cuda(
    torch::Tensor value,
    torch::Tensor objective_weight,
    torch::Tensor initial_anchor,
    torch::Tensor scale_lut,
    torch::Tensor bank_for_state,
    torch::Tensor codebooks,
    int64_t valid_width,
    int64_t refine_steps) {
    check_nvq2j_inputs(
        value,
        objective_weight,
        initial_anchor,
        scale_lut,
        bank_for_state,
        codebooks,
        valid_width,
        refine_steps);
    const int64_t rows = value.size(0);
    const int padded_width = static_cast<int>(value.size(1));
    const int groups_per_row = padded_width / kGroup;
    const int entries = static_cast<int>(codebooks.size(2));
    auto byte_options = value.options().dtype(torch::kUInt8);
    auto states = torch::empty({rows, groups_per_row}, byte_options);
    auto indices = torch::empty(
        {rows, groups_per_row, kVectorsPerGroup},
        entries == 256
            ? byte_options
            : value.options().dtype(torch::kInt32));
    auto first_anchor = initial_anchor.contiguous();
    auto second_anchor = torch::empty({rows}, value.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t blocks = rows * groups_per_row;

    auto launch_assign = [&](const torch::Tensor& current_anchor) {
        if (entries == 256) {
            nvq2j_assign_kernel<256, uint8_t>
                <<<blocks, kAssignThreads, 0, stream>>>(
                    value.data_ptr<float>(),
                    objective_weight.data_ptr<float>(),
                    current_anchor.data_ptr<float>(),
                    scale_lut.data_ptr<float>(),
                    bank_for_state.data_ptr<uint8_t>(),
                    codebooks.data_ptr<int8_t>(),
                    states.data_ptr<uint8_t>(),
                    indices.data_ptr<uint8_t>(),
                    padded_width,
                    static_cast<int>(valid_width),
                    groups_per_row);
        } else if (entries == 1024) {
            nvq2j_assign_kernel<1024, int32_t>
                <<<blocks, kAssignThreads, 0, stream>>>(
                    value.data_ptr<float>(),
                    objective_weight.data_ptr<float>(),
                    current_anchor.data_ptr<float>(),
                    scale_lut.data_ptr<float>(),
                    bank_for_state.data_ptr<uint8_t>(),
                    codebooks.data_ptr<int8_t>(),
                    states.data_ptr<uint8_t>(),
                    indices.data_ptr<int32_t>(),
                    padded_width,
                    static_cast<int>(valid_width),
                    groups_per_row);
        } else {
            nvq2j_assign_kernel<4096, int32_t>
                <<<blocks, kAssignThreads, 0, stream>>>(
                    value.data_ptr<float>(),
                    objective_weight.data_ptr<float>(),
                    current_anchor.data_ptr<float>(),
                    scale_lut.data_ptr<float>(),
                    bank_for_state.data_ptr<uint8_t>(),
                    codebooks.data_ptr<int8_t>(),
                    states.data_ptr<uint8_t>(),
                    indices.data_ptr<int32_t>(),
                    padded_width,
                    static_cast<int>(valid_width),
                    groups_per_row);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    auto launch_refit = [&](
        const torch::Tensor& current_anchor,
        torch::Tensor& fitted_anchor) {
        if (entries == 256) {
            nvq2j_refit_anchor_kernel<256, uint8_t>
                <<<rows, kRefitThreads, 0, stream>>>(
                    value.data_ptr<float>(),
                    objective_weight.data_ptr<float>(),
                    current_anchor.data_ptr<float>(),
                    scale_lut.data_ptr<float>(),
                    bank_for_state.data_ptr<uint8_t>(),
                    codebooks.data_ptr<int8_t>(),
                    states.data_ptr<uint8_t>(),
                    indices.data_ptr<uint8_t>(),
                    fitted_anchor.data_ptr<float>(),
                    padded_width,
                    static_cast<int>(valid_width),
                    groups_per_row);
        } else if (entries == 1024) {
            nvq2j_refit_anchor_kernel<1024, int32_t>
                <<<rows, kRefitThreads, 0, stream>>>(
                    value.data_ptr<float>(),
                    objective_weight.data_ptr<float>(),
                    current_anchor.data_ptr<float>(),
                    scale_lut.data_ptr<float>(),
                    bank_for_state.data_ptr<uint8_t>(),
                    codebooks.data_ptr<int8_t>(),
                    states.data_ptr<uint8_t>(),
                    indices.data_ptr<int32_t>(),
                    fitted_anchor.data_ptr<float>(),
                    padded_width,
                    static_cast<int>(valid_width),
                    groups_per_row);
        } else {
            nvq2j_refit_anchor_kernel<4096, int32_t>
                <<<rows, kRefitThreads, 0, stream>>>(
                    value.data_ptr<float>(),
                    objective_weight.data_ptr<float>(),
                    current_anchor.data_ptr<float>(),
                    scale_lut.data_ptr<float>(),
                    bank_for_state.data_ptr<uint8_t>(),
                    codebooks.data_ptr<int8_t>(),
                    states.data_ptr<uint8_t>(),
                    indices.data_ptr<int32_t>(),
                    fitted_anchor.data_ptr<float>(),
                    padded_width,
                    static_cast<int>(valid_width),
                    groups_per_row);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    launch_assign(first_anchor);
    for (int64_t refine = 0; refine < refine_steps; ++refine) {
        launch_refit(first_anchor, second_anchor);
        launch_assign(second_anchor);
        std::swap(first_anchor, second_anchor);
    }
    return {first_anchor, states, indices};
}

namespace {

void check_nvq2j_bank_search_inputs(
    const torch::Tensor& xgroup,
    const torch::Tensor& wgroup,
    const torch::Tensor& codebooks,
    int64_t groups_per_row,
    int64_t valid_last) {
    TORCH_CHECK(
        xgroup.is_cuda() && xgroup.is_contiguous()
            && xgroup.scalar_type() == torch::kFloat32
            && xgroup.dim() == 2 && xgroup.size(1) == kGroup,
        "nvq2j bank search: xgroup must be CUDA contiguous float32 [groups,24]");
    TORCH_CHECK(
        wgroup.is_cuda() && wgroup.is_contiguous()
            && wgroup.scalar_type() == torch::kFloat32
            && wgroup.sizes() == xgroup.sizes(),
        "nvq2j bank search: wgroup must match xgroup");
    TORCH_CHECK(
        codebooks.is_cuda() && codebooks.is_contiguous()
            && codebooks.scalar_type() == torch::kInt8
            && codebooks.dim() == 3
            && codebooks.size(0) == kBanks
            && (codebooks.size(1) == 1024
                || codebooks.size(1) == 4096)
            && codebooks.size(2) == kVector,
        "nvq2j bank search: codebooks must be CUDA int8 [4,1024|4096,8]");
    TORCH_CHECK(
        groups_per_row > 0
            && valid_last > 0 && valid_last <= kGroup,
        "nvq2j bank search: invalid row geometry");
    TORCH_CHECK(
        xgroup.size(0) % groups_per_row == 0,
        "nvq2j bank search: group count must be divisible by groups_per_row");
    TORCH_CHECK(
        xgroup.get_device() == wgroup.get_device()
            && xgroup.get_device() == codebooks.get_device(),
        "nvq2j bank search: tensors must share one CUDA device");
}

}  // namespace

std::vector<torch::Tensor> nvq2j_search_banks_cuda(
    torch::Tensor xgroup,
    torch::Tensor wgroup,
    torch::Tensor codebooks,
    torch::Tensor bank_qmax,
    int64_t groups_per_row,
    int64_t valid_last,
    int64_t search_steps) {
    check_nvq2j_bank_search_inputs(
        xgroup, wgroup, codebooks, groups_per_row, valid_last);
    TORCH_CHECK(
        bank_qmax.is_cuda() && bank_qmax.is_contiguous()
            && bank_qmax.scalar_type() == torch::kFloat32
            && bank_qmax.sizes() == torch::IntArrayRef({kBanks})
            && bank_qmax.get_device() == xgroup.get_device(),
        "nvq2j bank search: bank_qmax must be CUDA float32 [4]");
    TORCH_CHECK(
        search_steps > 0 && search_steps <= 64,
        "nvq2j bank search: search_steps must be in [1,64]");
    const int64_t groups = xgroup.size(0);
    const int chunks_per_bank =
        (static_cast<int>(search_steps) + kSearchStepsPerBlock - 1)
        / kSearchStepsPerBlock;
    const int64_t group_banks = groups * kBanks;
    const int64_t chunk_count = group_banks * chunks_per_bank;
    auto scales = torch::empty(
        {groups, kBanks}, xgroup.options());
    auto indices = torch::empty(
        {groups, kBanks, kVectorsPerGroup},
        xgroup.options().dtype(torch::kInt32));
    auto chunk_error = torch::empty(
        {chunk_count}, xgroup.options());
    auto chunk_scale = torch::empty(
        {chunk_count}, xgroup.options());
    auto chunk_indices = torch::empty(
        {chunk_count, kVectorsPerGroup},
        xgroup.options().dtype(torch::kInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (codebooks.size(1) == 1024) {
        nvq2j_search_bank_chunks_kernel<1024>
            <<<chunk_count, kParallelSearchThreads, 0, stream>>>(
                xgroup.data_ptr<float>(),
                wgroup.data_ptr<float>(),
                codebooks.data_ptr<int8_t>(),
                bank_qmax.data_ptr<float>(),
                chunk_error.data_ptr<float>(),
                chunk_scale.data_ptr<float>(),
                chunk_indices.data_ptr<int32_t>(),
                static_cast<int>(groups_per_row),
                static_cast<int>(valid_last),
                static_cast<int>(search_steps),
                chunks_per_bank);
    } else {
        nvq2j_search_bank_chunks_kernel<4096>
            <<<chunk_count, kParallelSearchThreads, 0, stream>>>(
                xgroup.data_ptr<float>(),
                wgroup.data_ptr<float>(),
                codebooks.data_ptr<int8_t>(),
                bank_qmax.data_ptr<float>(),
                chunk_error.data_ptr<float>(),
                chunk_scale.data_ptr<float>(),
                chunk_indices.data_ptr<int32_t>(),
                static_cast<int>(groups_per_row),
                static_cast<int>(valid_last),
                static_cast<int>(search_steps),
                chunks_per_bank);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t reduce_blocks =
        (group_banks + kSearchReduceThreads - 1)
        / kSearchReduceThreads;
    nvq2j_reduce_bank_chunks_kernel
        <<<reduce_blocks, kSearchReduceThreads, 0, stream>>>(
            chunk_error.data_ptr<float>(),
            chunk_scale.data_ptr<float>(),
            chunk_indices.data_ptr<int32_t>(),
            scales.data_ptr<float>(),
            indices.data_ptr<int32_t>(),
            group_banks,
            chunks_per_bank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scales, indices};
}

torch::Tensor nvq2j_reassign_banks_cuda(
    torch::Tensor xgroup,
    torch::Tensor wgroup,
    torch::Tensor scales,
    torch::Tensor codebooks,
    int64_t groups_per_row,
    int64_t valid_last) {
    check_nvq2j_bank_search_inputs(
        xgroup, wgroup, codebooks, groups_per_row, valid_last);
    TORCH_CHECK(
        scales.is_cuda() && scales.is_contiguous()
            && scales.scalar_type() == torch::kFloat32
            && scales.sizes()
                == torch::IntArrayRef({xgroup.size(0), kBanks})
            && scales.get_device() == xgroup.get_device(),
        "nvq2j bank reassign: scales must be CUDA float32 [groups,4]");
    const int64_t groups = xgroup.size(0);
    auto indices = torch::empty(
        {groups, kBanks, kVectorsPerGroup},
        xgroup.options().dtype(torch::kInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (codebooks.size(1) == 1024) {
        nvq2j_reassign_bank_blocks_kernel<1024>
            <<<groups * kBanks, kAssignThreads, 0, stream>>>(
                xgroup.data_ptr<float>(),
                wgroup.data_ptr<float>(),
                scales.data_ptr<float>(),
                codebooks.data_ptr<int8_t>(),
                indices.data_ptr<int32_t>(),
                static_cast<int>(groups_per_row),
                static_cast<int>(valid_last));
    } else {
        nvq2j_reassign_bank_blocks_kernel<4096>
            <<<groups * kBanks, kAssignThreads, 0, stream>>>(
                xgroup.data_ptr<float>(),
                wgroup.data_ptr<float>(),
                scales.data_ptr<float>(),
                codebooks.data_ptr<int8_t>(),
                indices.data_ptr<int32_t>(),
                static_cast<int>(groups_per_row),
                static_cast<int>(valid_last));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return indices;
}
