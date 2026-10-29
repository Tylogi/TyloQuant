#pragma once
#include <cuda_runtime.h>
#include <cstdint>

#ifdef _WIN32
#define MFQ_TRELLIS_API __declspec(dllexport)
#else
#define MFQ_TRELLIS_API
#endif

// Ordinary 2D projection view. No routing, expert axes or cache ownership.
// All row descriptors select their own 1..8-bit precision and packed offset.
struct TrellisProjectionView {
    const uint32_t* words;
    const uint64_t* descriptors;
    const void* scales;
    const void* relative;
    const float* bias;
    int rows;
    int stored_columns;  // Independent 256-element encoded rings, padded at tail.
    int input_columns;   // Logical input width; padding is masked in the kernel.
    int gslog;
    int columns;         // 0: subgroup selectors; 1: FP16 input-column scales.
};

// dtype=0: FP16; dtype=1: BF16; dtype=2: FP32 input/output. FP32 accumulation;
// FP16 factors shared scales after partial dots, BF16/FP32 reconstruct FP16
// weights. Workspaces and outputs are supplied by the caller.
extern "C" MFQ_TRELLIS_API int mfq_tq_gemv(TrellisProjectionView weight,const void* x,void* output,
    int dtype,int warps,int rows_per_block,cudaStream_t stream);
extern "C" MFQ_TRELLIS_API int mfq_tq_swiglu(TrellisProjectionView gate,TrellisProjectionView up,
    const void* x,void* hidden,int dtype,int warps,int rows_per_block,cudaStream_t stream);
extern "C" MFQ_TRELLIS_API int mfq_tq_ffn(TrellisProjectionView gate,TrellisProjectionView up,
    TrellisProjectionView down,const void* x,void* hidden,void* output,int dtype,cudaStream_t stream);

// Device arrays of ordinary projections, with per-row precision dispatch.
// The caller validates views before uploading them. max_rows must cover every
// view.rows. Strides count elements; input_stride=0 shares one input vector.
// This batches computations without introducing a routed weight wire format.
extern "C" MFQ_TRELLIS_API int mfq_tq_gemv_batch(const TrellisProjectionView* weights,
    const void* x,void* output,int count,int max_rows,int input_stride,int output_stride,
    int dtype,int warps,int rows_per_block,cudaStream_t stream);
extern "C" MFQ_TRELLIS_API int mfq_tq_swiglu_batch(const TrellisProjectionView* gate,
    const TrellisProjectionView* up,const void* x,void* hidden,int count,int max_rows,
    int input_stride,int output_stride,int dtype,int warps,int rows_per_block,cudaStream_t stream);
// Runtime view tables indexed by device expert IDs. Callers publish selected
// immutable views on this stream and retain their images until completion.
// This avoids rebuilding/uploading a selected-view array for every token.
extern "C" MFQ_TRELLIS_API int mfq_tq_routed_swiglu(const TrellisProjectionView* gate,
    const TrellisProjectionView* up,const int64_t* ids,const void* x,void* hidden,
    int routes,int hidden_rows,int dtype,float limit,cudaStream_t stream);
extern "C" MFQ_TRELLIS_API int mfq_tq_routed_down(const TrellisProjectionView* down,
    const int64_t* ids,const void* hidden,void* output,int routes,int output_rows,
    int hidden_width,int dtype,cudaStream_t stream);

// A zero-copy execution plan for independent projections sharing an input.
// block_end is the cumulative number of ceil(weight.rows/GROUP_ROWS) blocks.
// output_offset and strides count elements. Views may have different row
// precisions, scale modes, subgroup sizes and output lengths. Callers validate
// the plan and keep its weights alive; no per-matrix precision is involved.
inline constexpr int TRELLIS_GROUP_ROWS=4;
struct TrellisProjectionJob {
    TrellisProjectionView weight;
    int64_t output_offset;
    int block_end;
};
extern "C" MFQ_TRELLIS_API int mfq_tq_parallel(const TrellisProjectionJob* jobs,
    int count,int total_blocks,const void* x,void* output,int tokens,
    int input_stride,int output_stride,int dtype,cudaStream_t stream);
// Token reuse with projection-major output. Job i writes contiguous
// [tokens, weight.rows] at output + output_offset*tokens. The same static
// row plan is reused; input is contiguous [tokens,input_stride].
extern "C" MFQ_TRELLIS_API int mfq_tq_parallel_tokens(const TrellisProjectionJob* jobs,
    int count,int total_blocks,const void* x,void* output,int tokens,
    int input_stride,int dtype,cudaStream_t stream);

// Short projections with transient Q8 activations (16-element groups).
// The procedural Gaussian book is evaluated as centered 10-bit integers;
// dp2a accumulates their dot with Q8 before FP32 scaling and output.
// Storage stays TQ. Each 512-column input tile stores Q8 token planes,
// followed by paired gain/offset planes. Each input slot needs at least
// tokens*round_up(input_columns,512)*3/2 bytes, aligned to 16.
// The ordinary/grouped API needs groups slots. Parallel needs one shared
// slot when all projections use row scales, or count+1 slots when any
// projection uses column scales. Weights retain their original padding.
// Supports 1..4 input tokens, dtype 0 (FP16) or 2 (FP32); input and output
// must not alias workspace. Native short projections use this path for
// both FP16 and FP32, including a single input token.
// In parallel, joined_count leading projections can write a contiguous
// [tokens, joined_rows] prefix. The caller validates joined_rows equals the
// sum of their row counts; remaining projections stay projection-major.
extern "C" MFQ_TRELLIS_API int mfq_tq_q8_tokens(TrellisProjectionView weight,
    const void* x,void* output,int tokens,int groups,int dtype,void* workspace,
    uint64_t workspace_bytes,cudaStream_t stream);
// host_jobs and device_jobs describe identical immutable projections. Small
// batches of host descriptors become kernel parameters, avoiding per-block
// descriptor fetches without limiting the total projection count.
extern "C" MFQ_TRELLIS_API int mfq_tq_q8_parallel_tokens(const TrellisProjectionJob* host_jobs,const TrellisProjectionJob* device_jobs,
    int count,int total_blocks,const void* x,void* output,int tokens,int input_stride,int dtype,
    void* workspace,uint64_t workspace_bytes,int joined_count,int joined_rows,cudaStream_t stream);

// Indexed execution over ordinary TQ projections. One decoded row serves up
// to four arbitrary input vectors. Offsets count elements in the caller's
// input/output arrays. The optional second projection is used by SwiGLU only.
// The caller validates device plans: 1 <= count <= token_tile <= 4,
// offsets are nonnegative,
// output ranges do not overlap, views have at most max_rows rows, and the two
// SwiGLU views have identical dimensions. All referenced storage stays alive
// until execution completes. This plan never changes weight storage/layout.
struct TrellisIndexedProjectionJob {
    TrellisProjectionView weight,second;
    int64_t input_offsets[4],output_offsets[4];
    int count;
};
// A single ordinary projection reused across up to eight input rows. Dropping
// the unused SwiGLU view keeps this plan as small as the four-row pair plan.
struct TrellisIndexedWideProjectionJob {
    TrellisProjectionView weight;
    int64_t input_offsets[8],output_offsets[8];
    int count;
};
// Two ordinary projections sharing eight inputs. The native planner fills
// complete eight-row jobs, leaving short remainders to the SIMT pair kernel.
struct TrellisIndexedWidePairJob {
    TrellisProjectionView weight,second;
    int64_t input_offsets[8],output_offsets[8];
    int count;
};
extern "C" MFQ_TRELLIS_API int mfq_tq_indexed_wide_swiglu(const TrellisIndexedWidePairJob* jobs,
    const void* x,void* output,int count,int max_rows,int dtype,int token_tile,float limit,cudaStream_t stream);
extern "C" MFQ_TRELLIS_API int mfq_tq_indexed_wide(const TrellisIndexedWideProjectionJob* jobs,
    const void* x,void* output,int count,int max_rows,int dtype,int token_tile,cudaStream_t stream);
extern "C" MFQ_TRELLIS_API int mfq_tq_indexed(const TrellisIndexedProjectionJob* jobs,
    const void* x,void* output,int count,int max_rows,int dtype,int token_tile,cudaStream_t stream);
// Round both projections to dtype before activation. Positive limit clamps
// Gate above limit and Up to [-limit,limit], matching clipped SwiGLU.
extern "C" MFQ_TRELLIS_API int mfq_tq_indexed_swiglu(const TrellisIndexedProjectionJob* jobs,
    const void* x,void* output,int count,int max_rows,int dtype,int token_tile,float limit,cudaStream_t stream);

// FP16 prefill plan over ordinary Gate/Up/Down views. Each job owns count
// slots starting at first; slots select flattened [token,route] output pairs.
// The caller validates view geometry, 0<count<=capacity and nonoverlapping
// valid slots, and retains every compressed image until stream completion.
struct TrellisPrefillMoeJob {
    TrellisProjectionView gate,up,down;
    int64_t first;
    int count;
};
// FP16 WMMA requires SM70; older devices keep ordinary projection kernels.
// Scratch bytes: align16(B*T*max(K,H)*2)+align16(B*T*H*2)+B*T*max(K,H)*2.
// Scratch must not alias input, pairs, plans, slots or compressed images.
// There are no allocations, persistent expanded weights or BLAS dependencies.
extern "C" MFQ_TRELLIS_API int mfq_tq_prefill_moe(const TrellisPrefillMoeJob* jobs,
    const int64_t* slots,const void* input,void* pairs,int batch,int capacity,
    int input_width,int hidden_width,int routes,float limit,void* activations,
    uint64_t activation_bytes,cudaStream_t stream);

// The same plan with bounded decoded FP16 tiles and batched cuBLAS GEMM.
// All FP16 activations/bias semantics match mfq_tq_prefill_moe. The caller
// supplies a per-stream cublasHandle_t. Weight scratch needs B*max(K,H)*2
// bytes for at least one row; activation scratch is align16(B*T*max(K,H)*2)
// +B*T*max(K,2H)*4. Neither buffer may alias inputs, outputs, plans or weights.
extern "C" MFQ_TRELLIS_API int mfq_tq_prefill_moe_gemm(const TrellisPrefillMoeJob* jobs,
    const int64_t* slots,const void* input,void* pairs,int batch,int capacity,
    int input_width,int hidden_width,int routes,float limit,void* workspace,
    uint64_t workspace_bytes,void* activations,uint64_t activation_bytes,
    void* blas,cudaStream_t stream);

extern "C" MFQ_TRELLIS_API int mfq_tq_gemv_tokens(TrellisProjectionView weight,
    const void* x,void* output,int tokens,int dtype,cudaStream_t stream);
// FP16 prefill with bounded temporary decoded rows. Workspace never caches
// expanded weights; it is reused for each row tile and supplied by the caller.
// Bias uses an FP32 tile epilogue so it is added before output rounding.
// blas is the caller's cublasHandle_t; CUDA errors are returned directly,
// CUBLAS errors as 1000+status. No allocations or handles created in the call.
extern "C" MFQ_TRELLIS_API int mfq_tq_gemm(TrellisProjectionView weight,
    const void* x,void* output,int tokens,void* workspace,uint64_t workspace_bytes,
    void* blas,cudaStream_t stream);
// Each input group uses only its corresponding contiguous output-row range.
// Input [tokens,groups,input_columns], output [tokens,weight.rows].
extern "C" MFQ_TRELLIS_API int mfq_tq_grouped_tokens(TrellisProjectionView weight,
    const void* x,void* output,int tokens,int groups,int dtype,cudaStream_t stream);
extern "C" MFQ_TRELLIS_API int mfq_tq_grouped_gemm(TrellisProjectionView weight,
    const void* x,void* output,int tokens,int groups,void* workspace,uint64_t workspace_bytes,
    void* blas,cudaStream_t stream);
// BF16/FP32 prefill, retaining the source activation precision and exact FP16
// weight reconstruction. FP32 GEMM uses pedantic math (no TF32 truncation).
// Transient row/input/output tiles all fit in the supplied workspace.
extern "C" MFQ_TRELLIS_API int mfq_tq_float_gemm(TrellisProjectionView weight,
    const void* x,void* output,int tokens,int groups,int dtype,void* workspace,
    uint64_t workspace_bytes,void* blas,cudaStream_t stream);
extern "C" MFQ_TRELLIS_API int mfq_tq_swiglu_tokens(TrellisProjectionView gate,
    TrellisProjectionView up,const void* x,void* output,int tokens,int dtype,cudaStream_t stream);
// Contiguous [tokens, routes, width] expert outputs and FP32 [tokens, routes]
// routing weights. Accumulate in FP32 and round once to the output dtype.
extern "C" MFQ_TRELLIS_API int mfq_tq_weighted_reduce(const void* expert_output,
    const float* weights,void* output,int tokens,int routes,int width,int dtype,cudaStream_t stream);
// Null ids decodes consecutive rows; non-null ids is a device int64 array.
// Only selected embedding rows are materialized. Expert projections use GEMV.
extern "C" MFQ_TRELLIS_API int mfq_tq_embedding(TrellisProjectionView weight,
    const int64_t* ids,int count,void* output,int dtype,cudaStream_t stream);
// Each view selects one ordinary TQ row, potentially from different tables.
// Views and compressed payload may reside in a caller-owned mapped packet.
extern "C" MFQ_TRELLIS_API int mfq_tq_embedding_rows(const TrellisProjectionView* rows,
    int count,int columns,void* output,int dtype,cudaStream_t stream);
