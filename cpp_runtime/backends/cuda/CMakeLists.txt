# Native CUDA backend.

if(NOT TARGET mfq-runtime-communication)
    message(FATAL_ERROR
        "The CUDA runtime requires MFQ_BUILD_RUNTIME_COMMUNICATION=ON")
endif()

set(MFQ_CUDA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(MFQ_CUDA_KERNEL_ROOT "${MFQ_REPOSITORY_ROOT}/mfq/kernels/cuda")

set(MFQ_CUDA_ARCHITECTURES "" CACHE STRING
    "Optional CUDA architecture override; otherwise use CMAKE_CUDA_ARCHITECTURES, CUDAARCHS or native")
if(NOT "${MFQ_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}")
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(_mfq_cuda_architectures native)
    if(NOT "$ENV{CUDAARCHS}" STREQUAL "")
        set(_mfq_cuda_architectures "$ENV{CUDAARCHS}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_mfq_cuda_architectures}" CACHE STRING
        "CUDA architectures compiled into MFQ CUDA targets")
    unset(_mfq_cuda_architectures)
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Keep the override cache empty unless explicitly set, so later changes to the
# standard CMake option are honored on reconfigure.
set(MFQ_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(mfq-cuda-storage STATIC
    ${MFQ_CUDA_ROOT}/storage/mfe_expert_store.cpp
)
add_library(mfq::cuda-storage ALIAS mfq-cuda-storage)
target_include_directories(mfq-cuda-storage PUBLIC
    ${MFQ_CUDA_ROOT}/storage
)
target_link_libraries(mfq-cuda-storage PUBLIC Threads::Threads)
target_compile_features(mfq-cuda-storage PUBLIC cxx_std_20)
set_target_properties(mfq-cuda-storage PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
    target_compile_options(mfq-cuda-storage PRIVATE /utf-8 /EHsc)
endif()

add_library(mfq-cuda-core STATIC
    ${MFQ_CUDA_ROOT}/src/mfq_cuda_context.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor.cpp
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor_blas.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor_creation.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor_indexing.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor_ops.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor_reduction.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor_signal.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor_sort.cu
)
add_library(mfq::cuda-core ALIAS mfq-cuda-core)
target_include_directories(mfq-cuda-core PUBLIC
    ${MFQ_CUDA_ROOT}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(mfq-cuda-core PUBLIC CUDA::cudart CUDA::cublas mfq-host-parallel)
if(WIN32)
    target_link_libraries(mfq-cuda-core PUBLIC dxgi)
endif()
target_compile_features(mfq-cuda-core PUBLIC cxx_std_20)
target_compile_options(mfq-cuda-core PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
)
set_target_properties(mfq-cuda-core PROPERTIES
    CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

set(MFQ_CUDA_KERNEL_SOURCES
    ${MFQ_CUDA_KERNEL_ROOT}/acc.cu
    ${MFQ_CUDA_KERNEL_ROOT}/activation.cu
    ${MFQ_CUDA_KERNEL_ROOT}/attention.cu
    ${MFQ_CUDA_KERNEL_ROOT}/attention_mma.cu
    ${MFQ_CUDA_KERNEL_ROOT}/deepseek_v4_attention.cu
    ${MFQ_CUDA_KERNEL_ROOT}/deepseek_v4_hc.cu
    ${MFQ_CUDA_KERNEL_ROOT}/deepseek_v41.cu
    ${MFQ_CUDA_KERNEL_ROOT}/embedding.cu
    ${MFQ_CUDA_KERNEL_ROOT}/selected_attention.cu
    ${MFQ_CUDA_KERNEL_ROOT}/qsa_scores.cu
    ${MFQ_CUDA_KERNEL_ROOT}/qwen4_exp.cu
    ${MFQ_CUDA_KERNEL_ROOT}/ple_fused.cu
    ${MFQ_CUDA_KERNEL_ROOT}/glm5_next.cu
    ${MFQ_CUDA_KERNEL_ROOT}/gated_delta_net.cu
    ${MFQ_CUDA_KERNEL_ROOT}/gdn_prefill_chunk.cu
    ${MFQ_CUDA_KERNEL_ROOT}/gdn_decode_window.cu
    ${MFQ_CUDA_KERNEL_ROOT}/glm_dsa.cu
    ${MFQ_CUDA_KERNEL_ROOT}/kv_cache.cu
    ${MFQ_CUDA_KERNEL_ROOT}/paged_kv.cu
    ${MFQ_CUDA_KERNEL_ROOT}/dense_moe.cu
    ${MFQ_CUDA_KERNEL_ROOT}/qk_rope_kv.cu
    ${MFQ_CUDA_KERNEL_ROOT}/moe.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_compact.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_parity_read.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_cached_reduce.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_shared.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_token_groups.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_prepare.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_launch.cpp
    ${MFQ_CUDA_KERNEL_ROOT}/mx_matmul.cu
    ${MFQ_CUDA_KERNEL_ROOT}/fp8_sq.cu
    ${MFQ_CUDA_KERNEL_ROOT}/tq_ops.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mxfp4_sq.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nepq.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nepq_residual.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nint_matmul.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nint_short_decode.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nint_q8_decode.cu
    ${MFQ_CUDA_KERNEL_ROOT}/norm.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nvq_matmul.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nvq_routed_reuse.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nvq1_decode.cu
    ${MFQ_CUDA_KERNEL_ROOT}/rope.cu
    ${MFQ_CUDA_KERNEL_ROOT}/sampling.cu
    ${MFQ_CUDA_KERNEL_ROOT}/selective_state_scan.cu
    ${MFQ_CUDA_KERNEL_ROOT}/ssm_conv.cu
)

add_library(mfq-cuda-native-kernels STATIC
    ${MFQ_CUDA_KERNEL_SOURCES}
    ${MFQ_CUDA_ROOT}/kernels/float_projection.cu
    ${MFQ_CUDA_ROOT}/kernels/gr_prefill_blas.cu
    ${MFQ_CUDA_ROOT}/kernels/kv_quantization.cu
    ${MFQ_CUDA_ROOT}/kernels/batch_selected_attention.cu
    ${MFQ_CUDA_ROOT}/kernels/kv_hot_cache.cu
    ${MFQ_CUDA_ROOT}/kernels/gated_residual_fused.cu
    ${MFQ_CUDA_ROOT}/kernels/shared_gate_fused.cu
    ${MFQ_CUDA_ROOT}/runtime/moe_pipeline.cu
    ${MFQ_CUDA_ROOT}/runtime/batch_decode_graph.cu
    ${MFQ_CUDA_ROOT}/apps/ggml_cuda_compat.cu
)
add_library(mfq::cuda-native-kernels ALIAS mfq-cuda-native-kernels)

# Keep the shared packed math in incremental builds even when nvcc's depfile
# omits headers from a heavily inlined translation unit on Windows.
set_source_files_properties(${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn.cu PROPERTIES
    OBJECT_DEPENDS "${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${MFQ_CUDA_KERNEL_ROOT}/packed_nint.cuh;${MFQ_CUDA_KERNEL_ROOT}/packed_nvq.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn.h;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_launch.h;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_compact.h;${MFQ_CUDA_KERNEL_ROOT}/glu.cuh")
set_source_files_properties(${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_compact.cu PROPERTIES
    OBJECT_DEPENDS "${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_nvq_compact_dot.cuh;${MFQ_CUDA_KERNEL_ROOT}/nvq_jsc_bank_map.cuh;${MFQ_CUDA_KERNEL_ROOT}/nvq_dense_window32.cuh;${MFQ_CUDA_KERNEL_ROOT}/packed_nint.cuh;${MFQ_CUDA_KERNEL_ROOT}/packed_nvq.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn.h;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_launch.h;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_compact.h")
set_source_files_properties(${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_parity_read.cu PROPERTIES
    OBJECT_DEPENDS "${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_gu_prequantized.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_gu_group_quant.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_gu_transfer_routes.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_nvq_compact_dot.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_nvq_parity_read.cuh;${MFQ_CUDA_KERNEL_ROOT}/nvq_jsc_bank_map.cuh;${MFQ_CUDA_KERNEL_ROOT}/nvq_dense_window32.cuh;${MFQ_CUDA_KERNEL_ROOT}/packed_nint.cuh;${MFQ_CUDA_KERNEL_ROOT}/packed_nvq.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn.h;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_launch.h")
set_source_files_properties(${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_token_groups.cu PROPERTIES
    OBJECT_DEPENDS "${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn.h;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_token_groups.h;${MFQ_CUDA_KERNEL_ROOT}/mfe_nint_tokens.cuh;${MFQ_CUDA_KERNEL_ROOT}/packed_nint.cuh;${MFQ_CUDA_KERNEL_ROOT}/packed_nvq.cuh;${MFQ_CUDA_KERNEL_ROOT}/nvq_e8_dense_group.cuh;${MFQ_CUDA_KERNEL_ROOT}/nvq_d4_dense_group.cuh;${MFQ_CUDA_KERNEL_ROOT}/nvq_dense_window32.cuh;${MFQ_CUDA_KERNEL_ROOT}/nvq_jsc_bank_map.cuh;${MFQ_CUDA_KERNEL_ROOT}/nint_route_group.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_nint_whole.cuh")
set_source_files_properties(${MFQ_CUDA_KERNEL_ROOT}/nint_matmul.cu PROPERTIES
    OBJECT_DEPENDS "${MFQ_CUDA_KERNEL_ROOT}/packed_nint.cuh;${MFQ_CUDA_KERNEL_ROOT}/nint_route_group.cuh;${MFQ_CUDA_KERNEL_ROOT}/glu.cuh")
set_source_files_properties(${MFQ_CUDA_KERNEL_ROOT}/nvq_matmul.cu PROPERTIES
    OBJECT_DEPENDS "${MFQ_CUDA_KERNEL_ROOT}/packed_nvq.cuh;${MFQ_CUDA_KERNEL_ROOT}/glu.cuh")
set_source_files_properties(${MFQ_CUDA_ROOT}/kernels/gated_residual_fused.cu PROPERTIES
    COMPILE_OPTIONS "--maxrregcount=80")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set_property(SOURCE ${MFQ_CUDA_ROOT}/kernels/gated_residual_fused.cu APPEND
        PROPERTY COMPILE_OPTIONS "--split-compile=0")
endif()
target_compile_definitions(mfq-cuda-native-kernels PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1
)
target_include_directories(mfq-cuda-native-kernels PRIVATE
    ${MFQ_REPOSITORY_ROOT}
    ${MFQ_CUDA_ROOT}/include
    ${MFQ_CUDA_KERNEL_ROOT}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(mfq-cuda-native-kernels
    PUBLIC mfq-cuda-core CUDA::cuda_driver
    PRIVATE mfq-core mfq-ggml-internal-headers mfq-ggml
)
target_compile_options(mfq-cuda-native-kernels PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
)
set_target_properties(mfq-cuda-native-kernels PROPERTIES
    CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
    foreach(mfq_cuda_target mfq-cuda-core mfq-cuda-native-kernels)
        target_compile_options(${mfq_cuda_target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
        )
    endforeach()
endif()

if(BUILD_TESTING)
    add_executable(mfq-kv-vmm-failure-test ${MFQ_CUDA_ROOT}/tests/kv_vmm_failure_test.cpp)
    target_link_libraries(mfq-kv-vmm-failure-test PRIVATE CUDA::cudart)
    target_compile_features(mfq-kv-vmm-failure-test PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(mfq-kv-vmm-failure-test PRIVATE /EHsc)
    endif()
    add_test(NAME mfq-kv-vmm-failure-test COMMAND mfq-kv-vmm-failure-test)

    function(mfq_add_cuda_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE mfq-cuda-core ${ARGN})
        set_target_properties(${target} PROPERTIES
            CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
            CUDA_RUNTIME_LIBRARY Shared
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>"
            "$<$<COMPILE_LANGUAGE:CXX>:$<IF:$<CXX_COMPILER_ID:MSVC>,/UNDEBUG,-UNDEBUG>>")
        if(MSVC)
            target_compile_options(${target} PRIVATE
                "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
        endif()
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    add_executable(mfq-cuda-mfe-expert-store-test
        ${MFQ_CUDA_ROOT}/tests/mfq_mfe_expert_store_test.cpp)
    target_link_libraries(mfq-cuda-mfe-expert-store-test PRIVATE
        mfq-cuda-storage)
    target_compile_features(mfq-cuda-mfe-expert-store-test PRIVATE cxx_std_20)
    add_test(NAME mfq-cuda-mfe-expert-store-test
        COMMAND mfq-cuda-mfe-expert-store-test)

    add_executable(mfq-mxfp4-sq-test ${MFQ_CUDA_ROOT}/tests/mfq_mxfp4_sq_test.cu)
    target_compile_definitions(mfq-mxfp4-sq-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-mxfp4-sq-test PRIVATE ${MFQ_REPOSITORY_ROOT})
    target_link_libraries(mfq-mxfp4-sq-test PRIVATE
        mfq-core mfq-cuda-core mfq-cuda-native-kernels)
    set_target_properties(mfq-mxfp4-sq-test PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    add_executable(mfq-paged-kv-attention-bench
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_paged_kv_attention_bench.cu)
    target_compile_definitions(mfq-paged-kv-attention-bench PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1)
    target_link_libraries(mfq-paged-kv-attention-bench PRIVATE
        mfq-cuda-core mfq-cuda-native-kernels)
    set_target_properties(mfq-paged-kv-attention-bench PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    add_executable(mfq-cuda-nint-bench ${MFQ_REPOSITORY_ROOT}/bench/cuda_nint_bench.cpp)
    target_compile_definitions(mfq-cuda-nint-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_link_libraries(mfq-cuda-nint-bench PRIVATE mfq-cuda-core mfq-cuda-native-kernels)
    target_compile_features(mfq-cuda-nint-bench PRIVATE cxx_std_20)

    mfq_add_cuda_test(mfq-cuda-context-test
        ${MFQ_CUDA_ROOT}/tests/mfq_cuda_context_test.cu)
    mfq_add_cuda_test(mfq-cuda-activation-test
        ${MFQ_CUDA_ROOT}/tests/mfq_cuda_activation_test.cu
        mfq-cuda-native-kernels)
    mfq_add_cuda_test(mfq-recurrent-mixer-ops-test
        ${MFQ_CUDA_ROOT}/tests/recurrent_mixer_ops_test.cpp
        mfq-cuda-native-kernels)
    target_compile_definitions(mfq-recurrent-mixer-ops-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1)
    if(MSVC)
        target_compile_options(mfq-recurrent-mixer-ops-test PRIVATE /EHsc)
    endif()
    mfq_add_cuda_test(mfq-nint-q8-test
        ${MFQ_CUDA_ROOT}/tests/mfq_nint_q8_test.cu
        mfq-cuda-native-kernels)
    mfq_add_cuda_test(mfq-nint-rows-cuda-test
        ${MFQ_CUDA_ROOT}/tests/mfq_nint_rows_test.cu
        mfq-cuda-native-kernels mfq-core)
    target_compile_definitions(mfq-nint-rows-cuda-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_compile_definitions(mfq-nint-q8-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1)
    mfq_add_cuda_test(mfq-packed-prefill-test
        ${MFQ_CUDA_ROOT}/tests/mfq_packed_prefill_test.cu
        mfq-cuda-native-kernels mfq-core)
    target_compile_definitions(mfq-packed-prefill-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-packed-prefill-test PRIVATE ${MFQ_REPOSITORY_ROOT})
    mfq_add_cuda_test(mfq-paged-kv-test
        ${MFQ_CUDA_ROOT}/tests/mfq_paged_kv_test.cu
        mfq-cuda-native-kernels)
    target_compile_definitions(mfq-paged-kv-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1)
    mfq_add_cuda_test(mfq-deepseek-v41-hc-test
        ${MFQ_CUDA_ROOT}/tests/mfq_deepseek_v41_hc_test.cu
        mfq-cuda-native-kernels)
    target_compile_definitions(mfq-deepseek-v41-hc-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-deepseek-v41-hc-test PRIVATE
        ${MFQ_REPOSITORY_ROOT})
    mfq_add_cuda_test(mfq-native-tensor-cuda-test
        ${MFQ_CUDA_ROOT}/tests/mfq_native_tensor_cuda_test.cu)
    add_test(NAME mfq-contiguous-convert-test
        COMMAND mfq-native-tensor-cuda-test --contiguous-convert-check)
    set_tests_properties(mfq-contiguous-convert-test PROPERTIES SKIP_RETURN_CODE 77)
    mfq_add_cuda_test(mfq-cuda-storage-offset-test
        ${MFQ_CUDA_ROOT}/tests/mfq_cuda_storage_offset_test.cu
        mfq-cuda-native-kernels)
    target_compile_definitions(mfq-cuda-storage-offset-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    mfq_add_cuda_test(mfq-flash-next-test
        ${MFQ_CUDA_ROOT}/tests/mfq_flash_next_test.cu
        mfq-cuda-native-kernels mfq-json)
    target_compile_definitions(mfq-flash-next-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    mfq_add_cuda_test(mfq-nint-row-pipeline-test tests/nint_row_pipeline_test.cpp mfq-model-source mfq-cuda-native-kernels)
    target_compile_definitions(mfq-nint-row-pipeline-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-flash-next-test PRIVATE
        ${MFQ_CUDA_ROOT}
        ${PROJECT_SOURCE_DIR}
        ${MFQ_REPOSITORY_ROOT}
        ${MFQ_CUDA_ROOT}/models
        ${MFQ_CUDA_ROOT}/ops/include
        ${MFQ_CUDA_ROOT}/../../core)
endif()

set(MFQ_NCCL_ROOT "" CACHE PATH
    "Optional NCCL installation root for tensor-parallel inference")
if(UNIX)
    find_library(MFQ_NCCL_LIBRARY
        NAMES nccl libnccl.so.2
        HINTS
            "${MFQ_NCCL_ROOT}/lib"
            "${MFQ_NCCL_ROOT}/lib64"
            "$ENV{NCCL_ROOT}/lib"
            "$ENV{NCCL_ROOT}/lib64")
    find_path(MFQ_NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS
            "${MFQ_NCCL_ROOT}/include"
            "$ENV{NCCL_ROOT}/include")
endif()

find_package(OpenMP QUIET)

function(mfq_configure_cuda_backend_target target)
    target_include_directories(${target} PRIVATE
        ${MFQ_REPOSITORY_ROOT}
        ${MFQ_CUDA_ROOT}
        ${MFQ_CUDA_ROOT}/include
        ${MFQ_CUDA_ROOT}/ops/include
        ${MFQ_CUDA_ROOT}/storage
        ${MFQ_CUDA_KERNEL_ROOT}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE
        mfq-core
        mfq-cpu-ops
        mfq-json
        mfq-ggml-internal-headers
    )
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MFQ_NCCL_LIBRARY AND MFQ_NCCL_INCLUDE_DIR)
        target_include_directories(${target} PRIVATE "${MFQ_NCCL_INCLUDE_DIR}")
        target_link_libraries(${target} PRIVATE "${MFQ_NCCL_LIBRARY}")
        target_compile_definitions(${target} PRIVATE MFQ_HAVE_NCCL=1)
    endif()
endfunction()

# Shared Engine/model templates are instantiated only by backend runtime/loading
# units. Kernel/quant ops have no Engine, model-family, scheduler or transport link.
function(mfq_configure_cuda_runtime_target target)
    mfq_configure_cuda_backend_target(${target})
    target_include_directories(${target} PRIVATE
        ${MFQ_CUDA_ROOT}/models
        ${MFQ_CUDA_ROOT}/commands
        ${MFQ_CUDA_ROOT}/engine
    )
    target_link_libraries(${target} PRIVATE mfq-models mfq-engine)
endfunction()

set(MFQ_CUDA_MODEL_SOURCES
    ${MFQ_CUDA_ROOT}/core/axial_patch_vision.cpp
    ${MFQ_CUDA_ROOT}/models/deepseek_v4/ops.cpp
    ${MFQ_CUDA_ROOT}/models/deepseek_v41/ops.cpp
    ${MFQ_CUDA_ROOT}/models/deepseek_v41/session.cpp
    ${MFQ_CUDA_ROOT}/models/deepseek_v41/engram.cpp
    ${MFQ_CUDA_ROOT}/models/deepseek_v41/prefill.cpp
    ${MFQ_CUDA_ROOT}/models/deepseek_v41/dspark.cpp
    ${MFQ_CUDA_ROOT}/models/deepseek_v41/vision.cpp
    ${MFQ_CUDA_ROOT}/models/qwen4_exp/ops.cpp
    ${MFQ_CUDA_ROOT}/runtime/kv_offload.cpp
    ${MFQ_CUDA_ROOT}/runtime/kv_memory.cpp
    ${MFQ_CUDA_ROOT}/runtime/kv_vmm.cpp
    ${MFQ_CUDA_ROOT}/models/glm5_next/ops.cpp
    ${MFQ_CUDA_ROOT}/models/gemma4/ops.cpp
    ${MFQ_CUDA_ROOT}/models/glm_dsa/ops.cpp
    ${MFQ_CUDA_ROOT}/models/minicpmo45/ops.cpp
    ${MFQ_CUDA_ROOT}/models/nemotron_h/ops.cpp
    ${MFQ_CUDA_ROOT}/models/qwen35/ops.cpp
    ${MFQ_CUDA_ROOT}/models/qwen35/batch_state.cpp
    ${MFQ_CUDA_ROOT}/models/qwen35/batch_graph_state.cpp
)

set(MFQ_CUDA_TRANSFORMER_SOURCES
    ${MFQ_CUDA_ROOT}/core/causal_model.cpp
    ${MFQ_CUDA_ROOT}/core/decode_graph.cpp
    ${MFQ_CUDA_ROOT}/core/ffn.cpp
    ${MFQ_CUDA_ROOT}/core/full_block.cpp
    ${MFQ_CUDA_ROOT}/core/kv_cache.cpp
    ${MFQ_CUDA_ROOT}/core/rope.cpp
    ${MFQ_CUDA_ROOT}/core/pooled_cache.cpp
)

set(MFQ_CUDA_RUNTIME_OP_SOURCES
    ${MFQ_CUDA_ROOT}/ops/cuda_execution.cpp
    ${MFQ_CUDA_ROOT}/ops/batch_selected_attention.cpp
    ${MFQ_CUDA_ROOT}/ops/format.cpp
    ${MFQ_CUDA_ROOT}/ops/gated_residual.cpp
    ${MFQ_CUDA_ROOT}/ops/fp8_sq.cpp
    ${MFQ_CUDA_ROOT}/ops/tq.cpp
    ${MFQ_CUDA_ROOT}/ops/mx.cpp
    ${MFQ_CUDA_ROOT}/ops/mxfp4_sq.cpp
    ${MFQ_CUDA_ROOT}/ops/dense_moe.cpp
    ${MFQ_CUDA_ROOT}/ops/moe.cpp
    ${MFQ_CUDA_ROOT}/ops/mfe_ffn.cpp
    ${MFQ_CUDA_ROOT}/ops/nint.cpp
    ${MFQ_CUDA_ROOT}/ops/quant_linear.cpp
    ${MFQ_CUDA_ROOT}/ops/qsa_scores.cpp
    ${MFQ_CUDA_ROOT}/ops/vq.cpp
)

add_library(mfq-cuda-ops OBJECT ${MFQ_CUDA_RUNTIME_OP_SOURCES})
mfq_configure_cuda_backend_target(mfq-cuda-ops)
target_compile_definitions(mfq-cuda-ops PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-cuda-ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(MFQ_CUDA_LOADING_SOURCES
    ${MFQ_CUDA_ROOT}/storage/load_options.cpp
    ${MFQ_CUDA_ROOT}/storage/model_loader.cpp
    ${MFQ_CUDA_ROOT}/storage/moe_expert_cache.cpp
    ${MFQ_CUDA_ROOT}/storage/moe_ffn_pipeline.cpp
    ${MFQ_CUDA_ROOT}/storage/moe_quant_range_source.cpp
    ${MFQ_CUDA_ROOT}/runtime/moe_host_expert_cache.cpp
    ${MFQ_CUDA_ROOT}/runtime/graph_registry.cpp
    ${MFQ_CUDA_ROOT}/runtime/decode_window.cpp
    ${MFQ_CUDA_ROOT}/runtime/moe_residency.cpp
    ${MFQ_CUDA_ROOT}/runtime/nint_row_pipeline.cpp
    ${MFQ_CUDA_ROOT}/storage/weight_loader.cpp
    ${MFQ_CUDA_ROOT}/storage/transformer_loader.cpp
)
add_library(mfq-cuda-loading OBJECT ${MFQ_CUDA_LOADING_SOURCES})
mfq_configure_cuda_runtime_target(mfq-cuda-loading)
target_compile_definitions(mfq-cuda-loading PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-cuda-loading PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Concrete Engine/model bindings stay in this source group for both native and
# optional Torch builds. Extra targets do not reduce their header dependencies.
set(MFQ_CUDA_INSTANTIATION_SOURCES
    ${MFQ_CUDA_ROOT}/engine/cuda_engine.cpp
    ${MFQ_CUDA_ROOT}/engine/generation.cpp
    ${MFQ_CUDA_ROOT}/storage/text_session_cache.cpp
    ${MFQ_CUDA_ROOT}/storage/deepseek_v41_paged_session.cpp
    ${MFQ_CUDA_ROOT}/engine/mtp.cpp
    ${MFQ_CUDA_ROOT}/engine/cuda_batching.cpp
    ${MFQ_CUDA_ROOT}/engine/cuda_batch_admission.cpp
    ${MFQ_CUDA_ROOT}/engine/cuda_batch_decode.cpp
    ${MFQ_CUDA_ROOT}/engine/cuda_batch_mtp.cpp
    ${MFQ_CUDA_ROOT}/runtime/batch_decode_graph.cpp
    ${MFQ_CUDA_ROOT}/runtime/batch_sampling.cpp
    ${MFQ_CUDA_ROOT}/runtime/batch_mtp.cpp
    ${MFQ_CUDA_ROOT}/runtime/batch_mtp_predictor.cpp
    ${MFQ_CUDA_ROOT}/models/minicpmo45/components.cpp
    ${MFQ_CUDA_ROOT}/commands/cli.cpp
    ${MFQ_CUDA_ROOT}/commands/reference.cpp
    ${MFQ_CUDA_ROOT}/commands/minicpmo45.cpp
    ${MFQ_CUDA_MODEL_SOURCES}
    ${MFQ_CUDA_TRANSFORMER_SOURCES}
)
add_library(mfq-cuda-runtime STATIC
    $<TARGET_OBJECTS:mfq-cuda-ops>
    $<TARGET_OBJECTS:mfq-cuda-loading>
    ${MFQ_CUDA_INSTANTIATION_SOURCES}
)
add_library(mfq::cuda-runtime ALIAS mfq-cuda-runtime)
mfq_configure_cuda_runtime_target(mfq-cuda-runtime)

if(BUILD_TESTING)
    add_executable(mfq-cuda-format-bench
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_format_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-format-bench)
    target_compile_definitions(mfq-cuda-format-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-format-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-e8-dense-probe
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/e8_dense_group_probe.cpp
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/e8_dense_group_probe.cu
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/d4_dense_group_probe.cu)
    mfq_configure_cuda_runtime_target(mfq-cuda-e8-dense-probe)
    target_include_directories(mfq-cuda-e8-dense-probe PRIVATE ${MFQ_REPOSITORY_ROOT})
    target_compile_definitions(mfq-cuda-e8-dense-probe PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-e8-dense-probe PRIVATE mfq-cuda-runtime)
    set_target_properties(mfq-cuda-e8-dense-probe PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}" CUDA_RUNTIME_LIBRARY Shared
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    add_executable(mfq-mfe-e8-local-bounds-test
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_e8_local_bounds_test.cpp)
    target_include_directories(mfq-mfe-e8-local-bounds-test PRIVATE ${MFQ_REPOSITORY_ROOT})
    target_link_libraries(mfq-mfe-e8-local-bounds-test PRIVATE CUDA::cudart)
    target_compile_features(mfq-mfe-e8-local-bounds-test PRIVATE cxx_std_20)
    add_test(NAME mfq-mfe-e8-local-bounds-test COMMAND mfq-mfe-e8-local-bounds-test)
    add_executable(mfq-mfe-e8-default-dispatch-test
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_e8_default_dispatch_test.cpp)
    target_include_directories(mfq-mfe-e8-default-dispatch-test PRIVATE ${MFQ_REPOSITORY_ROOT})
    target_link_libraries(mfq-mfe-e8-default-dispatch-test PRIVATE mfq-cuda-native-kernels)
    target_compile_features(mfq-mfe-e8-default-dispatch-test PRIVATE cxx_std_20)
    add_test(NAME mfq-mfe-e8-default-dispatch-test COMMAND mfq-mfe-e8-default-dispatch-test)
    add_executable(mfq-cuda-format-moe-bench
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_moe_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-format-moe-bench)
    target_compile_definitions(mfq-cuda-format-moe-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-format-moe-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-real-mfe-bench
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_mfe_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-real-mfe-bench)
    target_compile_definitions(mfq-cuda-real-mfe-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-real-mfe-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-mfe-nvq-bank-map-probe
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_nvq_bank_map_probe.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-mfe-nvq-bank-map-probe)
    target_compile_definitions(mfq-cuda-mfe-nvq-bank-map-probe PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-mfe-nvq-bank-map-probe PRIVATE mfq-cuda-runtime)
    # Generated device snapshot keeps occupancy experiments out of the runtime.
    # Refresh with bench/cuda_format_comparison/generate_mfe_gu_register_probe.py.
    if(EXISTS "${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_gu_register_probe_body.inc")
        add_executable(mfq-cuda-mfe-gu-register-probe
            ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_gu_register_probe.cpp
            ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_gu_register_probe.cu)
        mfq_configure_cuda_runtime_target(mfq-cuda-mfe-gu-register-probe)
        target_include_directories(mfq-cuda-mfe-gu-register-probe PRIVATE ${MFQ_REPOSITORY_ROOT})
        target_compile_definitions(mfq-cuda-mfe-gu-register-probe PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
        target_link_libraries(mfq-cuda-mfe-gu-register-probe PRIVATE mfq-cuda-runtime)
        target_compile_options(mfq-cuda-mfe-gu-register-probe PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        set_target_properties(mfq-cuda-mfe-gu-register-probe PROPERTIES
            CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}" CUDA_RUNTIME_LIBRARY Shared
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    endif()
    add_executable(mfq-cuda-nvq-byte-signs-test
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/nvq_byte_signs_test.cu)
    target_link_libraries(mfq-cuda-nvq-byte-signs-test PRIVATE CUDA::cudart)
    set_target_properties(mfq-cuda-nvq-byte-signs-test PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}" CUDA_STANDARD 17 CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME mfq-cuda-nvq-byte-signs-test COMMAND mfq-cuda-nvq-byte-signs-test)
    add_executable(mfq-cuda-real-gr-bench
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_gr_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-real-gr-bench)
    target_compile_definitions(mfq-cuda-real-gr-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-real-gr-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-gr-batch-probe
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_gr_batch_probe.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-gr-batch-probe)
    target_compile_definitions(mfq-cuda-gr-batch-probe PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-gr-batch-probe PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-nint-dense-decode-probe
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/nint_dense_decode_probe.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-nint-dense-decode-probe)
    target_include_directories(mfq-cuda-nint-dense-decode-probe PRIVATE ${MFQ_REPOSITORY_ROOT}/bench)
    target_compile_definitions(mfq-cuda-nint-dense-decode-probe PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-nint-dense-decode-probe PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-shared-gate-bench
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_shared_gate_bench.cu)
    mfq_configure_cuda_runtime_target(mfq-cuda-shared-gate-bench)
    set_target_properties(mfq-cuda-shared-gate-bench PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(mfq-cuda-shared-gate-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-shared-gate-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-bf16-mmvf-bench
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_bf16_mmvf_bench.cu)
    mfq_configure_cuda_runtime_target(mfq-cuda-bf16-mmvf-bench)
    set_target_properties(mfq-cuda-bf16-mmvf-bench PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(mfq-cuda-bf16-mmvf-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-bf16-mmvf-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-nvq-pair-indices-probe
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/nvq_pair_indices_probe.cu)
    target_include_directories(mfq-cuda-nvq-pair-indices-probe PRIVATE ${MFQ_REPOSITORY_ROOT})
    set_target_properties(mfq-cuda-nvq-pair-indices-probe PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    add_executable(mfq-cuda-nvq-compressed-test
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/nvq_compressed_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-nvq-compressed-test)
    target_compile_definitions(mfq-cuda-nvq-compressed-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-nvq-compressed-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-nvq-compressed-test COMMAND mfq-cuda-nvq-compressed-test)
    set_tests_properties(mfq-cuda-nvq-compressed-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-cuda-format-nint-test
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/nint_dense_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-format-nint-test)
    target_compile_definitions(mfq-cuda-format-nint-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-format-nint-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-format-nint-test COMMAND mfq-cuda-format-nint-test)
    set_tests_properties(mfq-cuda-format-nint-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-cuda-routed-nint-test
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/nint_routed_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-routed-nint-test)
    target_compile_definitions(mfq-cuda-routed-nint-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-routed-nint-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-routed-nint-test COMMAND mfq-cuda-routed-nint-test)
    set_tests_properties(mfq-cuda-routed-nint-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-cuda-routed-nvq1-test
        ${MFQ_REPOSITORY_ROOT}/bench/cuda_format_comparison/nvq1_routed_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-routed-nvq1-test)
    target_compile_definitions(mfq-cuda-routed-nvq1-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-routed-nvq1-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-routed-nvq1-test COMMAND mfq-cuda-routed-nvq1-test)
    set_tests_properties(mfq-cuda-routed-nvq1-test PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME mfq-cuda-routed-nint-occupancy-test COMMAND mfq-cuda-routed-nint-test)
    set_tests_properties(mfq-cuda-routed-nint-occupancy-test PROPERTIES
        SKIP_RETURN_CODE 77 ENVIRONMENT "MFQ_NINT_ROUTE_BLOCKS=8;MFQ_NINT_ROUTE_LANES=32")

    add_executable(mfq-cuda-nvq-moe-bench ${MFQ_REPOSITORY_ROOT}/bench/cuda_nvq_moe_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-nvq-moe-bench)
    target_compile_definitions(mfq-cuda-nvq-moe-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-nvq-moe-bench PRIVATE mfq-cuda-runtime)

    add_executable(mfq-cuda-mfe-ffn-bench ${MFQ_REPOSITORY_ROOT}/bench/cuda_mfe_ffn_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-mfe-ffn-bench)
    target_compile_definitions(mfq-cuda-mfe-ffn-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-mfe-ffn-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-wire-bench ${MFQ_REPOSITORY_ROOT}/bench/cuda_moe_wire_bench.cu)
    mfq_configure_cuda_runtime_target(mfq-cuda-wire-bench)
    target_compile_definitions(mfq-cuda-wire-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-wire-bench PRIVATE mfq-cuda-runtime)
    target_compile_options(mfq-cuda-wire-bench PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3>")
    set_target_properties(mfq-cuda-wire-bench PROPERTIES CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}" CUDA_STANDARD 20)

    add_executable(mfq-cuda-mfe-released-test ${MFQ_REPOSITORY_ROOT}/bench/cuda_mfe_released_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-mfe-released-test)
    target_compile_definitions(mfq-cuda-mfe-released-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-mfe-released-test PRIVATE mfq-cuda-runtime)

    add_executable(mfq-sq-linear-test
        ${MFQ_CUDA_ROOT}/tests/mfq_sq_linear_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-sq-linear-test)
    target_compile_definitions(mfq-sq-linear-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-sq-linear-test PRIVATE mfq-cuda-runtime)

    add_executable(mfq-cpu-quant-linear-test
        ${MFQ_CUDA_ROOT}/tests/mfq_cpu_quant_linear_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cpu-quant-linear-test)
    target_compile_definitions(mfq-cpu-quant-linear-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cpu-quant-linear-test PRIVATE mfq-cuda-runtime)

    add_executable(mfq-quant-range-cache-test
        ${MFQ_CUDA_ROOT}/tests/mfq_quant_range_cache_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-quant-range-cache-test)
    target_compile_definitions(mfq-quant-range-cache-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-quant-range-cache-test PRIVATE mfq-cuda-runtime)
    add_executable(mfq-mxfp8-layout-test ${MFQ_CUDA_ROOT}/tests/mxfp8_layout_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-mxfp8-layout-test)
    target_compile_definitions(mfq-mxfp8-layout-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-mxfp8-layout-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-mxfp8-layout-test COMMAND mfq-mxfp8-layout-test)
    add_executable(mfq-nope-latent-moe-test ${MFQ_CUDA_ROOT}/tests/nope_latent_moe_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-nope-latent-moe-test)
    target_compile_definitions(mfq-nope-latent-moe-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-nope-latent-moe-test PRIVATE
        mfq-cuda-runtime mfq-cuda-native-kernels mfq-ggml)
    add_test(NAME mfq-nope-latent-moe-test COMMAND mfq-nope-latent-moe-test)
    set_tests_properties(mfq-nope-latent-moe-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-moe-route-map-test ${MFQ_CUDA_ROOT}/tests/moe_route_map_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-moe-route-map-test)
    target_compile_definitions(mfq-moe-route-map-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-moe-route-map-test PRIVATE
        mfq-cuda-runtime mfq-cuda-native-kernels)
    add_test(NAME mfq-moe-route-map-test COMMAND mfq-moe-route-map-test)
    set_tests_properties(mfq-moe-route-map-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-qk-rope-kv-test ${MFQ_CUDA_ROOT}/tests/qk_rope_kv_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-qk-rope-kv-test)
    target_compile_definitions(mfq-qk-rope-kv-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-qk-rope-kv-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-qk-rope-kv-test COMMAND mfq-qk-rope-kv-test)
    set_tests_properties(mfq-qk-rope-kv-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-moe-ffn-pipeline-test ${MFQ_CUDA_ROOT}/tests/moe_ffn_pipeline_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-moe-ffn-pipeline-test)
    target_compile_definitions(mfq-moe-ffn-pipeline-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-moe-ffn-pipeline-test PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-batch-decode-graph-test ${MFQ_CUDA_ROOT}/tests/batch_decode_graph_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-batch-decode-graph-test)
    target_compile_definitions(mfq-cuda-batch-decode-graph-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-batch-decode-graph-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-batch-decode-graph-test COMMAND mfq-cuda-batch-decode-graph-test)
    set_tests_properties(mfq-cuda-batch-decode-graph-test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(mfq-cuda-dma-batch-test ${MFQ_CUDA_ROOT}/tests/dma_copy_batch_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-dma-batch-test)
    target_compile_definitions(mfq-cuda-dma-batch-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-dma-batch-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-dma-batch-test COMMAND mfq-cuda-dma-batch-test)
    add_executable(mfq-cuda-warp-multi-sum-test ${MFQ_CUDA_ROOT}/tests/warp_multi_sum_test.cu)
    target_include_directories(mfq-cuda-warp-multi-sum-test PRIVATE ${MFQ_CUDA_KERNEL_ROOT})
    target_link_libraries(mfq-cuda-warp-multi-sum-test PRIVATE CUDA::cudart)
    set_target_properties(mfq-cuda-warp-multi-sum-test PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME mfq-cuda-warp-multi-sum-test COMMAND mfq-cuda-warp-multi-sum-test)
    add_executable(mfq-window-attention-test ${MFQ_CUDA_ROOT}/tests/decode_window_attention_test.cpp)
    add_executable(mfq-gdn-decode-window-test ${MFQ_CUDA_ROOT}/tests/gdn_decode_window_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-gdn-decode-window-test)
    target_compile_definitions(mfq-gdn-decode-window-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-gdn-decode-window-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-gdn-decode-window-test COMMAND mfq-gdn-decode-window-test)
    mfq_configure_cuda_runtime_target(mfq-window-attention-test)
    target_compile_definitions(mfq-window-attention-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-window-attention-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-window-attention-test COMMAND mfq-window-attention-test)
    foreach(bits 0 3.5)
        add_test(NAME mfq-cuda-qsa-independent-batch-${bits}-test
            COMMAND mfq-window-attention-test --qsa-independent-batch-check)
        set_tests_properties(mfq-cuda-qsa-independent-batch-${bits}-test PROPERTIES
            ENVIRONMENT "MFQ_KV_TURBOQUANT_BITS=${bits}")
    endforeach()
    add_test(NAME mfq-cuda-gdn-prefill-test COMMAND mfq-window-attention-test --gdn-prefill-check)
    foreach(check chunk chunk-varlen chunk-graph)
        add_test(NAME mfq-cuda-gdn-prefill-${check}-test
            COMMAND mfq-window-attention-test --gdn-prefill-${check}-check)
    endforeach()
    add_test(NAME mfq-cuda-qsa-live-scores-test COMMAND mfq-window-attention-test --qsa-live-score-check)
    add_test(NAME mfq-cuda-qsa-prefill-scores-test COMMAND mfq-window-attention-test --qsa-prefill-score-check)
    add_test(NAME mfq-cuda-qsa-pool-write-test COMMAND mfq-window-attention-test --qsa-pool-write-check)
    add_test(NAME mfq-cuda-qsa-incremental-pool-test COMMAND mfq-window-attention-test --qsa-incremental-pool-check)
    add_test(NAME mfq-cuda-qsa-prefill-test COMMAND mfq-window-attention-test --qsa-prefill-check)
    add_test(NAME mfq-cuda-qsa-history-prime-test COMMAND mfq-window-attention-test --qsa-history-prime-check)
    add_test(NAME mfq-cuda-qsa-kv-offload-test COMMAND mfq-window-attention-test --qsa-kv-offload-check)
    foreach(bits 0 2.5 4)
        add_test(NAME mfq-cuda-qsa-stream-graph-${bits}-test COMMAND mfq-window-attention-test --qsa-stream-graph-check)
        set_tests_properties(mfq-cuda-qsa-stream-graph-${bits}-test PROPERTIES
            ENVIRONMENT "MFQ_KV_TURBOQUANT_BITS=${bits};MFQ_KV_STREAM_GRAPHS=1;MFQ_MEMORY_ARCHITECTURE=discrete")
    endforeach()
    add_test(NAME mfq-cuda-turboquant-kv-test COMMAND mfq-window-attention-test --turboquant-kv-check)
    add_test(NAME mfq-cuda-turboquant-qsa-test COMMAND mfq-window-attention-test --turboquant-qsa-check)
    set_tests_properties(mfq-cuda-turboquant-qsa-test PROPERTIES ENVIRONMENT "MFQ_KV_TURBOQUANT_BITS=3.5")
    mfq_configure_cuda_runtime_target(mfq-nint-row-pipeline-test)
    target_link_libraries(mfq-nint-row-pipeline-test PRIVATE mfq-cuda-runtime)

    add_executable(mfq-host-expert-cache-test
        ${MFQ_CUDA_ROOT}/tests/moe_host_expert_cache_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-host-expert-cache-test)
    target_compile_definitions(mfq-host-expert-cache-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-host-expert-cache-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-host-expert-cache-test COMMAND mfq-host-expert-cache-test)

    add_executable(mfq-ssd-expert-streaming-test ${MFQ_CUDA_ROOT}/tests/ssd_expert_streaming_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-ssd-expert-streaming-test)
    target_compile_definitions(mfq-ssd-expert-streaming-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-ssd-expert-streaming-test PRIVATE mfq-cuda-runtime)
    if(MSVC)
        target_compile_options(mfq-ssd-expert-streaming-test PRIVATE /EHsc)
        target_compile_options(mfq-host-expert-cache-test PRIVATE /EHsc)
        # Cancellation checks must unwind CUDA stream guards before recovery.
        target_compile_options(mfq-nint-row-pipeline-test PRIVATE /EHsc)
    endif()

    add_executable(mfq-float-projection-test ${MFQ_CUDA_ROOT}/tests/float_projection_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-float-projection-test)
    target_compile_definitions(mfq-float-projection-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-float-projection-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-float-projection-test COMMAND mfq-float-projection-test)
    add_test(NAME mfq-gr-prefill-blas-test COMMAND mfq-float-projection-test --prefill-blas-check)

    add_executable(mfq-deepseek-v41-ced-test ${MFQ_CUDA_ROOT}/tests/deepseek_v41_ced_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-deepseek-v41-ced-test)
    target_link_libraries(mfq-deepseek-v41-ced-test PRIVATE mfq-cuda-runtime)
    target_compile_definitions(mfq-deepseek-v41-ced-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    add_test(NAME mfq-deepseek-v41-ced-test COMMAND mfq-deepseek-v41-ced-test)
    set_tests_properties(mfq-deepseek-v41-ced-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-deepseek-v41-engram-test ${MFQ_CUDA_ROOT}/tests/deepseek_v41_engram_test.cpp)
    target_link_libraries(mfq-deepseek-v41-engram-test PRIVATE mfq-cuda-runtime)
    mfq_configure_cuda_runtime_target(mfq-deepseek-v41-engram-test)
    target_compile_definitions(mfq-deepseek-v41-engram-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    add_test(NAME mfq-deepseek-v41-engram-test COMMAND mfq-deepseek-v41-engram-test)
    set_tests_properties(mfq-deepseek-v41-engram-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-deepseek-v41-vision-test ${MFQ_CUDA_ROOT}/tests/deepseek_v41_vision_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-deepseek-v41-vision-test)
    target_link_libraries(mfq-deepseek-v41-vision-test PRIVATE mfq-cuda-runtime)
    target_compile_definitions(mfq-deepseek-v41-vision-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    add_test(NAME mfq-deepseek-v41-vision-test COMMAND mfq-deepseek-v41-vision-test)
    set_tests_properties(mfq-deepseek-v41-vision-test PROPERTIES SKIP_RETURN_CODE 77)
    if(MSVC)
        target_compile_options(mfq-deepseek-v41-engram-test PRIVATE /EHsc)
        target_compile_options(mfq-deepseek-v41-vision-test PRIVATE /EHsc)
        target_compile_options(mfq-deepseek-v41-ced-test PRIVATE /EHsc)
    endif()
    add_executable(mfq-deepseek-v41-indexer-bench ${MFQ_REPOSITORY_ROOT}/bench/cuda_deepseek_v41_indexer_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-deepseek-v41-indexer-bench)
    target_compile_definitions(mfq-deepseek-v41-indexer-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-deepseek-v41-indexer-bench PRIVATE mfq-cuda-runtime)
    if(MSVC)
        target_compile_options(mfq-deepseek-v41-indexer-bench PRIVATE /EHsc)
    endif()
    add_executable(mfq-deepseek-v41-hc-bench ${MFQ_REPOSITORY_ROOT}/bench/cuda_deepseek_v41_hc_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-deepseek-v41-hc-bench)
    target_compile_definitions(mfq-deepseek-v41-hc-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-deepseek-v41-hc-bench PRIVATE mfq-cuda-runtime)
    if(MSVC)
        target_compile_options(mfq-deepseek-v41-hc-bench PRIVATE /EHsc)
    endif()
    add_executable(mfq-cuda-deepseek-v41-axial-vision-test
        ${MFQ_CUDA_ROOT}/tests/deepseek_v41_axial_vision_gpu_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-deepseek-v41-axial-vision-test)
    target_compile_definitions(mfq-cuda-deepseek-v41-axial-vision-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-deepseek-v41-axial-vision-test PRIVATE mfq-cuda-runtime)
    target_compile_features(mfq-cuda-deepseek-v41-axial-vision-test PRIVATE cxx_std_20)
    add_test(NAME mfq-cuda-deepseek-v41-axial-vision-test
        COMMAND mfq-cuda-deepseek-v41-axial-vision-test)
    set_tests_properties(mfq-cuda-deepseek-v41-axial-vision-test PROPERTIES SKIP_RETURN_CODE 77)
endif()
target_link_libraries(mfq-cuda-runtime PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    mfq-cuda-core
    mfq-cuda-native-kernels
    mfq-cuda-storage
)
target_compile_definitions(mfq-cuda-runtime PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1
    NOMINMAX
)
if(MSVC)
    foreach(target mfq-cuda-ops mfq-cuda-loading mfq-cuda-runtime)
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
            "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>"
            "$<$<COMPILE_LANGUAGE:CXX>:/bigobj>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
        )
    endforeach()
endif()
set_target_properties(mfq-cuda-runtime PROPERTIES
    CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)

if(BUILD_TESTING)
    add_executable(mfq-cuda-execution-scope-test
        ${MFQ_CUDA_ROOT}/tests/cuda_execution_scope_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-execution-scope-test)
    target_compile_definitions(mfq-cuda-execution-scope-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-execution-scope-test PRIVATE
        mfq-cuda-runtime)
    target_link_libraries(mfq-flash-next-test PRIVATE
        mfq-cuda-runtime)
    add_test(NAME mfq-cuda-execution-scope-test
        COMMAND mfq-cuda-execution-scope-test)

    add_executable(mfq-cuda-runtime-state-test
        ${MFQ_CUDA_ROOT}/tests/cuda_runtime_state_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-runtime-state-test)
    if(MSVC)
        target_compile_options(mfq-cuda-runtime-state-test PRIVATE /EHsc)
    endif()
    target_compile_definitions(mfq-cuda-runtime-state-test PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-runtime-state-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-runtime-state-test COMMAND mfq-cuda-runtime-state-test)
    add_test(NAME mfq-cuda-mfe-prepare-test COMMAND mfq-cuda-runtime-state-test --mfe-prepare-check)
    add_test(NAME mfq-cuda-mfe-transfer-test COMMAND mfq-cuda-runtime-state-test --mfe-transfer-check)
    add_test(NAME mfq-cuda-mapped-copy-test COMMAND mfq-cuda-runtime-state-test --mapped-copy-check)
    add_test(NAME mfq-cuda-batched-sampling-test COMMAND mfq-cuda-runtime-state-test --batched-sampling-check)
    set_tests_properties(mfq-cuda-batched-sampling-test PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME mfq-cuda-sampling-keys-test COMMAND mfq-cuda-runtime-state-test --sampling-check)
    add_test(NAME mfq-cuda-transfer-copy-test COMMAND mfq-cuda-runtime-state-test --transfer-copy-check)
    set_tests_properties(mfq-cuda-runtime-state-test PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(mfq-cuda-mfe-prepare-test mfq-cuda-mfe-transfer-test mfq-cuda-mapped-copy-test PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(mfq-cuda-sampling-keys-test PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(mfq-cuda-transfer-copy-test PROPERTIES SKIP_RETURN_CODE 77)
endif()

add_library(mfq-cuda-eval STATIC ${MFQ_CUDA_ROOT}/eval/kl.cpp)
mfq_configure_cuda_runtime_target(mfq-cuda-eval)
target_include_directories(mfq-cuda-eval PRIVATE ${MFQ_CUDA_ROOT}/storage)
target_compile_definitions(mfq-cuda-eval PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-cuda-eval PUBLIC mfq-cuda-runtime)

add_library(mfq-cuda-diagnostics STATIC
    ${MFQ_CUDA_ROOT}/diagnostics/attention_checks.cpp
    ${MFQ_CUDA_ROOT}/diagnostics/backend_checks.cpp
    ${MFQ_CUDA_ROOT}/diagnostics/linear_checks.cpp
    ${MFQ_CUDA_ROOT}/diagnostics/moe_checks.cpp
    ${MFQ_CUDA_ROOT}/diagnostics/runtime_checks.cpp
    ${MFQ_CUDA_ROOT}/diagnostics/session_checks.cpp
)
mfq_configure_cuda_runtime_target(mfq-cuda-diagnostics)
target_include_directories(mfq-cuda-diagnostics PRIVATE ${MFQ_CUDA_ROOT}/storage)
target_compile_definitions(mfq-cuda-diagnostics PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-cuda-diagnostics PUBLIC mfq-cuda-runtime)
if(MSVC)
    foreach(target mfq-cuda-diagnostics mfq-cuda-eval)
        target_compile_options(${target} PRIVATE /utf-8 /EHsc /bigobj)
    endforeach()
endif()

add_executable(mfq-runtime
    ${MFQ_CUDA_ROOT}/apps/runtime_main.cpp
    ${MFQ_CUDA_ROOT}/commands/runtime.cpp
)
target_link_libraries(mfq-runtime PRIVATE
    mfq-cuda-runtime mfq-runtime-communication)
target_include_directories(mfq-runtime PRIVATE
    ${MFQ_CUDA_ROOT}
    ${MFQ_CUDA_ROOT}/include
    ${MFQ_CUDA_ROOT}/engine
)
target_compile_features(mfq-runtime PRIVATE cxx_std_20)
set_target_properties(mfq-runtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(mfq-diagnostics
    ${MFQ_CUDA_ROOT}/apps/diagnostics_main.cpp
    ${MFQ_CUDA_ROOT}/commands/diagnostics.cpp
)
mfq_configure_cuda_runtime_target(mfq-diagnostics)
target_include_directories(mfq-diagnostics PRIVATE ${MFQ_CUDA_ROOT}/storage)
target_compile_definitions(mfq-diagnostics PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-diagnostics PRIVATE mfq-cuda-diagnostics)
target_compile_features(mfq-diagnostics PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(mfq-diagnostics PRIVATE /utf-8 /EHsc /bigobj)
endif()
set_target_properties(mfq-diagnostics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(mfq-eval
    ${MFQ_CUDA_ROOT}/apps/eval_main.cpp
    ${MFQ_CUDA_ROOT}/commands/eval.cpp
)
mfq_configure_cuda_runtime_target(mfq-eval)
target_include_directories(mfq-eval PRIVATE ${MFQ_CUDA_ROOT}/storage)
target_compile_definitions(mfq-eval PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-eval PRIVATE
    mfq-cuda-eval mfq-cuda-runtime mfq-cuda-native-kernels mfq-ggml)
target_compile_features(mfq-eval PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(mfq-eval PRIVATE /utf-8 /EHsc /bigobj)
endif()
set_target_properties(mfq-eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

if(BUILD_TESTING)
    add_test(
        NAME mfq-deepseek-v41-runtime-check
        COMMAND mfq-diagnostics --check-deepseek-v41)
    add_test(NAME mfq-diagnostics-help COMMAND mfq-diagnostics --help)
    add_test(NAME mfq-eval-help COMMAND mfq-eval --help)
    add_test(NAME mfq-eval-kl-help COMMAND mfq-eval kl --help)
    add_test(NAME mfq-runtime-cli-options COMMAND mfq-runtime --transport stdio --continuous-batching 2 --help)
    add_test(NAME mfq-diagnostics-cli-options COMMAND mfq-diagnostics --ids 1,2 --gen 3 --help)
    add_test(NAME mfq-eval-cli-options COMMAND mfq-eval kl --kl-base x --kl-evaluator optimized --help)
    add_test(NAME mfq-runtime-rejects-check
        COMMAND mfq-runtime --check-mfq-container)
    add_test(NAME mfq-runtime-rejects-kl
        COMMAND mfq-runtime --kl-base reference.bin)
    add_test(NAME mfq-runtime-rejects-bad-number
        COMMAND mfq-runtime --port 3x --help)
    add_test(NAME mfq-runtime-rejects-missing-value
        COMMAND mfq-runtime --model --help)
    add_test(NAME mfq-runtime-rejects-token-input
        COMMAND mfq-runtime --ids 1,2 --help)
    add_test(NAME mfq-diagnostics-rejects-bad-ids
        COMMAND mfq-diagnostics --ids 1,x --help)
    add_test(NAME mfq-diagnostics-rejects-kl
        COMMAND mfq-diagnostics --kl-base reference.bin)
    add_test(NAME mfq-diagnostics-rejects-missing-value
        COMMAND mfq-diagnostics --check-linear --check-backend-argmax)
    add_test(NAME mfq-eval-rejects-check
        COMMAND mfq-eval kl --check-mfq-container)
    add_test(NAME mfq-eval-rejects-bad-number
        COMMAND mfq-eval kl --kl-chunks 1x --help)
    set_tests_properties(
        mfq-runtime-rejects-check mfq-runtime-rejects-kl
        mfq-runtime-rejects-bad-number mfq-runtime-rejects-missing-value
        mfq-runtime-rejects-token-input mfq-diagnostics-rejects-bad-ids
        mfq-diagnostics-rejects-kl mfq-diagnostics-rejects-missing-value
        mfq-eval-rejects-check mfq-eval-rejects-bad-number
        PROPERTIES WILL_FAIL TRUE)
endif()

option(MFQ_BUILD_TORCH_REFERENCE_RUNTIME
    "Build the optional LibTorch CUDA runtime used only for A/B validation"
    OFF)
if(MFQ_BUILD_TORCH_REFERENCE_RUNTIME)
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    if(NOT TARGET CUDA::nvToolsExt)
        find_library(NVTOOLSEXT_LIBRARY
            NAMES nvToolsExt nvToolsExt64_1
            PATHS "${CUDAToolkit_LIBRARY_DIR}" "$ENV{CUDA_PATH}/lib/x64"
            NO_DEFAULT_PATH)
        if(NVTOOLSEXT_LIBRARY)
            add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
            set_target_properties(CUDA::nvToolsExt PROPERTIES
                IMPORTED_LOCATION "${NVTOOLSEXT_LIBRARY}")
        else()
            add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        endif()
    endif()
    find_package(Torch REQUIRED)
    add_executable(mfq-runtime-torch
        ${MFQ_CUDA_ROOT}/apps/runtime_main.cpp
        ${MFQ_CUDA_ROOT}/commands/runtime.cpp
        ${MFQ_CUDA_RUNTIME_OP_SOURCES}
        ${MFQ_CUDA_LOADING_SOURCES}
        ${MFQ_CUDA_INSTANTIATION_SOURCES}
        ${MFQ_CUDA_ROOT}/apps/ggml_cuda_compat.cu
        ${MFQ_CUDA_KERNEL_SOURCES}
    )
    mfq_configure_cuda_runtime_target(mfq-runtime-torch)
    target_include_directories(mfq-runtime-torch PRIVATE ${Python_INCLUDE_DIRS})
    target_link_libraries(mfq-runtime-torch PRIVATE
        mfq-runtime-communication
        ${TORCH_LIBRARIES}
        Python::Python
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::cublas
        mfq-cuda-storage
    )
    target_compile_definitions(mfq-runtime-torch PRIVATE NOMINMAX)
    target_compile_options(mfq-runtime-torch PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
    )
    set_target_properties(mfq-runtime-torch PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(NOT WIN32)
    set(MFQ_SYSTEM_RUNTIME_DIR
        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH
        "directory containing the host libstdc++ runtime")
    set_property(TARGET mfq-runtime PROPERTY
        BUILD_RPATH "${MFQ_SYSTEM_RUNTIME_DIR}")
endif()

set_source_files_properties(
    ${MFQ_CUDA_KERNEL_ROOT}/attention_mma.cu
    ${MFQ_CUDA_KERNEL_ROOT}/deepseek_v4_attention.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math;--extended-lambda"
)
set_source_files_properties(
    ${MFQ_CUDA_KERNEL_ROOT}/deepseek_v4_hc.cu
    PROPERTIES COMPILE_OPTIONS "--extended-lambda"
)

set_source_files_properties(${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_shared.cu PROPERTIES
    OBJECT_DEPENDS "${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_nint_tokens.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_shared.h;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_shared_short.cuh")

set_source_files_properties(${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_cached_reduce.cu PROPERTIES
    OBJECT_DEPENDS "${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_cached_reduce.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_nvq_compact_dot.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_nvq_parity_read.cuh;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn.h;${MFQ_CUDA_KERNEL_ROOT}/mfe_ffn_launch.h")
