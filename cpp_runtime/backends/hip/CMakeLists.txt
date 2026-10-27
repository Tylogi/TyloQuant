# Native HIP backend (AMD ROCm).
#
# Mirrors backends/cuda: the shared kernels live in the Python package
# (MFQ_HIP_KERNEL_ROOT), tensor/storage/engine/model code in this tree.
# Kernel sources were produced by hipify and are fixed up incrementally;
# tensor-core tile paths that rely on NVIDIA-only MMA intrinsics are
# excluded from the source lists until they gain MFMA or direct-path
# implementations.

if(NOT TARGET mfq-runtime-communication)
    message(FATAL_ERROR
        "The HIP runtime requires MFQ_BUILD_RUNTIME_COMMUNICATION=ON")
endif()

enable_language(HIP)

set(MFQ_HIP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(MFQ_HIP_KERNEL_ROOT "${MFQ_REPOSITORY_ROOT}/mfq/kernels/hip")

set(MFQ_HIP_ARCHITECTURES "gfx942" CACHE STRING
    "HIP architectures compiled into MFQ HIP targets")

find_package(hip REQUIRED CONFIG
    HINTS "${ROCM_PATH}" "$ENV{ROCM_PATH}" /opt/rocm)
find_package(hipblas REQUIRED CONFIG
    HINTS "${ROCM_PATH}" "$ENV{ROCM_PATH}" /opt/rocm)
find_package(Threads REQUIRED)

add_library(mfq-hip-storage STATIC
    ${MFQ_HIP_ROOT}/storage/mfe_expert_store.cpp
)
add_library(mfq::hip-storage ALIAS mfq-hip-storage)
target_include_directories(mfq-hip-storage PUBLIC
    ${MFQ_HIP_ROOT}/storage
)
target_link_libraries(mfq-hip-storage PUBLIC Threads::Threads)
target_compile_definitions(mfq-hip-storage PUBLIC __HIP_PLATFORM_AMD__=1)
target_compile_features(mfq-hip-storage PUBLIC cxx_std_20)
set_target_properties(mfq-hip-storage PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(mfq-hip-core STATIC
    ${MFQ_HIP_ROOT}/src/mfq_cuda_context.hip
    ${MFQ_HIP_ROOT}/src/mfq_native_tensor.cpp
    ${MFQ_HIP_ROOT}/src/mfq_native_tensor.hip
    ${MFQ_HIP_ROOT}/src/mfq_f32_matmul_lt.hip
    ${MFQ_HIP_ROOT}/src/mfq_native_tensor_blas.hip
    ${MFQ_HIP_ROOT}/src/mfq_native_tensor_creation.hip
    ${MFQ_HIP_ROOT}/src/mfq_native_tensor_indexing.hip
    ${MFQ_HIP_ROOT}/src/mfq_native_tensor_ops.hip
    ${MFQ_HIP_ROOT}/src/mfq_native_tensor_reduction.hip
    ${MFQ_HIP_ROOT}/src/mfq_native_tensor_signal.hip
    ${MFQ_HIP_ROOT}/src/mfq_native_tensor_sort.hip
)
add_library(mfq::hip-core ALIAS mfq-hip-core)
target_include_directories(mfq-hip-core PUBLIC
    ${MFQ_HIP_ROOT}/include
    ${MFQ_HIP_KERNEL_ROOT}
)
target_link_libraries(mfq-hip-core PUBLIC hip::host hipblas mfq-host-parallel)
find_package(hipblaslt QUIET CONFIG HINTS "${ROCM_PATH}" "$ENV{ROCM_PATH}" /opt/rocm)
if(TARGET roc::hipblaslt)
    target_link_libraries(mfq-hip-core PRIVATE roc::hipblaslt)
    target_compile_definitions(mfq-hip-core PRIVATE MFQ_HAVE_HIPBLASLT=1)
    # ROCm wheels copy development libraries without their kernel data. Load
    # the runtime copy so hipBLASLt can find its adjacent Tensile kernels.
    get_filename_component(MFQ_ROCM_WHEEL_BASE "${hipblaslt_DIR}/../../../.." ABSOLUTE)
    set(MFQ_ROCM_WHEEL_LIB "${MFQ_ROCM_WHEEL_BASE}/_rocm_sdk_libraries/lib")
    if(EXISTS "${MFQ_ROCM_WHEEL_LIB}/hipblaslt/library")
        # The wheel's LD_LIBRARY_PATH also contains the incomplete devel copy.
        target_link_options(mfq-hip-core INTERFACE
            "LINKER:--disable-new-dtags,-rpath,${MFQ_ROCM_WHEEL_LIB}")
    endif()
endif()
target_compile_features(mfq-hip-core PUBLIC cxx_std_20)
set_target_properties(mfq-hip-core PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES "${MFQ_HIP_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

set(MFQ_HIP_KERNEL_SOURCES
    ${MFQ_HIP_KERNEL_ROOT}/tq_gr.hip
    ${MFQ_HIP_KERNEL_ROOT}/tq_gr_dense.hip
    ${MFQ_HIP_KERNEL_ROOT}/tq_shared.hip
    ${MFQ_HIP_KERNEL_ROOT}/tq_moe.hip
    ${MFQ_HIP_KERNEL_ROOT}/tq_projection.hip
    ${MFQ_HIP_KERNEL_ROOT}/tq_prefill.hip
    ${MFQ_HIP_KERNEL_ROOT}/tq_q8.hip
    ${MFQ_HIP_KERNEL_ROOT}/tq_input_gate.hip
    ${MFQ_HIP_KERNEL_ROOT}/acc.hip
    ${MFQ_HIP_KERNEL_ROOT}/activation.hip
    ${MFQ_HIP_KERNEL_ROOT}/attention.hip
    ${MFQ_HIP_KERNEL_ROOT}/attention_mma.hip
    ${MFQ_HIP_KERNEL_ROOT}/embedding.hip
    ${MFQ_HIP_KERNEL_ROOT}/selected_attention.hip
    ${MFQ_HIP_KERNEL_ROOT}/qwen4_exp.hip
    ${MFQ_HIP_KERNEL_ROOT}/glm5_next.hip
    ${MFQ_HIP_KERNEL_ROOT}/gated_delta_net.hip
    ${MFQ_HIP_KERNEL_ROOT}/gdn_stateful_decode.hip
    ${MFQ_HIP_KERNEL_ROOT}/gdn_decode_window.hip
    ${MFQ_HIP_KERNEL_ROOT}/kv_cache.hip
    ${MFQ_HIP_KERNEL_ROOT}/paged_kv.hip
    ${MFQ_HIP_KERNEL_ROOT}/moe.hip
    ${MFQ_HIP_KERNEL_ROOT}/moe_route.hip
    ${MFQ_HIP_KERNEL_ROOT}/mfe_ffn.hip
    ${MFQ_HIP_KERNEL_ROOT}/mfe_ffn_launch.cpp
    ${MFQ_HIP_KERNEL_ROOT}/mx_matmul.hip
    ${MFQ_HIP_KERNEL_ROOT}/mxfp8_grouped.hip
    ${MFQ_HIP_KERNEL_ROOT}/fp8_sq.hip
    ${MFQ_HIP_KERNEL_ROOT}/mxfp4_sq.hip
    ${MFQ_HIP_KERNEL_ROOT}/nepq.hip
    ${MFQ_HIP_KERNEL_ROOT}/nepq_residual.hip
    ${MFQ_HIP_KERNEL_ROOT}/nint_matmul.hip
    ${MFQ_HIP_KERNEL_ROOT}/norm.hip
    ${MFQ_HIP_KERNEL_ROOT}/nvq_matmul.hip
    ${MFQ_HIP_KERNEL_ROOT}/rope.hip
    ${MFQ_HIP_KERNEL_ROOT}/sampling.hip
    ${MFQ_HIP_KERNEL_ROOT}/ssm_conv.hip
    ${MFQ_HIP_KERNEL_ROOT}/selective_state_scan.hip
    ${MFQ_HIP_KERNEL_ROOT}/deepseek_v41.hip
    ${MFQ_HIP_ROOT}/kernels/kv_quantization.hip
    ${MFQ_HIP_ROOT}/kernels/kv_hot_cache.hip
    ${MFQ_HIP_KERNEL_ROOT}/deepseek_v4_hc.hip
    ${MFQ_HIP_KERNEL_ROOT}/deepseek_v4_attention.hip
    ${MFQ_HIP_KERNEL_ROOT}/deepseek_v4_fp4.hip
    ${MFQ_HIP_KERNEL_ROOT}/deepseek_v41_index.hip
    ${MFQ_HIP_KERNEL_ROOT}/deepseek_v41_block_sort.hip
    ${MFQ_HIP_KERNEL_ROOT}/glm_dsa.hip
)

add_library(mfq-hip-native-kernels STATIC
    ${MFQ_HIP_KERNEL_SOURCES}
    ${MFQ_HIP_ROOT}/kernels/float_projection.hip
    ${MFQ_HIP_ROOT}/kernels/gated_residual_fused.hip
    ${MFQ_HIP_ROOT}/runtime/moe_pipeline.hip
    ${MFQ_HIP_ROOT}/apps/ggml_cuda_compat.hip
)
add_library(mfq::hip-native-kernels ALIAS mfq-hip-native-kernels)
target_compile_definitions(mfq-hip-native-kernels PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1
)
target_include_directories(mfq-hip-native-kernels PRIVATE
    ${MFQ_REPOSITORY_ROOT}
    ${MFQ_HIP_ROOT}/include
    ${MFQ_HIP_KERNEL_ROOT}
    ${MFQ_REPOSITORY_ROOT}/cpp_runtime/components/ggml/src/ggml-hip/vendors
)
target_link_libraries(mfq-hip-native-kernels
    PUBLIC mfq-hip-core
    PRIVATE mfq-core mfq-ggml-internal-headers
)
# nvq_matmul's heavily templated tile kernels take ~25 minutes at -O3 for
# gfx942 with negligible runtime difference on the memory-bound decode path;
# compile it at -O2 (the later flag wins over the config -O3).
set_source_files_properties(
    ${MFQ_HIP_KERNEL_ROOT}/nvq_matmul.hip
    PROPERTIES COMPILE_OPTIONS "-O2"
)
set_target_properties(mfq-hip-native-kernels PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES "${MFQ_HIP_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

find_package(OpenMP QUIET)

function(mfq_configure_hip_backend_target target)
    target_include_directories(${target} PRIVATE
        ${MFQ_REPOSITORY_ROOT}
        ${MFQ_HIP_ROOT}
        ${MFQ_HIP_ROOT}/include
        ${MFQ_HIP_ROOT}/ops/include
        ${MFQ_HIP_ROOT}/storage
        ${MFQ_HIP_KERNEL_ROOT}
    )
    target_link_libraries(${target} PRIVATE
        mfq-core
        mfq-cpu-ops
        mfq-json
        mfq-ggml-internal-headers
    )
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(${target} PRIVATE __HIP_PLATFORM_AMD__=1)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

function(mfq_configure_hip_runtime_target target)
    mfq_configure_hip_backend_target(${target})
    target_include_directories(${target} PRIVATE
        ${MFQ_HIP_ROOT}/models
        ${MFQ_HIP_ROOT}/commands
        ${MFQ_HIP_ROOT}/engine
    )
    target_link_libraries(${target} PRIVATE mfq-models mfq-engine)
endfunction()

set(MFQ_HIP_MODEL_SOURCES
    ${MFQ_HIP_ROOT}/models/deepseek_v4/ops.cpp
    ${MFQ_HIP_ROOT}/models/nemotron_h/ops.cpp
    ${MFQ_HIP_ROOT}/models/deepseek_v41/ops.cpp
    ${MFQ_HIP_ROOT}/models/deepseek_v41/engram.cpp
    ${MFQ_HIP_ROOT}/models/deepseek_v41/prefill.cpp
    ${MFQ_HIP_ROOT}/models/deepseek_v41/vision.cpp
    ${MFQ_HIP_ROOT}/models/deepseek_v41/session.cpp
    ${MFQ_HIP_ROOT}/models/deepseek_v41/dspark.cpp
    ${MFQ_HIP_ROOT}/models/qwen4_exp/ops.cpp
    ${MFQ_HIP_ROOT}/models/glm5_next/ops.cpp
    ${MFQ_HIP_ROOT}/models/gemma4/ops.cpp
    ${MFQ_HIP_ROOT}/models/glm_dsa/ops.cpp
    ${MFQ_HIP_ROOT}/models/minicpmo45/ops.cpp
    ${MFQ_HIP_ROOT}/models/minicpmo45/components.cpp
    ${MFQ_HIP_ROOT}/models/qwen35/ops.cpp
    ${MFQ_HIP_ROOT}/models/qwen35/batch_state.cpp
)

set(MFQ_HIP_TRANSFORMER_SOURCES
    ${MFQ_HIP_ROOT}/core/causal_model.cpp
    ${MFQ_HIP_ROOT}/core/decode_graph.cpp
    ${MFQ_HIP_ROOT}/core/ffn.cpp
    ${MFQ_HIP_ROOT}/core/full_block.cpp
    ${MFQ_HIP_ROOT}/core/axial_patch_vision.cpp
    ${MFQ_HIP_ROOT}/core/kv_cache.cpp
    ${MFQ_HIP_ROOT}/core/rope.cpp
)

set(MFQ_HIP_RUNTIME_OP_SOURCES
    ${MFQ_HIP_ROOT}/ops/cuda_execution.cpp
    ${MFQ_HIP_ROOT}/ops/format.cpp
    ${MFQ_HIP_ROOT}/ops/gated_residual.cpp
    ${MFQ_HIP_ROOT}/ops/fp8_sq.cpp
    ${MFQ_HIP_ROOT}/ops/mx.cpp
    ${MFQ_HIP_ROOT}/ops/mxfp4_sq.cpp
    ${MFQ_HIP_ROOT}/ops/moe.cpp
    ${MFQ_HIP_ROOT}/ops/mfe_ffn.cpp
    ${MFQ_HIP_ROOT}/ops/nint.cpp
    ${MFQ_HIP_ROOT}/ops/tq.cpp
    ${MFQ_HIP_ROOT}/ops/quant_linear.cpp
    ${MFQ_HIP_ROOT}/ops/vq.cpp
)

add_library(mfq-hip-ops OBJECT ${MFQ_HIP_RUNTIME_OP_SOURCES})
mfq_configure_hip_backend_target(mfq-hip-ops)
target_compile_definitions(mfq-hip-ops PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-hip-ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(MFQ_HIP_LOADING_SOURCES
    ${MFQ_HIP_ROOT}/storage/load_options.cpp
    ${MFQ_HIP_ROOT}/storage/model_loader.cpp
    ${MFQ_HIP_ROOT}/storage/moe_expert_cache.cpp
    ${MFQ_HIP_ROOT}/storage/moe_ffn_pipeline.cpp
    ${MFQ_HIP_ROOT}/storage/moe_quant_range_source.cpp
    ${MFQ_HIP_ROOT}/runtime/moe_host_expert_cache.cpp
    ${MFQ_HIP_ROOT}/runtime/graph_registry.cpp
    ${MFQ_HIP_ROOT}/runtime/decode_window.cpp
    ${MFQ_HIP_ROOT}/runtime/moe_residency.cpp
    ${MFQ_HIP_ROOT}/runtime/nint_row_pipeline.cpp
    ${MFQ_HIP_ROOT}/storage/weight_loader.cpp
    ${MFQ_HIP_ROOT}/storage/transformer_loader.cpp
)
add_library(mfq-hip-loading OBJECT ${MFQ_HIP_LOADING_SOURCES})
mfq_configure_hip_runtime_target(mfq-hip-loading)
target_compile_definitions(mfq-hip-loading PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-hip-loading PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(MFQ_HIP_INSTANTIATION_SOURCES
    ${MFQ_HIP_ROOT}/engine/cuda_engine.cpp
    ${MFQ_HIP_ROOT}/engine/generation.cpp
    ${MFQ_HIP_ROOT}/storage/text_session_cache.cpp
    ${MFQ_HIP_ROOT}/storage/qwen4_paged_session.cpp
    ${MFQ_HIP_ROOT}/storage/deepseek_v41_paged_session.cpp
    ${MFQ_HIP_ROOT}/engine/mtp.cpp
    ${MFQ_HIP_ROOT}/engine/cuda_batching.cpp
    ${MFQ_HIP_ROOT}/commands/cli.cpp
    ${MFQ_HIP_ROOT}/commands/minicpmo45.cpp
    ${MFQ_HIP_MODEL_SOURCES}
    ${MFQ_HIP_TRANSFORMER_SOURCES}
)
add_library(mfq-hip-runtime STATIC
    $<TARGET_OBJECTS:mfq-hip-ops>
    $<TARGET_OBJECTS:mfq-hip-loading>
    ${MFQ_HIP_INSTANTIATION_SOURCES}
)
add_library(mfq::hip-runtime ALIAS mfq-hip-runtime)
mfq_configure_hip_runtime_target(mfq-hip-runtime)
target_link_libraries(mfq-hip-runtime PRIVATE
    hip::host
    hipblas
    mfq-hip-core
    mfq-hip-native-kernels
    mfq-hip-storage
)
target_compile_definitions(mfq-hip-runtime PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1
    NOMINMAX
)
set_target_properties(mfq-hip-runtime PROPERTIES
    HIP_ARCHITECTURES "${MFQ_HIP_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

add_library(mfq-hip-eval STATIC ${MFQ_HIP_ROOT}/eval/kl.cpp)
mfq_configure_hip_runtime_target(mfq-hip-eval)
target_include_directories(mfq-hip-eval PRIVATE ${MFQ_HIP_ROOT}/storage)
target_compile_definitions(mfq-hip-eval PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-hip-eval PUBLIC mfq-hip-runtime)

add_library(mfq-hip-diagnostics STATIC
    ${MFQ_HIP_ROOT}/diagnostics/attention_checks.cpp
    ${MFQ_HIP_ROOT}/diagnostics/backend_checks.cpp
    ${MFQ_HIP_ROOT}/diagnostics/linear_checks.cpp
    ${MFQ_HIP_ROOT}/diagnostics/moe_checks.cpp
    ${MFQ_HIP_ROOT}/diagnostics/runtime_checks.cpp
    ${MFQ_HIP_ROOT}/diagnostics/session_checks.cpp
)
mfq_configure_hip_runtime_target(mfq-hip-diagnostics)
target_include_directories(mfq-hip-diagnostics PRIVATE ${MFQ_HIP_ROOT}/storage)
target_compile_definitions(mfq-hip-diagnostics PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-hip-diagnostics PUBLIC mfq-hip-runtime)

add_executable(mfq-runtime
    ${MFQ_HIP_ROOT}/apps/runtime_main.cpp
    ${MFQ_HIP_ROOT}/commands/runtime.cpp
)
target_link_libraries(mfq-runtime PRIVATE
    mfq-hip-runtime mfq-runtime-communication)
target_include_directories(mfq-runtime PRIVATE
    ${MFQ_HIP_ROOT}
    ${MFQ_HIP_ROOT}/include
    ${MFQ_HIP_ROOT}/engine
)
target_compile_features(mfq-runtime PRIVATE cxx_std_20)
set_target_properties(mfq-runtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(mfq-diagnostics
    ${MFQ_HIP_ROOT}/apps/diagnostics_main.cpp
    ${MFQ_HIP_ROOT}/commands/diagnostics.cpp
)
mfq_configure_hip_runtime_target(mfq-diagnostics)
target_include_directories(mfq-diagnostics PRIVATE ${MFQ_HIP_ROOT}/storage)
target_compile_definitions(mfq-diagnostics PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-diagnostics PRIVATE mfq-hip-diagnostics)
target_compile_features(mfq-diagnostics PRIVATE cxx_std_20)
set_target_properties(mfq-diagnostics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(mfq-eval
    ${MFQ_HIP_ROOT}/apps/eval_main.cpp
    ${MFQ_HIP_ROOT}/commands/eval.cpp
)
mfq_configure_hip_runtime_target(mfq-eval)
target_include_directories(mfq-eval PRIVATE ${MFQ_HIP_ROOT}/storage)
target_compile_definitions(mfq-eval PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-eval PRIVATE
    mfq-hip-eval mfq-hip-runtime mfq-hip-native-kernels mfq-ggml)
target_compile_features(mfq-eval PRIVATE cxx_std_20)
set_target_properties(mfq-eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

if(BUILD_TESTING)
    add_executable(mfq-hip-decode-test ${MFQ_HIP_ROOT}/tests/decode_hotspots_test.cpp)
    mfq_configure_hip_backend_target(mfq-hip-decode-test)
    target_compile_definitions(mfq-hip-decode-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_compile_features(mfq-hip-decode-test PRIVATE cxx_std_20)
    target_link_libraries(mfq-hip-decode-test PRIVATE
        mfq-hip-runtime mfq-hip-native-kernels mfq-hip-core mfq-core)
    add_test(NAME mfq-hip-decode-test COMMAND mfq-hip-decode-test)
    set_tests_properties(mfq-hip-decode-test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(mfq-hip-tq-test ${MFQ_HIP_ROOT}/tests/tq_projection_test.cpp)
    mfq_configure_hip_backend_target(mfq-hip-tq-test)
    target_compile_definitions(mfq-hip-tq-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_compile_features(mfq-hip-tq-test PRIVATE cxx_std_20)
    target_link_libraries(mfq-hip-tq-test PRIVATE
        mfq-hip-runtime mfq-hip-native-kernels mfq-hip-core mfq-core)
    add_test(NAME mfq-hip-tq-test COMMAND mfq-hip-tq-test
        ${MFQ_REPOSITORY_ROOT}/tests/fixtures/hip-tq-native-subgroup.bin
        ${MFQ_REPOSITORY_ROOT}/tests/fixtures/hip-tq-native-short-groups.bin
        ${MFQ_REPOSITORY_ROOT}/tests/fixtures/hip-tq-native-columns.bin)
    set_tests_properties(mfq-hip-tq-test PROPERTIES SKIP_RETURN_CODE 77)

    # The CUDA backend's test sources compile directly as HIP translation
    # units: hipcc accepts <<<>>> launches, and tests/cuda_compat maps the
    # cuda_* runtime API names onto hip.
    set(MFQ_HIP_TEST_INCLUDES
        ${MFQ_REPOSITORY_ROOT}/cpp_runtime/core
        ${MFQ_HIP_ROOT}/tests/cuda_compat
        ${MFQ_REPOSITORY_ROOT}/cpp_runtime/components/ggml/src/ggml-hip/vendors
        ${MFQ_REPOSITORY_ROOT}
        ${MFQ_HIP_ROOT}
        ${MFQ_HIP_ROOT}/include
        ${MFQ_HIP_ROOT}/ops/include
        ${MFQ_HIP_ROOT}/storage
        ${MFQ_HIP_KERNEL_ROOT}
    )
    function(mfq_add_hip_test target source)
        add_executable(${target} ${source})
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
        target_include_directories(${target} PRIVATE ${MFQ_HIP_TEST_INCLUDES})
        target_compile_definitions(${target} PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
        target_compile_features(${target} PRIVATE cxx_std_20)
        set_target_properties(${target} PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${MFQ_HIP_ARCHITECTURES}"
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/hip-tests"
        )
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    mfq_add_hip_test(mfq-hip-context-test
        ${MFQ_HIP_ROOT}/tests/mfq_cuda_context_test.cu)
    target_link_libraries(mfq-hip-context-test PRIVATE mfq-hip-core)

    mfq_add_hip_test(mfq-hip-wave-test
        ${MFQ_HIP_ROOT}/tests/mfq_hip_wave_test.hip)
    target_link_libraries(mfq-hip-wave-test PRIVATE hip::host)

    mfq_add_hip_test(mfq-hip-activation-test
        ${MFQ_HIP_ROOT}/tests/mfq_cuda_activation_test.cu)
    target_link_libraries(mfq-hip-activation-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels)

    mfq_add_hip_test(mfq-hip-attention-mma-test
        ${MFQ_HIP_ROOT}/tests/attention_mma_test.cu)
    target_link_libraries(mfq-hip-attention-mma-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels mfq-core mfq-ggml)

    mfq_add_hip_test(mfq-hip-attention-decode-test
        ${MFQ_HIP_ROOT}/tests/attention_decode_test.cu)
    target_link_libraries(mfq-hip-attention-decode-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels)

    mfq_add_hip_test(mfq-hip-native-tensor-test
        ${MFQ_HIP_ROOT}/tests/mfq_native_tensor_cuda_test.cu)
    target_link_libraries(mfq-hip-native-tensor-test PRIVATE mfq-hip-core)

    mfq_add_hip_test(mfq-hip-nint-q8-test
        ${MFQ_HIP_ROOT}/tests/mfq_nint_q8_test.cu)
    target_link_libraries(mfq-hip-nint-q8-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels mfq-hip-storage
        mfq-hip-runtime mfq-runtime-communication mfq-model-graph)

    add_executable(mfq-hip-nint-bench ${MFQ_REPOSITORY_ROOT}/bench/hip_nint_bench.cpp)
    target_include_directories(mfq-hip-nint-bench PRIVATE ${MFQ_HIP_TEST_INCLUDES})
    target_compile_definitions(mfq-hip-nint-bench PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_link_libraries(mfq-hip-nint-bench PRIVATE
        mfq-hip-core mfq-hip-native-kernels mfq-hip-storage
        mfq-hip-runtime mfq-runtime-communication mfq-model-graph)
    target_compile_features(mfq-hip-nint-bench PRIVATE cxx_std_20)

    mfq_add_hip_test(mfq-hip-nint-rows-test
        ${MFQ_HIP_ROOT}/tests/mfq_nint_rows_test.cu)
    target_link_libraries(mfq-hip-nint-rows-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels mfq-core)

    mfq_add_hip_test(mfq-hip-packed-prefill-test
        ${MFQ_HIP_ROOT}/tests/mfq_packed_prefill_test.cu)
    target_link_libraries(mfq-hip-packed-prefill-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels mfq-core)

    mfq_add_hip_test(mfq-hip-paged-kv-test
        ${MFQ_HIP_ROOT}/tests/mfq_paged_kv_test.cu)
    target_link_libraries(mfq-hip-paged-kv-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels)
endif()

set(MFQ_SYSTEM_RUNTIME_DIR
    "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH
    "directory containing the host libstdc++ runtime")
set_property(TARGET mfq-runtime PROPERTY
    BUILD_RPATH "${MFQ_SYSTEM_RUNTIME_DIR}")

if(BUILD_TESTING)
    # This bridge contains host code only. Match the runtime's C++ compiler
    # for the model/template ABI; GPU work is supplied by the linked kernels.
    set_source_files_properties(${MFQ_HIP_ROOT}/tests/qwen38_flash_next_test.hip
        PROPERTIES LANGUAGE CXX)
    add_executable(mfq-hip-flash-next-test ${MFQ_HIP_ROOT}/tests/qwen38_flash_next_test.hip
        ${MFQ_HIP_ROOT}/tests/quant_kv_check.cpp
        ${MFQ_HIP_ROOT}/tests/hot_kv_check.cpp
        ${MFQ_HIP_ROOT}/tests/qwen4_graph_reuse_check.cpp
        ${MFQ_HIP_ROOT}/tests/qwen4_session_check.cpp
        ${MFQ_HIP_ROOT}/tests/qwen4_predictor_graph_check.cpp)
    target_compile_definitions(mfq-hip-flash-next-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-hip-flash-next-test PRIVATE ${MFQ_HIP_TEST_INCLUDES}
        ${MFQ_REPOSITORY_ROOT}/cpp_runtime ${MFQ_HIP_ROOT}/models ${MFQ_HIP_ROOT}/runtime)
    target_compile_features(mfq-hip-flash-next-test PRIVATE cxx_std_20)
    set_target_properties(mfq-hip-flash-next-test PROPERTIES HIP_STANDARD 20
        HIP_ARCHITECTURES "${MFQ_HIP_ARCHITECTURES}" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/hip-tests")
    target_link_libraries(mfq-hip-flash-next-test PRIVATE mfq-hip-runtime mfq-hip-native-kernels
        mfq-hip-core mfq-core mfq-json mfq-model-graph mfq-runtime-communication)
    add_test(NAME mfq-hip-flash-next COMMAND mfq-hip-flash-next-test)
    set_tests_properties(mfq-hip-flash-next PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()

if(BUILD_TESTING)
    mfq_add_hip_test(mfq-hip-recurrent-mixer-test ${MFQ_HIP_ROOT}/tests/recurrent_mixer_ops_test.hip)
    target_link_libraries(mfq-hip-recurrent-mixer-test PRIVATE mfq-hip-core mfq-hip-native-kernels)
endif()
