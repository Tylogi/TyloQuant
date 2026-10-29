find_package(Threads REQUIRED)

set(MFQ_METAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(MFQ_METAL_INCLUDE_DIRS
    ${MFQ_METAL_ROOT}/storage
    ${MFQ_METAL_ROOT}/storage/compat
    ${MFQ_METAL_ROOT}/runtime
    ${MFQ_METAL_ROOT}/ops
    ${MFQ_METAL_ROOT}/models/deepseek_v4
    ${MFQ_METAL_ROOT}/models/deepseek_v41
    ${MFQ_METAL_ROOT}/models/flash_next
    ${MFQ_METAL_ROOT}/models/qwen35
    ${MFQ_METAL_ROOT}/models/minicpmo45
    ${MFQ_METAL_ROOT}/models/nemotron_h
)

set(MFQ_MLX_ROOT "" CACHE PATH
    "MLX package root containing include/, lib/libmlx.dylib, and lib/mlx.metallib")
if(NOT MFQ_MLX_ROOT)
    file(GLOB _MFQ_LOCAL_MLX_ROOTS
        "${MFQ_REPOSITORY_ROOT}/.venv/lib/python*/site-packages/mlx")
    list(LENGTH _MFQ_LOCAL_MLX_ROOTS _MFQ_LOCAL_MLX_ROOT_COUNT)
    if(_MFQ_LOCAL_MLX_ROOT_COUNT GREATER 0)
        list(GET _MFQ_LOCAL_MLX_ROOTS 0 MFQ_MLX_ROOT)
    endif()
endif()

set(MFQ_MLX_METALLIB_DEFAULT "${MFQ_MLX_ROOT}/lib/mlx.metallib" CACHE STRING
    "Default mlx.metallib path embedded in the native Metal runtime; leave empty for relocatable packages")

foreach(_MFQ_MLX_REQUIRED
        "${MFQ_MLX_ROOT}/include/mlx/mlx.h"
        "${MFQ_MLX_ROOT}/lib/libmlx.dylib"
        "${MFQ_MLX_ROOT}/lib/libjaccl.dylib"
        "${MFQ_MLX_ROOT}/lib/mlx.metallib")
    if(NOT EXISTS "${_MFQ_MLX_REQUIRED}")
        message(FATAL_ERROR
            "MLX native runtime file not found: ${_MFQ_MLX_REQUIRED}. "
            "Set -DMFQ_MLX_ROOT=/path/to/site-packages/mlx")
    endif()
endforeach()

add_library(mfq-mlx SHARED IMPORTED GLOBAL)
set_target_properties(mfq-mlx PROPERTIES
    IMPORTED_LOCATION "${MFQ_MLX_ROOT}/lib/libmlx.dylib"
    INTERFACE_INCLUDE_DIRECTORIES
        "${MFQ_MLX_ROOT}/include;${MFQ_MLX_ROOT}/include/metal_cpp"
    INTERFACE_COMPILE_DEFINITIONS
        "_METAL_;ACCELERATE_NEW_LAPACK"
)
if(EXISTS "${MFQ_MLX_ROOT}/include/mlx/backend/metal/kernels/metal_3_1")
    set_property(TARGET mfq-mlx APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES
        "${MFQ_MLX_ROOT}/include/mlx/backend/metal/kernels/metal_3_1")
endif()

add_library(mfq-jaccl SHARED IMPORTED GLOBAL)
set_target_properties(mfq-jaccl PROPERTIES
    IMPORTED_LOCATION "${MFQ_MLX_ROOT}/lib/libjaccl.dylib"
)

find_library(MFQ_METAL_FRAMEWORK Metal REQUIRED)
find_library(MFQ_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(MFQ_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(MFQ_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(MFQ_AVFOUNDATION_FRAMEWORK AVFoundation REQUIRED)
find_library(MFQ_COREMEDIA_FRAMEWORK CoreMedia REQUIRED)
find_library(MFQ_COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
find_library(MFQ_IOSURFACE_FRAMEWORK IOSurface REQUIRED)

add_library(mfq-ane STATIC platform/mfq_ane.mm runtime/mtp_lora.cpp)
target_include_directories(mfq-ane PUBLIC ${MFQ_METAL_ROOT}/platform)
target_include_directories(mfq-ane PUBLIC ${MFQ_METAL_ROOT}/runtime)
target_compile_features(mfq-ane PUBLIC cxx_std_20)
target_compile_options(mfq-ane PRIVATE -fobjc-arc)
target_link_libraries(mfq-ane PRIVATE ${MFQ_FOUNDATION_FRAMEWORK} ${MFQ_IOSURFACE_FRAMEWORK})

add_executable(mfq-ane-test tests/mfq_ane_test.cpp)
target_link_libraries(mfq-ane-test PRIVATE mfq-ane)
target_compile_features(mfq-ane-test PRIVATE cxx_std_20)

add_library(mfq-avfoundation-video SHARED
    platform/mfq_avfoundation_video.mm
)
target_include_directories(mfq-avfoundation-video PUBLIC
    ${MFQ_METAL_ROOT}/platform
)
target_compile_features(mfq-avfoundation-video PRIVATE cxx_std_20)
target_compile_options(mfq-avfoundation-video PRIVATE -fobjc-arc)
target_link_libraries(mfq-avfoundation-video PRIVATE
    ${MFQ_FOUNDATION_FRAMEWORK}
    ${MFQ_AVFOUNDATION_FRAMEWORK}
    ${MFQ_COREMEDIA_FRAMEWORK}
    ${MFQ_COREVIDEO_FRAMEWORK}
)
set_target_properties(mfq-avfoundation-video PROPERTIES
    OUTPUT_NAME "mfq_avfoundation_video"
)

# MLX's runtime Metal compiler does not expose filesystem include paths.  Fold
# the Steel MMA implementation into the MFE prefill source at configure time
# so the native primitive can use the same tiled MMA machinery as MLX/oMLX
# without requiring a full Xcode Metal command-line toolchain at build time.
set(_MFQ_STEEL_STUB_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/metal-include-stubs")
file(MAKE_DIRECTORY "${_MFQ_STEEL_STUB_DIR}")
foreach(_MFQ_METAL_SYSTEM_HEADER
        metal_stdlib metal_math metal_simdgroup metal_simdgroup_matrix)
    file(TOUCH
        "${_MFQ_STEEL_STUB_DIR}/${_MFQ_METAL_SYSTEM_HEADER}")
endforeach()
file(MAKE_DIRECTORY
    "${_MFQ_STEEL_STUB_DIR}/MetalPerformancePrimitives")
file(TOUCH
    "${_MFQ_STEEL_STUB_DIR}/MetalPerformancePrimitives/MetalPerformancePrimitives.h")
set(_MFQ_STEEL_ENTRY
    "${CMAKE_CURRENT_BINARY_DIR}/mfq_steel_mma_entry.h")
set(MFQ_MLX_UTILS_SOURCE "#define MLX_METAL_JIT\n")
foreach(_MFQ_UTILS_HEADER bf16.h bf16_math.h complex.h defines.h logging.h utils.h)
    set(_MFQ_UTILS_PATH "${MFQ_MLX_ROOT}/include/mlx/backend/metal/kernels/${_MFQ_UTILS_HEADER}")
    file(READ "${_MFQ_UTILS_PATH}" _MFQ_UTILS_CONTENT)
    string(REGEX REPLACE "#[ \t]*include[ \t]+\"[^\"]*\"" "" _MFQ_UTILS_CONTENT "${_MFQ_UTILS_CONTENT}")
    string(REPLACE "#pragma once" "" _MFQ_UTILS_CONTENT "${_MFQ_UTILS_CONTENT}")
    string(APPEND MFQ_MLX_UTILS_SOURCE "\n${_MFQ_UTILS_CONTENT}\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_MFQ_UTILS_PATH}")
endforeach()
configure_file("${MFQ_METAL_ROOT}/kernels/mfq_mlx_utils_embedded.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mfq_mlx_utils_embedded.h" @ONLY)
file(WRITE "${_MFQ_STEEL_ENTRY}"
    "#include \"mlx/backend/metal/kernels/utils.h\"\n"
    "#include \"mlx/backend/metal/kernels/steel/gemm/gemm.h\"\n"
    "#include \"mlx/backend/metal/kernels/fp_quantized.h\"\n"
    "#include \"mlx/backend/metal/kernels/quantized_utils.h\"\n")
execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -E -P -x c++
        -D__METAL_VERSION__=310
        -DMLX_METAL_JIT
        "-I${MFQ_MLX_ROOT}/include"
        "-I${_MFQ_STEEL_STUB_DIR}"
        "${_MFQ_STEEL_ENTRY}"
    RESULT_VARIABLE _MFQ_STEEL_EXPAND_RESULT
    OUTPUT_VARIABLE MFQ_STEEL_MMA_SOURCE
    ERROR_VARIABLE _MFQ_STEEL_EXPAND_ERROR)
if(NOT _MFQ_STEEL_EXPAND_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Failed to embed MLX Steel MMA headers: "
        "${_MFQ_STEEL_EXPAND_ERROR}")
endif()
set(_MFQ_STEEL_NAX_ENTRY
    "${CMAKE_CURRENT_BINARY_DIR}/mfq_steel_nax_entry.h")
file(WRITE "${_MFQ_STEEL_NAX_ENTRY}"
    "#include \"mlx/backend/metal/kernels/complex.h\"\n"
    "#include \"mlx/backend/metal/kernels/steel/gemm/mma.h\"\n"
    "#include \"mlx/backend/metal/kernels/steel/gemm/nax.h\"\n")
execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -E -P -x c++
        -D__METAL_VERSION__=320
        "-I${MFQ_MLX_ROOT}/include"
        "-I${_MFQ_STEEL_STUB_DIR}"
        "${_MFQ_STEEL_NAX_ENTRY}"
    RESULT_VARIABLE _MFQ_STEEL_NAX_EXPAND_RESULT
    OUTPUT_VARIABLE MFQ_STEEL_NAX_SOURCE
    ERROR_VARIABLE _MFQ_STEEL_NAX_EXPAND_ERROR)
if(NOT _MFQ_STEEL_NAX_EXPAND_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Failed to embed MLX Steel NAX headers: "
        "${_MFQ_STEEL_NAX_EXPAND_ERROR}")
endif()
set(_MFQ_STEEL_ATTN_ENTRY
    "${CMAKE_CURRENT_BINARY_DIR}/mfq_steel_attn_entry.h")
file(WRITE "${_MFQ_STEEL_ATTN_ENTRY}"
    "#include \"mlx/backend/metal/kernels/utils.h\"\n"
    "#include \"mlx/backend/metal/kernels/steel/attn/attn.h\"\n")
execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -E -P -x c++
        -D__METAL_VERSION__=310
        -DMLX_METAL_JIT
        "-I${MFQ_MLX_ROOT}/include"
        "-I${_MFQ_STEEL_STUB_DIR}"
        "${_MFQ_STEEL_ATTN_ENTRY}"
    RESULT_VARIABLE _MFQ_STEEL_ATTN_EXPAND_RESULT
    OUTPUT_VARIABLE MFQ_STEEL_ATTN_SOURCE
    ERROR_VARIABLE _MFQ_STEEL_ATTN_EXPAND_ERROR)
if(NOT _MFQ_STEEL_ATTN_EXPAND_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Failed to embed MLX Steel attention headers: "
        "${_MFQ_STEEL_ATTN_EXPAND_ERROR}")
endif()
set(_MFQ_MFE_PREFILL_METAL
    "${MFQ_METAL_ROOT}/kernels/mfq_mfe_prefill.metal")
file(READ "${_MFQ_MFE_PREFILL_METAL}"
    MFQ_MFE_PREFILL_SOURCE)
string(REPLACE
    "#include \"mlx/backend/metal/kernels/steel/gemm/gemm.h\""
    ""
    MFQ_MFE_PREFILL_SOURCE
    "${MFQ_MFE_PREFILL_SOURCE}")
set(_MFQ_NINT_PREFILL_METAL
    "${MFQ_METAL_ROOT}/kernels/mfq_nint_prefill.metal")
file(READ "${_MFQ_NINT_PREFILL_METAL}"
    MFQ_NINT_PREFILL_SOURCE)
string(REPLACE
    "#include \"mlx/backend/metal/kernels/steel/gemm/gemm.h\""
    ""
    MFQ_NINT_PREFILL_SOURCE
    "${MFQ_NINT_PREFILL_SOURCE}")
set(_MFQ_DSA_SPARSE_PREFILL_METAL
    "${MFQ_METAL_ROOT}/kernels/mfq_dsa_sparse_prefill.metal")
file(READ "${_MFQ_DSA_SPARSE_PREFILL_METAL}"
    MFQ_DSA_SPARSE_PREFILL_SOURCE)
string(REPLACE
    "#include \"mlx/backend/metal/kernels/steel/attn/attn.h\""
    ""
    MFQ_DSA_SPARSE_PREFILL_SOURCE
    "${MFQ_DSA_SPARSE_PREFILL_SOURCE}")
set(_MFQ_SPARSE_BLOCK_GQA_METAL
    "${MFQ_METAL_ROOT}/kernels/mfq_sparse_block_gqa.metal")
file(READ "${_MFQ_SPARSE_BLOCK_GQA_METAL}"
    MFQ_SPARSE_BLOCK_GQA_SOURCE)
string(REPLACE
    "#include \"mlx/backend/metal/kernels/steel/attn/attn.h\""
    ""
    MFQ_SPARSE_BLOCK_GQA_SOURCE
    "${MFQ_SPARSE_BLOCK_GQA_SOURCE}")
set(_MFQ_CSA2_PREFILL_METAL
    "${MFQ_METAL_ROOT}/kernels/mfq_csa2_prefill.metal")
file(READ "${_MFQ_CSA2_PREFILL_METAL}" MFQ_CSA2_PREFILL_SOURCE)
string(REPLACE
    "#include \"mlx/backend/metal/kernels/steel/attn/attn.h\"" ""
    MFQ_CSA2_PREFILL_SOURCE "${MFQ_CSA2_PREFILL_SOURCE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_MFQ_CSA2_PREFILL_METAL}"
    "${_MFQ_MFE_PREFILL_METAL}"
    "${_MFQ_NINT_PREFILL_METAL}"
    "${_MFQ_DSA_SPARSE_PREFILL_METAL}"
    "${_MFQ_SPARSE_BLOCK_GQA_METAL}"
    "${MFQ_MLX_ROOT}/include/mlx/backend/metal/kernels/fp_quantized.h"
    "${MFQ_MLX_ROOT}/include/mlx/backend/metal/kernels/quantized_utils.h"
    "${MFQ_MLX_ROOT}/include/mlx/backend/metal/kernels/utils.h"
    "${MFQ_MLX_ROOT}/include/mlx/backend/metal/kernels/steel/attn/attn.h"
    "${MFQ_MLX_ROOT}/include/mlx/backend/metal/kernels/steel/gemm/gemm.h"
    "${MFQ_MLX_ROOT}/include/mlx/backend/metal/kernels/steel/gemm/mma.h"
    "${MFQ_MLX_ROOT}/include/mlx/backend/metal/kernels/steel/gemm/nax.h")
configure_file(
    "${MFQ_METAL_ROOT}/kernels/mfq_mfe_prefill_embedded.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mfq_mfe_prefill_embedded.h"
    @ONLY)

file(READ "${MFQ_REPOSITORY_ROOT}/mfq/kernels/metal/tq_decode.h" MFQ_TQ_DECODE_SOURCE)
file(READ "${MFQ_REPOSITORY_ROOT}/mfq/kernels/metal/tq_matmul.h" MFQ_TQ_MATMUL_SOURCE)
set(_MFQ_TQ_NAX_ENTRY "${CMAKE_CURRENT_BINARY_DIR}/mfq_tq_nax_entry.h")
file(WRITE "${_MFQ_TQ_NAX_ENTRY}"
    "#include \"mlx/backend/metal/kernels/steel/gemm/nax.h\"\n")
execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -E -P -x c++
        -D__METAL_VERSION__=320
        "-I${MFQ_MLX_ROOT}/include"
        "-I${_MFQ_STEEL_STUB_DIR}"
        "${_MFQ_TQ_NAX_ENTRY}"
    RESULT_VARIABLE _MFQ_TQ_NAX_EXPAND_RESULT
    OUTPUT_VARIABLE MFQ_TQ_NAX_SOURCE
    ERROR_VARIABLE _MFQ_TQ_NAX_EXPAND_ERROR)
if(NOT _MFQ_TQ_NAX_EXPAND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to embed TQ NAX headers: ${_MFQ_TQ_NAX_EXPAND_ERROR}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${MFQ_REPOSITORY_ROOT}/mfq/kernels/metal/tq_decode.h"
    "${MFQ_REPOSITORY_ROOT}/mfq/kernels/metal/tq_matmul.h")
configure_file(kernels/mfq_tq_embedded.h.in mfq_tq_embedded.h @ONLY)

add_library(mfq-metal-runtime STATIC
    storage/hf_safetensors_store.cpp
    storage/mfq_container.cpp
    storage/mfe_expert_store.cpp
    storage/mlx_quantized_row_store.cpp
    runtime/mlx_tq_expert_cache.cpp
    storage/mlx_tq_rows.cpp
    storage/compat/mlx_legacy_tensor_compat.cpp
    runtime/mlx_axial_patch_vision.cpp
    runtime/mlx_grid_vision.cpp
    runtime/mlx_mtp.cpp
    runtime/mlx_mtp_lora.cpp
    runtime/mlx_ane_matmul.cpp
    runtime/mtp_lora_sessions.cpp
    runtime/mlx_multimodal.cpp
    runtime/mlx_pre_ln_vision.cpp
    runtime/mlx_sampling.cpp
    runtime/mlx_kernel_prepare.cpp
    runtime/mlx_kernel_cache.cpp
    runtime/mlx_builtin_kernels.cpp
    runtime/mlx_inference_warmup.cpp
    runtime/mlx_ssd_expert_arena.cpp
    runtime/mlx_ssd_expert_cache.cpp
    runtime/mlx_tensor.cpp
    runtime/mlx_transformer.cpp
    runtime/mlx_kv_quantization.cpp
    runtime/mlx_qsa_kv_offload.cpp
    ops/mlx_grouped_linear.cpp
    ops/mlx_linear_attention.cpp
    ops/mlx_moe.cpp
    ops/mlx_moe_ops.cpp
    ops/mlx_fp8_sq.cpp
    ops/mlx_mx.cpp
    ops/mlx_mxfp4_sq.cpp
    ops/mlx_nint.cpp
    ops/mlx_nint_rows.cpp
    ops/mlx_tq.cpp
    ops/mlx_nint8_zero.cpp
    ops/mlx_sparse_attention.cpp
    ops/mlx_csa2.cpp
    ops/mlx_dsa_cache_ops.cpp
    ops/mlx_vq.cpp
    models/deepseek_v4/deepseek_v4_model.cpp
    models/deepseek_v4/mlx_deepseek_v4_attention.cpp
    models/deepseek_v4/mlx_deepseek_v4_causal_lm.cpp
    models/deepseek_v4/mlx_deepseek_v4_dspark.cpp
    models/deepseek_v4/mlx_deepseek_v4_hc.cpp
    models/deepseek_v4/mlx_deepseek_v4_moe.cpp
    ops/mlx_dsa.cpp
    models/deepseek_v4/mlx_deepseek_v4_vision.cpp
    models/deepseek_v41/mlx_deepseek_v41_attention.cpp
    models/deepseek_v41/mlx_deepseek_v41_causal_lm.cpp
    models/deepseek_v41/mlx_deepseek_v41_dspark.cpp
    models/deepseek_v41/mlx_deepseek_v41_engram.cpp
    models/deepseek_v41/mlx_deepseek_v41_mhc.cpp
    models/deepseek_v41/mlx_deepseek_v41_moe.cpp
    models/deepseek_v41/mlx_deepseek_v41_vision.cpp
    models/flash_next/mlx_qwen4_causal_lm.cpp
    models/flash_next/qwen4_ops.cpp
    models/minicpmo45/mlx_minicpmo45.cpp
    models/qwen35/mlx_qwen35_causal_lm.cpp
    models/qwen35/mlx_qwen35_full_attention.cpp
    models/qwen35/mlx_qwen35_linear_attention.cpp
    models/qwen35/qwen35_model.cpp
    models/nemotron_h/mlx_nemotron_h_attention.cpp
    models/nemotron_h/mlx_nemotron_h_causal_lm.cpp
    models/nemotron_h/mlx_nemotron_h_linear_attention.cpp
    models/nemotron_h/mlx_nemotron_h_moe.cpp
    models/nemotron_h/mlx_nemotron_h_ops.cpp
    models/nemotron_h/mlx_nemotron_h_vision.cpp
    models/nemotron_h/nemotron_h_model.cpp
)
add_library(mfq::metal-runtime ALIAS mfq-metal-runtime)
target_include_directories(mfq-metal-runtime PUBLIC
    ${MFQ_METAL_INCLUDE_DIRS}
)
target_include_directories(mfq-metal-runtime PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(mfq-metal-runtime PUBLIC
    mfq-core
    mfq-models
    mfq-engine
    mfq-mlx
    mfq-jaccl
    mfq-ane
    Threads::Threads
    ${MFQ_METAL_FRAMEWORK}
    ${MFQ_FOUNDATION_FRAMEWORK}
    ${MFQ_QUARTZCORE_FRAMEWORK}
    ${MFQ_ACCELERATE_FRAMEWORK}
)
target_link_libraries(mfq-metal-runtime PRIVATE mfq-json)
# MTP LoRA sessions and the Qwen4 MTP cache key use block_hash_hex from the
# paged prefix cache; without it, targets linking only this library fail.
if(TARGET mfq-paged-prefix-cache)
    target_link_libraries(mfq-metal-runtime PUBLIC mfq-paged-prefix-cache)
else()
    target_sources(mfq-metal-runtime PRIVATE ${PROJECT_SOURCE_DIR}/core/mfq_paged_prefix_cache.cpp)
endif()
target_compile_features(mfq-metal-runtime PUBLIC cxx_std_20)
target_compile_definitions(mfq-metal-runtime PUBLIC
    MFQ_NATIVE_MLX_RUNTIME=1
    MFQ_MLX_METALLIB_DEFAULT="${MFQ_MLX_METALLIB_DEFAULT}"
)

add_executable(mfq-metal-tq-test tests/mlx_tq_test.cpp)
target_link_libraries(mfq-metal-tq-test PRIVATE mfq-metal-runtime)
target_include_directories(mfq-metal-tq-test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(mfq-metal-tq-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-tq-test PROPERTIES BUILD_RPATH "${MFQ_MLX_ROOT}/lib")

if(BUILD_TESTING)
    add_test(NAME mfq-metal-tq-container-test COMMAND mfq-metal-tq-test unused --expert-containers)
    add_test(NAME mfq-metal-tq-prefill-reduction-test COMMAND mfq-metal-tq-test --prefill-reduction)
    add_test(NAME mfq-metal-tq-projection-cache-test COMMAND mfq-metal-tq-test --projection-cache)
    add_test(NAME mfq-metal-tq-expansion-vector-test COMMAND mfq-metal-tq-test --expansion-vectors)
    add_test(NAME mfq-metal-tq-column-input-test COMMAND mfq-metal-tq-test --column-inputs)
    add_test(NAME mfq-metal-tq-compact-reuse-test COMMAND mfq-metal-tq-test --compact-reuse)
    add_test(NAME mfq-metal-tq-streamed-experts-test COMMAND mfq-metal-tq-test --streamed-experts)
    add_test(NAME mfq-metal-tq-routed-tiles-test COMMAND mfq-metal-tq-test unused --routed-tiles)
endif()

if(TARGET mfq-paged-prefix-cache)
    add_executable(mfq-metal-tq-model-benchmark tests/mlx_tq_model_benchmark.cpp)
    target_link_libraries(mfq-metal-tq-model-benchmark PRIVATE mfq-metal-runtime mfq-paged-prefix-cache)
    target_compile_features(mfq-metal-tq-model-benchmark PRIVATE cxx_std_20)
    set_target_properties(mfq-metal-tq-model-benchmark PROPERTIES BUILD_RPATH "${MFQ_MLX_ROOT}/lib")
endif()

add_library(mfq-metal-engine STATIC
    runtime/mlx_engine.cpp
    runtime/mlx_generation_job.cpp
    runtime/mlx_server_components.cpp
)
target_link_libraries(mfq-metal-engine PUBLIC mfq-metal-runtime mfq-engine)
target_compile_features(mfq-metal-engine PUBLIC cxx_std_20)

add_executable(mfq-metal-engine-test tests/mlx_engine_test.cpp)
target_link_libraries(mfq-metal-engine-test PRIVATE mfq-metal-engine)
target_compile_features(mfq-metal-engine-test PRIVATE cxx_std_20)
add_test(NAME mfq-metal-engine-test COMMAND mfq-metal-engine-test)

add_executable(mfq-qsa-kv-offload-test tests/qsa_kv_offload_test.cpp runtime/mlx_paged_session_codec.cpp)
target_link_libraries(mfq-qsa-kv-offload-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-qsa-kv-offload-test PRIVATE cxx_std_20)
add_test(NAME mfq-qsa-kv-offload-test COMMAND mfq-qsa-kv-offload-test)

add_executable(mfq-decode-metal
    apps/mfq_decode_mlx.cpp
)
add_dependencies(mfq-decode-metal mfq-avfoundation-video)
target_link_libraries(mfq-decode-metal PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-decode-metal PRIVATE cxx_std_20)

if(TARGET mfq-paged-prefix-cache)
    target_sources(mfq-decode-metal PRIVATE
        runtime/mlx_paged_session_codec.cpp
    )

    add_executable(mfq-metal-paged-session-codec-test
        runtime/mlx_paged_session_codec.cpp
        tests/mlx_paged_session_codec_test.cpp
    )
    target_link_libraries(mfq-metal-paged-session-codec-test PRIVATE
        mfq-metal-runtime
        mfq-paged-prefix-cache
    )
    target_compile_features(
        mfq-metal-paged-session-codec-test PRIVATE cxx_std_20)
    add_test(
        NAME mfq-metal-paged-session-codec-test
        COMMAND mfq-metal-paged-session-codec-test
    )
endif()

set_target_properties(mfq-decode-metal PROPERTIES
    BUILD_RPATH "@executable_path/lib"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "@executable_path/lib"
)

# Both native executables live in this binary directory and use the same
# relocatable MLX runtime bundle. A single dependency target prevents parallel
# links from copying the same runtime files at once.
set(_MFQ_METAL_RUNTIME_BUNDLE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/lib")
option(MFQ_BUILD_PRECOMPILED_METAL "Build bundled MFQ Metal libraries" OFF)
add_executable(mfq-metal-source-export
    apps/mfq_metal_source_export.cpp runtime/mlx_builtin_kernels.cpp)
target_include_directories(mfq-metal-source-export PRIVATE
    "${MFQ_METAL_ROOT}/runtime" "${MFQ_METAL_ROOT}/ops" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(mfq-metal-source-export PRIVATE cxx_std_20)
if(MFQ_BUILD_PRECOMPILED_METAL)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND xcrun --find metal OUTPUT_VARIABLE _MFQ_METAL_COMPILER
        RESULT_VARIABLE _MFQ_METAL_COMPILER_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND xcrun --find metallib OUTPUT_VARIABLE _MFQ_METALLIB_COMPILER
        RESULT_VARIABLE _MFQ_METALLIB_COMPILER_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND xcrun --sdk macosx --show-sdk-path OUTPUT_VARIABLE _MFQ_METAL_SDK
        RESULT_VARIABLE _MFQ_METAL_SDK_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _MFQ_METAL_COMPILER_RESULT EQUAL 0 OR NOT _MFQ_METALLIB_COMPILER_RESULT EQUAL 0
            OR NOT _MFQ_METAL_SDK_RESULT EQUAL 0)
        message(FATAL_ERROR "Precompiled Metal libraries require the Xcode Metal Toolchain (metal and metallib)")
    endif()
    set(_MFQ_AOT_ROOT "${_MFQ_METAL_RUNTIME_BUNDLE_DIR}/mfq-metal")
    add_custom_command(OUTPUT "${_MFQ_AOT_ROOT}/manifest.json"
        COMMAND $<TARGET_FILE:mfq-metal-source-export> "${CMAKE_CURRENT_BINARY_DIR}/metal-sources"
        COMMAND "${Python3_EXECUTABLE}" "${MFQ_REPOSITORY_ROOT}/packaging/compile_metal_libraries.py"
            "${CMAKE_CURRENT_BINARY_DIR}/metal-sources" "${_MFQ_AOT_ROOT}"
            --metal "${_MFQ_METAL_COMPILER}" --metallib "${_MFQ_METALLIB_COMPILER}" --sdk "${_MFQ_METAL_SDK}"
        DEPENDS mfq-metal-source-export "${MFQ_REPOSITORY_ROOT}/packaging/compile_metal_libraries.py"
        VERBATIM)
    add_custom_target(mfq-metal-precompiled DEPENDS "${_MFQ_AOT_ROOT}/manifest.json")
    add_dependencies(mfq-decode-metal mfq-metal-precompiled)
    if(TARGET mfq-perplexity)
        add_dependencies(mfq-perplexity mfq-metal-precompiled)
    endif()
endif()
set(_MFQ_METAL_RUNTIME_BUNDLE_STAMP
    "${_MFQ_METAL_RUNTIME_BUNDLE_DIR}/.mfq-runtime-bundle.stamp")
add_custom_command(
    OUTPUT "${_MFQ_METAL_RUNTIME_BUNDLE_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${_MFQ_METAL_RUNTIME_BUNDLE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MFQ_MLX_ROOT}/lib/libmlx.dylib"
        "${_MFQ_METAL_RUNTIME_BUNDLE_DIR}/libmlx.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MFQ_MLX_ROOT}/lib/libjaccl.dylib"
        "${_MFQ_METAL_RUNTIME_BUNDLE_DIR}/libjaccl.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MFQ_MLX_ROOT}/lib/mlx.metallib"
        "${_MFQ_METAL_RUNTIME_BUNDLE_DIR}/mlx.metallib"
    DEPENDS
        "${MFQ_MLX_ROOT}/lib/libmlx.dylib"
        "${MFQ_MLX_ROOT}/lib/libjaccl.dylib"
        "${MFQ_MLX_ROOT}/lib/mlx.metallib"
    COMMAND_EXPAND_LISTS
    COMMENT "Building the shared native Metal runtime bundle"
)

if(TARGET mfq-runtime-communication)
    target_link_libraries(mfq-decode-metal PRIVATE mfq-metal-engine mfq-runtime-communication)
    target_compile_definitions(
        mfq-decode-metal PRIVATE MFQ_METAL_RUNTIME_COMMUNICATION=1)

    add_executable(mfq-perplexity
        apps/mfq_perplexity_mlx.cpp
    )
    target_link_libraries(mfq-perplexity PRIVATE
        mfq-metal-runtime
        mfq-runtime-communication
    )
    target_compile_features(mfq-perplexity PRIVATE cxx_std_20)
    set_target_properties(mfq-perplexity PROPERTIES
        BUILD_RPATH "@executable_path/lib"
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "@executable_path/lib"
    )
endif()

add_custom_command(
    OUTPUT "${_MFQ_METAL_RUNTIME_BUNDLE_STAMP}"
    APPEND
    COMMAND ${CMAKE_COMMAND} -E touch
        "${_MFQ_METAL_RUNTIME_BUNDLE_STAMP}"
)
add_custom_target(mfq-metal-runtime-bundle
    DEPENDS "${_MFQ_METAL_RUNTIME_BUNDLE_STAMP}")
add_dependencies(mfq-decode-metal mfq-metal-runtime-bundle)
if(TARGET mfq-perplexity)
    add_dependencies(mfq-perplexity mfq-metal-runtime-bundle)
    if(TARGET mfq-metal-precompiled)
        add_dependencies(mfq-perplexity mfq-metal-precompiled)
    endif()
endif()

add_executable(mfq-metal-container-test
    tests/mfq_container_test.cpp
)
target_link_libraries(mfq-metal-container-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-container-test PRIVATE cxx_std_20)

enable_testing()
add_test(NAME mfq-metal-container-test COMMAND mfq-metal-container-test)

add_executable(mfq-metal-nint-test
    tests/mlx_nint_test.cpp
)
target_link_libraries(mfq-metal-nint-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-nint-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-nint-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-nint-test COMMAND mfq-metal-nint-test)
add_test(NAME mfq-metal-nint-portable-test
    COMMAND ${CMAKE_COMMAND} -E env MFQ_METAL_DISABLE_NAX=1
        $<TARGET_FILE:mfq-metal-nint-test>)

add_executable(mfq-metal-kernel-prepare-test tests/mlx_kernel_prepare_test.cpp)
target_link_libraries(mfq-metal-kernel-prepare-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-kernel-prepare-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-kernel-prepare-test PROPERTIES BUILD_RPATH "${MFQ_MLX_ROOT}/lib")
add_test(NAME mfq-metal-kernel-prepare-test COMMAND mfq-metal-kernel-prepare-test)
add_test(NAME mfq-metal-kernel-prepare-portable-test
    COMMAND ${CMAKE_COMMAND} -E env MFQ_METAL_DISABLE_NAX=1
        $<TARGET_FILE:mfq-metal-kernel-prepare-test>)

add_executable(mfq-metal-inference-warmup-test tests/mlx_inference_warmup_test.cpp)
target_link_libraries(mfq-metal-inference-warmup-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-inference-warmup-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-inference-warmup-test PROPERTIES BUILD_RPATH "${MFQ_MLX_ROOT}/lib")
add_test(NAME mfq-metal-inference-warmup-test COMMAND mfq-metal-inference-warmup-test)

add_executable(mfq-metal-nint-rows-test tests/mlx_nint_rows_test.cpp)
target_link_libraries(mfq-metal-nint-rows-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-nint-rows-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-nint-rows-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib")
add_test(NAME mfq-metal-nint-rows-test COMMAND mfq-metal-nint-rows-test)

add_executable(mfq-metal-memory-residency-test tests/mlx_memory_residency_test.cpp)
target_link_libraries(mfq-metal-memory-residency-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-memory-residency-test PRIVATE cxx_std_20)
target_compile_options(mfq-metal-memory-residency-test PRIVATE -UNDEBUG)
set_target_properties(mfq-metal-memory-residency-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib")
add_test(NAME mfq-metal-memory-residency-test COMMAND mfq-metal-memory-residency-test)

add_executable(mfq-metal-mx-test
    tests/mlx_mx_test.cpp
)
target_link_libraries(mfq-metal-mx-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mx-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mx-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-mx-test COMMAND mfq-metal-mx-test)

add_executable(mfq-metal-fp8-sq-test
    tests/mlx_fp8_sq_test.cpp
)
target_link_libraries(mfq-metal-fp8-sq-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-fp8-sq-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-fp8-sq-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-fp8-sq-test COMMAND mfq-metal-fp8-sq-test)

add_executable(mfq-metal-mxfp4-sq2-test
    tests/mlx_mxfp4_sq2_test.cpp
)
target_link_libraries(mfq-metal-mxfp4-sq2-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mxfp4-sq2-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mxfp4-sq2-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-mxfp4-sq2-test
    COMMAND mfq-metal-mxfp4-sq2-test
)

add_executable(mfq-metal-mxfp4-sq2-benchmark
    benchmarks/mlx_mxfp4_sq2_benchmark.cpp
)
target_link_libraries(mfq-metal-mxfp4-sq2-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mxfp4-sq2-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mxfp4-sq2-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-mxfp4-sq3-test
    tests/mlx_mxfp4_sq3_test.cpp
)
target_link_libraries(mfq-metal-mxfp4-sq3-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mxfp4-sq3-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mxfp4-sq3-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-mxfp4-sq3-test
    COMMAND mfq-metal-mxfp4-sq3-test
)

add_executable(mfq-metal-mxfp4-sq3-benchmark
    benchmarks/mlx_mxfp4_sq3_benchmark.cpp
)
target_link_libraries(mfq-metal-mxfp4-sq3-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mxfp4-sq3-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mxfp4-sq3-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-mx-benchmark
    benchmarks/mlx_mx_benchmark.cpp
)
target_link_libraries(mfq-metal-mx-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mx-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mx-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-nint-benchmark
    benchmarks/mlx_nint_benchmark.cpp
)
target_link_libraries(mfq-metal-nint-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-nint-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-nint-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-dense-smallm-benchmark
    benchmarks/mlx_dense_smallm_benchmark.cpp
)
target_link_libraries(mfq-metal-dense-smallm-benchmark PRIVATE
    mfq-metal-runtime
)
target_include_directories(mfq-metal-dense-smallm-benchmark PRIVATE
    ${MFQ_REPOSITORY_ROOT}/bench
)
target_compile_features(
    mfq-metal-dense-smallm-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-dense-smallm-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-mfe-smallm-benchmark
    benchmarks/mlx_mfe_smallm_benchmark.cpp
)
target_link_libraries(mfq-metal-mfe-smallm-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mfe-smallm-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mfe-smallm-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-vq-smallm-benchmark
    benchmarks/mlx_vq_smallm_benchmark.cpp
)
target_link_libraries(mfq-metal-vq-smallm-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-vq-smallm-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-vq-smallm-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-mfe-prefill-benchmark
    benchmarks/mlx_mfe_prefill_benchmark.cpp
)
target_link_libraries(mfq-metal-mfe-prefill-benchmark PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-mfe-prefill-benchmark PRIVATE cxx_std_20)

add_executable(mfq-metal-vq-benchmark
    benchmarks/mlx_vq_benchmark.cpp
)
target_link_libraries(mfq-metal-vq-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-vq-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-vq-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-grouped-linear-benchmark
    benchmarks/mlx_grouped_linear_benchmark.cpp
)
target_link_libraries(mfq-metal-grouped-linear-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-grouped-linear-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-grouped-linear-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-qwen35-projection-profile
    tools/qwen35_projection_profile.cpp
)
target_link_libraries(
    mfq-metal-qwen35-projection-profile PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-projection-profile PRIVATE cxx_std_20
)
set_target_properties(
    mfq-metal-qwen35-projection-profile PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-qwen35-decode-overhead-profile
    tools/qwen35_decode_overhead_profile.cpp
)
target_link_libraries(
    mfq-metal-qwen35-decode-overhead-profile PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-decode-overhead-profile PRIVATE cxx_std_20
)
set_target_properties(
    mfq-metal-qwen35-decode-overhead-profile PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-nint8-zero-test
    tests/mlx_nint8_zero_test.cpp
)
target_link_libraries(mfq-metal-nint8-zero-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-nint8-zero-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-nint8-zero-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-nint8-zero-test
    COMMAND mfq-metal-nint8-zero-test
)

add_executable(mfq-metal-grouped-linear-test
    tests/mlx_grouped_linear_test.cpp
)
target_link_libraries(mfq-metal-grouped-linear-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-grouped-linear-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-grouped-linear-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-grouped-linear-test
    COMMAND mfq-metal-grouped-linear-test
)

add_executable(mfq-metal-deepseek-v4-hc-test
    tests/mlx_deepseek_v4_hc_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-hc-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-hc-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-hc-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-hc-test
    COMMAND mfq-metal-deepseek-v4-hc-test
)

add_executable(mfq-metal-deepseek-v41-fusions-test
    tests/mlx_deepseek_v41_fusions_test.cpp
)
target_link_libraries(mfq-metal-deepseek-v41-fusions-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-deepseek-v41-fusions-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-deepseek-v41-fusions-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-deepseek-v41-fusions-test COMMAND mfq-metal-deepseek-v41-fusions-test)

add_executable(mfq-metal-deepseek-v4-model-test
    tests/deepseek_v4_model_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-model-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-model-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-model-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-model-test
    COMMAND mfq-metal-deepseek-v4-model-test
)

add_executable(mfq-metal-hf-safetensors-store-test
    storage/hf_safetensors_store.cpp
    tests/hf_safetensors_store_test.cpp
)

add_executable(mfq-metal-quantized-row-store-test
    tests/mlx_quantized_row_store_test.cpp
)
target_link_libraries(mfq-metal-quantized-row-store-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-quantized-row-store-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-quantized-row-store-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-quantized-row-store-test
    COMMAND mfq-metal-quantized-row-store-test
)
target_include_directories(mfq-metal-hf-safetensors-store-test PRIVATE
    ${MFQ_METAL_ROOT}/storage
)
target_link_libraries(mfq-metal-hf-safetensors-store-test PRIVATE mfq-model-source)
target_compile_features(
    mfq-metal-hf-safetensors-store-test PRIVATE cxx_std_20)
add_test(
    NAME mfq-metal-hf-safetensors-store-test
    COMMAND mfq-metal-hf-safetensors-store-test
)

add_executable(mfq-metal-ssd-expert-benchmark
    benchmarks/ssd_expert_benchmark.cpp
)
target_link_libraries(
    mfq-metal-ssd-expert-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-ssd-expert-benchmark PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-ssd-expert-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-deepseek-v4-moe-smallm-benchmark
    benchmarks/mlx_deepseek_v4_moe_smallm_benchmark.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-moe-smallm-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-moe-smallm-benchmark PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-moe-smallm-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-ssd-expert-arena-test
    tests/mlx_ssd_expert_arena_test.cpp
)
target_link_libraries(
    mfq-metal-ssd-expert-arena-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-ssd-expert-arena-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-ssd-expert-arena-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-ssd-expert-arena-test
    COMMAND mfq-metal-ssd-expert-arena-test
)

add_executable(mfq-metal-ssd-expert-real-smoke
    tools/ssd_expert_real_smoke.cpp
)
target_link_libraries(
    mfq-metal-ssd-expert-real-smoke PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-ssd-expert-real-smoke PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-ssd-expert-real-smoke PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-ssd-hf-model-smoke
    tools/ssd_hf_model_smoke.cpp
)
target_link_libraries(
    mfq-metal-ssd-hf-model-smoke PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-ssd-hf-model-smoke PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-ssd-hf-model-smoke PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-ssd-hf-prefill-smoke
    tools/ssd_hf_prefill_smoke.cpp
)
target_link_libraries(
    mfq-metal-ssd-hf-prefill-smoke PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-ssd-hf-prefill-smoke PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-ssd-hf-prefill-smoke PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-deepseek-v4-attention-test
    tests/mlx_deepseek_v4_attention_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-attention-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-attention-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-attention-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-attention-test
    COMMAND mfq-metal-deepseek-v4-attention-test
)

add_executable(mfq-metal-deepseek-v4-causal-lm-test
    tests/mlx_deepseek_v4_causal_lm_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-causal-lm-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-causal-lm-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-causal-lm-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-causal-lm-test
    COMMAND mfq-metal-deepseek-v4-causal-lm-test
)

add_executable(mfq-metal-deepseek-v4-sparse-test
    tests/mlx_deepseek_v4_sparse_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-sparse-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-sparse-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-sparse-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-sparse-test
    COMMAND mfq-metal-deepseek-v4-sparse-test
)

add_executable(mfq-metal-deepseek-v4-moe-test
    tests/mlx_deepseek_v4_moe_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-moe-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-moe-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-moe-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-moe-test
    COMMAND mfq-metal-deepseek-v4-moe-test
)

add_executable(mfq-metal-mtp-test tests/mlx_mtp_test.cpp)
target_link_libraries(mfq-metal-mtp-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-mtp-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mtp-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-mtp-test COMMAND mfq-metal-mtp-test)

add_executable(mfq-mtp-lora-sessions-test tests/mtp_lora_sessions_test.cpp)
target_link_libraries(mfq-mtp-lora-sessions-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-mtp-lora-sessions-test PRIVATE cxx_std_20)
add_test(NAME mfq-mtp-lora-sessions-test COMMAND mfq-mtp-lora-sessions-test)

add_executable(mfq-metal-deepseek-v4-dspark-test
    tests/mlx_deepseek_v4_dspark_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-dspark-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-dspark-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-dspark-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-dspark-test
    COMMAND mfq-metal-deepseek-v4-dspark-test
)

add_executable(mfq-metal-deepseek-v4-vision-test
    tests/mlx_deepseek_v4_vision_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-vision-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-vision-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-vision-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-vision-test
    COMMAND mfq-metal-deepseek-v4-vision-test
)

add_executable(mfq-metal-deepseek-v41-vision-test
    tests/mlx_deepseek_v41_vision_test.cpp
)
target_link_libraries(mfq-metal-deepseek-v41-vision-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-deepseek-v41-vision-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-deepseek-v41-vision-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-deepseek-v41-vision-test COMMAND mfq-metal-deepseek-v41-vision-test)
add_test(NAME mfq-metal-deepseek-v41-vision-routing-test COMMAND mfq-metal-moe-test --test-v41-vision-routing)
add_test(NAME mfq-metal-deepseek-v41-prefix-test COMMAND mfq-metal-moe-test --test-v41-prefix)

add_executable(mfq-metal-moe-ops-test
    tests/mlx_moe_ops_test.cpp
)
target_link_libraries(mfq-metal-moe-ops-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-moe-ops-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-moe-ops-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-moe-ops-test
    COMMAND mfq-metal-moe-ops-test
)

add_executable(mfq-metal-moe-test
    runtime/mlx_paged_session_codec.cpp
    tests/mlx_moe_test.cpp
)
target_link_libraries(mfq-metal-moe-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-moe-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-moe-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-moe-test
    COMMAND mfq-metal-moe-test
)
add_test(NAME mfq-metal-moe-compat-test COMMAND mfq-metal-moe-test --test-compat)
add_test(NAME mfq-metal-moe-portable-test COMMAND mfq-metal-moe-test)
set_tests_properties(mfq-metal-moe-portable-test PROPERTIES ENVIRONMENT "MFQ_METAL_DISABLE_NAX=1")
add_test(NAME mfq-metal-moe-prepare-test COMMAND mfq-metal-moe-test --test-prepare)
add_test(NAME mfq-metal-model-prepare-test COMMAND mfq-metal-moe-test --test-model-prepare)
option(MFQ_TEST_ANE "Run private ANE hardware tests" OFF)
if(MFQ_TEST_ANE)
    add_test(NAME mfq-ane-test COMMAND mfq-ane-test)
    add_test(NAME mfq-metal-ane-autograd-test COMMAND mfq-metal-moe-test --test-ane-autograd)
    add_test(NAME mfq-metal-mtp-ttt-test COMMAND mfq-metal-moe-test --test-mtp-ttt)
endif()

add_executable(mfq-metal-vq-test
    tests/mlx_vq_test.cpp
)
target_link_libraries(mfq-metal-vq-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-vq-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-vq-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-vq-test
    COMMAND mfq-metal-vq-test
)

add_executable(mfq-metal-tensor-test
    tests/mlx_tensor_test.cpp
)
target_link_libraries(mfq-metal-tensor-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-tensor-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-tensor-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-tensor-test COMMAND mfq-metal-tensor-test)

add_executable(mfq-metal-dynamic-geometry-test
    tests/mlx_dynamic_geometry_test.cpp
)
target_link_libraries(mfq-metal-dynamic-geometry-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-dynamic-geometry-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-dynamic-geometry-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-dynamic-geometry-test COMMAND mfq-metal-dynamic-geometry-test)

add_executable(mfq-metal-transformer-test
    tests/mlx_transformer_test.cpp
)
target_link_libraries(mfq-metal-transformer-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-transformer-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-transformer-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-transformer-test
    COMMAND mfq-metal-transformer-test
)

add_executable(mfq-metal-multimodal-test
    tests/mlx_multimodal_test.cpp
)
target_link_libraries(mfq-metal-multimodal-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-multimodal-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-multimodal-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-multimodal-test
    COMMAND mfq-metal-multimodal-test
)

add_executable(mfq-metal-linear-attention-test
    tests/mlx_linear_attention_test.cpp
)
target_link_libraries(mfq-metal-linear-attention-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-linear-attention-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-linear-attention-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-linear-attention-test
    COMMAND mfq-metal-linear-attention-test
)

add_executable(mfq-metal-qwen35-config-test
    tests/qwen35_config_test.cpp
)
target_link_libraries(mfq-metal-qwen35-config-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-config-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-qwen35-config-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen35-config-test
    COMMAND mfq-metal-qwen35-config-test
)

add_executable(mfq-metal-qwen4-config-test
    tests/qwen4_config_test.cpp
)

target_link_libraries(mfq-metal-qwen4-config-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen4-config-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-qwen4-config-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen4-config-test
    COMMAND mfq-metal-qwen4-config-test
)

add_executable(mfq-metal-qwen4-batching-test tests/mlx_qwen4_batching_test.cpp)
target_link_libraries(mfq-metal-qwen4-batching-test PRIVATE mfq-metal-runtime mfq-engine)
target_compile_features(mfq-metal-qwen4-batching-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-qwen4-batching-test PROPERTIES BUILD_RPATH "${MFQ_MLX_ROOT}/lib")

add_executable(mfq-metal-sparse-attention-test
    tests/mlx_sparse_attention_test.cpp
)
target_link_libraries(mfq-metal-sparse-attention-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-sparse-attention-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-sparse-attention-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-sparse-attention-test
    COMMAND mfq-metal-sparse-attention-test
)

add_executable(mfq-metal-qwen-gated-hc-test
    tests/qwen_gated_hc_test.cpp
)
target_link_libraries(mfq-metal-qwen-gated-hc-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen-gated-hc-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-qwen-gated-hc-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen-gated-hc-test
    COMMAND mfq-metal-qwen-gated-hc-test
)

add_executable(mfq-metal-grid-vision-test
    tests/mlx_grid_vision_test.cpp
)
target_link_libraries(
    mfq-metal-grid-vision-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-grid-vision-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-grid-vision-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-grid-vision-test
    COMMAND mfq-metal-grid-vision-test
)

add_executable(mfq-metal-qwen35-full-attention-test
    tests/mlx_qwen35_full_attention_test.cpp
)
target_link_libraries(
    mfq-metal-qwen35-full-attention-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-full-attention-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-qwen35-full-attention-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen35-full-attention-test
    COMMAND mfq-metal-qwen35-full-attention-test
)

add_executable(mfq-metal-qwen35-linear-attention-test
    tests/mlx_qwen35_linear_attention_test.cpp
)
target_link_libraries(
    mfq-metal-qwen35-linear-attention-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-linear-attention-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-qwen35-linear-attention-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen35-linear-attention-test
    COMMAND mfq-metal-qwen35-linear-attention-test
)

add_executable(mfq-metal-sampling-test
    tests/mlx_sampling_test.cpp
)
target_link_libraries(mfq-metal-sampling-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-sampling-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-sampling-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-sampling-test
    COMMAND mfq-metal-sampling-test
)

add_executable(mfq-metal-qwen35-causal-lm-test
    tests/mlx_qwen35_causal_lm_test.cpp
)
target_link_libraries(
    mfq-metal-qwen35-causal-lm-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-causal-lm-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-qwen35-causal-lm-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen35-causal-lm-test
    COMMAND mfq-metal-qwen35-causal-lm-test
)

add_executable(mfq-metal-nemotron-test
    tests/mlx_nemotron_h_test.cpp
)
target_link_libraries(
    mfq-metal-nemotron-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-nemotron-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-nemotron-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-nemotron-conv-test
    COMMAND mfq-metal-nemotron-test --check-cached-conv)
add_executable(mfq-metal-minicpmo45-test
    tests/mlx_minicpmo45_test.cpp
)
target_link_libraries(
    mfq-metal-minicpmo45-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-minicpmo45-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-minicpmo45-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-minicpmo45-test
    COMMAND mfq-metal-minicpmo45-test
)

add_executable(mfq-metal-qwen35-grouped-parity
    tools/qwen35_grouped_parity.cpp
)
target_link_libraries(
    mfq-metal-qwen35-grouped-parity PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-grouped-parity PRIVATE cxx_std_20
)
set_target_properties(
    mfq-metal-qwen35-grouped-parity PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
