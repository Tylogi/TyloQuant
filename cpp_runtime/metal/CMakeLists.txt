find_package(Threads REQUIRED)

set(MFQ_MLX_ROOT "" CACHE PATH
    "MLX package root containing include/, lib/libmlx.dylib, and lib/mlx.metallib")
if(NOT MFQ_MLX_ROOT)
    file(GLOB _MFQ_LOCAL_MLX_ROOTS
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/python*/site-packages/mlx")
    list(LENGTH _MFQ_LOCAL_MLX_ROOTS _MFQ_LOCAL_MLX_ROOT_COUNT)
    if(_MFQ_LOCAL_MLX_ROOT_COUNT GREATER 0)
        list(GET _MFQ_LOCAL_MLX_ROOTS 0 MFQ_MLX_ROOT)
    endif()
endif()

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

add_library(mfq-metal-runtime STATIC
    deepseek_v4_model.cpp
    mfq_container.cpp
    mlx_cccp.cpp
    mlx_tensor.cpp
    mlx_grouped_linear.cpp
    mlx_deepseek_v4_attention.cpp
    mlx_deepseek_v4_causal_lm.cpp
    mlx_deepseek_v4_hc.cpp
    mlx_deepseek_v4_moe.cpp
    mlx_deepseek_v4_sparse.cpp
    mlx_moe.cpp
    mlx_moe_ops.cpp
    mlx_mx.cpp
    mlx_nint.cpp
    mlx_nint8_zero.cpp
    mlx_vq.cpp
    mlx_linear_attention.cpp
    mlx_qwen35_full_attention.cpp
    mlx_qwen35_linear_attention.cpp
    mlx_qwen35_causal_lm.cpp
    mlx_sampling.cpp
    mlx_transformer.cpp
    qwen35_model.cpp
)
target_include_directories(mfq-metal-runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(mfq-metal-runtime PUBLIC
    mfq-mlx
    mfq-jaccl
    Threads::Threads
    ${MFQ_METAL_FRAMEWORK}
    ${MFQ_FOUNDATION_FRAMEWORK}
    ${MFQ_QUARTZCORE_FRAMEWORK}
    ${MFQ_ACCELERATE_FRAMEWORK}
)
target_compile_features(mfq-metal-runtime PUBLIC cxx_std_20)
target_compile_definitions(mfq-metal-runtime PUBLIC
    MFQ_NATIVE_MLX_RUNTIME=1
    MFQ_MLX_METALLIB_DEFAULT="${MFQ_MLX_ROOT}/lib/mlx.metallib"
)

add_executable(mfq-decode-metal
    mfq_decode_mlx.cpp
)
target_link_libraries(mfq-decode-metal PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-decode-metal PRIVATE cxx_std_20)

set_target_properties(mfq-decode-metal PROPERTIES
    BUILD_RPATH "@executable_path/lib"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "@executable_path/lib"
)

add_custom_command(TARGET mfq-decode-metal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:mfq-decode-metal>/lib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MFQ_MLX_ROOT}/lib/libmlx.dylib"
        "$<TARGET_FILE_DIR:mfq-decode-metal>/lib/libmlx.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MFQ_MLX_ROOT}/lib/libjaccl.dylib"
        "$<TARGET_FILE_DIR:mfq-decode-metal>/lib/libjaccl.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${MFQ_MLX_ROOT}/lib/mlx.metallib"
        "$<TARGET_FILE_DIR:mfq-decode-metal>/lib/mlx.metallib"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../web"
        "$<TARGET_FILE_DIR:mfq-decode-metal>/web"
    COMMENT "Copying MLX runtime and MFQ WebUI assets"
)

if(TARGET mfq-server)
    file(GLOB _MFQ_LLAMA_RUNTIME_DYLIBS
        "${MFQ_LLAMA_RUNTIME_DIR}/*.dylib")
    if(NOT _MFQ_LLAMA_RUNTIME_DYLIBS)
        message(FATAL_ERROR
            "No llama.cpp runtime dylibs found in "
            "${MFQ_LLAMA_RUNTIME_DIR}")
    endif()

    add_executable(mfq-perplexity
        mfq_perplexity_mlx.cpp
    )
    target_link_libraries(mfq-perplexity PRIVATE
        mfq-metal-runtime
        mfq-server
    )
    target_compile_features(mfq-perplexity PRIVATE cxx_std_20)
    set_target_properties(mfq-perplexity PROPERTIES
        BUILD_RPATH "@executable_path/lib"
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "@executable_path/lib"
    )
    add_custom_command(TARGET mfq-perplexity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:mfq-perplexity>/lib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MFQ_MLX_ROOT}/lib/libmlx.dylib"
            "$<TARGET_FILE_DIR:mfq-perplexity>/lib/libmlx.dylib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MFQ_MLX_ROOT}/lib/libjaccl.dylib"
            "$<TARGET_FILE_DIR:mfq-perplexity>/lib/libjaccl.dylib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MFQ_MLX_ROOT}/lib/mlx.metallib"
            "$<TARGET_FILE_DIR:mfq-perplexity>/lib/mlx.metallib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_MFQ_LLAMA_RUNTIME_DYLIBS}
            "$<TARGET_FILE_DIR:mfq-perplexity>/lib"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying mfq-perplexity MLX and tokenizer runtime files"
    )

    add_custom_command(TARGET mfq-decode-metal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_MFQ_LLAMA_RUNTIME_DYLIBS}
            "$<TARGET_FILE_DIR:mfq-decode-metal>/lib"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying llama.cpp tokenizer runtime dylibs"
    )
    foreach(_MFQ_LLAMA_RUNTIME_DYLIB
            IN LISTS _MFQ_LLAMA_RUNTIME_DYLIBS)
        get_filename_component(
            _MFQ_LLAMA_RUNTIME_DYLIB_NAME
            "${_MFQ_LLAMA_RUNTIME_DYLIB}"
            NAME)
        add_custom_command(TARGET mfq-decode-metal POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DMFQ_DYLIB=$<TARGET_FILE_DIR:mfq-decode-metal>/lib/${_MFQ_LLAMA_RUNTIME_DYLIB_NAME}"
                "-DMFQ_INSTALL_NAME_TOOL=${CMAKE_INSTALL_NAME_TOOL}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/bundle_macos_dylib.cmake"
            COMMENT
                "Making ${_MFQ_LLAMA_RUNTIME_DYLIB_NAME} relocatable"
        )
        add_custom_command(TARGET mfq-perplexity POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DMFQ_DYLIB=$<TARGET_FILE_DIR:mfq-perplexity>/lib/${_MFQ_LLAMA_RUNTIME_DYLIB_NAME}"
                "-DMFQ_INSTALL_NAME_TOOL=${CMAKE_INSTALL_NAME_TOOL}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/bundle_macos_dylib.cmake"
            COMMENT
                "Making ${_MFQ_LLAMA_RUNTIME_DYLIB_NAME} relocatable for mfq-perplexity"
        )
    endforeach()
endif()

add_executable(mfq-metal-container-test
    mfq_container_test.cpp
)
target_link_libraries(mfq-metal-container-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-container-test PRIVATE cxx_std_20)

enable_testing()
add_test(NAME mfq-metal-container-test COMMAND mfq-metal-container-test)

add_executable(mfq-metal-nint-test
    mlx_nint_test.cpp
)
target_link_libraries(mfq-metal-nint-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-nint-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-nint-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-nint-test COMMAND mfq-metal-nint-test)

add_executable(mfq-metal-mx-test
    mlx_mx_test.cpp
)
target_link_libraries(mfq-metal-mx-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mx-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mx-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-mx-test COMMAND mfq-metal-mx-test)

add_executable(mfq-metal-nint-benchmark
    mlx_nint_benchmark.cpp
)
target_link_libraries(mfq-metal-nint-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-nint-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-nint-benchmark PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)

add_executable(mfq-metal-qwen35-projection-profile
    qwen35_projection_profile.cpp
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
    qwen35_decode_overhead_profile.cpp
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
    mlx_nint8_zero_test.cpp
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
    mlx_grouped_linear_test.cpp
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

add_executable(mfq-metal-cccp-test
    mlx_cccp_test.cpp
)
target_link_libraries(
    mfq-metal-cccp-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-cccp-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-cccp-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-cccp-test
    COMMAND mfq-metal-cccp-test
)

add_executable(mfq-metal-deepseek-v4-hc-test
    mlx_deepseek_v4_hc_test.cpp
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

add_executable(mfq-metal-deepseek-v4-model-test
    deepseek_v4_model_test.cpp
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

add_executable(mfq-metal-deepseek-v4-attention-test
    mlx_deepseek_v4_attention_test.cpp
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
    mlx_deepseek_v4_causal_lm_test.cpp
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
    mlx_deepseek_v4_sparse_test.cpp
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
    mlx_deepseek_v4_moe_test.cpp
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

add_executable(mfq-metal-moe-ops-test
    mlx_moe_ops_test.cpp
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
    mlx_moe_test.cpp
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

add_executable(mfq-metal-vq-test
    mlx_vq_test.cpp
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
    mlx_tensor_test.cpp
)
target_link_libraries(mfq-metal-tensor-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-tensor-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-tensor-test PROPERTIES
    BUILD_RPATH "${MFQ_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-tensor-test COMMAND mfq-metal-tensor-test)

add_executable(mfq-metal-transformer-test
    mlx_transformer_test.cpp
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

add_executable(mfq-metal-linear-attention-test
    mlx_linear_attention_test.cpp
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
    qwen35_config_test.cpp
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

add_executable(mfq-metal-qwen35-full-attention-test
    mlx_qwen35_full_attention_test.cpp
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
    mlx_qwen35_linear_attention_test.cpp
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
    mlx_sampling_test.cpp
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
    mlx_qwen35_causal_lm_test.cpp
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
