cmake_minimum_required(VERSION 3.26)
project(mfq_cpp_runtime LANGUAGES CXX)

include(CTest)

if(BUILD_TESTING)
    add_executable(mfq-cuda-tensor-view-test
        cuda/mfq_cuda_tensor_view_test.cpp
    )
    target_include_directories(mfq-cuda-tensor-view-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    )
    target_compile_features(mfq-cuda-tensor-view-test PRIVATE cxx_std_20)
    add_test(NAME mfq-cuda-tensor-view-test COMMAND mfq-cuda-tensor-view-test)

    add_executable(mfq-native-tensor-test
        cuda/mfq_native_tensor.cpp
        cuda/mfq_native_tensor_cpu_test_stubs.cpp
        cuda/mfq_native_tensor_test.cpp
    )
    target_include_directories(mfq-native-tensor-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    )
    target_compile_features(mfq-native-tensor-test PRIVATE cxx_std_20)
    add_test(NAME mfq-native-tensor-test COMMAND mfq-native-tensor-test)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(
    MFQ_BUILD_METAL_RUNTIME
    "Build the native MLX/Metal runtime on Apple Silicon"
    ${APPLE}
)

if(APPLE)
    set(_MFQ_BUILD_CPP_SERVER_DEFAULT OFF)
else()
    set(_MFQ_BUILD_CPP_SERVER_DEFAULT ON)
endif()
option(
    MFQ_BUILD_CPP_SERVER
    "Build the backend-neutral C++ HTTP/tokenizer server library"
    ${_MFQ_BUILD_CPP_SERVER_DEFAULT}
)
unset(_MFQ_BUILD_CPP_SERVER_DEFAULT)

if(MFQ_BUILD_METAL_RUNTIME AND NOT APPLE)
    message(FATAL_ERROR "MFQ_BUILD_METAL_RUNTIME requires macOS")
endif()

if(MFQ_BUILD_CPP_SERVER)
    find_package(Threads REQUIRED)
    set(MFQ_LLAMA_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama-runtime")
    if(NOT EXISTS "${MFQ_LLAMA_SOURCE_DIR}/common/chat.h")
        message(FATAL_ERROR
            "Bundled llama runtime is incomplete: ${MFQ_LLAMA_SOURCE_DIR}")
    endif()

    # Tokenizer, GGUF metadata and chat-template support are built from the
    # exact vendored source revision.  A clean MFQ checkout therefore needs
    # no external llama.cpp checkout, build directory, or runtime dylibs.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_MTMD OFF CACHE BOOL "" FORCE)
    set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    set(GGML_RPC OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${MFQ_LLAMA_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/llama-runtime"
        EXCLUDE_FROM_ALL)

    add_library(mfq_llama ALIAS llama)
    add_library(mfq_ggml_base ALIAS ggml-base)
    add_library(mfq_llama_common ALIAS llama-common)

    add_library(mfq_httplib STATIC
        ../third_party/cpp-httplib/httplib.cpp
        ../third_party/cpp-httplib/httplib.h
    )
    target_include_directories(mfq_httplib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cpp-httplib
    )
    target_compile_definitions(mfq_httplib PRIVATE
        CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576
        CPPHTTPLIB_LISTEN_BACKLOG=512
        CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
        CPPHTTPLIB_TCP_NODELAY=1
    )
    target_link_libraries(mfq_httplib PRIVATE Threads::Threads)
    set_target_properties(mfq_httplib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    if(WIN32)
        target_link_libraries(mfq_httplib PUBLIC ws2_32)
    endif()
    if(MSVC)
        target_compile_options(mfq_httplib PRIVATE /w /utf-8 /EHsc)
    endif()

    add_library(mfq-server STATIC
        mfq_server.cpp
        mfq_server.h
    )
    add_library(mfq::server ALIAS mfq-server)
    target_include_directories(mfq-server
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
        PRIVATE
            ${MFQ_LLAMA_SOURCE_DIR}
            ${MFQ_LLAMA_SOURCE_DIR}/common
            ${MFQ_LLAMA_SOURCE_DIR}/include
            ${MFQ_LLAMA_SOURCE_DIR}/ggml/include
            ${MFQ_LLAMA_SOURCE_DIR}/ggml/src
            ${MFQ_LLAMA_SOURCE_DIR}/vendor
            ${MFQ_LLAMA_SOURCE_DIR}/vendor/nlohmann
    )
    target_link_libraries(mfq-server PRIVATE
        mfq_httplib
        mfq_llama_common
        mfq_llama
        mfq_ggml_base
        Threads::Threads
    )
    target_compile_definitions(mfq-server PRIVATE
        NOMINMAX
        LLAMA_SHARED
    )
    target_compile_features(mfq-server PUBLIC cxx_std_17)
    set_target_properties(mfq-server PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    if(MSVC)
        target_compile_options(mfq-server PRIVATE /utf-8 /EHsc)
    endif()
endif()

if(MFQ_BUILD_METAL_RUNTIME)
    enable_language(OBJCXX)
    enable_testing()
    add_subdirectory(metal)

    add_executable(mfq-metal-qwen35-real-layer-smoke
        metal/qwen35_real_layer_smoke.cpp
    )
    target_link_libraries(
        mfq-metal-qwen35-real-layer-smoke PRIVATE
        mfq-metal-runtime
    )
    target_compile_features(
        mfq-metal-qwen35-real-layer-smoke PRIVATE
        cxx_std_20
    )

    add_executable(mfq-metal-qwen35-real-model-smoke
        metal/qwen35_real_model_smoke.cpp
    )
    target_link_libraries(
        mfq-metal-qwen35-real-model-smoke PRIVATE
        mfq-metal-runtime
    )
    target_compile_features(
        mfq-metal-qwen35-real-model-smoke PRIVATE
        cxx_std_20
    )

    if(TARGET mfq-server)
        target_link_libraries(mfq-decode-metal PRIVATE
            mfq-server
        )
        target_compile_definitions(mfq-decode-metal PRIVATE
            MFQ_METAL_SERVER=1
        )

        add_executable(mfq-metal-tokenizer-test
            metal/mfq_tokenizer_test.cpp
        )

        add_executable(mfq-runtime-profile-test
            mfq_runtime_profile_test.cpp
        )
        target_link_libraries(mfq-runtime-profile-test PRIVATE
            mfq-server
        )
        target_compile_features(mfq-runtime-profile-test PRIVATE
            cxx_std_17
        )
        add_test(
            NAME mfq-runtime-profile-test
            COMMAND mfq-runtime-profile-test
        )
        target_link_libraries(mfq-metal-tokenizer-test PRIVATE
            mfq-metal-runtime
            mfq-server
        )
        target_compile_features(
            mfq-metal-tokenizer-test PRIVATE
            cxx_std_20
        )

        add_executable(mfq-metal-deepseek-v4-real-cache-audit
            metal/deepseek_v4_real_cache_audit.cpp
        )
        target_link_libraries(
            mfq-metal-deepseek-v4-real-cache-audit PRIVATE
            mfq-metal-runtime
            mfq-server
        )
        target_compile_features(
            mfq-metal-deepseek-v4-real-cache-audit PRIVATE
            cxx_std_20
        )
    endif()
endif()

if(NOT APPLE)
    if(NOT MFQ_BUILD_CPP_SERVER)
        message(FATAL_ERROR
            "The CUDA mfq-decode CLI currently requires "
            "MFQ_BUILD_CPP_SERVER=ON")
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(MFQ_CUDA_ARCHITECTURES "86" CACHE STRING
        "CUDA architectures compiled into MFQ CUDA targets")

    find_package(CUDAToolkit REQUIRED)

    add_library(mfq-cuda-core STATIC
        cuda/mfq_cuda_context.cu
        cuda/mfq_native_tensor.cpp
        cuda/mfq_native_tensor.cu
        cuda/mfq_native_tensor_ops.cu
    )
    target_include_directories(mfq-cuda-core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(mfq-cuda-core PUBLIC
        CUDA::cudart
        CUDA::cublas
    )
    target_compile_features(mfq-cuda-core PUBLIC cxx_std_20)
    set_target_properties(mfq-cuda-core PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    set(MFQ_CUDA_KERNEL_SOURCES
        ../mfq/kernels/cuda/acc.cu
        ../mfq/kernels/cuda/activation.cu
        ../mfq/kernels/cuda/attention.cu
        ../mfq/kernels/cuda/attention_llama.cu
        ../mfq/kernels/cuda/deepseek_v4_attention.cu
        ../mfq/kernels/cuda/deepseek_v4_hc.cu
        ../mfq/kernels/cuda/embedding.cu
        ../mfq/kernels/cuda/gated_delta_net.cu
        ../mfq/kernels/cuda/glm_dsa.cu
        ../mfq/kernels/cuda/kv_cache.cu
        ../mfq/kernels/cuda/moe.cu
        ../mfq/kernels/cuda/mx_matmul.cu
        ../mfq/kernels/cuda/nepq.cu
        ../mfq/kernels/cuda/nepq_residual.cu
        ../mfq/kernels/cuda/nint_matmul.cu
        ../mfq/kernels/cuda/norm.cu
        ../mfq/kernels/cuda/nvq_matmul.cu
        ../mfq/kernels/cuda/rope.cu
        ../mfq/kernels/cuda/sampling.cu
        ../mfq/kernels/cuda/ssm_conv.cu
        ../mfq/kernels/cuda/tpq_matmul.cu
    )
    add_library(mfq-cuda-native-kernels STATIC
        ${MFQ_CUDA_KERNEL_SOURCES}
    )
    target_compile_definitions(mfq-cuda-native-kernels PRIVATE
        MFQ_NATIVE_CUDA_RUNTIME=1
    )
    target_include_directories(mfq-cuda-native-kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/../mfq/kernels/cuda
        ${CUDAToolkit_INCLUDE_DIRS}
        ${MFQ_LLAMA_SOURCE_DIR}
        ${MFQ_LLAMA_SOURCE_DIR}/ggml/include
        ${MFQ_LLAMA_SOURCE_DIR}/ggml/src
        ${MFQ_LLAMA_SOURCE_DIR}/ggml/src/ggml-cuda
    )
    target_link_libraries(mfq-cuda-native-kernels PUBLIC mfq-cuda-core)
    set_target_properties(mfq-cuda-native-kernels PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    if(BUILD_TESTING)
        add_executable(mfq-cuda-context-test
            cuda/mfq_cuda_context_test.cu
        )
        target_link_libraries(mfq-cuda-context-test PRIVATE mfq-cuda-core)
        set_target_properties(mfq-cuda-context-test PROPERTIES
            CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        )
        add_test(NAME mfq-cuda-context-test COMMAND mfq-cuda-context-test)
        set_tests_properties(mfq-cuda-context-test PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(mfq-cuda-activation-test
            cuda/mfq_cuda_activation_test.cu
        )
        target_link_libraries(mfq-cuda-activation-test PRIVATE
            mfq-cuda-core
            mfq-cuda-native-kernels
        )
        set_target_properties(mfq-cuda-activation-test PROPERTIES
            CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        )
        add_test(NAME mfq-cuda-activation-test COMMAND mfq-cuda-activation-test)
        set_tests_properties(mfq-cuda-activation-test PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(mfq-native-tensor-cuda-test
            cuda/mfq_native_tensor_cuda_test.cu
        )
        target_link_libraries(mfq-native-tensor-cuda-test PRIVATE mfq-cuda-core)
        set_target_properties(mfq-native-tensor-cuda-test PROPERTIES
            CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )
        add_test(NAME mfq-native-tensor-cuda-test COMMAND mfq-native-tensor-cuda-test)
        set_tests_properties(mfq-native-tensor-cuda-test PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    find_package(Threads REQUIRED)
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

add_executable(mfq-decode
    mfq_decode.cpp
    ggml_cuda_compat.cu
)

target_include_directories(mfq-decode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nlohmann
    ${MFQ_LLAMA_SOURCE_DIR}
    ${MFQ_LLAMA_SOURCE_DIR}/common
    ${MFQ_LLAMA_SOURCE_DIR}/include
    ${MFQ_LLAMA_SOURCE_DIR}/ggml/include
    ${MFQ_LLAMA_SOURCE_DIR}/ggml/src
    ${MFQ_LLAMA_SOURCE_DIR}/ggml/src/ggml-cuda
    ${MFQ_LLAMA_SOURCE_DIR}/vendor
)
target_link_libraries(mfq-decode PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    mfq-cuda-core
    mfq-cuda-native-kernels
    mfq-server
)
target_compile_definitions(mfq-decode PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1
    NOMINMAX
)
target_compile_features(mfq-decode PRIVATE cxx_std_20)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mfq-decode PRIVATE OpenMP::OpenMP_CXX)
endif()
if (MFQ_NCCL_LIBRARY AND MFQ_NCCL_INCLUDE_DIR)
    target_include_directories(mfq-decode PRIVATE
        "${MFQ_NCCL_INCLUDE_DIR}")
    target_link_libraries(mfq-decode PRIVATE "${MFQ_NCCL_LIBRARY}")
    target_compile_definitions(mfq-decode PRIVATE MFQ_HAVE_NCCL=1)
endif()
if (MSVC)
    target_compile_options(mfq-decode PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>" "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
endif()

option(
    MFQ_BUILD_TORCH_REFERENCE_RUNTIME
    "Build the optional LibTorch CUDA runtime used only for A/B validation"
    OFF
)
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
    add_executable(mfq-decode-torch
        mfq_decode.cpp
        ggml_cuda_compat.cu
        ${MFQ_CUDA_KERNEL_SOURCES}
    )
    target_include_directories(mfq-decode-torch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        ${CUDAToolkit_INCLUDE_DIRS}
        ${Python_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nlohmann
        ${MFQ_LLAMA_SOURCE_DIR}
        ${MFQ_LLAMA_SOURCE_DIR}/common
        ${MFQ_LLAMA_SOURCE_DIR}/include
        ${MFQ_LLAMA_SOURCE_DIR}/ggml/include
        ${MFQ_LLAMA_SOURCE_DIR}/ggml/src
        ${MFQ_LLAMA_SOURCE_DIR}/ggml/src/ggml-cuda
        ${MFQ_LLAMA_SOURCE_DIR}/vendor
    )
    target_link_libraries(mfq-decode-torch PRIVATE
        ${TORCH_LIBRARIES}
        Python::Python
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::cublas
        mfq-server
    )
    target_compile_definitions(mfq-decode-torch PRIVATE NOMINMAX)
    target_compile_features(mfq-decode-torch PRIVATE cxx_std_17)
    set_property(TARGET mfq-decode-torch PROPERTY
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}")
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mfq-decode-torch PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MFQ_NCCL_LIBRARY AND MFQ_NCCL_INCLUDE_DIR)
        target_include_directories(mfq-decode-torch PRIVATE
            "${MFQ_NCCL_INCLUDE_DIR}")
        target_link_libraries(mfq-decode-torch PRIVATE "${MFQ_NCCL_LIBRARY}")
        target_compile_definitions(mfq-decode-torch PRIVATE MFQ_HAVE_NCCL=1)
    endif()
    if(MSVC)
        target_compile_options(mfq-decode-torch PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
            "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
    endif()
endif()

set_property(TARGET mfq-decode PROPERTY
    CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}")
if (NOT WIN32)
    set(MFQ_SYSTEM_RUNTIME_DIR
        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH
        "directory containing the host libstdc++ runtime")
    set_property(TARGET mfq-decode PROPERTY
        BUILD_RPATH
            "${MFQ_SYSTEM_RUNTIME_DIR};${MFQ_LLAMA_RUNTIME_DIR}")
endif()

set_source_files_properties(../mfq/kernels/cuda/attention_llama.cu PROPERTIES
    COMPILE_OPTIONS "--use_fast_math;--extended-lambda"
)
set_source_files_properties(../mfq/kernels/cuda/deepseek_v4_attention.cu PROPERTIES
    COMPILE_OPTIONS "--use_fast_math;--extended-lambda"
)
set_source_files_properties(../mfq/kernels/cuda/deepseek_v4_hc.cu PROPERTIES
    COMPILE_OPTIONS "--extended-lambda"
)

if(WIN32)
    foreach(_dll IN LISTS _MFQ_LLAMA_RUNTIME_DLLS)
        add_custom_command(TARGET mfq-decode POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MFQ_LLAMA_RUNTIME_DIR}/${_dll}"
                "$<TARGET_FILE_DIR:mfq-decode>/${_dll}"
        )
    endforeach()
endif()

add_executable(mfq-moe-cache-policy-test
    moe_cache_policy_test.cpp
)
target_include_directories(mfq-moe-cache-policy-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if (MSVC)
    target_compile_options(mfq-moe-cache-policy-test PRIVATE /utf-8 /EHsc)
endif()

add_executable(mfq-moe-cache-profile-test
    moe_cache_profile_test.cpp
)
target_include_directories(mfq-moe-cache-profile-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nlohmann
)
if (MSVC)
    target_compile_options(mfq-moe-cache-profile-test PRIVATE /utf-8 /EHsc)
endif()

add_executable(mfq-tensor-parallel-test
    tensor_parallel_test.cpp
)
target_include_directories(mfq-tensor-parallel-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if (MSVC)
    target_compile_options(mfq-tensor-parallel-test PRIVATE /utf-8 /EHsc)
endif()

endif()
