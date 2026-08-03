cmake_minimum_required(VERSION 3.26)
project(mfq_cpp_runtime LANGUAGES CXX)

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

    set(_MFQ_LLAMA_BUILD_DEFAULT "")
    if(EXISTS
           "${CMAKE_CURRENT_SOURCE_DIR}/../references/llamacpp/build_mfq_bench/CMakeCache.txt")
        set(_MFQ_LLAMA_BUILD_DEFAULT
            "${CMAKE_CURRENT_SOURCE_DIR}/../references/llamacpp/build_mfq_bench")
    endif()
    set(MFQ_LLAMA_BUILD_DIR "${_MFQ_LLAMA_BUILD_DEFAULT}" CACHE PATH
        "Matching llama.cpp shared-library build used by mfq-server")
    unset(_MFQ_LLAMA_BUILD_DEFAULT)

    set(_MFQ_LLAMA_SOURCE_DEFAULT "")
    if(EXISTS
           "${CMAKE_CURRENT_SOURCE_DIR}/../references/llamacpp/common/chat.h")
        set(_MFQ_LLAMA_SOURCE_DEFAULT
            "${CMAKE_CURRENT_SOURCE_DIR}/../references/llamacpp")
    endif()
    set(MFQ_LLAMA_SOURCE_DIR "${_MFQ_LLAMA_SOURCE_DEFAULT}" CACHE PATH
        "llama.cpp source tree matching MFQ_LLAMA_BUILD_DIR")
    unset(_MFQ_LLAMA_SOURCE_DEFAULT)

    if(NOT EXISTS "${MFQ_LLAMA_SOURCE_DIR}/common/chat.h")
        message(FATAL_ERROR
            "MFQ_LLAMA_SOURCE_DIR does not contain common/chat.h: "
            "${MFQ_LLAMA_SOURCE_DIR}")
    endif()
    if(NOT EXISTS "${MFQ_LLAMA_BUILD_DIR}/CMakeCache.txt")
        message(FATAL_ERROR
            "MFQ_LLAMA_BUILD_DIR is not a configured llama.cpp build: "
            "${MFQ_LLAMA_BUILD_DIR}")
    endif()

    # common/chat.h exposes C++ types in the llama-common ABI. Do not mix
    # headers from one llama.cpp checkout with libraries built from another.
    file(STRINGS "${MFQ_LLAMA_BUILD_DIR}/CMakeCache.txt"
        _MFQ_LLAMA_HOME_LINE
        REGEX "^CMAKE_HOME_DIRECTORY:[^=]*="
        LIMIT_COUNT 1)
    if(NOT _MFQ_LLAMA_HOME_LINE)
        message(FATAL_ERROR
            "Cannot identify the llama.cpp source used by "
            "${MFQ_LLAMA_BUILD_DIR}")
    endif()
    string(REGEX REPLACE "^CMAKE_HOME_DIRECTORY:[^=]*=" ""
        _MFQ_LLAMA_CONFIGURED_SOURCE "${_MFQ_LLAMA_HOME_LINE}")
    file(REAL_PATH "${MFQ_LLAMA_SOURCE_DIR}" _MFQ_LLAMA_SOURCE_REAL)
    file(REAL_PATH "${_MFQ_LLAMA_CONFIGURED_SOURCE}"
        _MFQ_LLAMA_CONFIGURED_SOURCE_REAL)
    if(NOT _MFQ_LLAMA_SOURCE_REAL STREQUAL
       _MFQ_LLAMA_CONFIGURED_SOURCE_REAL)
        message(FATAL_ERROR
            "MFQ_LLAMA_SOURCE_DIR (${_MFQ_LLAMA_SOURCE_REAL}) does not "
            "match MFQ_LLAMA_BUILD_DIR's source "
            "(${_MFQ_LLAMA_CONFIGURED_SOURCE_REAL})")
    endif()
    unset(_MFQ_LLAMA_HOME_LINE)
    unset(_MFQ_LLAMA_CONFIGURED_SOURCE)
    unset(_MFQ_LLAMA_SOURCE_REAL)
    unset(_MFQ_LLAMA_CONFIGURED_SOURCE_REAL)

    set(MFQ_LLAMA_RUNTIME_DIR "${MFQ_LLAMA_BUILD_DIR}/bin")
    if(WIN32)
        set(MFQ_LLAMA_LIBRARY
            "${MFQ_LLAMA_BUILD_DIR}/src/llama.lib")
        set(MFQ_GGML_BASE_LIBRARY
            "${MFQ_LLAMA_BUILD_DIR}/ggml/src/ggml-base.lib")
        set(MFQ_LLAMA_COMMON_LIBRARY
            "${MFQ_LLAMA_BUILD_DIR}/common/llama-common.lib")
        set(_MFQ_LLAMA_RUNTIME_DLLS
            llama.dll
            llama-common.dll
            ggml.dll
            ggml-base.dll
            ggml-cpu.dll)
    else()
        if(APPLE)
            set(_MFQ_LLAMA_SHARED_SUFFIX ".dylib")
        else()
            set(_MFQ_LLAMA_SHARED_SUFFIX ".so")
        endif()
        set(MFQ_LLAMA_LIBRARY
            "${MFQ_LLAMA_RUNTIME_DIR}/libllama${_MFQ_LLAMA_SHARED_SUFFIX}")
        set(MFQ_GGML_BASE_LIBRARY
            "${MFQ_LLAMA_RUNTIME_DIR}/libggml-base${_MFQ_LLAMA_SHARED_SUFFIX}")
        set(MFQ_LLAMA_COMMON_LIBRARY
            "${MFQ_LLAMA_RUNTIME_DIR}/libllama-common${_MFQ_LLAMA_SHARED_SUFFIX}")
        unset(_MFQ_LLAMA_SHARED_SUFFIX)
    endif()

    foreach(_MFQ_LLAMA_LIBRARY_PATH
            "${MFQ_LLAMA_LIBRARY}"
            "${MFQ_GGML_BASE_LIBRARY}"
            "${MFQ_LLAMA_COMMON_LIBRARY}")
        if(NOT EXISTS "${_MFQ_LLAMA_LIBRARY_PATH}")
            message(FATAL_ERROR
                "Required llama.cpp library does not exist: "
                "${_MFQ_LLAMA_LIBRARY_PATH}")
        endif()
    endforeach()
    unset(_MFQ_LLAMA_LIBRARY_PATH)
    if(WIN32)
        foreach(_MFQ_LLAMA_DLL IN LISTS _MFQ_LLAMA_RUNTIME_DLLS)
            if(NOT EXISTS "${MFQ_LLAMA_RUNTIME_DIR}/${_MFQ_LLAMA_DLL}")
                message(FATAL_ERROR
                    "Required llama.cpp runtime DLL does not exist: "
                    "${MFQ_LLAMA_RUNTIME_DIR}/${_MFQ_LLAMA_DLL}")
            endif()
        endforeach()
    endif()

    add_library(mfq_llama SHARED IMPORTED GLOBAL)
    add_library(mfq_ggml_base SHARED IMPORTED GLOBAL)
    add_library(mfq_llama_common SHARED IMPORTED GLOBAL)
    if(WIN32)
        set_target_properties(mfq_llama PROPERTIES
            IMPORTED_IMPLIB "${MFQ_LLAMA_LIBRARY}"
            IMPORTED_LOCATION "${MFQ_LLAMA_RUNTIME_DIR}/llama.dll")
        set_target_properties(mfq_ggml_base PROPERTIES
            IMPORTED_IMPLIB "${MFQ_GGML_BASE_LIBRARY}"
            IMPORTED_LOCATION "${MFQ_LLAMA_RUNTIME_DIR}/ggml-base.dll")
        set_target_properties(mfq_llama_common PROPERTIES
            IMPORTED_IMPLIB "${MFQ_LLAMA_COMMON_LIBRARY}"
            IMPORTED_LOCATION
                "${MFQ_LLAMA_RUNTIME_DIR}/llama-common.dll")
    else()
        set_target_properties(mfq_llama PROPERTIES
            IMPORTED_LOCATION "${MFQ_LLAMA_LIBRARY}")
        set_target_properties(mfq_ggml_base PROPERTIES
            IMPORTED_LOCATION "${MFQ_GGML_BASE_LIBRARY}")
        set_target_properties(mfq_llama_common PROPERTIES
            IMPORTED_LOCATION "${MFQ_LLAMA_COMMON_LIBRARY}")
    endif()

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
        target_link_libraries(mfq-metal-tokenizer-test PRIVATE
            mfq-metal-runtime
            mfq-server
        )
        target_compile_features(
            mfq-metal-tokenizer-test PRIVATE
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

    find_package(CUDAToolkit REQUIRED)
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    find_package(Threads REQUIRED)
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
    if(UNIX)
        get_filename_component(MFQ_PYTHON_SITE_PACKAGES
            "${Torch_DIR}/../../../.." ABSOLUTE)
        find_library(MFQ_NCCL_LIBRARY
            NAMES nccl libnccl.so.2
            HINTS "${MFQ_PYTHON_SITE_PACKAGES}/nvidia/nccl/lib")
        find_path(MFQ_NCCL_INCLUDE_DIR
            NAMES nccl.h
            HINTS "${MFQ_PYTHON_SITE_PACKAGES}/nvidia/nccl/include")
    endif()

add_executable(mfq-decode
    mfq_decode.cpp
    ggml_cuda_compat.cu
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
    ../mfq/kernels/cuda/nint_matmul.cu
    ../mfq/kernels/cuda/nvq_matmul.cu
    ../mfq/kernels/cuda/nepq.cu
    ../mfq/kernels/cuda/norm.cu
    ../mfq/kernels/cuda/rope.cu
    ../mfq/kernels/cuda/sampling.cu
    ../mfq/kernels/cuda/ssm_conv.cu
)

target_include_directories(mfq-decode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
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
target_link_libraries(mfq-decode PRIVATE
    ${TORCH_LIBRARIES}
    Python::Python
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    mfq-server
)
if (MFQ_NCCL_LIBRARY AND MFQ_NCCL_INCLUDE_DIR)
    target_include_directories(mfq-decode PRIVATE
        "${MFQ_NCCL_INCLUDE_DIR}")
    target_link_libraries(mfq-decode PRIVATE "${MFQ_NCCL_LIBRARY}")
    target_compile_definitions(mfq-decode PRIVATE MFQ_HAVE_NCCL=1)
endif()
target_compile_definitions(mfq-decode PRIVATE
    NOMINMAX
)

if (MSVC)
    target_compile_options(mfq-decode PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>" "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
endif()

set(MFQ_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures compiled into mfq-decode")
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

add_custom_target(mfq-web-assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/web"
        "$<TARGET_FILE_DIR:mfq-decode>/web"
    COMMENT "Copying MFQ WebUI assets"
)
add_dependencies(mfq-decode mfq-web-assets)

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
