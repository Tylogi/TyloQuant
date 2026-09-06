# Native CUDA backend.

if(NOT TARGET mfq-server)
    message(FATAL_ERROR
        "The CUDA mfq-decode CLI currently requires MFQ_BUILD_CPP_SERVER=ON")
endif()

set(MFQ_CUDA_ROOT "${PROJECT_SOURCE_DIR}/backends/cuda")
set(MFQ_CUDA_KERNEL_ROOT "${MFQ_REPOSITORY_ROOT}/mfq/kernels/cuda")

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(MFQ_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures compiled into MFQ CUDA targets")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(mfq-cuda-core STATIC
    ${MFQ_CUDA_ROOT}/src/mfq_cuda_context.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor.cpp
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor.cu
    ${MFQ_CUDA_ROOT}/src/mfq_native_tensor_ops.cu
)
add_library(mfq::cuda-core ALIAS mfq-cuda-core)
target_include_directories(mfq-cuda-core PUBLIC
    ${MFQ_CUDA_ROOT}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(mfq-cuda-core PUBLIC CUDA::cudart CUDA::cublas)
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
    ${MFQ_CUDA_KERNEL_ROOT}/embedding.cu
    ${MFQ_CUDA_KERNEL_ROOT}/gated_delta_net.cu
    ${MFQ_CUDA_KERNEL_ROOT}/glm_dsa.cu
    ${MFQ_CUDA_KERNEL_ROOT}/kv_cache.cu
    ${MFQ_CUDA_KERNEL_ROOT}/moe.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mx_matmul.cu
    ${MFQ_CUDA_KERNEL_ROOT}/mxfp4_sq.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nepq.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nepq_residual.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nint_matmul.cu
    ${MFQ_CUDA_KERNEL_ROOT}/norm.cu
    ${MFQ_CUDA_KERNEL_ROOT}/nvq_matmul.cu
    ${MFQ_CUDA_KERNEL_ROOT}/rope.cu
    ${MFQ_CUDA_KERNEL_ROOT}/sampling.cu
    ${MFQ_CUDA_KERNEL_ROOT}/ssm_conv.cu
    # Temporary old-artifact compatibility only; TPQ is not a forward format.
    ${MFQ_CUDA_KERNEL_ROOT}/tpq_matmul.cu
)

add_library(mfq-cuda-native-kernels STATIC ${MFQ_CUDA_KERNEL_SOURCES})
add_library(mfq::cuda-native-kernels ALIAS mfq-cuda-native-kernels)
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
    PRIVATE mfq-ggml-internal-headers
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
    function(mfq_add_cuda_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE mfq-cuda-core ${ARGN})
        set_target_properties(${target} PROPERTIES
            CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
            CUDA_RUNTIME_LIBRARY Shared
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    add_executable(mfq-mxfp4-sq-test ${MFQ_CUDA_ROOT}/tests/mfq_mxfp4_sq_test.cu)
    target_compile_definitions(mfq-mxfp4-sq-test PRIVATE MFQ_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-mxfp4-sq-test PRIVATE ${MFQ_REPOSITORY_ROOT})
    target_link_libraries(mfq-mxfp4-sq-test PRIVATE mfq-cuda-core mfq-cuda-native-kernels)
    set_target_properties(mfq-mxfp4-sq-test PROPERTIES
        CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    mfq_add_cuda_test(mfq-cuda-context-test
        ${MFQ_CUDA_ROOT}/tests/mfq_cuda_context_test.cu)
    mfq_add_cuda_test(mfq-cuda-activation-test
        ${MFQ_CUDA_ROOT}/tests/mfq_cuda_activation_test.cu
        mfq-cuda-native-kernels)
    mfq_add_cuda_test(mfq-native-tensor-cuda-test
        ${MFQ_CUDA_ROOT}/tests/mfq_native_tensor_cuda_test.cu)
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

function(mfq_configure_cuda_decode_target target)
    target_include_directories(${target} PRIVATE
        ${MFQ_REPOSITORY_ROOT}
        ${MFQ_CUDA_ROOT}/include
        ${MFQ_CUDA_ROOT}/models
        ${MFQ_CUDA_KERNEL_ROOT}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE
        mfq-core
        mfq-json
        mfq-server
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

add_executable(mfq-decode
    ${MFQ_CUDA_ROOT}/apps/mfq_decode.cpp
    ${MFQ_CUDA_ROOT}/apps/ggml_cuda_compat.cu
)
mfq_configure_cuda_decode_target(mfq-decode)
target_link_libraries(mfq-decode PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    mfq-cuda-core
    mfq-cuda-native-kernels
)
target_compile_definitions(mfq-decode PRIVATE
    MFQ_NATIVE_CUDA_RUNTIME=1
    NOMINMAX
)
if(MSVC)
    target_compile_options(mfq-decode PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
    )
endif()
set_target_properties(mfq-decode PROPERTIES
    CUDA_ARCHITECTURES "${MFQ_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)

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
    add_executable(mfq-decode-torch
        ${MFQ_CUDA_ROOT}/apps/mfq_decode.cpp
        ${MFQ_CUDA_ROOT}/apps/ggml_cuda_compat.cu
        ${MFQ_CUDA_KERNEL_SOURCES}
    )
    mfq_configure_cuda_decode_target(mfq-decode-torch)
    target_include_directories(mfq-decode-torch PRIVATE ${Python_INCLUDE_DIRS})
    target_link_libraries(mfq-decode-torch PRIVATE
        ${TORCH_LIBRARIES}
        Python::Python
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::cublas
    )
    target_compile_definitions(mfq-decode-torch PRIVATE NOMINMAX)
    target_compile_options(mfq-decode-torch PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
    )
    set_target_properties(mfq-decode-torch PROPERTIES
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
    set_property(TARGET mfq-decode PROPERTY
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
