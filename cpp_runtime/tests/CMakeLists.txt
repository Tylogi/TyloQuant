# Backend-independent native tests.

add_executable(mfq-cuda-tensor-view-test
    ${PROJECT_SOURCE_DIR}/backends/cuda/tests/mfq_cuda_tensor_view_test.cpp
)
target_include_directories(mfq-cuda-tensor-view-test PRIVATE
    ${PROJECT_SOURCE_DIR}/backends/cuda/include
)
target_compile_features(mfq-cuda-tensor-view-test PRIVATE cxx_std_20)
add_test(NAME mfq-cuda-tensor-view-test COMMAND mfq-cuda-tensor-view-test)

add_executable(mfq-native-tensor-test
    ${PROJECT_SOURCE_DIR}/backends/cuda/src/mfq_native_tensor.cpp
    ${PROJECT_SOURCE_DIR}/backends/cuda/tests/mfq_native_tensor_cpu_test_stubs.cpp
    ${PROJECT_SOURCE_DIR}/backends/cuda/tests/mfq_native_tensor_test.cpp
)
target_include_directories(mfq-native-tensor-test PRIVATE
    ${PROJECT_SOURCE_DIR}/backends/cuda/include
)
target_compile_features(mfq-native-tensor-test PRIVATE cxx_std_20)
add_test(NAME mfq-native-tensor-test COMMAND mfq-native-tensor-test)

function(mfq_add_core_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE mfq-core)
    target_compile_features(${target} PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8 /EHsc)
    endif()
    add_test(NAME ${target} COMMAND ${target})
endfunction()

mfq_add_core_test(mfq-model-graph-test mfq_model_graph_test.cpp)
mfq_add_core_test(mfq-moe-cache-policy-test moe_cache_policy_test.cpp)
mfq_add_core_test(mfq-tensor-parallel-test tensor_parallel_test.cpp)

add_executable(mfq-cuda-model-plan-test
    ${PROJECT_SOURCE_DIR}/backends/cuda/tests/mfq_cuda_model_plan_test.cpp
)
target_include_directories(mfq-cuda-model-plan-test PRIVATE
    ${PROJECT_SOURCE_DIR}/backends/cuda/include
)
target_link_libraries(mfq-cuda-model-plan-test PRIVATE mfq-core)
target_compile_features(mfq-cuda-model-plan-test PRIVATE cxx_std_20)
add_test(NAME mfq-cuda-model-plan-test COMMAND mfq-cuda-model-plan-test)

add_executable(mfq-moe-cache-profile-test moe_cache_profile_test.cpp)
target_link_libraries(mfq-moe-cache-profile-test PRIVATE mfq-core mfq-json)
target_compile_features(mfq-moe-cache-profile-test PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(mfq-moe-cache-profile-test PRIVATE /utf-8 /EHsc)
endif()
add_test(NAME mfq-moe-cache-profile-test COMMAND mfq-moe-cache-profile-test)

if(TARGET mfq-paged-prefix-cache)
    add_executable(mfq-paged-prefix-cache-test
        mfq_paged_prefix_cache_test.cpp
    )
    target_link_libraries(mfq-paged-prefix-cache-test PRIVATE
        mfq-paged-prefix-cache
    )
    target_compile_features(mfq-paged-prefix-cache-test PRIVATE cxx_std_17)
    add_test(NAME mfq-paged-prefix-cache-test COMMAND mfq-paged-prefix-cache-test)
endif()

if(TARGET mfq-server)
    add_executable(mfq-runtime-profile-test mfq_runtime_profile_test.cpp)
    target_link_libraries(mfq-runtime-profile-test PRIVATE mfq-server)
    target_compile_features(mfq-runtime-profile-test PRIVATE cxx_std_17)
    add_test(NAME mfq-runtime-profile-test COMMAND mfq-runtime-profile-test)
endif()
