# Backend-neutral graph, policies, interfaces, and generated tables.

find_package(Threads REQUIRED)
add_library(mfq-host-parallel STATIC host_parallel.cpp include/mfq/host_parallel.h
    host_copy.cpp include/mfq/host_copy.h
    cpu_expert_pool.cpp include/mfq/cpu_expert_pool.h)
target_include_directories(mfq-host-parallel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mfq-host-parallel PUBLIC cxx_std_17)
if(MSVC)
    # Worker failures must unwind dispatch locks before the caller retries.
    target_compile_options(mfq-host-parallel PRIVATE /EHsc)
endif()
target_link_libraries(mfq-host-parallel PUBLIC Threads::Threads)
set_target_properties(mfq-host-parallel PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(mfq::host-parallel ALIAS mfq-host-parallel)

add_library(mfq-model-graph STATIC
    grid_vision.cpp
    grid_vision.h
    mfq_model_graph.cpp
    mfq_model_graph.h
    compat/mfq_legacy_model_graph.cpp
    compat/mfq_legacy_model_graph.h
    compat/mfq_legacy_tensor_names.cpp
    compat/mfq_legacy_tensor_names.h
    compat/mfq_format_compat.h
)
add_library(mfq::model-graph ALIAS mfq-model-graph)
target_include_directories(mfq-model-graph PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
)
target_link_libraries(mfq-model-graph PUBLIC mfq-json)
target_compile_features(mfq-model-graph PUBLIC cxx_std_17)
set_target_properties(mfq-model-graph PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(mfq-model-source STATIC
    file_range_reader.cpp
    async_range_reader.cpp
    include/mfq/file_range_reader.h
    mfe_quant_expert_store.cpp
    include/mfq/mfe_quant_expert_store.h
    nint_rows.cpp
    include/mfq/nint_rows.h
    nvq_rows.cpp
    include/mfq/nvq_rows.h
    include/mfq/packed_row_range.h
    hf_model_source.cpp
    hf_safetensors_source.cpp
    include/mfq/hf_model_source.h
    include/mfq/hf_safetensors_source.h
    include/mfq/model_graph.h
    include/mfq/model_source.h
    include/mfq/mfq_model_source.h
    mfq_model_source.cpp
    model_source.cpp
    tq_fixture_source.cpp
    include/mfq/tq_fixture_source.h
    include/mfq/tq_rows.h
)
add_library(mfq::model-source ALIAS mfq-model-source)
target_include_directories(mfq-model-source PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
)
target_link_libraries(mfq-model-source PUBLIC mfq-model-graph)
target_link_libraries(mfq-model-source PRIVATE mfq-host-parallel)
target_link_libraries(mfq-model-source PRIVATE mfq-json)
target_compile_features(mfq-model-source PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(mfq-model-source PRIVATE /EHsc)
endif()
set_target_properties(mfq-model-source PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(mfq-core INTERFACE)
add_library(mfq-qsa-kv-store STATIC qsa_kv_store.cpp include/mfq/qsa_kv_store.h)
target_include_directories(mfq-qsa-kv-store PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mfq-qsa-kv-store PUBLIC Threads::Threads)
target_compile_features(mfq-qsa-kv-store PUBLIC cxx_std_17)
set_target_properties(mfq-qsa-kv-store PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(mfq-qsa-kv-store PRIVATE /utf-8 /EHsc)
endif()
add_library(mfq::core ALIAS mfq-core)
target_include_directories(mfq-core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
)
target_link_libraries(mfq-core INTERFACE
    mfq-host-parallel
    mfq-model-graph
    mfq-model-source
    mfq-json
    mfq-qsa-kv-store
)

if(MFQ_BUILD_RUNTIME_COMMUNICATION)
    find_package(Threads REQUIRED)

    add_library(mfq-paged-prefix-cache STATIC
        mfq_paged_prefix_cache.cpp
        mfq_paged_prefix_cache.h
    )
    target_include_directories(mfq-paged-prefix-cache PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(mfq-paged-prefix-cache PUBLIC Threads::Threads)
    target_compile_features(mfq-paged-prefix-cache PUBLIC cxx_std_17)
    set_target_properties(mfq-paged-prefix-cache PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()
