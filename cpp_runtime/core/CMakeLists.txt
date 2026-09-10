# Backend-neutral graph, policies, interfaces, and generated tables.

add_library(mfq-model-graph STATIC
    mfq_model_graph.cpp
    mfq_model_graph.h
    models/deepseek_v41.cpp
    models/deepseek_v41.h
    compat/mfq_legacy_model_graph.cpp
    compat/mfq_legacy_model_graph.h
    compat/mfq_legacy_tensor_names.cpp
    compat/mfq_legacy_tensor_names.h
)
add_library(mfq::model-graph ALIAS mfq-model-graph)
target_include_directories(mfq-model-graph PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
)
target_link_libraries(mfq-model-graph PRIVATE mfq-json)
target_compile_features(mfq-model-graph PUBLIC cxx_std_17)
set_target_properties(mfq-model-graph PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(mfq-core INTERFACE)
add_library(mfq::core ALIAS mfq-core)
target_include_directories(mfq-core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compat
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
)
target_link_libraries(mfq-core INTERFACE
    mfq-model-graph
    mfq-json
)

if(MFQ_BUILD_CPP_SERVER)
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
