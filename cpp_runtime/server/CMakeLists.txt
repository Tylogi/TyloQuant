# Backend-neutral HTTP/tokenizer server.

find_package(Threads REQUIRED)

add_library(mfq-server STATIC
    include/mfq/server.h
    include/mfq_server.h
    src/server.cpp
)
add_library(mfq::server ALIAS mfq-server)
target_include_directories(mfq-server
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/components/tokenizer/src
)
target_link_libraries(mfq-server
    PUBLIC
        mfq-core
    PRIVATE
        mfq-http
        mfq-tokenizer
        mfq-json
        mfq-paged-prefix-cache
        Threads::Threads
)
target_compile_definitions(mfq-server PRIVATE NOMINMAX)
target_compile_features(mfq-server PUBLIC cxx_std_17)
set_target_properties(mfq-server PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
    target_compile_options(mfq-server PRIVATE /utf-8 /EHsc)
endif()
