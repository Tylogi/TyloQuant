# Integrated cpp-httplib transport.
# SPDX-License-Identifier: MIT

find_package(Threads REQUIRED)

add_library(mfq-http STATIC
    httplib.cpp
    httplib.h
)
add_library(mfq::http ALIAS mfq-http)
# Preserve the pre-reorganization target name for downstream CMake users.
add_library(mfq_httplib ALIAS mfq-http)
target_include_directories(mfq-http PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(mfq-http PRIVATE
    CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576
    CPPHTTPLIB_LISTEN_BACKLOG=512
    CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
    CPPHTTPLIB_TCP_NODELAY=1
)
target_link_libraries(mfq-http PRIVATE Threads::Threads)
set_target_properties(mfq-http PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
if(WIN32)
    target_link_libraries(mfq-http PUBLIC ws2_32)
endif()
if(MSVC)
    target_compile_options(mfq-http PRIVATE /w /utf-8 /EHsc)
endif()
