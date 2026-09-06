# Header-only nlohmann/json integration.
# SPDX-License-Identifier: MIT

add_library(mfq-json INTERFACE)
add_library(mfq::json ALIAS mfq-json)
target_include_directories(mfq-json INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
