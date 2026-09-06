# Minimal GGML/GGUF support retained by MFQ.
# SPDX-License-Identifier: MIT

enable_language(C)
find_package(Threads REQUIRED)

add_library(mfq-ggml STATIC
    include/ggml.h
    include/ggml-alloc.h
    include/ggml-backend.h
    include/ggml-cpp.h
    include/gguf.h
    src/ggml.c
    src/ggml-alloc.c
    src/ggml-backend.cpp
    src/ggml-backend-meta.cpp
    src/ggml-threading.cpp
    src/ggml-quants.c
    src/gguf.cpp
)
add_library(mfq::ggml ALIAS mfq-ggml)
# Preserve the pre-reorganization target name for downstream CMake users.
add_library(mfq-gguf ALIAS mfq-ggml)
target_include_directories(mfq-ggml
    PUBLIC include
    PRIVATE src
)
target_compile_features(mfq-ggml PUBLIC cxx_std_17)
target_compile_definitions(mfq-ggml PRIVATE
    GGML_VERSION="mfq-integrated"
    GGML_COMMIT="mfq-integrated"
    GGML_SCHED_MAX_COPIES=4
)
target_link_libraries(mfq-ggml PRIVATE Threads::Threads)
set_target_properties(mfq-ggml PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# CUDA kernels consume a small set of implementation/MMA headers without
# linking a GGML compute backend.
add_library(mfq-ggml-internal-headers INTERFACE)
target_include_directories(mfq-ggml-internal-headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-cuda
)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_compile_definitions(mfq-ggml PRIVATE _GNU_SOURCE _XOPEN_SOURCE=600)
    target_link_libraries(mfq-ggml PRIVATE dl)
elseif(APPLE)
    target_compile_definitions(mfq-ggml PRIVATE _DARWIN_C_SOURCE _XOPEN_SOURCE=600)
elseif(WIN32)
    target_compile_definitions(mfq-ggml PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
