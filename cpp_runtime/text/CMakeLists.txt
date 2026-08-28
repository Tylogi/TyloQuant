# MFQ's integrated GGUF tokenizer, grammar, and chat-template runtime.
# SPDX-License-Identifier: Apache-2.0

enable_language(C)
find_package(Threads REQUIRED)

add_library(mfq-gguf STATIC
    ggml/include/ggml.h
    ggml/include/ggml-alloc.h
    ggml/include/ggml-backend.h
    ggml/include/ggml-cpp.h
    ggml/include/gguf.h
    ggml/src/ggml.c
    ggml/src/ggml-alloc.c
    ggml/src/ggml-backend.cpp
    ggml/src/ggml-backend-meta.cpp
    ggml/src/ggml-threading.cpp
    ggml/src/ggml-quants.c
    ggml/src/gguf.cpp
)
target_include_directories(mfq-gguf
    PUBLIC ggml/include
    PRIVATE ggml/src
)
target_compile_features(mfq-gguf PUBLIC cxx_std_17)
target_compile_definitions(mfq-gguf PRIVATE
    GGML_VERSION="mfq-integrated"
    GGML_COMMIT="mfq-integrated"
    GGML_SCHED_MAX_COPIES=4
)
target_link_libraries(mfq-gguf PRIVATE Threads::Threads)
set_target_properties(mfq-gguf PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_compile_definitions(mfq-gguf PRIVATE _GNU_SOURCE _XOPEN_SOURCE=600)
    target_link_libraries(mfq-gguf PRIVATE dl)
elseif(APPLE)
    target_compile_definitions(mfq-gguf PRIVATE _DARWIN_C_SOURCE _XOPEN_SOURCE=600)
elseif(WIN32)
    target_compile_definitions(mfq-gguf PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_library(mfq-text-runtime STATIC
    include/mfq_text.h
    src/mfq_text.cpp
    src/mfq_vocab.cpp
    src/mfq_vocab.h
    src/mfq_grammar.cpp
    src/mfq_grammar.h
    src/mfq_chat.cpp
    src/mfq_chat.h
    src/unicode.cpp
    src/unicode.h
    src/unicode-data.cpp
    src/unicode-data.h
    chat/common.cpp
    chat/common.h
    chat/chat-auto-parser-generator.cpp
    chat/chat-auto-parser-helpers.cpp
    chat/chat-diff-analyzer.cpp
    chat/chat-peg-parser.cpp
    chat/chat.cpp
    chat/json-schema-to-grammar.cpp
    chat/peg-parser.cpp
    chat/unicode.cpp
    chat/unicode.h
    chat/jinja/caps.cpp
    chat/jinja/lexer.cpp
    chat/jinja/parser.cpp
    chat/jinja/runtime.cpp
    chat/jinja/string.cpp
    chat/jinja/value.cpp
)
target_include_directories(mfq-text-runtime
    PUBLIC
        include
        chat
        ${CMAKE_CURRENT_SOURCE_DIR}/../json
    PRIVATE
        src
        ggml/include
        ggml/src
)
target_compile_features(mfq-text-runtime PUBLIC cxx_std_17)
target_link_libraries(mfq-text-runtime PUBLIC mfq-gguf Threads::Threads)
set_target_properties(mfq-text-runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(mfq-text-runtime PRIVATE /utf-8 /EHsc)
endif()

if(BUILD_TESTING)
    add_executable(mfq-text-runtime-test mfq_text_runtime_test.cpp)
    target_include_directories(mfq-text-runtime-test PRIVATE src)
    target_link_libraries(mfq-text-runtime-test PRIVATE mfq-text-runtime)
    target_compile_features(mfq-text-runtime-test PRIVATE cxx_std_17)
    add_test(NAME mfq-text-runtime-test COMMAND mfq-text-runtime-test)
endif()
