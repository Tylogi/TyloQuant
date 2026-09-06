# MFQ tokenizer, grammar, and chat-template component.
# SPDX-License-Identifier: MIT AND Apache-2.0

find_package(Threads REQUIRED)

add_library(mfq-tokenizer STATIC
    include/mfq_text.h
    src/mfq_text.cpp
    src/mfq_vocab.cpp
    src/mfq_vocab.h
    src/mfq_grammar.cpp
    src/mfq_grammar.h
    src/mfq_chat.cpp
    src/mfq_chat.h
    src/mfq_gguf_loader.h
    src/mfq_impl.h
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
add_library(mfq::tokenizer ALIAS mfq-tokenizer)
# Keep the established target name while downstream build files migrate.
add_library(mfq-text-runtime ALIAS mfq-tokenizer)
target_include_directories(mfq-tokenizer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/chat
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(mfq-tokenizer PUBLIC cxx_std_17)
target_link_libraries(mfq-tokenizer
    PUBLIC
        mfq-ggml
        mfq-json
        Threads::Threads
    PRIVATE
        mfq-ggml-internal-headers
)
set_target_properties(mfq-tokenizer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
    target_compile_options(mfq-tokenizer PRIVATE /utf-8 /EHsc)
endif()

if(BUILD_TESTING)
    add_executable(mfq-tokenizer-test
        tests/mfq_text_runtime_test.cpp
    )
    target_include_directories(mfq-tokenizer-test PRIVATE src)
    target_link_libraries(mfq-tokenizer-test PRIVATE mfq-tokenizer)
    target_compile_features(mfq-tokenizer-test PRIVATE cxx_std_17)
    add_test(NAME mfq-tokenizer-test COMMAND mfq-tokenizer-test)
endif()
