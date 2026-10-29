// SPDX-License-Identifier: Apache-2.0

#include "chat.h"
#include "gguf.h"
#include "mfq_text.h"
#include "mfq_grammar.h"
#include "unicode.h"
#include "jinja/parser.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void test_jinja_dictionary_arguments() {
    const auto render = [](const std::string & source) {
        jinja::context context(source);
        const auto program = jinja::parse_from_tokens(jinja::lexer().tokenize(source));
        return jinja::runtime::gather_string_parts(jinja::runtime(context).execute(program))->as_string().str();
    };
    require(render("{% set key='b' %}{% set d={'a':1} %}{{ dict(d, **{key:2}, c=3)|tojson }}{{ d|tojson }}")
        == "{\"a\": 1, \"b\": 2, \"c\": 3}{\"a\": 1}", "Jinja mapping unpack/copy mismatch");
    require(render("{{ dict([['a',1],['b',2]], **{'a':3})|tojson }}")
        == "{\"a\": 3, \"b\": 2}", "Jinja dictionary replacement mismatch");
    require(render("{% macro f(a,b) %}{{a}}:{{b}}{% endmacro %}{{f(**{'a':1,'b':2})}}")
        == "1:2", "Jinja macro keyword unpack mismatch");
    require(render("{{ namespace(**{'a':7}).a }}") == "7", "Jinja namespace keyword unpack mismatch");
    for (const std::string source : {
        "{{ dict(**1) }}", "{{ dict(**{1:'a'}) }}", "{{ dict(a=1, **{'a':2}) }}",
        "{{ dict(**{'a':1}, a=2) }}", "{{ dict({}, {}) }}", "{{ dict([['a']]) }}"}) {
        bool failed = false;
        try { render(source); } catch (const std::exception &) { failed = true; }
        require(failed, "Jinja invalid dictionary arguments were accepted");
    }
}

constexpr const char * kTekkenRegex =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

void require_tekken_parts(
        const std::string & text,
        const std::vector<std::string> & expected) {
    const auto actual = unicode_regex_split(text, {kTekkenRegex}, false);
    require(actual == expected, "Tekken regex split mismatch");
}

void test_tekken_regex_split() {
    require_tekken_parts("abcDEF", {"abc", "DEF"});
    require_tekken_parts("ABCdef", {"ABCdef"});
    require_tekken_parts("aBcDeF", {"a", "Bc", "De", "F"});
    require_tekken_parts("Hello, world!", {"Hello", ",", " world", "!"});
    require_tekken_parts("국É한ʰ", {"국É한ʰ"});
    require_tekken_parts("aBcDeF 123\r\n",
        {"a", "Bc", "De", "F", " ", "1", "2", "3", "\r\n"});
    require_tekken_parts("Ünicode ÄÖÜ äöü ß",
        {"Ünicode", " ÄÖÜ", " äöü", " ß"});
    // General categories differ from case-conversion targets. Titlecase is
    // upper-only; mathematical letters retain Lu/Ll without case mappings.
    require_tekken_parts("ǅA", {"ǅA"});
    require_tekken_parts("aǅA", {"a", "ǅA"});
    require_tekken_parts("𝐀A", {"𝐀A"});
    require_tekken_parts("a𝐚A", {"a𝐚", "A"});
    require_tekken_parts("中A", {"中", "A"});
    require_tekken_parts("aʰA", {"aʰ", "A"});
    require_tekken_parts("ʕAa", {"ʕ", "Aa"});  // Ll in the runtime's Unicode version
    // The Tekken punctuation tail includes '/', unlike the Qwen splitter.
    require_tekken_parts("_\n/'", {"_\n/", "'"});

    // GPQA question 74 contains a long DNA sequence. A std::regex fallback
    // exhausted its stack on this shape; the custom splitter must stay linear.
    std::string dna;
    const std::string codon =
        "ATGTACCCATACGATGTTCCAGATTACGCCAAATGACTCTGGAAGAAGTCCGCGGCCAGGACACAGTTCCGGAAAGCACAGCCAGGATGCAGGGTGCCGGGAA";
    for (int repeat = 0; repeat < 40; ++repeat) dna += codon;
    const auto pieces = unicode_regex_split(dna, {kTekkenRegex}, false);
    std::string joined;
    for (const auto & piece : pieces) joined += piece;
    require(joined == dna, "Tekken long DNA split did not preserve the input");
    require(pieces.size() == 1, "Tekken long DNA split changed the uppercase run");

    // Many short matches in one letter run must not repeatedly scan its suffix.
    std::string alternating;
    for (int repeat = 0; repeat < 16384; ++repeat) alternating += "aB";
    const auto mixed = unicode_regex_split(alternating, {kTekkenRegex}, false);
    require(mixed.size() == 16385, "Tekken alternating-case boundaries changed");
    require(mixed.front() == "a" && mixed.back() == "B",
        "Tekken alternating-case edge tokens changed");
    for (size_t i = 1; i + 1 < mixed.size(); ++i)
        require(mixed[i] == "Ba", "Tekken alternating-case middle tokens changed");
}


std::filesystem::path write_tokenizer_fixture() {
    std::vector<std::string> token_storage = {"<unk>", "<s>", "</s>"};
    std::vector<int32_t> token_types = {
        MFQ_TEXT_TOKEN_TYPE_UNKNOWN,
        MFQ_TEXT_TOKEN_TYPE_CONTROL,
        MFQ_TEXT_TOKEN_TYPE_CONTROL,
    };
    for (int value = 0; value < 256; ++value) {
        char token[7];
        std::snprintf(token, sizeof(token), "<0x%02X>", value);
        token_storage.emplace_back(token);
        token_types.push_back(MFQ_TEXT_TOKEN_TYPE_BYTE);
    }
    const int32_t hello_token = static_cast<int32_t>(token_storage.size());
    token_storage.emplace_back("hello");
    token_types.push_back(MFQ_TEXT_TOKEN_TYPE_NORMAL);

    std::vector<const char *> tokens;
    tokens.reserve(token_storage.size());
    for (const auto & token : token_storage) tokens.push_back(token.c_str());
    std::vector<float> scores(token_storage.size(), 0.0f);
    scores[static_cast<size_t>(hello_token)] = 10.0f;

    gguf_context * metadata = gguf_init_empty();
    require(metadata != nullptr, "cannot create GGUF metadata");
    gguf_set_val_str(metadata, "general.architecture", "mfq-test");
    gguf_set_val_str(metadata, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(
        metadata, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(
        metadata, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32,
        scores.data(), scores.size());
    gguf_set_arr_data(
        metadata, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
        token_types.data(), token_types.size());
    gguf_set_val_u32(metadata, "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(metadata, "tokenizer.ggml.eos_token_id", 2);
    gguf_set_val_bool(metadata, "tokenizer.ggml.add_space_prefix", false);
    gguf_set_val_str(
        metadata,
        "tokenizer.chat_template",
        "{% for message in messages %}{{ message['role'] + ': ' + "
        "message['content'] + '\\n' }}{% endfor %}{% if "
        "add_generation_prompt %}assistant: {% endif %}");

    const auto path = std::filesystem::temp_directory_path() /
        "mfq-integrated-tokenizer-test.gguf";
    require(
        gguf_write_to_file(metadata, path.string().c_str(), true),
        "cannot write GGUF fixture");
    gguf_free(metadata);
    return path;
}

}  // namespace

int main() try {
    test_jinja_dictionary_arguments();
    test_tekken_regex_split();
    const auto path = write_tokenizer_fixture();
    std::ifstream input(path, std::ios::binary);
    std::vector<uint8_t> encoded(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    require(!encoded.empty(), "cannot read GGUF fixture");
    input.close();

    mfq_text_context * context = mfq_text_load_file(path.string().c_str());
    require(context != nullptr, "cannot load integrated tokenizer");

    const mfq_text_vocab * vocab = mfq_text_get_vocab(context);
    require(vocab != nullptr, "missing vocabulary");
    require(mfq_text_vocab_n_tokens(vocab) == 260, "vocabulary size mismatch");

    mfq_text_token token = MFQ_TEXT_TOKEN_NULL;
    const int32_t count = mfq_text_tokenize(
        vocab, "h", 1, &token, 1, false, false);
    require(count == 1 && token == 107, "SPM tokenization mismatch");

    const char * source = mfq_text_get_chat_template(context, nullptr);
    require(source != nullptr, "missing chat template");
    auto templates = common_chat_templates_init(context, "");
    common_chat_templates_inputs inputs;
    inputs.messages.push_back({"user", "hello"});
    const auto applied = common_chat_templates_apply(templates.get(), inputs);
    require(
        applied.prompt == "user: hello\nassistant: ",
        "chat template output mismatch");
    common_chat_templates_inputs flexible_inputs;
    flexible_inputs.messages = {
        {"user", "first"},
        {"system", "late but supported"},
    };
    const auto flexible_applied =
        common_chat_templates_apply(templates.get(), flexible_inputs);
    require(
        flexible_applied.prompt ==
            "user: first\nsystem: late but supported\nassistant: ",
        "supported non-leading system message was reordered");

    const std::string strict_template =
        "{% for message in messages %}"
        "{% if message['role'] == 'system' and not loop.first %}"
        "{{ raise_exception('System message must be at the beginning.') }}"
        "{% endif %}"
        "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}";
    auto strict_templates =
        common_chat_templates_init(context, strict_template);
    const auto strict_caps =
        common_chat_templates_get_caps(strict_templates.get());
    require(
        !strict_caps.at("supports_non_leading_system") &&
        !strict_caps.at("supports_multiple_system_messages"),
        "strict system-message capabilities were not detected");
    common_chat_templates_inputs late_system_inputs;
    late_system_inputs.messages = {
        {"system", "base"},
        {"user", "first"},
        {"assistant", "reply"},
        {"system", "late"},
        {"user", "second"},
    };
    const auto late_system_applied =
        common_chat_templates_apply(strict_templates.get(), late_system_inputs);
    require(
        late_system_applied.prompt ==
            "system: base\n\nlate\nuser: first\nassistant: reply\n"
            "user: second\nassistant: ",
        "late system-message fallback mismatch");

    common_chat_templates_inputs distinct_role_inputs;
    distinct_role_inputs.messages = {
        {"system", "root policy"},
        {"user", "first"},
        {"developer", "application policy"},
        {"user", "second"},
    };
    const auto distinct_role_applied =
        common_chat_templates_apply(strict_templates.get(), distinct_role_inputs);
    require(
        distinct_role_applied.prompt ==
            "system: System instructions (higher priority):\nroot policy\n\n"
            "Developer instructions:\napplication policy\nuser: first\n"
            "user: second\nassistant: ",
        "system/developer compatibility fallback mismatch");
    strict_templates.reset();

    const std::string developer_template =
        "{# <|channel|> #}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' and not loop.first %}"
        "{{ raise_exception('System message must be at the beginning.') }}"
        "{% endif %}"
        "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}";
    auto developer_templates =
        common_chat_templates_init(context, developer_template);
    common_chat_templates_inputs developer_inputs;
    developer_inputs.messages = {
        {"system", "root policy"},
        {"developer", "application policy"},
        {"user", "hello"},
    };
    const auto developer_applied =
        common_chat_templates_apply(developer_templates.get(), developer_inputs);
    require(
        developer_applied.prompt ==
            "system: root policy\ndeveloper: application policy\n"
            "user: hello\nassistant: ",
        "native developer role was collapsed");

    common_chat_templates_inputs late_system_with_developer_inputs;
    late_system_with_developer_inputs.messages = {
        {"system", "root policy"},
        {"user", "first"},
        {"developer", "application policy"},
        {"system", "late root policy"},
        {"user", "second"},
    };
    const auto late_system_with_developer_applied = common_chat_templates_apply(
        developer_templates.get(), late_system_with_developer_inputs);
    require(
        late_system_with_developer_applied.prompt ==
            "system: root policy\n\nlate root policy\nuser: first\n"
            "developer: application policy\nuser: second\nassistant: ",
        "native developer role changed during system fallback");
    developer_templates.reset();

    const std::string deepseek_single_turn_template =
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<｜User｜>' + message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<｜Assistant｜>' }}"
        "{% if enable_thinking %}{{ '<think>' }}"
        "{% else %}{{ '</think>' }}{% endif %}"
        "{% endif %}";
    auto deepseek_templates =
        common_chat_templates_init(context, deepseek_single_turn_template);
    common_chat_templates_inputs deepseek_inputs;
    deepseek_inputs.messages = {{"user", "hello"}};
    deepseek_inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    deepseek_inputs.enable_thinking = true;
    const auto deepseek_applied =
        common_chat_templates_apply(deepseek_templates.get(), deepseek_inputs);
    require(
        deepseek_applied.generation_prompt == "<｜Assistant｜><think>",
        "DeepSeek single-turn generation prompt mismatch");
    common_chat_parser_params deepseek_parser(deepseek_applied);
    deepseek_parser.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    deepseek_parser.parser.load(deepseek_applied.parser);
    const auto deepseek_partial =
        common_chat_parse("analysis", true, deepseek_parser);
    require(
        deepseek_partial.reasoning_content == "analysis" &&
            deepseek_partial.content.empty(),
        "DeepSeek partial reasoning leaked into content");
    const auto deepseek_complete = common_chat_parse(
        "analysis</think>answer", false, deepseek_parser);
    require(
        deepseek_complete.reasoning_content == "analysis" &&
            deepseek_complete.content == "answer",
        "DeepSeek single-turn close marker was not parsed");
    deepseek_templates.reset();

    mfq_text_grammar * grammar = mfq_text_grammar_init_impl(
        vocab, "root ::= \"h\"", "root", false, nullptr, 0, nullptr, 0);
    require(grammar != nullptr, "cannot create grammar");
    mfq_text_token_data candidates[] = {
        {107, 0.0f, 0.0f},
        {108, 0.0f, 0.0f},
    };
    mfq_text_token_data_array candidate_array = {
        candidates, 2, -1, false};
    mfq_text_grammar_apply_impl(*grammar, &candidate_array);
    require(
        std::isfinite(candidates[0].logit) &&
        !std::isfinite(candidates[1].logit),
        "grammar filtering mismatch");
    mfq_text_grammar_accept_impl(*grammar, 107);
    mfq_text_grammar_free_impl(grammar);
    templates.reset();
    mfq_text_free(context);
    std::filesystem::remove(path);

    mfq_text_context * embedded = mfq_text_load_buffer(
        encoded.data(), encoded.size());
    require(embedded != nullptr, "cannot load embedded tokenizer");
    require(
        mfq_text_vocab_n_tokens(mfq_text_get_vocab(embedded)) == 260,
        "embedded vocabulary size mismatch");
    mfq_text_free(embedded);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "mfq-tokenizer-test: %s\n", error.what());
    return 1;
}
