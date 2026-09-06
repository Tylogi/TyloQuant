#include "mfq_chat.h"

#include "mfq_text.h"

#include <map>
#include <sstream>
#include <algorithm>

#if __cplusplus >= 202000L
    #define LU8(x) (const char*)(u8##x)
#else
    #define LU8(x) u8##x
#endif

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && isspace(static_cast<unsigned char>(str[start]))) {
        start += 1;
    }
    while (end > start && isspace(static_cast<unsigned char>(str[end - 1]))) {
        end -= 1;
    }
    return str.substr(start, end - start);
}

static const std::map<std::string, mfq_text_chat_template> MFQ_TEXT_CHAT_TEMPLATES = {
    { "chatml",            MFQ_TEXT_CHAT_TEMPLATE_CHATML            },
    { "llama2",            MFQ_TEXT_CHAT_TEMPLATE_META_2           },
    { "llama2-sys",        MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS       },
    { "llama2-sys-bos",    MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS_BOS   },
    { "llama2-sys-strip",  MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS_STRIP },
    { "mistral-v1",        MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V1        },
    { "mistral-v3",        MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V3        },
    { "mistral-v3-tekken", MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN },
    { "mistral-v7",        MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V7        },
    { "mistral-v7-tekken", MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN },
    { "phi3",              MFQ_TEXT_CHAT_TEMPLATE_PHI_3             },
    { "phi4",              MFQ_TEXT_CHAT_TEMPLATE_PHI_4             },
    { "falcon3",           MFQ_TEXT_CHAT_TEMPLATE_FALCON_3          },
    { "zephyr",            MFQ_TEXT_CHAT_TEMPLATE_ZEPHYR            },
    { "monarch",           MFQ_TEXT_CHAT_TEMPLATE_MONARCH           },
    { "gemma",             MFQ_TEXT_CHAT_TEMPLATE_GEMMA             },
    { "orion",             MFQ_TEXT_CHAT_TEMPLATE_ORION             },
    { "openchat",          MFQ_TEXT_CHAT_TEMPLATE_OPENCHAT          },
    { "vicuna",            MFQ_TEXT_CHAT_TEMPLATE_VICUNA            },
    { "vicuna-orca",       MFQ_TEXT_CHAT_TEMPLATE_VICUNA_ORCA       },
    { "deepseek",          MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK          },
    { "deepseek2",         MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK_2        },
    { "deepseek3",         MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK_3        },
    { "deepseek-ocr",      MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK_OCR      },
    { "command-r",         MFQ_TEXT_CHAT_TEMPLATE_COMMAND_R         },
    { "llama3",            MFQ_TEXT_CHAT_TEMPLATE_META_3           },
    { "chatglm3",          MFQ_TEXT_CHAT_TEMPLATE_CHATGLM_3         },
    { "chatglm4",          MFQ_TEXT_CHAT_TEMPLATE_CHATGLM_4         },
    { "glmedge",           MFQ_TEXT_CHAT_TEMPLATE_GLMEDGE           },
    { "minicpm",           MFQ_TEXT_CHAT_TEMPLATE_MINICPM           },
    { "exaone3",           MFQ_TEXT_CHAT_TEMPLATE_EXAONE_3          },
    { "exaone4",           MFQ_TEXT_CHAT_TEMPLATE_EXAONE_4          },
    { "exaone-moe",        MFQ_TEXT_CHAT_TEMPLATE_EXAONE_MOE        },
    { "rwkv-world",        MFQ_TEXT_CHAT_TEMPLATE_RWKV_WORLD        },
    { "granite",           MFQ_TEXT_CHAT_TEMPLATE_GRANITE_3_X       },
    { "granite-4.0",       MFQ_TEXT_CHAT_TEMPLATE_GRANITE_4_0       },
    { "granite-4.1",       MFQ_TEXT_CHAT_TEMPLATE_GRANITE_4_1       },
    { "gigachat",          MFQ_TEXT_CHAT_TEMPLATE_GIGACHAT          },
    { "megrez",            MFQ_TEXT_CHAT_TEMPLATE_MEGREZ            },
    { "yandex",            MFQ_TEXT_CHAT_TEMPLATE_YANDEX            },
    { "bailing",           MFQ_TEXT_CHAT_TEMPLATE_BAILING           },
    { "bailing-think",     MFQ_TEXT_CHAT_TEMPLATE_BAILING_THINK     },
    { "bailing2",          MFQ_TEXT_CHAT_TEMPLATE_BAILING2          },
    { "llama4",            MFQ_TEXT_CHAT_TEMPLATE_META_4            },
    { "smolvlm",           MFQ_TEXT_CHAT_TEMPLATE_SMOLVLM           },
    { "hunyuan-moe",       MFQ_TEXT_CHAT_TEMPLATE_HUNYUAN_MOE       },
    { "gpt-oss",           MFQ_TEXT_CHAT_TEMPLATE_OPENAI_MOE        },
    { "hunyuan-dense",     MFQ_TEXT_CHAT_TEMPLATE_HUNYUAN_DENSE     },
    { "hunyuan-vl",        MFQ_TEXT_CHAT_TEMPLATE_HUNYUAN_VL        },
    { "kimi-k2",           MFQ_TEXT_CHAT_TEMPLATE_KIMI_K2           },
    { "seed_oss",          MFQ_TEXT_CHAT_TEMPLATE_SEED_OSS          },
    { "grok-2",            MFQ_TEXT_CHAT_TEMPLATE_GROK_2            },
    { "pangu-embedded",    MFQ_TEXT_CHAT_TEMPLATE_PANGU_EMBED       },
    { "solar-open",        MFQ_TEXT_CHAT_TEMPLATE_SOLAR_OPEN        },
};

mfq_text_chat_template mfq_text_chat_template_from_str(const std::string & name) {
    return MFQ_TEXT_CHAT_TEMPLATES.at(name);
}

mfq_text_chat_template mfq_text_chat_detect_template(const std::string & tmpl) {
    try {
        return mfq_text_chat_template_from_str(tmpl);
    } catch (const std::out_of_range &) {
        // ignore
    }

    auto tmpl_contains = [&tmpl](const char * haystack) -> bool {
        return tmpl.find(haystack) != std::string::npos;
    };
    if (tmpl_contains("<|im_start|>")) {
        return tmpl_contains("<|im_sep|>")
            ? MFQ_TEXT_CHAT_TEMPLATE_PHI_4
            : tmpl_contains("<end_of_utterance>")
                ? MFQ_TEXT_CHAT_TEMPLATE_SMOLVLM // SmolVLM uses <|im_start|> as BOS, but it is NOT chatml
                : MFQ_TEXT_CHAT_TEMPLATE_CHATML;
    } else if (tmpl.find("mistral") == 0 || tmpl_contains("[INST]")) {
        if (tmpl_contains("[SYSTEM_PROMPT]")) {
            return MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V7;
        } else if (
            // catches official 'v1' template
            tmpl_contains("' [INST] ' + system_message")
            // catches official 'v3' and 'v3-tekken' templates
            || tmpl_contains("[AVAILABLE_TOOLS]")
        ) {
            // Official mistral 'v1', 'v3' and 'v3-tekken' templates
            // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md
            // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
            if (tmpl_contains(" [INST]")) {
                return MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V1;
            } else if (tmpl_contains("\"[INST]\"")) {
                return MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN;
            }
            return MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V3;
        } else {
            // llama2 template and its variants
            // [variant] support system message
            // See: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            bool support_system_message = tmpl_contains("<<SYS>>");
            bool add_bos_inside_history = tmpl_contains("bos_token + '[INST]");
            bool strip_message = tmpl_contains("content.strip()");
            if (strip_message) {
                return MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS_STRIP;
            } else if (add_bos_inside_history) {
                return MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS_BOS;
            } else if (support_system_message) {
                return MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS;
            } else {
                return MFQ_TEXT_CHAT_TEMPLATE_META_2;
            }
        }
    } else if (tmpl_contains("<|assistant|>") && tmpl_contains("<|end|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_PHI_3;
    } else if (tmpl_contains("[gMASK]<sop>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_CHATGLM_4;
    } else if (tmpl_contains("<|assistant|>") && tmpl_contains("<|user|>")) {
        if (tmpl_contains("<|tool_declare|>")) {
            return MFQ_TEXT_CHAT_TEMPLATE_EXAONE_MOE;
        }
        return tmpl_contains("</s>") ? MFQ_TEXT_CHAT_TEMPLATE_FALCON_3 : MFQ_TEXT_CHAT_TEMPLATE_GLMEDGE;
    } else if (tmpl_contains("<|{{ item['role'] }}|>") && tmpl_contains("<|begin_of_image|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_GLMEDGE;
    } else if (tmpl_contains("<|user|>") && tmpl_contains("<|endoftext|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_ZEPHYR;
    } else if (tmpl_contains("bos_token + message['role']")) {
        return MFQ_TEXT_CHAT_TEMPLATE_MONARCH;
    } else if (tmpl_contains("<start_of_turn>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_GEMMA;
    } else if (tmpl_contains("'\\n\\nAssistant: ' + eos_token")) {
        // OrionStarAI/Orion-14B-Chat
        return MFQ_TEXT_CHAT_TEMPLATE_ORION;
    } else if (tmpl_contains("GPT4 Correct ")) {
        // openchat/openchat-3.5-0106
        return MFQ_TEXT_CHAT_TEMPLATE_OPENCHAT;
    } else if (tmpl_contains("USER: ") && tmpl_contains("ASSISTANT: ")) {
        // eachadea/vicuna-13b-1.1 (and Orca variant)
        if (tmpl_contains("SYSTEM: ")) {
            return MFQ_TEXT_CHAT_TEMPLATE_VICUNA_ORCA;
        }
        return MFQ_TEXT_CHAT_TEMPLATE_VICUNA;
    } else if (tmpl_contains("### Instruction:") && tmpl_contains("<|EOT|>")) {
        // deepseek-ai/deepseek-coder-33b-instruct
        return MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK;
    } else if (tmpl_contains("<|START_OF_TURN_TOKEN|>") && tmpl_contains("<|USER_TOKEN|>")) {
        // CohereForAI/c4ai-command-r-plus
        return MFQ_TEXT_CHAT_TEMPLATE_COMMAND_R;
    } else if (tmpl_contains("<|start_header_id|>") && tmpl_contains("<|end_header_id|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_META_3;
    } else if (tmpl_contains("[gMASK]sop")) {
        // chatglm3-6b
        return MFQ_TEXT_CHAT_TEMPLATE_CHATGLM_3;
    } else if (tmpl_contains(LU8("<用户>"))) {
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        return MFQ_TEXT_CHAT_TEMPLATE_MINICPM;
    } else if (tmpl_contains("'Assistant: ' + message['content'] + eos_token")) {
        return MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK_2;
    } else if (tmpl_contains(LU8("<｜Assistant｜>")) && tmpl_contains(LU8("<｜User｜>")) && tmpl_contains(LU8("<｜end▁of▁sentence｜>"))) {
        return MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK_3;
    } else if (tmpl_contains("[|system|]") && tmpl_contains("[|assistant|]") && tmpl_contains("[|endofturn|]")) {
        if (tmpl_contains("[|tool|]")) {
            return MFQ_TEXT_CHAT_TEMPLATE_EXAONE_4;
        }
        // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
        // EXAONE-3.0-7.8B-Instruct
        return MFQ_TEXT_CHAT_TEMPLATE_EXAONE_3;
    } else if (tmpl_contains("rwkv-world") || tmpl_contains("{{- 'User: ' + message['content']|trim + '\\n\\n' -}}")) {
        return MFQ_TEXT_CHAT_TEMPLATE_RWKV_WORLD;
    } else if (tmpl_contains("<|start_of_role|>")) {
        if (tmpl_contains("<tool_call>") || tmpl_contains("<tools>")) {
            if (tmpl_contains("g4_default_system_message")) {
                return MFQ_TEXT_CHAT_TEMPLATE_GRANITE_4_0;
            }
            return MFQ_TEXT_CHAT_TEMPLATE_GRANITE_4_1;
        }
        return MFQ_TEXT_CHAT_TEMPLATE_GRANITE_3_X;
    } else if (tmpl_contains("message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]")) {
        return MFQ_TEXT_CHAT_TEMPLATE_GIGACHAT;
    } else if (tmpl_contains("<|role_start|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_MEGREZ;
    } else if (tmpl_contains(" Ассистент:")) {
        return MFQ_TEXT_CHAT_TEMPLATE_YANDEX;
    } else if (tmpl_contains("<role>ASSISTANT</role>") && tmpl_contains("'HUMAN'")) {
        return MFQ_TEXT_CHAT_TEMPLATE_BAILING;
    } else if (tmpl_contains("<role>ASSISTANT</role>") && tmpl_contains("\"HUMAN\"") && tmpl_contains("<think>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_BAILING_THINK;
    } else if (tmpl_contains("<role>ASSISTANT</role>") && tmpl_contains("<role>HUMAN</role>") && tmpl_contains("<|role_end|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_BAILING2;
    } else if (tmpl_contains("<|header_start|>") && tmpl_contains("<|header_end|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_META_4;
    } else if (tmpl_contains("<|endofuserprompt|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_DOTS1;
    } else if (tmpl_contains("<|extra_0|>") && tmpl_contains("<|extra_4|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_HUNYUAN_MOE;
    } else if (tmpl_contains("<|start|>") && tmpl_contains("<|channel|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_OPENAI_MOE;
    } else if (tmpl_contains("<｜hy_Assistant｜>") && tmpl_contains("<｜hy_begin▁of▁sentence｜>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_HUNYUAN_VL;
    } else if (tmpl_contains("<｜hy_Assistant｜>") && tmpl_contains("<｜hy_place▁holder▁no▁3｜>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_HUNYUAN_DENSE;
    } else if (tmpl_contains("<|im_assistant|>assistant<|im_middle|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_KIMI_K2;
    } else if (tmpl_contains("<seed:bos>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_SEED_OSS;
    } else if (tmpl_contains("'Assistant: '  + message['content'] + '<|separator|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_GROK_2;
    } else if (tmpl_contains(LU8("[unused9]系统：[unused10]"))) {
        return MFQ_TEXT_CHAT_TEMPLATE_PANGU_EMBED;
    } else if (tmpl_contains("<|begin|>") && tmpl_contains("<|end|>") && tmpl_contains("<|content|>")) {
        return MFQ_TEXT_CHAT_TEMPLATE_SOLAR_OPEN;
    }
    return MFQ_TEXT_CHAT_TEMPLATE_UNKNOWN;
}

// Simple version of "mfq_text_apply_chat_template" that only works with strings
// This function uses heuristic checks to determine commonly used template. It is not a jinja parser.
int32_t mfq_text_chat_apply_template(
    mfq_text_chat_template tmpl,
    const std::vector<const mfq_text_chat_message *> & chat,
    std::string & dest, bool add_ass) {
    // Taken from the research: https://github.com/ggml-org/llama.cpp/issues/5527
    std::stringstream ss;
    if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_CHATML) {
        // chatml template
        for (auto message : chat) {
            ss << "<|im_start|>" << message->role << "\n" << message->content << "<|im_end|>\n";
        }
        if (add_ass) {
            ss << "<|im_start|>assistant\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V7 || tmpl == MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN) {
        // Official mistral 'v7' template
        // See: https://huggingface.co/mistralai/Mistral-Large-Instruct-2411#basic-instruct-template-v7
        //      https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503#basic-instruct-template-v7-tekken
        const char * trailing_space = tmpl == MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V7 ? " " : "";
        for (auto message : chat) {
            std::string role(message->role);
            std::string content(message->content);
            if (role == "system") {
                ss << "[SYSTEM_PROMPT]" << trailing_space << content << "[/SYSTEM_PROMPT]";
            } else if (role == "user") {
                ss << "[INST]" << trailing_space << content << "[/INST]";
            } else {
                ss << trailing_space << content << "</s>";
            }
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V1
            || tmpl == MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V3
            || tmpl == MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN) {
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
        std::string leading_space = tmpl == MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V1 ? " " : "";
        std::string trailing_space = tmpl == MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN ? "" : " ";
        bool trim_assistant_message = tmpl == MFQ_TEXT_CHAT_TEMPLATE_MISTRAL_V3;
        bool is_inside_turn = false;
        for (auto message : chat) {
            if (!is_inside_turn) {
                ss << leading_space << "[INST]" << trailing_space;
                is_inside_turn = true;
            }
            std::string role(message->role);
            std::string content(message->content);
            if (role == "system") {
                ss << content << "\n\n";
            } else if (role == "user") {
                ss << content << leading_space << "[/INST]";
            } else {
                ss << trailing_space << (trim_assistant_message ? trim(content) : content) << "</s>";
                is_inside_turn = false;
            }
        }
    } else if (
            tmpl == MFQ_TEXT_CHAT_TEMPLATE_META_2
            || tmpl == MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS
            || tmpl == MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS_BOS
            || tmpl == MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS_STRIP) {
        // llama2 template and its variants
        // [variant] support system message
        // See: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        bool support_system_message = tmpl != MFQ_TEXT_CHAT_TEMPLATE_META_2;
        // [variant] add BOS inside history
        bool add_bos_inside_history = tmpl == MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS_BOS;
        // [variant] trim spaces from the input message
        bool strip_message = tmpl == MFQ_TEXT_CHAT_TEMPLATE_META_2_SYS_STRIP;
        // construct the prompt
        bool is_inside_turn = true; // skip BOS at the beginning
        ss << "[INST] ";
        for (auto message : chat) {
            std::string content = strip_message ? trim(message->content) : message->content;
            std::string role(message->role);
            if (!is_inside_turn) {
                is_inside_turn = true;
                ss << (add_bos_inside_history ? "<s>[INST] " : "[INST] ");
            }
            if (role == "system") {
                if (support_system_message) {
                    ss << "<<SYS>>\n" << content << "\n<</SYS>>\n\n";
                } else {
                    // if the model does not support system message, we still include it in the first message, but without <<SYS>>
                    ss << content << "\n";
                }
            } else if (role == "user") {
                ss << content << " [/INST]";
            } else {
                ss << content << "</s>";
                is_inside_turn = false;
            }
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_PHI_3) {
        // Phi 3
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>\n" << message->content << "<|end|>\n";
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_PHI_4) {
        // chatml template
        for (auto message : chat) {
            ss << "<|im_start|>" << message->role << "<|im_sep|>" << message->content << "<|im_end|>";
        }
        if (add_ass) {
            ss << "<|im_start|>assistant<|im_sep|>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_FALCON_3) {
        // Falcon 3
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>\n" << message->content << "\n";
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_ZEPHYR) {
        // zephyr template
        for (auto message : chat) {
            ss << "<|" << message->role << "|>" << "\n" << message->content << "<|endoftext|>\n";
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_MONARCH) {
        // mlabonne/AlphaMonarch-7B template (the <s> is included inside history)
        for (auto message : chat) {
            std::string bos = (message == chat.front()) ? "" : "<s>"; // skip BOS for first message
            ss << bos << message->role << "\n" << message->content << "</s>\n";
        }
        if (add_ass) {
            ss << "<s>assistant\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_GEMMA) {
        // google/gemma-7b-it
        std::string system_prompt = "";
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                // there is no system message for gemma, but we will merge it with user prompt, so nothing is broken
                system_prompt += trim(message->content);
                continue;
            }
            // in gemma, "assistant" is "model"
            role = role == "assistant" ? "model" : message->role;
            ss << "<start_of_turn>" << role << "\n";
            if (!system_prompt.empty() && role != "model") {
                ss << system_prompt << "\n\n";
                system_prompt = "";
            }
            ss << trim(message->content) << "<end_of_turn>\n";
        }
        if (add_ass) {
            ss << "<start_of_turn>model\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_ORION) {
        // OrionStarAI/Orion-14B-Chat
        std::string system_prompt = "";
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                // there is no system message support, we will merge it with user prompt
                system_prompt += message->content;
                continue;
            } else if (role == "user") {
                ss << "Human: ";
                if (!system_prompt.empty()) {
                    ss << system_prompt << "\n\n";
                    system_prompt = "";
                }
                ss << message->content << "\n\nAssistant: </s>";
            } else {
                ss << message->content << "</s>";
            }
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_OPENCHAT) {
        // openchat/openchat-3.5-0106,
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "<|end_of_turn|>";
            } else {
                role[0] = toupper(role[0]);
                ss << "GPT4 Correct " << role << ": " << message->content << "<|end_of_turn|>";
            }
        }
        if (add_ass) {
            ss << "GPT4 Correct Assistant:";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_VICUNA || tmpl == MFQ_TEXT_CHAT_TEMPLATE_VICUNA_ORCA) {
        // eachadea/vicuna-13b-1.1 (and Orca variant)
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                // Orca-Vicuna variant uses a system prefix
                if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_VICUNA_ORCA) {
                    ss << "SYSTEM: " << message->content << "\n";
                } else {
                    ss << message->content << "\n\n";
                }
            } else if (role == "user") {
                ss << "USER: " << message->content << "\n";
            } else if (role == "assistant") {
                ss << "ASSISTANT: " << message->content << "</s>\n";
            }
        }
        if (add_ass) {
            ss << "ASSISTANT:";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK) {
        // deepseek-ai/deepseek-coder-33b-instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content;
            } else if (role == "user") {
                ss << "### Instruction:\n" << message->content << "\n";
            } else if (role == "assistant") {
                ss << "### Response:\n" << message->content << "\n<|EOT|>\n";
            }
        }
        if (add_ass) {
            ss << "### Response:\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_COMMAND_R) {
        // CohereForAI/c4ai-command-r-plus
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>" << trim(message->content) << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "user") {
                ss << "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" << trim(message->content) << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "assistant") {
                ss << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" << trim(message->content) << "<|END_OF_TURN_TOKEN|>";
            }
        }
        if (add_ass) {
            ss << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_META_3) {
        // Llama 3
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|start_header_id|>" << role << "<|end_header_id|>\n\n" << trim(message->content) << "<|eot_id|>";
        }
        if (add_ass) {
            ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_CHATGLM_3) {
        // chatglm3-6b
        ss << "[gMASK]" << "sop";
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>" << "\n " << message->content;
        }
        if (add_ass) {
            ss << "<|assistant|>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_CHATGLM_4) {
        ss << "[gMASK]" << "<sop>";
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>" << "\n" << message->content;
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_GLMEDGE) {
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>" << "\n" << message->content;
        }
        if (add_ass) {
            ss << "<|assistant|>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_MINICPM) {
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "user") {
                ss << LU8("<用户>");
                ss << trim(message->content);
                ss << "<AI>";
            } else {
                ss << trim(message->content);
            }
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK_2) {
        // DeepSeek-V2
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "\n\n";
            } else if (role == "user") {
                ss << "User: " << message->content << "\n\n";
            } else if (role == "assistant") {
                ss << "Assistant: " << message->content << LU8("<｜end▁of▁sentence｜>");
            }
        }
        if (add_ass) {
            ss << "Assistant:";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK_3) {
        // DeepSeek-V3
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "\n\n";
            } else if (role == "user") {
                ss << LU8("<｜User｜>") << message->content;
            } else if (role == "assistant") {
                ss << LU8("<｜Assistant｜>") << message->content << LU8("<｜end▁of▁sentence｜>");
            }
        }
        if (add_ass) {
            ss << LU8("<｜Assistant｜>");
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_DEEPSEEK_OCR) {
        for (auto message : chat) {
            // no template
            ss << message->content;
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_EXAONE_3) {
        // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
        // EXAONE-3.0-7.8B-Instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "[|system|]" << trim(message->content) << "[|endofturn|]\n";
            } else if (role == "user") {
                ss << "[|user|]" << trim(message->content) << "\n";
            } else if (role == "assistant") {
                ss << "[|assistant|]" << trim(message->content) << "[|endofturn|]\n";
            }
        }
        if (add_ass) {
            ss << "[|assistant|]";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_EXAONE_4) {
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "[|system|]" << trim(message->content) << "[|endofturn|]\n";
            } else if (role == "user") {
                ss << "[|user|]" << trim(message->content) << "\n";
            } else if (role == "assistant") {
                ss << "[|assistant|]" << trim(message->content) << "[|endofturn|]\n";
            } else if (role == "tool") {
                ss << "[|tool|]" << trim(message->content) << "[|endofturn|]\n";
            }
        }
        if (add_ass) {
            ss << "[|assistant|]";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_EXAONE_MOE) {
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|system|>\n" << trim(message->content) << "<|endofturn|>\n";
            } else if (role == "user") {
                ss << "<|user|>\n" << trim(message->content) << "<|endofturn|>\n";
            } else if (role == "assistant") {
                ss << "<|assistant|>\n" << trim(message->content) << "<|endofturn|>\n";
            } else if (role == "tool") {
                ss << "<|tool|>\n" << trim(message->content) << "<|endofturn|>\n";
            }
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_RWKV_WORLD) {
        // this template requires the model to have "\n\n" as EOT token
        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "system") {
                ss << "System: " << trim(chat[i]->content) << "\n\n";
            } else if (role == "user") {
                ss << "User: " << trim(chat[i]->content) << "\n\n";
                if (i == chat.size() - 1) {
                    ss << "Assistant:";
                }
            } else if (role == "assistant") {
                ss << "Assistant: " << trim(chat[i]->content) << "\n\n";
            }
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_GRANITE_3_X) {
        // IBM Granite 3.x template
        for (const auto & message : chat) {
            std::string role(message->role);
            ss << "<|start_of_role|>" << role << "<|end_of_role|>";
            if (role == "assistant_tool_call") {
                ss << "<|tool_call|>";
            }
            ss << message->content << "<|end_of_text|>\n";
        }
        if (add_ass) {
            ss << "<|start_of_role|>assistant<|end_of_role|>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_GRANITE_4_0) {
        // IBM Granite 4.0 template
        for (const auto & message : chat) {
            std::string role(message->role);
            if (role == "assistant_tool_call") {
                ss << "<|start_of_role|>assistant<|end_of_role|><|tool_call|>";
            } else {
                ss << "<|start_of_role|>" << role << "<|end_of_role|>";
            }
            ss << message->content << "<|end_of_text|>\n";
        }
        if (add_ass) {
            ss << "<|start_of_role|>assistant<|end_of_role|>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_GRANITE_4_1) {
        // IBM Granite 4.1 template
        for (const auto & message : chat) {
            std::string role(message->role);
            if (role == "assistant_tool_call") {
                ss << "<|start_of_role|>assistant<|end_of_role|><|tool_call|>";
            } else {
                ss << "<|start_of_role|>" << role << "<|end_of_role|>";
            }
            ss << message->content << "<|end_of_text|>\n";
        }
        if (add_ass) {
            ss << "<|start_of_role|>assistant<|end_of_role|>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_GIGACHAT) {
        // GigaChat template
        bool has_system = !chat.empty() && std::string(chat[0]->role) == "system";

        // Handle system message if present
        if (has_system) {
            ss << "<s>" << chat[0]->content << "<|message_sep|>";
        } else {
            ss << "<s>";
        }

        // Process remaining messages
        for (size_t i = has_system ? 1 : 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "user") {
                ss << "user<|role_sep|>" << chat[i]->content << "<|message_sep|>"
                << "available functions<|role_sep|>[]<|message_sep|>";
            } else if (role == "assistant") {
                ss << "assistant<|role_sep|>" << chat[i]->content << "<|message_sep|>";
            }
        }

        // Add generation prompt if needed
        if (add_ass) {
            ss << "assistant<|role_sep|>";
        }
    }  else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_MEGREZ) {
        // Megrez template
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|role_start|>" << role << "<|role_end|>" << message->content << "<|turn_end|>";
        }

        if (add_ass) {
            ss << "<|role_start|>assistant<|role_end|>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_YANDEX) {
        // Yandex template ("\n\n" is defined as EOT token)

        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "user") {
                ss << " Пользователь: " << chat[i]->content << "\n\n";
            } else if (role == "assistant") {
                ss << " Ассистент: " << chat[i]->content << "\n\n";
            }
        }

        // Add generation prompt if needed
        if (add_ass) {
            ss << " Ассистент:[SEP]";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_BAILING || tmpl == MFQ_TEXT_CHAT_TEMPLATE_BAILING_THINK) {
        // Bailing (Ling/Ring) template
        for (auto message : chat) {
            std::string role(message->role);

            if (role == "user") {
                role = "HUMAN";
            } else {
                std::transform(role.begin(), role.end(), role.begin(), ::toupper);
            }

            ss << "<role>" << role << "</role>" << message->content;
        }

        if (add_ass) {
            ss << "<role>ASSISTANT</role>";

            if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_BAILING_THINK) {
                ss << "<think>";
            }
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_BAILING2) {
        // Bailing2 (Ling 2.0) template
        bool has_system = !chat.empty() && std::string(chat[0]->role) == "system";

        if (!has_system) {
            ss << "<role>SYSTEM</role>detailed thinking off<|role_end|>";
        }

        for (auto message : chat) {
            std::string role(message->role);

            if (role == "user") {
                role = "HUMAN";
            } else {
                std::transform(role.begin(), role.end(), role.begin(), ::toupper);
            }

            ss << "<role>" << role << "</role>" << message->content << "<|role_end|>";
        }

        if (add_ass) {
            ss << "<role>ASSISTANT</role>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_META_4) {
        // Llama 4
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|header_start|>" << role << "<|header_end|>\n\n" << trim(message->content) << "<|eot|>";
        }
        if (add_ass) {
            ss << "<|header_start|>assistant<|header_end|>\n\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_SMOLVLM) {
        // SmolVLM
        ss << "<|im_start|>"; // uses <|im_start|> as BOS, but the actual content is NOT chatml
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "\n\n";
            } else if (role == "user") {
                ss << "User: " << message->content << "<end_of_utterance>\n";
            } else {
                ss << "Assistant: " << message->content << "<end_of_utterance>\n";
            }
        }
        if (add_ass) {
            ss << "Assistant:";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_DOTS1) {
        // dots.llm1.inst (DOTS1)
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|system|>" << message->content << "<|endofsystem|>";
            } else if (role == "user") {
                ss << "<|userprompt|>" << message->content << "<|endofuserprompt|>";
            } else {
                ss << "<|response|>" << message->content << "<|endofresponse|>";
            }
        }
        if (add_ass) {
            ss << "<|response|>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_HUNYUAN_MOE) {
        // tencent/Hunyuan-A13B-Instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|startoftext|>" << message->content << "<|extra_4|>";
            } else if (role == "assistant") {
                ss << message->content << "<|eos|>";
            } else {
                ss << "<|startoftext|>" << message->content << "<|extra_0|>";
            }
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_OPENAI_MOE) {
        // OpenAI MoE (based on Harmony chat template)
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|start|>" << role << "<|message|>" << message->content;
            ss << (role == "assistant" ? "<|return|>" : "<|end|>");
        }
        if (add_ass) {
            ss << "<|start|>assistant";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_HUNYUAN_DENSE) {
        // tencent/Hunyuan-4B-Instruct
        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (i == 0) {
                if (role == "system") {
                    ss << chat[i]->content << "<｜hy_place▁holder▁no▁3｜>";
                }
            }

            if (role == "assistant") {
                ss << "<｜hy_Assistant｜>" << chat[i]->content << "<｜hy_place▁holder▁no▁2｜>";
            } else if (role == "user") {
                ss << "<｜hy_User｜>" << chat[i]->content << "<｜hy_Assistant｜>";
            }
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_HUNYUAN_VL) {
        // tencent/HunyuanOCR & tencent/HunyuanVL
        ss << "<｜hy_begin▁of▁sentence｜>";
        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (i == 0 && role == "system") {
                ss << chat[i]->content << "<｜hy_place▁holder▁no▁3｜>";
                continue;
            }

            if (role == "user") {
                ss << chat[i]->content << "<｜hy_User｜>";
            } else if (role == "assistant") {
                ss << chat[i]->content << "<｜hy_Assistant｜>";
            }
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_KIMI_K2) {
        // moonshotai/Kimi-K2-Instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|im_system|>system<|im_middle|>";
            } else if (role == "user") {
                ss << "<|im_user|>user<|im_middle|>";
            } else if (role == "assistant") {
                ss << "<|im_assistant|>assistant<|im_middle|>";
            } else if (role == "tool") {
                ss << "<|im_system|>tool<|im_middle|>";
            }

            ss << message->content << "<|im_end|>";
        }
        if (add_ass) {
            ss << "<|im_assistant|>assistant<|im_middle|>";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_SEED_OSS) {
        for (auto message: chat) {
            std::string role(message->role);
            ss << "<seed:bos>" << role << "\n" << (role == "assistant" ? trim(message->content) : message->content) << "<seed:eos>";
        }
        if (add_ass) {
            ss << "<seed:bos>assistant\n";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_GROK_2) {
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "System: " << trim(message->content) << "<|separator|>\n\n";
            } else if (role == "user") {
                ss << "Human: " << trim(message->content) << "<|separator|>\n\n";
            } else if (role == "assistant") {
                ss << "Assistant: " << message->content << "<|separator|>\n\n";
            }
        }
        if (add_ass) {
            ss << "Assistant:";
        }
    }else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_PANGU_EMBED) {
        // [unused9]系统：xxx[unused10]
        // [unused9]用户：xxx[unused10]
        // [unused9]助手：xxx[unused10]
        // ...
        for (size_t i = 0; i < chat.size(); ++i) {
            const auto & msg = chat[i];
            const std::string & role = msg->role;
            const std::string & content = msg->content;

            if (i == 0 && role != "system") {
                ss << "[unused9]系统：[unused10]";
            }

            if (role == "system") {
                ss << "[unused9]系统：" << content << "[unused10]";
            } else if (role == "user") {
                ss << "[unused9]用户：" << content << "[unused10]";
            } else if (role == "assistant") {
                ss << "[unused9]助手：" << content << "[unused10]";
            } else if (role == "tool") {
                ss << "[unused9]工具：" << content << "[unused10]";
            } else if (role == "function") {
                ss << "[unused9]方法：" << content << "[unused10]";
            }
        }
        if (add_ass) {
            ss << "[unused9]助手：";
        }
    } else if (tmpl == MFQ_TEXT_CHAT_TEMPLATE_SOLAR_OPEN) {
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|begin|>" << role << "<|content|>" << message->content << "<|end|>";
        }
        if (add_ass) {
            ss << "<|begin|>assistant";
        }
    } else {
        // template not supported
        return -1;
    }
    dest = ss.str();
    return dest.size();
}
