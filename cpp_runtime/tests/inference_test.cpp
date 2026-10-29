#include "inference.h"
#include "generation_policy.h"
#include "gguf.h"
#include <cassert>
int main() {
    using namespace mfq::engine;
    InferenceMetrics metrics;
    metrics.mark_token();
    assert(metrics.saw_token && metrics.first_token == metrics.last_token);
    const auto first = metrics.first_token;
    metrics.mark_token();
    assert(metrics.first_token == first && metrics.last_token >= first);
    InferenceRequest request; request.prompt = {1, 2, 3}; request.sampling.max_tokens = 3;
    auto plan = plan_generation(request.prompt, 32, 8, 20, 2);
    assert(plan.generation_tokens == 5 && plan.stable_prefix_tokens == 2);
    auto chunk = next_prefill_chunk(9, 4, 3);
    assert(chunk.count == 3 && !chunk.last);
    InferenceOutput output(request, nullptr, "raw");
    auto delta = output.append({4, 5, 6, 7});
    assert((delta.token_ids == std::vector<int64_t>{4, 5, 6}));
    assert(output.stopped() && output.result.completion_tokens == 3);
    assert(output.metrics.saw_token && output.metrics.last_token >= output.metrics.first_token);
    const auto last = output.metrics.last_token;
    assert(output.append({8}).token_ids.empty());
    assert(output.metrics.last_token == last);
    output.result.cancelled = true;
    output.finish();
    assert(output.result.finish_reason == "cancelled");
    assert(output.metrics.last_token == last);
    assert(output.finish().diffs.empty());
    assert(model_stop_token_ids(R"({"eos_token_id":[0,7,7,-1,32,"8",true,1.5,18446744073709551615]})", 32) ==
        (std::unordered_set<std::int64_t>{0, 7}));
    assert(model_stop_token_ids(R"({"eos_token_id":3,"text_config":{"eos_token_id":[4,5]}})", 32) ==
        (std::unordered_set<std::int64_t>{4, 5}));
    assert(model_stop_token_ids(R"({"eos_token_id":0,"text_config":{}})", 32).contains(0));
    assert(model_stop_token_ids(R"({"eos_token_id":7,"text_config":{"eos_token_id":[]}})", 32).empty());
    assert(model_stop_token_ids("{}", 32).empty());
    assert(model_stop_token_ids("invalid", 32).empty());
    for (const auto eos : {0, 7}) {
        InferenceRequest raw;
        raw.sampling.max_tokens = 64;
        raw.stop_token_ids = {0, 7};
        InferenceOutput stopped(raw, nullptr, "eos");
        assert((stopped.append({4, eos, 5}).token_ids == std::vector<int64_t>{4, eos}));
        assert(stopped.stopped() && stopped.result.completion_tokens == 2);
        stopped.finish();
        assert(stopped.result.finish_reason == "stop" && stopped.result.text.empty());
        assert(stopped.append({5}).token_ids.empty());
    }
    InferenceRequest single;
    single.sampling.max_tokens = 1;
    single.stop_token_ids = {7};
    InferenceOutput first_eos(single, nullptr, "first-eos");
    assert((first_eos.append({7, 8}).token_ids == std::vector<int64_t>{7}));
    first_eos.finish();
    assert(first_eos.result.finish_reason == "stop");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_empty(), &gguf_free);
    const char* pieces[] = {"a", "b", "c", "d", "e", "f", "g", "<eos>"};
    const int32_t types[] = {1, 1, 1, 1, 1, 1, 1, 3};
    gguf_set_val_str(metadata.get(), "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(metadata.get(), "tokenizer.ggml.pre", "gpt-2");
    gguf_set_arr_str(metadata.get(), "tokenizer.ggml.tokens", pieces, 8);
    gguf_set_arr_str(metadata.get(), "tokenizer.ggml.merges", nullptr, 0);
    gguf_set_arr_data(metadata.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types, 8);
    gguf_set_val_u32(metadata.get(), "tokenizer.ggml.eos_token_id", 7);
    const uint32_t eog_ids[] = {5, 7};
    gguf_set_arr_data(metadata.get(), "tokenizer.ggml.eog_token_ids", GGUF_TYPE_UINT32, eog_ids, 2);
    gguf_set_val_bool(metadata.get(), "tokenizer.ggml.add_bos_token", false);
    std::vector<uint8_t> bytes(gguf_get_meta_size(metadata.get()));
    gguf_get_meta_data(metadata.get(), bytes.data());
    MfqTokenizer tokenizer(bytes);
    assert(tokenizer.is_eog(5));
    assert(tokenizer.is_eog(7));
    InferenceRequest text;
    text.chat = false;
    text.sampling.max_tokens = 64;
    text.stop_token_ids = {4};
    InferenceOutput tokenized(text, &tokenizer, "tokenizer-priority");
    assert((tokenized.append({4}).token_ids == std::vector<int64_t>{4}));
    assert(!tokenized.stopped());
    assert((tokenized.append({5, 7}).token_ids == std::vector<int64_t>{5}));
    assert(tokenized.stopped() && tokenized.result.finish_reason == "stop");
    tokenized.finish();
    auto original = text;
    original.stop_token_ids.clear();
    InferenceOutput unchanged(original, &tokenizer, "tokenizer-original");
    assert((unchanged.append({4, 5, 7}).token_ids == std::vector<int64_t>{4, 5}));
    unchanged.finish();
    assert(tokenized.result.text == unchanged.result.text);
    assert(tokenized.result.finish_reason == unchanged.result.finish_reason);
    assert(tokenized.result.completion_tokens == unchanged.result.completion_tokens);
    for(bool tokenized_mode:{false,true}) {
        InferenceRequest fixed;
        fixed.benchmark_fixed_tokens=true;fixed.chat=false;
        fixed.sampling.max_tokens=4;fixed.stop_token_ids={5,7};
        InferenceOutput bench(fixed,tokenized_mode?&tokenizer:nullptr,"fixed-benchmark");
        const auto ids=bench.append({4,5,7,1,2}).token_ids;
        if(ids!=std::vector<int64_t>{4,5,7,1} || !bench.stopped() || bench.result.completion_tokens!=4)
            throw std::runtime_error("fixed benchmark stopped early at EOS or exceeded max_tokens");
        bench.finish();
        if(bench.result.finish_reason!="length")throw std::runtime_error("fixed benchmark finish reason differs");
    }

}
