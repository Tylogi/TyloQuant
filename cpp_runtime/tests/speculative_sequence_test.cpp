#include "speculative_sequence.h"
#include "mtp_batch_policy.h"
#include "batched_speculative_sequence.h"

#include <cmath>
#include <iostream>
#include <thread>

namespace {
using namespace mfq::engine;

struct Tensor {
    std::vector<int64_t> shape;
    bool proposal = false;
    int64_t start = 0;
};

struct Model {
    int64_t cache_pos = 0, decode_position_delta = 0, speculative_start = 0;
    int vocab_size() const { return 3; }
    int max_position_embeddings() const { return 1024; }
    void begin_speculative_suffix(int) { speculative_start = cache_pos-1; }
    void commit_speculative() {}
    void rollback_speculative(int keep) { cache_pos = speculative_start+1+keep; }
};

struct Predictor {
    mtp::GenerationStats last_stats;
    mtp::PolicyState policy_state;
    uint64_t last_cycles = 0, last_accepted = 0, last_rejected = 0;
    int64_t position = 0;
    bool legacy = false;
    bool bootstrap = false;
    bool split = false;
    bool blockwise = false;
    std::vector<std::pair<int64_t,int64_t>> context_appends;
    int maximum_draft_depth() const { return 3; }
    bool dspark_policy() const { return legacy; }
    bool blockwise_drafting() const { return blockwise; }
    bool supports_session_state() const { return true; }
    bool teacher_forced_prompt_prime() const { return !blockwise; }
    bool target_bootstrap_decode() const { return bootstrap; }
    bool split_target_verification() const { return split; }
    bool retains_partial_target_prefix() const { return true; }
    int64_t cache_position() const { return position; }
    void trim_cache_to(int64_t value) { position = value; }
    void append_target_context(Tensor value, int64_t start) {
        if(!blockwise || start<position || (start>position && value.shape.at(1)<2))
            throw std::logic_error("invalid blockwise context append");
        context_appends.emplace_back(start,value.shape.at(1));position=start+value.shape.at(1);
    }
    void seed_target_context(Tensor value,int64_t tokens) {
        if(!blockwise || position || value.shape.at(1)>tokens)throw std::logic_error("invalid context seed");
        position=tokens-value.shape.at(1);append_target_context(value,position);
    }
    Tensor draft_block(int, Tensor, int depth) {
        if(!blockwise)throw std::logic_error("not blockwise");
        return {{1,depth,1},true};
    }
    Tensor draft_next(Tensor hidden, Tensor) {
        if(!blockwise)throw std::logic_error("not blockwise");
        hidden.shape.back()=3;hidden.proposal=true;return hidden;
    }
};

struct Prepared {
    std::vector<int64_t> token_ids;
    Tensor embeddings, positions;
    int decode_position_delta = 0;
    bool transformed() const { return !embeddings.shape.empty() || !positions.shape.empty() || decode_position_delta!=0; }
};

struct Sampler {
    bool greedy() const { return true; }
    double next_uniform() { return 0.5; }
};

struct Ops {
    using Tensor = ::Tensor;
    Model& model;
    Predictor& mtp;
    const Prepared* prepared = nullptr;
    int target = 0;
    Tensor input_ids, counts;
    bool penalties = false;
    Sampler sampler_;
    void initialize(const std::vector<int64_t>& prompt, const MfqSamplingParams&) {
        input_ids = ids_for(prompt);
    }
    Sampler& sampler() { return sampler_; }
    static bool defined(const Tensor& t) { return !t.shape.empty(); }
    static int rank(const Tensor& t) { return t.shape.size(); }
    static int64_t size(const Tensor& t, int axis) {
        return t.shape.at(axis < 0 ? t.shape.size()+axis : axis);
    }
    static Tensor clone(const Tensor& t) { return t; }
    static Tensor slice(Tensor t, int axis, int64_t, int64_t length) {
        t.shape.at(axis < 0 ? t.shape.size()+axis : axis) = length;
        return t;
    }
    static Tensor reshape(Tensor t, std::initializer_list<int64_t> shape) {
        int64_t total = 1, fixed = 1;
        for (int64_t dim : t.shape) total *= dim;
        t.shape = shape;
        for (int64_t dim : t.shape) if (dim > 0) fixed *= dim;
        for (auto& dim : t.shape) if (dim < 0) dim = total/fixed;
        return t;
    }
    static Tensor concatenate(const std::vector<Tensor>& values, int axis) {
        auto result = values.at(0);
        result.shape.at(axis) = 0;
        for (const auto& value : values) result.shape.at(axis) += value.shape.at(axis);
        return result;
    }
    static Tensor ids_for(const std::vector<int64_t>& ids) {
        return {{1, static_cast<int64_t>(ids.size())}};
    }
    static void add_counts(Tensor, Tensor) {}
    static Tensor decode_positions(int64_t, int64_t count) { return {{count}}; }
    Tensor target_forward(Tensor ids, Tensor* raw = nullptr, int64_t confirmed = 0) {
        const auto count = size(ids, 1);
        if (confirmed) model.speculative_start = model.cache_pos;
        model.cache_pos += count;
        Tensor hidden{{1, count, 1}};
        if (raw) *raw = hidden;
        std::this_thread::sleep_for(std::chrono::milliseconds(count > 1 ? 12 : 1));
        return hidden;
    }
    Tensor target_suffix(Tensor ids, Tensor* raw) { return target_forward(ids, raw); }
    struct Prefill { Tensor hidden, raw; double elapsed; };
    Prefill prefill(PrefillChunk chunk) {
        auto hidden = target_forward({{1, chunk.count}});
        return {hidden, hidden, 1.0};
    }
    struct Head { Tensor sample_hidden, chain_hidden; };
    Head predictor_step(Tensor hidden, Tensor ids, Tensor) {
        mtp.position += size(ids, 1);
        hidden.proposal = true;
        return {hidden, hidden};
    }
    static Tensor logits_for(Tensor hidden) {
        hidden.shape.back() = 3;
        return hidden;
    }
    static int32_t sample_normal(Tensor logits, Tensor) { return logits.proposal ? 1 : 0; }
    static int32_t sample_constrained(Tensor logits, Tensor counts, const MfqTokenConstraintPtr&) {
        return sample_normal(logits, counts);
    }
    static std::vector<float> probabilities(Tensor logits, Tensor, const MfqSamplingParams&) {
        return logits.proposal ? std::vector<float>{0, 1, 0} : std::vector<float>{1, 0, 0};
    }
    static mtp::CompactDistribution compact_probabilities(Tensor logits, Tensor counts,
        const MfqSamplingParams& sampling) { return {{0, 1, 2}, probabilities(logits, counts, sampling)}; }
    static std::vector<mtp::CompactDistribution> compact_probability_rows(Tensor logits,
        const MfqSamplingParams& sampling) {
        return std::vector<mtp::CompactDistribution>(size(logits, 0), compact_probabilities(logits, {}, sampling));
    }
};

void run(bool dspark,bool metal_policy=false,bool inherited_park=false) {
    struct PolicyOps : Ops {
        bool metal_policy;
        bool metal_mtp_policy() const {return metal_policy;}
        Head predictor_step(Tensor hidden,Tensor ids,Tensor positions) {
            if(!mtp.legacy)std::this_thread::sleep_for(std::chrono::milliseconds(32));
            return Ops::predictor_step(hidden,ids,positions);
        }
    };
    Model model;
    Predictor mtp;
    mtp.legacy = dspark;
    if(inherited_park) {
        mtp::v1::BatchDepthPolicy previous({"previous"},3);
        previous.start_parked(128);
        mtp.policy_state.park_batch(previous);
    }
    InferenceRequest request;
    request.chat = false;
    request.prompt = {0, 0, 0};
    request.sampling.max_tokens = inherited_park?64:256;
    request.sampling.mtp_max_draft_tokens = 3;
    InferenceOutput output(request, nullptr, "mtp-test");
    Tensor last_hidden;
    auto generation = speculative_sequence(PolicyOps{Ops{model, mtp},metal_policy}, request, output, 512, 0, Tensor{}, &last_hidden);
    std::vector<int64_t> tokens;
    while (auto step = generation.next()) {
        if (!step.value) continue;
        if (const auto* delta = std::get_if<TokenOutput>(&*step.value))
            tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
    }
    if (tokens != std::vector<int64_t>(request.sampling.max_tokens, 0) ||
            model.cache_pos != request.sampling.max_tokens+2 || !Ops::defined(last_hidden))
        throw std::runtime_error("shared MTP output/cache boundary mismatch");
    const auto& stats = output.metrics.mtp;
    if(stats.target_ms<=0 || stats.head_ms<=0 || stats.rollback_ms<0)
        throw std::runtime_error("shared MTP phases were not measured");
    if(inherited_park && (stats.park_count!=1 || stats.reentry_probes!=0 || stats.standard_tokens==0))
        throw std::runtime_error("inherited Metal park was omitted from a short response's metrics");
    if (dspark ? stats.park_count != 0 || stats.standard_tokens != 0 :
        stats.park_count == 0 || stats.standard_tokens == 0)
        throw std::runtime_error("shared MTP policy dispatch mismatch: dspark="+std::to_string(dspark)+
            " metal="+std::to_string(metal_policy)+" parks="+std::to_string(stats.park_count)+
            " standard="+std::to_string(stats.standard_tokens));
}

void run_dspark_draft_cost() {
    struct DelayedDraftOps : Ops {
        Head predictor_step(Tensor hidden,Tensor ids,Tensor positions) {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            return Ops::predictor_step(hidden,ids,positions);
        }
    };
    Model model;Predictor predictor;predictor.legacy=true;
    InferenceRequest request;
    request.chat=false;request.prompt={0,0,0};
    request.sampling.max_tokens=10;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"dspark-draft-cost");
    auto generation=speculative_sequence(DelayedDraftOps{{model,predictor}},
        request,output,512,0,Tensor{},nullptr);
    std::vector<int64_t> tokens;
    while(auto step=generation.next())if(step.value)
        if(const auto* delta=std::get_if<TokenOutput>(&*step.value))
            tokens.insert(tokens.end(),delta->token_ids.begin(),delta->token_ids.end());
    bool sampled_drafts=false;
    for(int depth=1;depth<=3;++depth)if(output.metrics.mtp.measured_depth_ms[depth]>0) {
        sampled_drafts=true;
        if(output.metrics.mtp.measured_depth_ms[depth]<25.0*depth)
            throw std::runtime_error("DSpark measured cycle omitted completed draft work");
    }
    if(tokens!=std::vector<int64_t>(10,0) || output.metrics.mtp.head_ms<75.0 || !sampled_drafts)
        throw std::runtime_error("DSpark depth selection omitted completed draft work from its cycle cost");
}

void run_chunked_prompt_history(bool streaming, int reused) {
    struct StreamOps : Ops {
        bool streaming;
        std::vector<int64_t>* boundaries;
        bool stream_prompt_history(int64_t) const { return streaming; }
        static Tensor slice(Tensor value, int axis, int64_t begin, int64_t length) {
            value=Ops::slice(value,axis,begin,length);value.start+=begin;return value;
        }
        Prefill prefill(PrefillChunk chunk) {
            auto result=Ops::prefill(chunk);result.raw.start=chunk.offset;return result;
        }
        double predictor_prefill(Tensor hidden, Tensor ids, Tensor positions) {
            if(hidden.start+1!=ids.start || size(hidden,1)!=size(ids,1))
                throw std::runtime_error("streamed prompt lost a hidden/input pair at a chunk or reused boundary");
            boundaries->push_back(model.cache_pos);
            (void)predictor_step(hidden,ids,positions);return 0;
        }
    };
    Model model;model.cache_pos=reused;
    Predictor mtp;mtp.position=reused?reused-1:0;
    InferenceRequest request;request.chat=false;request.prompt=std::vector<int64_t>(13,0);
    request.sampling.max_tokens=1;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"mtp-chunked-prompt-history");
    Tensor last_hidden,restored{{1,1,1},false,reused-1};
    std::vector<int64_t> boundaries;
    auto generation=speculative_sequence(StreamOps{Ops{model,mtp},streaming,&boundaries},
        request,output,4,reused,restored,&last_hidden);
    while(auto step=generation.next()) {}
    if(boundaries.empty() || (streaming?boundaries.front()>=13:boundaries.front()!=13) ||
        mtp.position!=12 || last_hidden.start!=12 || output.result.completion_tokens!=1)
        throw std::runtime_error("streamed prompt history changed the final cache boundary");
}

void run_prompt_snapshot_boundary() {
    const auto wall_started=InferenceMetrics::Clock::now();
    Model model;Predictor mtp;
    InferenceRequest request;request.chat=false;request.prompt={0,0,0};
    request.sampling.max_tokens=8;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"mtp-snapshot-timing-test");
    Tensor last_hidden;
    auto generation=speculative_sequence(Ops{model,mtp},request,output,512,0,Tensor{},&last_hidden);
    bool saw_boundary=false;
    InferenceMetrics::Clock::time_point cache_finished;
    while(auto step=generation.next())if(step.value) {
        if(const auto* progress=std::get_if<PrefillProgress>(&*step.value);progress && progress->session_boundary) {
            if(saw_boundary || output.metrics.saw_token || model.cache_pos!=3 ||
                    mtp.cache_position()!=2 || !Ops::defined(last_hidden))
                throw std::runtime_error("MTP prompt snapshot boundary is incomplete");
            saw_boundary=true;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            cache_finished=InferenceMetrics::Clock::now();
        }
        if(std::holds_alternative<TokenOutput>(*step.value) &&
                (!saw_boundary || output.metrics.first_token<cache_finished))
            throw std::runtime_error("initial snapshot entered decode timing");
    }
    if(!saw_boundary || output.result.completion_tokens!=8 || model.cache_pos!=10)
        throw std::runtime_error("prompt snapshot changed generation history");
    const auto wall_ms=std::chrono::duration<double,std::milli>(InferenceMetrics::Clock::now()-wall_started).count();
    const auto& phases=output.metrics.mtp;
    if(phases.target_ms+phases.head_ms+phases.rollback_ms>wall_ms-15)
        throw std::runtime_error("MTP phase timing included the paused output consumer");
}

void run_bootstrap_publication() {
    struct BootstrapOps : Ops {
        bool early;
        bool bootstrap_before_publication() const {return early;}
    };
    const auto run_bootstrap=[](bool early,int limit) {
        Model model;Predictor mtp;mtp.bootstrap=true;
        InferenceRequest request;request.chat=false;request.prompt={0,0,0};
        request.sampling.max_tokens=limit;request.sampling.mtp_max_draft_tokens=3;
        InferenceOutput output(request,nullptr,"mtp-bootstrap-publication-test");
        Tensor last_hidden;
        auto generation=speculative_sequence(BootstrapOps{{model,mtp},early},request,output,512,0,Tensor{},&last_hidden);
        std::vector<int64_t> tokens;bool boundary=false;
        while(auto step=generation.next())if(step.value) {
            if(const auto* progress=std::get_if<PrefillProgress>(&*step.value);progress && progress->session_boundary) {
                if(model.cache_pos!=3 || mtp.cache_position()!=2 || output.metrics.saw_token)
                    throw std::runtime_error("bootstrap changed prompt snapshot boundary");
                boundary=true;
            }
            if(const auto* delta=std::get_if<TokenOutput>(&*step.value)) {
                if(tokens.empty() && (!boundary || delta->token_ids.size()!=(early && limit>1?2:1)))
                    throw std::runtime_error("bootstrap publication emitted an incomplete first batch");
                tokens.insert(tokens.end(),delta->token_ids.begin(),delta->token_ids.end());
            }
        }
        if(tokens.size()!=std::size_t(limit) || model.cache_pos!=limit+2 || !Ops::defined(last_hidden))
            throw std::runtime_error("bootstrap publication changed output/cache length");
        return tokens;
    };
    if(run_bootstrap(false,8)!=run_bootstrap(true,8) || run_bootstrap(false,1)!=run_bootstrap(true,1))
        throw std::runtime_error("bootstrap publication changed sampled tokens");
}

void run_target_initialization() {
    struct InitializingOps : Ops {
        int* observations;
        bool take_target_maintenance() { ++*observations;return true; }
    };
    Model model;Predictor mtp;int observations=0;
    InferenceRequest request;request.chat=false;request.prompt={0,0,0};
    request.sampling.max_tokens=32;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"mtp-initialization-test");
    auto generation=speculative_sequence(InitializingOps{{model,mtp},&observations},request,output,512,0,Tensor{},nullptr);
    std::vector<int64_t> tokens;
    while(auto step=generation.next())if(step.value)
        if(const auto* delta=std::get_if<TokenOutput>(&*step.value))
            tokens.insert(tokens.end(),delta->token_ids.begin(),delta->token_ids.end());
    if(tokens!=std::vector<int64_t>(32,0) || model.cache_pos!=34 || observations==0)
        throw std::runtime_error("target initialization changed token/cache boundaries");
    for(double ms:output.metrics.mtp.measured_depth_ms)if(ms!=0)
        throw std::runtime_error("target initialization entered steady MTP timing");
}

void run_streaming_publication(bool accepted) {
    struct StreamingOps : Ops {
        int* predictor_calls;
        bool accepted;
        Head predictor_step(Tensor hidden,Tensor ids,Tensor positions) {
            ++*predictor_calls;return Ops::predictor_step(hidden,ids,positions);
        }
        std::vector<int32_t> greedy_tokens(Tensor logits) {
            return std::vector<int32_t>(size(logits,0),accepted?1:0);
        }
    };
    Model model;Predictor mtp;int calls=0;
    InferenceRequest request;request.chat=false;request.prompt={0,0,0};
    request.sampling.max_tokens=32;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"mtp-publication-test");
    Tensor last_hidden;
    auto generation=speculative_sequence(StreamingOps{{model,mtp},&calls,accepted},
        request,output,512,0,Tensor{},&last_hidden);
    int deltas=0;
    while(auto step=generation.next())if(step.value)
        if(const auto* delta=std::get_if<TokenOutput>(&*step.value);delta && !delta->token_ids.empty()) {
            if(++deltas==2)break;
        }
    // One prompt-priming call and three calls for the first draft. Computing
    // any part of the next speculative chain would delay this publication.
    if(deltas!=2 || calls!=4)
        throw std::runtime_error("completed MTP tokens waited for the next draft");
    output.result.cancelled=true;
    while(auto step=generation.next())if(step.value && std::holds_alternative<TokenOutput>(*step.value))
        throw std::runtime_error("cancelled MTP stream emitted another token batch");
    if(calls!=5 || mtp.cache_position()!=model.cache_pos || !Ops::defined(last_hidden))
        throw std::runtime_error("MTP cancellation failed to commit predictor history");
}

void run_history_batching() {
    struct HistoryOps : Ops {
        std::vector<int64_t>* widths;
        int* history_calls;
        bool gradual_mtp_warmup() const {return true;}
        std::size_t predictor_history_merge_limit() const {return 128;}
        std::size_t predictor_history_flush_threshold() const {return 4;}
        std::size_t predictor_parked_history_flush_threshold() const {return 16;}
        Head predictor_step(Tensor hidden,Tensor ids,Tensor positions) {
            if(size(hidden,1)!=size(ids,1))throw std::runtime_error("merged MTP history lost token alignment");
            widths->push_back(size(ids,1));
            return Ops::predictor_step(hidden,ids,positions);
        }
        void predictor_history_step(Tensor hidden,Tensor ids,Tensor positions) {
            if(size(hidden,1)!=size(ids,1))throw std::runtime_error("headless history lost token alignment");
            ++*history_calls;
            for(int64_t offset=0;offset<size(ids,1);offset+=4) {
                const auto rows=std::min<int64_t>(4,size(ids,1)-offset);
                widths->push_back(rows);
                (void)Ops::predictor_step(slice(hidden,1,offset,rows),slice(ids,1,offset,rows),positions);
            }
        }
        std::vector<int32_t> greedy_tokens(Tensor logits) {return std::vector<int32_t>(size(logits,0),1);}
    };
    Model model;Predictor mtp;std::vector<int64_t> widths;int history_calls=0;
    InferenceRequest request;request.chat=false;request.prompt={0,0,0};
    request.sampling.max_tokens=96;request.sampling.mtp_max_draft_tokens=3;
    InferenceOutput output(request,nullptr,"mtp-history-test");
    auto generation=speculative_sequence(HistoryOps{{model,mtp},&widths,&history_calls},request,output,512,0,Tensor{},nullptr);
    std::vector<int64_t> tokens;
    while(auto step=generation.next())if(step.value)
        if(const auto* delta=std::get_if<TokenOutput>(&*step.value))
            tokens.insert(tokens.end(),delta->token_ids.begin(),delta->token_ids.end());
    if(tokens.size()!=96 || tokens.front()!=0 || model.cache_pos!=98 || mtp.cache_position()!=98 ||
            output.metrics.mtp.depth_cycles[0]<3 || history_calls==0 ||
            std::find(widths.begin(),widths.end(),4)==widths.end() ||
            std::any_of(widths.begin(),widths.end(),[](auto n){return n>4;})) {
        std::cerr<<"history tokens="<<tokens.size()<<" model="<<model.cache_pos<<" predictor="<<mtp.cache_position()
            <<" baseline="<<output.metrics.mtp.depth_cycles[0]<<" widths=";
        for(auto n:widths)std::cerr<<n<<',';std::cerr<<std::endl;
        throw std::runtime_error("MTP history batching changed output/cache boundaries or left a re-entry burst");
    }
}

void run_batched_greedy(bool penalties) {
    struct BatchedOps : Ops {
        int* batches;
        bool use_penalties;
        void initialize(const std::vector<int64_t>& prompt, const MfqSamplingParams& sampling) {
            Ops::initialize(prompt, sampling);
            this->penalties = use_penalties;
        }
        std::vector<int32_t> greedy_tokens(Tensor logits) {
            ++*batches;
            return std::vector<int32_t>(size(logits, 0), 0);
        }
    };
    Model model; Predictor mtp; mtp.legacy = true; int batches = 0;
    InferenceRequest request; request.chat = false; request.prompt = {0, 0, 0};
    request.sampling.max_tokens = 32; request.sampling.mtp_max_draft_tokens = 3;
    InferenceOutput output(request, nullptr, "mtp-batched-greedy-test");
    auto generation = speculative_sequence(BatchedOps{{model, mtp}, &batches, penalties},
        request, output, 512, 0, Tensor{}, nullptr);
    std::vector<int64_t> tokens;
    while (auto step = generation.next()) if (step.value)
        if (const auto* delta = std::get_if<TokenOutput>(&*step.value))
            tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
    if (tokens != std::vector<int64_t>(32, 0) || model.cache_pos != 34 ||
        (penalties ? batches != 0 : batches == 0))
        throw std::runtime_error("batched greedy sampling changed rejection/cache/penalty behavior");
}

void run_committed_route_hooks(bool accept_all,bool cancel_during_target=false,bool partial=false,bool retain_prefix=true,
        bool split=false,int depth=3) {
    struct Journal {int pending=0,begins=0,commits=0,cancels=0,rows=0;};
    struct JournalOps : Ops {
        Journal* journal;bool accept_all,cancel_target,partial,retain_prefix;InferenceOutput* output;
        bool retains_partial_target_prefix(int)const{return retain_prefix;}
        void begin_target_routes(int rows) {
            if(journal->pending || rows<1)throw std::runtime_error("route journal nested or entered ordinary prefill");
            journal->pending=rows;++journal->begins;
        }
        void resolve_target_routes(int rows,bool cancelled) {
            if(!journal->pending)return;
            if(rows<0 || rows>journal->pending)throw std::runtime_error("route journal committed an invalid prefix");
            if(cancelled)++journal->cancels;
            else {++journal->commits;journal->rows+=rows;}
            journal->pending=0;
        }
        Tensor target_forward(Tensor ids,Tensor* raw=nullptr,int64_t confirmed=0) {
            if(confirmed && !journal->pending)throw std::runtime_error("route journal missed verification");
            auto result=Ops::target_forward(ids,raw,confirmed);
            if(confirmed && cancel_target)output->result.cancelled=true;
            return result;
        }
        Tensor target_suffix(Tensor ids,Tensor* raw) {
            if(journal->pending!=size(ids,1))throw std::runtime_error("route journal missed split verification");
            auto result=Ops::target_forward(ids,raw);
            if(cancel_target)output->result.cancelled=true;
            return result;
        }
        std::vector<int32_t> greedy_tokens(Tensor logits) {
            std::vector<int32_t> tokens(size(logits,0),accept_all?1:0);
            if(partial)tokens.front()=1;
            return tokens;
        }
    };
    Model model;Predictor mtp;mtp.legacy=true;mtp.split=split;Journal journal;
    InferenceRequest request;request.chat=false;request.prompt={0,0,0};
    request.sampling.max_tokens=9;request.sampling.mtp_max_draft_tokens=depth;
    InferenceOutput output(request,nullptr,"route-journal-hooks");
    auto generation=speculative_sequence(JournalOps{{model,mtp},&journal,accept_all,cancel_during_target,partial,retain_prefix,&output},
        request,output,512,0,Tensor{},nullptr);
    while(generation.next().value){}
    if(journal.pending || !journal.begins || journal.begins!=journal.commits+journal.cancels)
        throw std::runtime_error("MTP route journal was left pending");
    if(cancel_during_target ? journal.cancels!=1 || journal.commits!=0 :
            journal.cancels!=0 || (partial?journal.rows!=((retain_prefix?2:1)-int(split))*journal.commits:
                accept_all?journal.rows<=(1-int(split))*journal.commits:journal.rows!=(1-int(split))*journal.commits))
        throw std::runtime_error("MTP route journal lost rejection/cancellation/truncated output semantics");
    if(!cancel_during_target && model.cache_pos!=int64_t(request.prompt.size())+request.sampling.max_tokens-1)
        throw std::runtime_error("route journal changed the committed model cache boundary");
}
void early_target_prefetch() {
    struct HintOps : Ops {
        std::vector<char>* events;
        static int32_t sample_normal(Tensor,Tensor) {return 0;}
        void prefetch_target_first(int32_t token) {
            if(token!=0)throw std::runtime_error("early target hint received an uncommitted proposal");
            events->push_back('h');
        }
        void prefetch_committed_token(int32_t token) {
            if(token!=0)throw std::runtime_error("PLE publication hint received an uncommitted token");
            events->push_back('b');
        }
        void prefetch_target_prefix(int32_t token,const std::vector<int32_t>& drafts) {
            if(token!=0 || drafts.empty() || drafts.size()>=3 ||
               !std::all_of(drafts.begin(),drafts.end(),[](int32_t value){return value==0;}))
                throw std::runtime_error("PLE draft prefix does not match the upcoming verification");
            events->push_back('c');
        }
        Head predictor_step(Tensor hidden,Tensor ids,Tensor positions) {
            events->push_back('p');
            // Keep the parked-path fixture slower than plain decode with coarse host timers.
            if(!mtp.legacy)std::this_thread::sleep_for(std::chrono::milliseconds(32));
            return Ops::predictor_step(hidden,ids,positions);
        }
        Tensor target_forward(Tensor ids,Tensor* raw=nullptr,int64_t confirmed=0) {
            events->push_back('t');return Ops::target_forward(ids,raw,confirmed);
        }
    };
    for(int count:{1,32,96}) {
        Model model;Predictor mtp;mtp.legacy=count!=96;std::vector<char> events;
        InferenceRequest request;request.chat=false;request.prompt={0,0,0};
        request.sampling.max_tokens=count;request.sampling.mtp_max_draft_tokens=3;
        InferenceOutput output(request,nullptr,"early-target-prefetch");
        auto generation=speculative_sequence(HintOps{Ops{model,mtp},&events},request,output,512,0,Tensor{},nullptr);
        while(auto step=generation.next()) {
            if(step.value && std::get_if<TokenOutput>(&*step.value) && !output.stopped() &&
               (events.empty() || events.back()!='b'))
                throw std::runtime_error("PLE hint was issued after token publication");
        }
        int hints=0,extensions=0,publications=0;
        for(char event:events)if(event=='b')++publications;
        for(std::size_t i=0;i<events.size();++i)if(events[i]=='h' || events[i]=='c') {
            if(events[i]=='h')++hints;else ++extensions;
            if(i+1>=events.size() || events[i+1]!='p')
                throw std::runtime_error("target hint did not precede drafting");
        }
        if((count==1 && (hints || extensions || publications)) ||
           (count>1 && (!hints || !extensions || !publications)) || model.cache_pos!=count+2 ||
           (count==96 && !output.metrics.mtp.standard_tokens))
            throw std::runtime_error("early target hint boundary: count="+std::to_string(count)+
                " hints="+std::to_string(hints)+" extensions="+std::to_string(extensions)+
                " cache="+std::to_string(model.cache_pos)+" events="+std::string(events.begin(),events.end()));
    }
}
void batch_policy() {
    mtp::BatchDepthPolicy policy({"one", "two"}, 1);
    auto now = mtp::BatchDepthPolicy::Clock::time_point{};
    const auto observe = [&](int depth, int milliseconds, bool stable = true, bool interrupted = false) {
        const auto started = now;
        now += std::chrono::milliseconds(milliseconds);
        const std::array<mtp::BatchAcceptance, 2> rejected{{{depth, 0}, {depth, 0}}};
        policy.observe(rejected, started, now, stable, interrupted);
    };
    observe(0, 500, false);
    observe(0, 500, true, true);
    if (policy.cost(0) || policy.depth()) throw std::runtime_error("interrupted batch polluted normal calibration");
    for (int index = 0; index < 5; ++index) observe(0, 10);
    if (policy.depth() != 1 || policy.cost(0) != 10)
        throw std::runtime_error("batch normal warmup/median mismatch");
    policy.interrupt();
    now += std::chrono::seconds(3);
    observe(0, 10);
    if (policy.cost(0) != 10) throw std::runtime_error("prefill gap entered batch timing");
    for (int index = 0; index < 30 && !policy.parks(); ++index)
        observe(policy.depth(), policy.depth() ? 40 : 10);
    if (policy.depth() != 0 || policy.parks() != 1 || policy.remaining() != 128)
        throw std::runtime_error("losing batch MTP did not park");
    mtp::BatchParkMemory memory;
    memory.parked(policy);
    memory.tick();
    mtp::BatchDepthPolicy replacement({"three", "four"}, 1);
    memory.seed(replacement);
    if (replacement.remaining() != 127 || replacement.cooldown() != 256)
        throw std::runtime_error("batch membership reset park countdown");
    mtp::BatchDepthPolicy larger({"one", "two", "three"}, 1);
    memory.seed(larger);
    if (larger.remaining()) throw std::runtime_error("smaller-batch loss parked a larger cohort");
    mtp::BatchDepthPolicy smaller({"one"}, 1);
    memory.seed(smaller);
    if (smaller.remaining()) throw std::runtime_error("larger-batch loss parked a smaller cohort");
    mtp::BatchDepthPolicy mixed({"one", "two"}, 1, 1);
    memory.seed(mixed);
    if (mixed.remaining()) throw std::runtime_error("pure MTP loss parked a mixed cohort");
    mixed.start_parked(64);
    memory.parked(mixed);
    mtp::BatchDepthPolicy pure({"one", "two"}, 1);
    memory.seed(pure);
    if (pure.remaining() != 127) throw std::runtime_error("mixed MTP verdict replaced pure MTP verdict");
    mtp::BatchDepthPolicy deeper({"one", "two"}, 3);
    memory.seed(deeper);
    if (deeper.remaining()) throw std::runtime_error("different draft sweep inherited a park verdict");
    for (int index = 0; index < 127; ++index) memory.tick();
    mtp::BatchDepthPolicy expired({"one", "two"}, 1);
    memory.seed(expired);
    if (expired.remaining()) throw std::runtime_error("batch park verdict did not expire");
    for (int index = 0; index < 128; ++index) observe(0, 10);
    if (policy.remaining() || policy.depth() != 1 || policy.cost(1))
        throw std::runtime_error("batch MTP did not restart a clean depth sweep");
}
void batch_policy_short_response() {
    mtp::BatchDepthPolicy policy({"one"}, 3);
    auto now = mtp::BatchDepthPolicy::Clock::time_point{};
    const std::array<double, 4> costs{22, 42.853896, 62.069792, 74.042291};
    int produced = 0;
    for (int cycle = 0; produced < 128; ++cycle) {
        const int depth = policy.depth();
        const std::array<mtp::BatchAcceptance, 1> accepted{{{depth, cycle == 7 && depth > 1 ? 1 : depth}}};
        const auto started = now;
        now += std::chrono::duration_cast<mtp::BatchDepthPolicy::Clock::duration>(
            std::chrono::duration<double, std::milli>(costs[depth]));
        policy.observe(accepted, started, now, true, false);
        produced += accepted[0].accepted + 1;
        if (policy.parks()) throw std::runtime_error("profitable short high-acceptance MTP was parked");
    }
}
void batch_policy_adaptation() {
    using Clock = mtp::BatchDepthPolicy::Clock;
    const auto advance = [](mtp::BatchDepthPolicy& policy, Clock::time_point& now, int milliseconds) {
        const int depth = policy.depth();
        const auto started = now;
        now += std::chrono::milliseconds(milliseconds);
        const std::array<mtp::BatchAcceptance, 1> full{{{depth, depth}}};
        policy.observe(full, started, now, true, false);
    };
    {
        mtp::BatchDepthPolicy policy({"mtp"}, 3, 7);
        auto now = Clock::time_point{};
        for (int cycle = 0; cycle < 500; ++cycle)
            advance(policy, now, 10 + policy.depth() * 5);
        if (!policy.parks()) throw std::runtime_error("mixed batch ignored ordinary-row throughput loss");
    }
    {
        mtp::BatchDepthPolicy policy({"one"}, 1);
        auto now = Clock::time_point{};
        for (int cycle = 0; cycle < 500 && !policy.parks(); ++cycle)
            advance(policy, now, policy.depth() ? 25 : 10);
        if (!policy.parks()) throw std::runtime_error("unprofitable MTP did not park");
        mtp::BatchParkMemory memory;
        memory.parked(policy);
        for (int cycle = 0; cycle < 10 && policy.remaining(); ++cycle) advance(policy, now, 40);
        if (policy.remaining() || !policy.depth())
            throw std::runtime_error("refreshed profitable MTP was trapped in cooldown");
        memory.resumed(policy);
        mtp::BatchDepthPolicy next({"next"}, 1);
        memory.seed(next);
        if (next.remaining()) throw std::runtime_error("resumed MTP left an obsolete park verdict");
    }
    {
        mtp::BatchDepthPolicy policy({"one"}, 1);
        auto now = Clock::time_point{};
        int plain = 0;
        for (int cycle = 0; cycle < 100; ++cycle) {
            const auto ms = policy.depth() ? 25 : ++plain <= 5 ? 10 : 40;
            advance(policy, now, ms);
            if (policy.parks()) throw std::runtime_error("stale scalar baseline parked profitable MTP");
        }
        if (plain <= 5 || policy.cost(0) != 40)
            throw std::runtime_error("scalar baseline was not revalidated before parking");
    }
}
void batch_policy_observations() {
    using Sample = mtp::BatchAcceptance;
    using Clock = mtp::BatchDepthPolicy::Clock;
    const auto near = [](double actual, double expected) {
        if (std::abs(actual - expected) > 1e-10)
            throw std::runtime_error("batch MTP conditional acceptance is inconsistent");
    };
    {
        mtp::BatchDepthPolicy policy({"one"}, 3);
        const auto start = Clock::time_point{};
        const std::array<Sample, 1> full{{{3, 3}}};
        policy.observe(full, start, start + std::chrono::seconds(10), false, false);
        near(policy.conditional_acceptance(0), 1);
        near(policy.conditional_acceptance(1), 1);
        near(policy.conditional_acceptance(2), 1);
        policy.observe(full, start, start + std::chrono::seconds(20), true, true);
        near(policy.conditional_acceptance(0), 1);
        near(policy.conditional_acceptance(1), 1);
        near(policy.conditional_acceptance(2), 1);
        for (int depth = 0; depth <= 3; ++depth)
            if (policy.cost(depth)) throw std::runtime_error("maintenance entered batch cost calibration");
        const std::array<Sample, 1> invalid{{{1, 2}}};
        bool rejected = false;
        try { policy.observe(invalid, start, start, true, false); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid batch acceptance was admitted");
        near(policy.conditional_acceptance(0), 1);
        mtp::BatchDepthPolicy replacement({"new"}, 3);
        near(replacement.conditional_acceptance(2), .6);
    }
    {
        mtp::BatchDepthPolicy policy({"zero", "one", "two", "three"}, 3);
        const auto start = Clock::time_point{};
        const std::array<Sample, 4> mixed{{{0, 0}, {1, 1}, {3, 1}, {3, 3}}};
        policy.observe(mixed, start, start + std::chrono::seconds(10), true, false);
        near(policy.conditional_acceptance(0), 1);
        near(policy.conditional_acceptance(1), .5);
        near(policy.conditional_acceptance(2), 1);
        for (int depth = 0; depth <= 3; ++depth)
            if (policy.cost(depth)) throw std::runtime_error("mixed draft widths polluted a single-width cost");
    }
    {
        mtp::BatchDepthPolicy policy({"one"}, 3);
        auto now = Clock::time_point{};
        const auto observe = [&](int depth, int milliseconds, int accepted = -1) {
            const auto start = now;
            now += std::chrono::milliseconds(milliseconds);
            const std::array<Sample, 1> full{{{depth, accepted < 0 ? depth : accepted}}};
            policy.observe(full, start, now, true, false);
        };
        for (int index = 0; index < 5; ++index) observe(0, 10);
        for (int index = 0; index < 4; ++index) observe(3, 25);
        if (policy.depth() != 2) throw std::runtime_error("batch depth transition was not exercised");
        const auto previous = policy.conditional_acceptance(2);
        observe(3, 25, 2);
        if (policy.cost(2) || policy.cost(3) != 25 || policy.conditional_acceptance(2) >= previous)
            throw std::runtime_error("queued old-depth round was attributed to the newly selected depth");
    }
    {
        mtp::BatchDepthPolicy policy({"one"}, 1);
        const auto start = Clock::time_point{};
        const std::array<Sample, 1> rejected{{{1, 0}}}, accepted{{{1, 1}}};
        policy.observe(rejected, start, start, false, false);
        near(policy.conditional_acceptance(0), 0);
        policy.observe(accepted, start, start, false, false);
        near(policy.conditional_acceptance(0), 1 / 1.92);
    }
}
}

void run_prepared_default_positions() {
    for(bool positioned:{false,true}) {
        Model model;Predictor predictor;
        InferenceRequest request;request.prompt={1,1,1};request.sampling.max_tokens=8;
        Prepared prepared;prepared.token_ids=request.prompt;prepared.embeddings={{1,3,1}};
        if(positioned)prepared.positions={{3}};
        int calls=0;
        struct ImageOps:Ops {
            bool positioned=false;int* calls=nullptr;
            Head predictor_step(Tensor hidden,Tensor ids,Tensor positions) {
                assert(defined(positions)==positioned);++*calls;
                return Ops::predictor_step(hidden,ids,positions);
            }
        };
        ImageOps ops{{model,predictor},positioned,&calls};ops.prepared=&prepared;
        InferenceOutput output(request,nullptr,"prepared-default-positions");
        auto generation=speculative_sequence(std::move(ops),request,output,2,0,Tensor{},nullptr);
        while(generation.next()){}
        assert(output.result.completion_tokens==8 && calls>0);
    }
    Model model;Predictor predictor;InferenceRequest request;request.prompt={1,1,1};
    Prepared bad;bad.token_ids=request.prompt;bad.embeddings={{1,3,1}};bad.decode_position_delta=1;
    Ops ops{model,predictor};ops.prepared=&bad;
    InferenceOutput output(request,nullptr,"missing-shifted-positions");
    auto generation=speculative_sequence(ops,request,output,2,0,Tensor{},nullptr);
    bool rejected=false;
    try{(void)generation.next();}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);
}

void run_restored_blockwise_context() {
    for(int reused:{0,4})for(bool tail_only:{false,true}) {
        Model model;model.cache_pos=reused;
        Predictor predictor;predictor.blockwise=true;predictor.position=reused;
        InferenceRequest request;request.prompt=std::vector<int64_t>(10,1);
        request.sampling.max_tokens=8;request.stop_token_ids={0};
        struct TailOps:Ops {
            bool tail_only=false;
            Prefill prefill(PrefillChunk chunk) {
                auto result=Ops::prefill(chunk);
                if(tail_only)result.raw.shape[1]=2;
                return result;
            }
        };
        InferenceOutput output(request,nullptr,"restored-blockwise-context");
        auto generation=speculative_sequence(TailOps{{model,predictor},tail_only},request,output,
            16,reused,Tensor{{1,1,1}},nullptr);
        while(generation.next()){}
        assert(output.result.completion_tokens==1 && predictor.position==10);
        const auto expected_start=tail_only?8:reused;
        assert(predictor.context_appends.size()==1 && predictor.context_appends.front().first==expected_start);
    }
    Model model;model.cache_pos=4;Predictor predictor;predictor.blockwise=true;predictor.position=3;
    InferenceRequest request;request.prompt=std::vector<int64_t>(10,1);
    InferenceOutput output(request,nullptr,"invalid-blockwise-context");
    auto generation=speculative_sequence(Ops{model,predictor},request,output,16,4,Tensor{{1,1,1}},nullptr);
    bool rejected=false;try{(void)generation.next();}catch(const std::runtime_error&){rejected=true;}
    assert(rejected && predictor.context_appends.empty() && model.cache_pos==4);
}

void run_blockwise_prompt_checkpoint() {
    for(int reused:{0,4})for(int stable:{5,9})for(int chunk:{3,64})for(bool tail_only:{false,true}) {
        if(tail_only && chunk<64)continue;
        Model model;model.cache_pos=reused;
        Predictor predictor;predictor.blockwise=true;predictor.position=reused;
        InferenceRequest request;request.prompt=std::vector<int64_t>(10,1);
        request.sampling.max_tokens=8;request.stop_token_ids={0};
        struct CheckpointOps:Ops {
            bool tail_only;
            std::vector<PrefillChunk>* chunks;
            int* primes;
            double blockwise_prompt_context(Tensor context,int64_t position,bool seed) {
                ++*primes;
                if(seed)mtp.seed_target_context(context,position);
                else mtp.append_target_context(context,position-size(context,1));
                return 3.0;
            }
            Prefill prefill(PrefillChunk chunk) {
                chunks->push_back(chunk);
                auto result=Ops::prefill(chunk);
                if(tail_only)result.raw.shape[1]=std::min<int64_t>(2,chunk.count);
                return result;
            }
        };
        InferenceOutput output(request,nullptr,"blockwise-prompt-checkpoint");
        std::vector<PrefillChunk> chunks;std::vector<int64_t> checkpoints;Tensor hidden;int primes=0;
        auto generation=speculative_sequence(CheckpointOps{{model,predictor},tail_only,&chunks,&primes},
            request,output,chunk,reused,Tensor{{1,1,1}},&hidden,stable);
        while(auto step=generation.next())if(step.value)
            if(const auto* progress=std::get_if<PrefillProgress>(&*step.value);progress && progress->session_boundary) {
                if(output.metrics.saw_token || predictor.cache_position()!=model.cache_pos ||
                        !Ops::defined(hidden) || Ops::size(hidden,1)!=1)
                    throw std::runtime_error("blockwise prompt checkpoint lost its committed predictor boundary");
                checkpoints.push_back(model.cache_pos);
                if(primes!=int(checkpoints.size()) || progress->timing.model_ms<primes*3.0+1.0)
                    throw std::runtime_error("blockwise predictor prompt preparation was omitted from PP timing");
            }
        if(checkpoints!=std::vector<int64_t>{stable,10} || output.result.completion_tokens!=1 ||
                model.cache_pos!=10 || predictor.cache_position()!=10)
            throw std::runtime_error("stable MTP prompt checkpoint changed the target/predictor history");
        int64_t position=reused;
        for(const auto& part:chunks) {
            if(part.offset!=position || part.count<=0 || part.last!=(position+part.count==10))
                throw std::runtime_error("MTP checkpoint duplicated prefill rows or changed the final-chunk flag");
            position+=part.count;
        }
        if(position!=10)throw std::runtime_error("MTP checkpoint omitted a prompt suffix");
    }
}
void run_blockwise_final_context() {
    for(bool accepted:{false,true})for(bool eos:{false,true})for(int limit:{2,3,4,7,8}) {
        struct BlockOps:Ops {
            bool accepted;
            std::vector<int32_t> greedy_tokens(Tensor logits) {
                return std::vector<int32_t>(size(logits,0),accepted?1:0);
            }
        };
        Model model;Predictor predictor;predictor.blockwise=true;
        InferenceRequest request;request.prompt={0,0,0};request.sampling.max_tokens=limit;
        if(eos)request.stop_token_ids={1};
        InferenceOutput output(request,nullptr,"blockwise-final-context");
        Tensor hidden;
        auto generation=speculative_sequence(BlockOps{{model,predictor},accepted},request,output,
            16,0,Tensor{},&hidden);
        while(generation.next()){}
        const int expected=eos && accepted?2:limit;
        if(output.result.completion_tokens!=expected || predictor.cache_position()!=model.cache_pos ||
                !Ops::defined(hidden))
            throw std::runtime_error("block predictor final snapshot lost committed target context");
        int64_t position=0;
        for(const auto& [start,rows]:predictor.context_appends) {
            if(start!=position || rows<=0)throw std::runtime_error("block predictor duplicated or skipped final history");
            position+=rows;
        }
        if(position!=model.cache_pos)throw std::runtime_error("block predictor final history is incomplete");
    }
}

void run_continuous_speculative_publication() {
    for (bool cancelled : {false, true}) {
        InferenceRequest input;
        input.prompt = {0};
        input.sampling.max_tokens = 4;
        input.stop_token_ids = {2};
        InferenceOutput output(input, nullptr, "batched-publication");
        struct Row {
            int produced = 0;
            int64_t pending_token = 0;
            std::vector<int64_t> accepted_tokens;
        } row;
        bool scheduled = false;
        int retained = -1, publications = 0, initial_accepts = 0;
        auto generation = batched_speculative_sequence(row, output, 1,
            [&] { scheduled = true; }, [&] { return scheduled; }, [] {},
            [&](auto& destination) {
                ++publications;
                auto delta = destination.append(std::vector<int64_t>{1, 2, 0});
                retained = std::min<int>(2, delta.token_ids.size());
                return delta;
            }, [&](const TokenOutput& delta) { initial_accepts += delta.token_ids.size(); });
        auto first = generation.next();
        assert(first && row.produced == 1 && row.pending_token == 1 && !scheduled && initial_accepts == 1);
        assert((row.accepted_tokens == std::vector<int64_t>{1}));
        auto scheduling = generation.next();
        assert(scheduling && scheduled && publications == 0);
        if (cancelled) output.result.cancelled = true;
        while (generation.next()) {}
        assert(publications == (cancelled ? 0 : 1));
        if (!cancelled) {
            assert(row.produced == 3 && row.pending_token == 2 && retained == 2);
            assert((row.accepted_tokens == std::vector<int64_t>{1, 1, 2}));
            assert(output.result.finish_reason == "stop");
        } else {
            assert(row.produced == 1 && retained == -1);
            assert((row.accepted_tokens == std::vector<int64_t>{1}));
        }
    }
}

int main() try {
    run_continuous_speculative_publication();
    for (const bool dspark : {false, true}) {
        Model model;
        model.speculative_start = -1;
        Predictor predictor;
        predictor.legacy = dspark;
        InferenceRequest request;
        request.prompt = {1, 1, 1};
        request.sampling.max_tokens = 64;
        request.stop_token_ids = {0};
        InferenceOutput output(request, nullptr, "mtp-raw-eos");
        auto generation = speculative_sequence(Ops{model, predictor}, request, output, 512, 0, Tensor{}, nullptr);
        std::vector<int64_t> tokens;
        while (auto step = generation.next())
            if (step.value)
                if (const auto* delta = std::get_if<TokenOutput>(&*step.value))
                    tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
        assert((tokens == std::vector<int64_t>{0}));
        assert(output.result.finish_reason == "stop" && output.result.completion_tokens == 1);
        assert(model.speculative_start < 0 && model.cache_pos == 3);
    }
    run_prepared_default_positions();
    run_restored_blockwise_context();
    run_blockwise_prompt_checkpoint();
    run_blockwise_final_context();
    run(false);
    run(false,true);
    run(false,true,true);
    run(true);
    run_dspark_draft_cost();
    for(bool streaming:{false,true})for(int reused:{0,4})run_chunked_prompt_history(streaming,reused);
    run_prompt_snapshot_boundary();
    run_bootstrap_publication();
    run_target_initialization();
    run_streaming_publication(false);
    run_streaming_publication(true);
    run_history_batching();
    run_batched_greedy(false);
    run_batched_greedy(true);
    run_committed_route_hooks(false);
    run_committed_route_hooks(true);
    run_committed_route_hooks(false,true);
    run_committed_route_hooks(false,false,true);
    run_committed_route_hooks(false,false,true,false);
    run_committed_route_hooks(false,false,false,true,true);
    run_committed_route_hooks(true,false,false,true,true);
    run_committed_route_hooks(false,true,false,true,true);
    run_committed_route_hooks(false,false,true,true,true);
    run_committed_route_hooks(false,false,true,false,true);
    run_committed_route_hooks(false,false,false,true,true,1);
    run_committed_route_hooks(true,false,false,true,true,1);
    early_target_prefetch();
    batch_policy();
    batch_policy_short_response();
    batch_policy_adaptation();
    batch_policy_observations();
    std::cout << "shared speculative sequence output/cache/dispatch tests passed\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
