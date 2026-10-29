#pragma once

#include "core/causal_model.h"
#include "core/block.h"
#include "models/qwen4_exp/causal_lm.h"
#include "models/qwen4_exp/config.h"
#include "quant_linear.h"

#include <memory>
#include <functional>
#include <iosfwd>
#include <vector>

namespace mfq::cuda::qwen4_exp { struct Gr; struct Qwen4Block; }
struct MtpModule;

namespace mfq::cuda {
struct Qwen4DecodeGraph;
class Qwen4BatchDecodeGraph;
class Qwen4BatchRequestState;
// Recurrent checkpoints for batched verification, sized once. Its slot count
// bounds the draft tokens one window may verify, whatever the row count.
class Qwen4BatchCheckpointPool {
  public:
    explicit Qwen4BatchCheckpointPool(int64_t tokens);
    int64_t tokens() const noexcept { return int64_t(busy_.size()); }
    int64_t available() const noexcept;
    std::vector<int64_t> acquire(int64_t count);
    void release(const std::vector<int64_t>& slots) noexcept;
  private:
    void allocate(Qwen4BatchRequestState&);
    std::vector<bool> busy_;
    // One [tokens, ...] tensor per layer state; undefined where a layer has none.
    std::vector<mfq_tensor_backend::Tensor> gdn_conv_, gdn_state_, ple_conv_;
    friend mfq_tensor_backend::Tensor qwen4_batch_decode(
        Qwen4CausalLm&, const std::vector<Qwen4BatchRequestState*>&,
        mfq_tensor_backend::Tensor, const mfq_tensor_backend::Tensor&,
        mfq_tensor_backend::Tensor*, const std::vector<int>&, bool,
        const std::shared_ptr<Qwen4BatchCheckpointPool>&);
};
// A request owns recurrent, positional and attention state; weights stay shared.
class Qwen4BatchRequestState {
  public:
    explicit Qwen4BatchRequestState(Qwen4CausalLm&);
    ~Qwen4BatchRequestState();
    Qwen4BatchRequestState(const Qwen4BatchRequestState&) = delete;
    Qwen4BatchRequestState& operator=(const Qwen4BatchRequestState&) = delete;
    mfq_tensor_backend::Tensor prefill(Qwen4CausalLm&, mfq_tensor_backend::Tensor,
        mfq_tensor_backend::Tensor* raw = nullptr);
    int64_t position() const;
    std::vector<mfq_tensor_backend::Tensor> recurrent_state_tensors() const;
    std::size_t resident_bytes() const;
    std::size_t prefill_growth_bytes(int64_t tokens) const;
    void resolve_verified(int64_t retained_tokens);
  private:
    struct Impl;
    std::unique_ptr<Impl> state_;
    friend class Qwen4BatchDecodeGraph;
    friend std::vector<mfq_tensor_backend::Tensor> qwen4_batch_prefill(
        Qwen4CausalLm&,const std::vector<Qwen4BatchRequestState*>&,
        const std::vector<mfq_tensor_backend::Tensor>&);
    friend mfq_tensor_backend::Tensor qwen4_batch_decode(
        Qwen4CausalLm&, const std::vector<Qwen4BatchRequestState*>&,
        mfq_tensor_backend::Tensor, const mfq_tensor_backend::Tensor&,
        mfq_tensor_backend::Tensor*, const std::vector<int>&, bool,
        const std::shared_ptr<Qwen4BatchCheckpointPool>&);
    friend class Qwen4BatchCheckpointPool;
};
std::vector<mfq_tensor_backend::Tensor> qwen4_batch_prefill(
    Qwen4CausalLm&,const std::vector<Qwen4BatchRequestState*>&,
    const std::vector<mfq_tensor_backend::Tensor>&);
// Speculative verification takes recurrent checkpoints from `checkpoints`
// when given; the caller keeps the window's draft tokens within its slots.
mfq_tensor_backend::Tensor qwen4_batch_decode(
    Qwen4CausalLm&, const std::vector<Qwen4BatchRequestState*>&,
    mfq_tensor_backend::Tensor ids, const mfq_tensor_backend::Tensor& host_ids,
    mfq_tensor_backend::Tensor* raw = nullptr, const std::vector<int>& widths = {}, bool speculative = false,
    const std::shared_ptr<Qwen4BatchCheckpointPool>& checkpoints = {});
class Qwen4BatchDecodeGraph {
  public:
    explicit Qwen4BatchDecodeGraph(Qwen4CausalLm&);
    ~Qwen4BatchDecodeGraph();
    std::optional<mfq_tensor_backend::Tensor> run(
        const std::vector<Qwen4BatchRequestState*>&,
        const mfq_tensor_backend::Tensor&,const mfq_tensor_backend::Tensor&,bool greedy=false,
        int64_t maximum_rows=32);
    void clear();
    bool has_graphs() const;
    int64_t captures() const;
    int64_t replays() const;
    int64_t membership_captures() const;
    int64_t bucket_reuses() const;
    double capture_ms() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> graph_;
};
void qwen4_begin_router_lookahead_audit(Qwen4CausalLm&,int samples);
void qwen4_finish_router_lookahead_audit(Qwen4CausalLm&,const std::string& prefix);
void qwen4_set_rotary_fusion(Qwen4CausalLm&, bool enabled);
void qwen4_set_attention_grouping(Qwen4CausalLm&, bool enabled);
bool qwen4_prefetch_decode_ple(Qwen4CausalLm&,const mfq_tensor_backend::Tensor& host_ids);
std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_logits(
    Qwen4CausalLm&,mfq_tensor_backend::Tensor,int,const mfq_tensor_backend::Tensor& host_ids={});
std::optional<mfq_tensor_backend::Tensor> qwen4_decode_graph_hidden(
    Qwen4CausalLm&,mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor* raw_hidden,int64_t confirmed=0,
    bool* initialized=nullptr,const mfq_tensor_backend::Tensor& host_ids={},
    mfq_tensor_backend::Tensor* logits=nullptr,std::vector<int32_t>* greedy_tokens=nullptr,bool compact_sampling=true);
void qwen4_report_decode_graph(const Qwen4CausalLm&,std::ostream&);
std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(const Qwen4CausalLm&,const MtpModule*);
mfq_tensor_backend::Tensor qwen4_decode_graph_compact_topk(const Qwen4CausalLm&);
void qwen4_release_decode_graphs(Qwen4CausalLm&,MtpModule*);
void qwen4_warmup_decode_graphs(Qwen4CausalLm&,MtpModule*,int maximum_draft_depth=3);
std::size_t qwen4_mtp_graph_reserve(const Qwen4CausalLm&,const MtpModule*,int maximum_draft_depth=5);
std::size_t qwen4_prefix_snapshot_bytes(const Qwen4CausalLm&,const MtpModule*);
std::vector<std::pair<std::string,double>> qwen4_cache_memory_metrics(const Qwen4CausalLm&,const MtpModule*);
bool qwen4_layer_major_prefill_eligible(const Qwen4CausalLm&, int64_t tokens);
mfq_tensor_backend::Tensor qwen4_layer_major_prefill_hidden(
    Qwen4CausalLm&, mfq_tensor_backend::Tensor ids, int64_t compute_chunk = 1024,
    bool last_only = true, const std::function<bool()>& cancelled = {},
    mfq_tensor_backend::Tensor* raw_hidden = nullptr, bool raw_hidden_host = false);
// Report completed prompt computation before restoring lent expert storage.
mfq_tensor_backend::Tensor qwen4_layer_major_prefill_hidden(
    Qwen4CausalLm&, mfq_tensor_backend::Tensor ids, int64_t compute_chunk,
    bool last_only, const std::function<bool()>& cancelled,
    mfq_tensor_backend::Tensor* raw_hidden, bool raw_hidden_host,
    const std::function<void()>& computation_finished);

struct Qwen4Model : CausalResources {
    bool adapter_prefetch_token_rows(Qwen4CausalLm&,const mfq_tensor_backend::Tensor&);
    Qwen4Model();
    ~Qwen4Model();
    Qwen4Model(Qwen4Model&&) noexcept;
    Qwen4Model& operator=(Qwen4Model&&) noexcept;
    template <class Backend> using CausalModel = mfq::models::qwen4_exp::CausalLm<Backend>;
    mfq::models::qwen4_exp::Config config;
    std::unique_ptr<qwen4_exp::Gr> final_mixer;
    mfq_tensor_backend::Tensor positions;
    int64_t batch = 0;
    std::optional<bool> router_lookahead;
    std::unique_ptr<Qwen4DecodeGraph> graph_registry;
    bool reuse_decode_graphs=false;
    uint64_t decode_graph_captures=0;
    // Bounded request lookahead, independent of the configured context ceiling.
    int64_t decode_reserve_until=0;
    bool speculative_prefix_retention=false;
    std::vector<qwen4_exp::Qwen4Block*> loaded_blocks;
    std::function<void(std::size_t)> reserve_kv_workspace;
    template<class Model> std::optional<mfq_tensor_backend::Tensor>
    adapter_graph_logits(Model& model,mfq_tensor_backend::Tensor ids,int kind) {
        if (ids.dim()==2 && ids.size(0)==1 &&
            qwen4_layer_major_prefill_eligible(model,ids.size(1))) {
            auto hidden=qwen4_layer_major_prefill_hidden(model,std::move(ids),1024,kind!=0);
            if(kind==2)return model.adapter_next_token(model.lm_head,model.last_hidden(std::move(hidden)));
            if(kind==1)return model.apply_final_logit_softcap(
                model.adapter_last_logits(model.lm_head,model.last_hidden(std::move(hidden))));
            return model.logits_from_hidden(std::move(hidden));
        }
        return qwen4_decode_graph_logits(model,std::move(ids),kind);
    }

    void adapter_validate_load_options() const;
    void adapter_load_final_state(const mfq::ModelSource &source,
                                  mfq_tensor_backend::Tensor &output_norm);
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);

    void adapter_reset(int64_t batch);
    void adapter_begin_forward(bool capture_raw_hidden);

    mfq_tensor_backend::Tensor
    adapter_block_positions(const mfq_tensor_backend::Tensor &full_positions,
                            const mfq_tensor_backend::Tensor &local_positions, int device) const;
    void adapter_finish_forward(const mfq_tensor_backend::Tensor &full_positions, int64_t batch,
                                int64_t tokens);
    mfq_tensor_backend::Tensor
    adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                            const mfq_tensor_backend::Tensor &output_norm, int64_t batch,
                            int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_last_logits(const QuantLinear &lm_head,
                                                   mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(const QuantLinear &lm_head,
                                                  mfq_tensor_backend::Tensor hidden) const;
    void adapter_rollback_speculative(int64_t keep);
    void adapter_commit_speculative() { speculative_prefix_retention=false; }
};

template <> struct CudaSessionCodec<Qwen4Model> {
    static TextSessionStateKind kind(const Qwen4CausalLm&);
    static bool supports_paged(const Qwen4CausalLm&);
    static TextSessionState capture(const Qwen4CausalLm&,const std::vector<int64_t>&);
    static void restore(Qwen4CausalLm&,const TextSessionState&);
};

} // namespace mfq::cuda

namespace mfq::models {
extern template struct qwen4_exp::CausalLm<cuda::CudaCausalOps<cuda::Qwen4Model>>;
} // namespace mfq::models
