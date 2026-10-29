#pragma once
#include "cuda_execution.h"
#include "mfq/tq_rows.h"
#include "mfq/kernels/cuda/tq_ops.h"
#include "runtime/row_stage.h"

class TqGemmResources;
struct TqWeight {
    // Keep execution streams alive until image storage has fenced its users.
    std::shared_ptr<TqGemmResources> gemm;
    mfq_tensor_backend::Tensor image;
    TrellisProjectionView view{};
    std::int64_t out=0,neuron_len=0;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor)const;
    mfq_tensor_backend::Tensor forward_groupwise(mfq_tensor_backend::Tensor,std::int64_t groups)const;
    mfq_tensor_backend::Tensor embedding(mfq_tensor_backend::Tensor)const;
};
TqWeight to_device_tq(const std::vector<std::uint8_t>&,std::int64_t input_columns,int device,bool cuda=true);
TqWeight load_tq_weight(const mfq::ModelSource&,const std::string&,int device,bool cuda=true);
bool is_tq_row_table(const mfq::ModelSource&,const std::string&);

// Device row plan only; compressed weights retain their original storage.
class TqProjectionGroup {
    std::vector<TqWeight> weights_;
    std::vector<TrellisProjectionJob> host_plan_;
    mfq_tensor_backend::Tensor plan_;
    std::int64_t input_=0,output_=0;
    std::size_t joined_prefix_=1;
    std::int64_t joined_rows_=0;
    int blocks_=0;
public:
    TqProjectionGroup()=default;
    explicit TqProjectionGroup(std::vector<TqWeight>);
    void join_prefix(std::size_t count);
    bool defined()const noexcept{return !weights_.empty();}
    std::vector<mfq_tensor_backend::Tensor> forward(mfq_tensor_backend::Tensor)const;
    mfq_tensor_backend::Tensor swiglu(mfq_tensor_backend::Tensor)const;
};

// Table rows remain in their source. Only requested rows enter device memory.
class TqRowTable {
    std::vector<std::shared_ptr<mfq::TqRows>> segments_;
    std::vector<std::uint64_t> ends_;
    std::int64_t columns_=0;
    std::uint64_t storage_bytes_=0;
public:
    TqRowTable(const mfq::ModelSource&,const std::string&);
    std::uint64_t rows()const noexcept{return ends_.back();}
    std::int64_t width()const noexcept{return columns_;}
    std::uint64_t storage_bytes()const noexcept{return storage_bytes_;}
    std::size_t row_image_bytes_bound()const;
    std::vector<std::uint8_t> slice_rows(std::uint64_t begin,std::uint64_t end)const;
    mfq_tensor_backend::Tensor lookup(const std::int64_t*,std::size_t,const mfq_tensor_backend::Device&)const;
    mfq_tensor_backend::Tensor lookup(mfq_tensor_backend::Tensor)const;
};
mfq_tensor_backend::Tensor lookup_tq_shards(const std::vector<std::shared_ptr<TqRowTable>>&,
    std::int64_t rows_per_shard,const std::vector<std::int64_t>& ids,
    const mfq_tensor_backend::Device&);
std::shared_ptr<RowStage> make_tq_row_stage(const std::vector<std::shared_ptr<TqRowTable>>&,
    std::int64_t rows_per_shard,const std::vector<std::int64_t>& shape,
    const mfq_tensor_backend::Device&);

struct TqRoutedWeight {
    std::shared_ptr<TqRowTable> table;
    std::shared_ptr<TqExpertCache> cache;
    std::uint64_t source_id=0;
    std::int64_t experts=0,out=0,input=0;
    TqRoutedWeight(CudaExecutionContext&,const mfq::ModelSource&,const std::string&,
                   std::int64_t experts,std::int64_t out,std::int64_t input);
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor)const;
};
std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(CudaExecutionContext&,
    const mfq::ModelSource&,const std::string&);
bool prepare_tq_graph_pool(const std::shared_ptr<TqExpertCache>&);
bool tq_graph_supports_tokens(std::int64_t tokens)noexcept;
std::size_t tq_expert_cache_budget_bytes(const std::shared_ptr<TqExpertCache>&);
void resize_tq_expert_cache(const std::shared_ptr<TqExpertCache>&,std::size_t bytes);
std::size_t reclaim_tq_expert_host(const std::shared_ptr<TqExpertCache>&,std::size_t bytes);
std::size_t reserve_tq_expert_host(const std::shared_ptr<TqExpertCache>&,std::size_t allocation_bytes);
std::vector<std::pair<std::string,double>> tq_expert_memory_metrics(const std::shared_ptr<TqExpertCache>&);

// Three ordinary [experts, rows, columns] TQ tensors. Expert ID selects
// a row range; every row retains its own bit width, scale and encoded data.
struct TqMoeBanks {
    std::shared_ptr<TqRoutedWeight> gate,up,down;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor weights,float swiglu_limit=0,
        CudaProfiler* profiler=nullptr)const;
};
