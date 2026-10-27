#pragma once
#include "mfq_tensor_backend.h"
#include "mfq/model_source.h"
#include "tq_projection.h"
#include "mfq/tq_rows.h"
#include "runtime/row_stage.h"

struct TqWeight {
    mfq_tensor_backend::Tensor image;
    ProjectionView view{};
    std::int64_t out=0, neuron_len=0;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor input) const;
    mfq_tensor_backend::Tensor embedding(mfq_tensor_backend::Tensor)const;
    mfq_tensor_backend::Tensor materialize(mfq_tensor_backend::ScalarType type) const;
};

TqWeight load_tq_weight(const mfq::ModelSource&,const std::string&,int device);
TqWeight to_device_tq(const std::vector<std::uint8_t>&,std::int64_t columns,int device);
bool tq_q8_enabled();
std::vector<mfq_tensor_backend::Tensor> tq_forward_group(
    const std::vector<const TqWeight*>&,mfq_tensor_backend::Tensor);
// GDN auxiliaries retain BF16 activation/output rounding, promoted to FP32.
std::vector<mfq_tensor_backend::Tensor> tq_forward_group_bf16(
    const std::vector<const TqWeight*>&,mfq_tensor_backend::Tensor,
    const mfq_tensor_backend::Tensor&,const mfq_tensor_backend::Tensor&);
mfq_tensor_backend::Tensor tq_swiglu(
    const TqWeight&,const TqWeight&,mfq_tensor_backend::Tensor);
mfq_tensor_backend::Tensor tq_forward_residual(
    const TqWeight&,mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor);

mfq_tensor_backend::Tensor tq_gr_activation(const TqWeight&,mfq_tensor_backend::Tensor,int64_t,bool);
mfq_tensor_backend::Tensor tq_gr_mix(const TqWeight&,mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor,int64_t);
mfq_tensor_backend::Tensor dense_gr_injection_cuda(
    const mfq_tensor_backend::Tensor&,const mfq_tensor_backend::Tensor&,int64_t);
std::vector<mfq_tensor_backend::Tensor> tq_gr_down_injection_cuda(ProjectionView,
    const mfq_tensor_backend::Tensor&,const mfq_tensor_backend::Tensor&,int64_t);
mfq_tensor_backend::Tensor tq_shared_swiglu_cuda(ProjectionView,ProjectionView,
    const mfq_tensor_backend::Tensor&);
std::vector<mfq_tensor_backend::Tensor> tq_router_gate_cuda(ProjectionView,
    const mfq_tensor_backend::Tensor&,const mfq_tensor_backend::Tensor&);
std::vector<mfq_tensor_backend::Tensor> tq_moe_inputs_cuda(ProjectionView,
    const mfq_tensor_backend::Tensor&,ProjectionView,ProjectionView,const mfq_tensor_backend::Tensor&);
std::vector<mfq_tensor_backend::Tensor> tq_moe_inputs_batch_cuda(ProjectionView,
    const mfq_tensor_backend::Tensor&,ProjectionView,ProjectionView,const mfq_tensor_backend::Tensor&);
mfq_tensor_backend::Tensor tq_shared_add_cuda(const mfq_tensor_backend::Tensor&,
        const mfq_tensor_backend::Tensor&,const mfq_tensor_backend::Tensor&);
mfq_tensor_backend::Tensor tq_shared_finish_cuda(ProjectionView,
        const mfq_tensor_backend::Tensor&,const mfq_tensor_backend::Tensor&,
        const mfq_tensor_backend::Tensor&,const mfq_tensor_backend::Tensor&);

// The same three ordinary TQ banks used by CUDA. Compressed rows retain their
// standard layout and independent bit widths, without an MFE wrapper.
struct CudaExecutionContext;
struct TqRoutedWeight {
    struct Shard { int device,begin,end; std::shared_ptr<TqRoutedWeight> weight; };
    TqWeight weight;
    std::int64_t experts=0,out=0,input=0;
    std::vector<Shard> shards;
    CudaExecutionContext* execution=nullptr;
    TqRoutedWeight()=default;
    TqRoutedWeight(const mfq::ModelSource&,const std::string&,int device,
        std::int64_t experts,std::int64_t out,std::int64_t input,
        std::int64_t begin=0,std::int64_t end=-1);
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor)const;
};
struct TqMoePeer;
struct TqMoeBanks {
    std::shared_ptr<TqRoutedWeight> gate,up,down;
    mutable std::shared_ptr<TqMoePeer> peer;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor,
        mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor,float limit=0.f,
        const TqWeight* shared_down=nullptr,mfq_tensor_backend::Tensor shared_hidden={},
        mfq_tensor_backend::Tensor shared_coefficient={})const;
};

// Table rows remain in their source. Only requested rows enter device memory.
class TqRowTable {
    std::vector<std::shared_ptr<mfq::TqRows>> segments_;
    std::vector<std::uint64_t> ends_;
    std::int64_t columns_=0;
    std::uint64_t storage_bytes_=0;
    bool parallel_reads_=true;
public:
    TqRowTable(const mfq::ModelSource&,const std::string&);
    std::uint64_t rows()const noexcept{return ends_.back();}
    std::int64_t width()const noexcept{return columns_;}
    std::uint64_t storage_bytes()const noexcept{return storage_bytes_;}
    bool parallel_reads()const noexcept{return parallel_reads_;}
    std::size_t row_image_bytes_bound()const;
    ProjectionView stage_row(std::uint64_t,std::uint8_t*,std::uint8_t*,std::size_t)const;
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


bool is_tq_row_table(const mfq::ModelSource&,const std::string&);

std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(const mfq::ModelSource&,const std::string&,int,
    CudaExecutionContext* execution=nullptr);
