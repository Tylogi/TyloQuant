#pragma once
#include "mfq_tensor_backend.h"
#include <cstdint>
#include <vector>

// A prepared table-row transfer. Encodings stay in their own loader; graph
// consumers use stable output storage and never allocate inside upload/decode.
class RowStage {
public:
    virtual ~RowStage()=default;
    virtual mfq_tensor_backend::Tensor output()const=0;
    virtual bool mapped_rows()const=0;
    // Return true only when the host packet is reusable and its device-side
    // consumers remain ordered before the next upload on the current stream.
    virtual bool wait_upload_reuse(){return false;}
    virtual void upload(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape)=0;
    virtual void decode()=0;
};

#include "nint_row_pipeline.h"
class NintStage final:public RowStage {
    std::shared_ptr<NintRowPipeline> pipeline_;
    std::shared_ptr<NintRowStage> stage_;
public:
    NintStage(std::shared_ptr<NintRowPipeline> pipeline,const std::vector<int64_t>& shape,
        const mfq_tensor_backend::Device& device):pipeline_(std::move(pipeline)),stage_(pipeline_->make_stage(shape,device)){}
    mfq_tensor_backend::Tensor output()const override{return stage_->output();}
    bool mapped_rows()const override{return false;}
    void upload(const std::vector<int64_t>& ids,const std::vector<int64_t>& shape)override{pipeline_->upload(*stage_,ids,shape);}
    void decode()override{stage_->decode();}
};
