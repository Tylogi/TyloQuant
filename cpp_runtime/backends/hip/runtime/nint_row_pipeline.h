#pragma once
#include "mfq_tensor_backend.h"
#include "mfq/nint_rows.h"
#include <memory>
namespace mfq {class NintRowCache;}
class NintRowStage {
public:
    ~NintRowStage();
    mfq_tensor_backend::Tensor output() const;
    bool mapped_rows() const;
    void decode();
private:
    friend class NintRowPipeline;
    struct Impl;
    explicit NintRowStage(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
class NintRowPipeline {
public:
    explicit NintRowPipeline(std::vector<std::shared_ptr<mfq::NintRows>>,
        std::shared_ptr<mfq::NintRowCache> cache={});
    ~NintRowPipeline();
    void issue(const std::vector<int64_t>& ids,const std::vector<int64_t>& shape,bool early_prefix=false);
    mfq_tensor_backend::Tensor collect(const std::vector<int64_t>& ids,const std::vector<int64_t>& shape,
        const mfq_tensor_backend::Device& device);
    std::shared_ptr<NintRowStage> make_stage(const std::vector<int64_t>& shape,
        const mfq_tensor_backend::Device& device);
    void upload(NintRowStage&,const std::vector<int64_t>& ids,const std::vector<int64_t>& shape);
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
