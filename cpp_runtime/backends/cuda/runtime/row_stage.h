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
    virtual void upload(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape)=0;
    virtual void decode()=0;
};
