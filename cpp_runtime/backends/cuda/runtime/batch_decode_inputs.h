#pragma once

#include "mfq_tensor_backend.h"

namespace mfq::cuda {

struct BatchDecodeRow {
    int64_t token = 0, position = 0, slot = -1;
};

// Stable device inputs. Null state pointers and disabled rows never commit writes.
// State entries contain a base address and the extent of their sequence axis.
class BatchDecodeInputs {
public:
    using Tensor = mfq_tensor_backend::Tensor;
    BatchDecodeInputs(int64_t rows, int64_t fields, int device);
    void update_rows(const std::vector<BatchDecodeRow>&);
    void row(int64_t index, int64_t token, int64_t position, int64_t slot = -1, bool enabled = true);
    void state(int64_t row, int64_t field, const Tensor& value, int64_t axis = -1);
    void validate_position(int64_t row, int64_t maximum) const;
    void upload();
    void copy_from(const BatchDecodeInputs&);
    int64_t rows() const { return rows_; }
    Tensor ids() const { return metadata_.narrow(0, 0, 1).reshape({rows_, 1}); }
    Tensor positions() const { return metadata_.narrow(0, 1, 1).reshape({rows_}); }
    Tensor lengths() const { return metadata_.narrow(0, 2, 1).reshape({rows_}); }
    Tensor enabled() const { return metadata_.narrow(0, 3, 1).reshape({rows_}); }
    Tensor slots() const { return metadata_.narrow(0, 4, 1).reshape({rows_}); }
    const Tensor& states() const { return states_; }
private:
    int64_t rows_, fields_;
    Tensor host_metadata_, metadata_, host_states_, states_;
};

// Indirect transfers are also used by unfused recurrent operators. Only their
// small working state is staged; request KV stays in its original allocation.
void batch_state_load(const BatchDecodeInputs&, int64_t field, int64_t row,
                      const mfq_tensor_backend::Tensor& output);
void batch_state_store(const BatchDecodeInputs&, int64_t field, int64_t row,
                       const mfq_tensor_backend::Tensor& input);
mfq_tensor_backend::Tensor batch_state_gather(const BatchDecodeInputs&, int64_t field, int64_t row,
    const mfq_tensor_backend::Tensor& indices, int64_t heads, int64_t width,
    const mfq_tensor_backend::TensorOptions& options, int64_t hot_field = -1, bool values = false);
void batch_state_scatter(const BatchDecodeInputs&, int64_t field, int64_t row,
    const mfq_tensor_backend::Tensor& indices, const mfq_tensor_backend::Tensor& values);
void mask_decode_routes(const mfq_tensor_backend::Tensor& routes,
    const mfq_tensor_backend::Tensor& weights, const mfq_tensor_backend::Tensor& enabled, int64_t offset);

} // namespace mfq::cuda
