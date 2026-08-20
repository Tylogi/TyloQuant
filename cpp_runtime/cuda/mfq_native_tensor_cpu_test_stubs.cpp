#include "mfq_native_tensor.h"

#include <stdexcept>

namespace mfq::cuda {

Tensor empty_cuda(std::span<const std::int64_t>, const TensorOptions&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

Tensor empty_pinned(std::span<const std::int64_t>, const TensorOptions&) {
    throw std::runtime_error("pinned memory is unavailable in the host-only tensor test");
}

Tensor copy_or_convert_cuda(const Tensor&, const TensorOptions&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

void fill_cuda(Tensor&, double) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

void copy_cuda(Tensor&, const Tensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

Tensor make_contiguous_cuda(const Tensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

Tensor mean_cuda(const Tensor&, std::int64_t, bool) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

#define MFQ_HOST_CUDA_STUB(name, signature) \
    Tensor name signature { \
        throw std::runtime_error("CUDA is unavailable in the host-only tensor test"); \
    }

MFQ_HOST_CUDA_STUB(reduce_cuda,
    (const Tensor&, std::int64_t, bool, int))
MFQ_HOST_CUDA_STUB(unary_cuda,
    (const Tensor&, int, double, double))
MFQ_HOST_CUDA_STUB(binary_cuda,
    (const Tensor&, const Tensor&, int))
MFQ_HOST_CUDA_STUB(scalar_binary_cuda,
    (const Tensor&, double, int, bool))
MFQ_HOST_CUDA_STUB(index_select_cuda,
    (const Tensor&, std::int64_t, const Tensor&))
MFQ_HOST_CUDA_STUB(gather_cuda,
    (const Tensor&, std::int64_t, const Tensor&))
MFQ_HOST_CUDA_STUB(repeat_cuda,
    (const Tensor&, std::span<const std::int64_t>))
MFQ_HOST_CUDA_STUB(repeat_interleave_cuda,
    (const Tensor&, std::int64_t, std::int64_t))
MFQ_HOST_CUDA_STUB(masked_select_cuda,
    (const Tensor&, const Tensor&))

#undef MFQ_HOST_CUDA_STUB

void scatter_cuda(Tensor&, std::int64_t, const Tensor&, const Tensor&, bool) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
void index_copy_cuda(Tensor&, std::int64_t, const Tensor&, const Tensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
void index_fill_cuda(Tensor&, std::int64_t, const Tensor&, double) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
void masked_fill_cuda(Tensor&, const Tensor&, double) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

bool equal(const Tensor&, const Tensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
Tensor where(const Tensor&, const Tensor&, double) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
Tensor arange(std::int64_t, const TensorOptions&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
Tensor operator<=(const Tensor&, const Tensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
Tensor operator+(const Tensor&, const Tensor&) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}
Tensor operator+(const Tensor&, double) {
    throw std::runtime_error("CUDA is unavailable in the host-only tensor test");
}

}  // namespace mfq::cuda
