#include "mlx_multimodal.h"

#include <stdexcept>

namespace mfq::metal {

mlx::core::array replace_multimodal_embedding_span(
    const mlx::core::array& embeddings,
    const mlx::core::array& replacement,
    int batch,
    int begin,
    int end) {
    if (embeddings.ndim() != 3 || replacement.ndim() != 3 ||
        batch < 0 || batch >= embeddings.shape(0) || begin < 0 ||
        end <= begin || end > embeddings.shape(1) ||
        replacement.shape(0) != 1 ||
        replacement.shape(1) != end - begin ||
        replacement.shape(2) != embeddings.shape(2)) {
        throw std::invalid_argument(
            "multimodal embedding replacement span is incompatible");
    }
    auto value = replacement;
    if (value.dtype() != embeddings.dtype()) {
        value = mlx::core::astype(value, embeddings.dtype());
    }
    return mlx::core::slice_update(
        embeddings,
        value,
        mlx::core::Shape{batch, begin, 0},
        mlx::core::Shape{
            batch + 1, end, embeddings.shape(2)});
}

} // namespace mfq::metal
