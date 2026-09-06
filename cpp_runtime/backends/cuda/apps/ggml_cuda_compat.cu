#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

// MFQ's custom kernels use the integrated GGML CUDA helper headers without
// linking the complete ggml-cuda backend. Provide the host symbols they need.
void ggml_cuda_error(
    const char * statement,
    const char * function,
    const char * file,
    int line,
    const char * message) {
    std::fprintf(
        stderr,
        "CUDA error: %s\n  statement: %s\n  function: %s\n  at %s:%d\n",
        message != nullptr ? message : "unknown",
        statement != nullptr ? statement : "unknown",
        function != nullptr ? function : "unknown",
        file != nullptr ? file : "unknown",
        line);
    std::abort();
}

int ggml_cuda_get_device() {
    int device = 0;
    const cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        ggml_cuda_error(
            "cudaGetDevice(&device)",
            __func__,
            __FILE__,
            __LINE__,
            cudaGetErrorString(status));
    }
    return device;
}
