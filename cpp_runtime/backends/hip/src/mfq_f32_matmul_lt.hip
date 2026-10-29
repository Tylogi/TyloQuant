#include "mfq_native_tensor.h"
#include "mfq_hip_context.h"
#include <array>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#ifdef MFQ_HAVE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif

namespace mfq::cuda {
#ifdef MFQ_HAVE_HIPBLASLT
namespace {
void check_lt(hipblasStatus_t status) {
    if(status!=HIPBLAS_STATUS_SUCCESS)
        throw Error("hipBLASLt status "+std::to_string(int(status)));
}
struct LtPlan {
    hipblasLtMatmulDesc_t operation{};
    hipblasLtMatrixLayout_t a{},b{},c{};
    hipblasLtMatmulHeuristicResult_t algorithm{};
    bool valid=false;
    ~LtPlan() {
        if(operation)(void)hipblasLtMatmulDescDestroy(operation);
        if(a)(void)hipblasLtMatrixLayoutDestroy(a);
        if(b)(void)hipblasLtMatrixLayoutDestroy(b);
        if(c)(void)hipblasLtMatrixLayoutDestroy(c);
    }
};
struct LtDevice {
    hipblasLtHandle_t handle{};
    std::map<std::array<int,8>,std::unique_ptr<LtPlan>> plans;
    std::map<hipStream_t,Tensor> workspaces;
    LtDevice(){check_lt(hipblasLtCreate(&handle));}
    ~LtDevice(){if(handle)(void)hipblasLtDestroy(handle);}
};
}
#endif

bool f32_matmul_lt(const float* a,const float* b,float* c,int m,int n,int k,
    bool transpose_a,bool transpose_b,int lda,int ldb,int ldc,hipStream_t stream) {
#ifdef MFQ_HAVE_HIPBLASLT
    static const bool enabled=[] {const char* x=std::getenv("MFQ_HIP_F32_LT");return !x || x[0]!='0';}();
    if(!enabled || m<16 || n<32 || k<64 ||
        ((reinterpret_cast<uintptr_t>(a)|reinterpret_cast<uintptr_t>(b)|reinterpret_cast<uintptr_t>(c))&255))return false;
    int device=0;MFQ_NATIVE_CUDA_CHECK(hipGetDevice(&device));
    static thread_local std::map<int,std::unique_ptr<LtDevice>> devices;
    const std::array<int,8> key{m,n,k,lda,ldb,ldc,int(transpose_a),int(transpose_b)};
    auto di=devices.find(device);
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    MFQ_NATIVE_CUDA_CHECK(hipStreamIsCapturing(stream,&capture));
    if(di==devices.end()) {
        if(capture!=hipStreamCaptureStatusNone)return false;
        di=devices.emplace(device,std::make_unique<LtDevice>()).first;
    }
    auto& state=*di->second;
    auto found=state.plans.find(key);
    constexpr size_t workspace_bytes=32*1024*1024;
    if(found==state.plans.end()) {
        if(capture!=hipStreamCaptureStatusNone)return false;
        auto plan=std::make_unique<LtPlan>();
        check_lt(hipblasLtMatmulDescCreate(&plan->operation,HIPBLAS_COMPUTE_32F,HIP_R_32F));
        const hipblasOperation_t op_a=transpose_a?HIPBLAS_OP_T:HIPBLAS_OP_N;
        const hipblasOperation_t op_b=transpose_b?HIPBLAS_OP_T:HIPBLAS_OP_N;
        check_lt(hipblasLtMatmulDescSetAttribute(plan->operation,HIPBLASLT_MATMUL_DESC_TRANSA,&op_a,sizeof(op_a)));
        check_lt(hipblasLtMatmulDescSetAttribute(plan->operation,HIPBLASLT_MATMUL_DESC_TRANSB,&op_b,sizeof(op_b)));
        check_lt(hipblasLtMatrixLayoutCreate(&plan->a,HIP_R_32F,transpose_a?k:m,transpose_a?m:k,lda));
        check_lt(hipblasLtMatrixLayoutCreate(&plan->b,HIP_R_32F,transpose_b?n:k,transpose_b?k:n,ldb));
        check_lt(hipblasLtMatrixLayoutCreate(&plan->c,HIP_R_32F,m,n,ldc));
        hipblasLtMatmulPreference_t preference{};
        check_lt(hipblasLtMatmulPreferenceCreate(&preference));
        const auto preference_status=hipblasLtMatmulPreferenceSetAttribute(preference,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes));
        if(preference_status!=HIPBLAS_STATUS_SUCCESS) {
            (void)hipblasLtMatmulPreferenceDestroy(preference);check_lt(preference_status);
        }
        int count=0;
        const auto status=hipblasLtMatmulAlgoGetHeuristic(state.handle,plan->operation,
            plan->a,plan->b,plan->c,plan->c,preference,1,&plan->algorithm,&count);
        (void)hipblasLtMatmulPreferenceDestroy(preference);
        plan->valid=status==HIPBLAS_STATUS_SUCCESS && count>0 &&
            plan->algorithm.state==HIPBLAS_STATUS_SUCCESS && plan->algorithm.workspaceSize<=workspace_bytes;
        static const bool trace=[] {const char* x=std::getenv("MFQ_HIP_F32_LT_TRACE");return x && x[0]=='1';}();
        if(trace)std::fprintf(stderr,"hip_f32_lt m=%d n=%d k=%d selected=%d workspace=%zu\n",m,n,k,int(plan->valid),plan->algorithm.workspaceSize);
        found=state.plans.emplace(key,std::move(plan)).first;
    }
    auto& plan=*found->second;if(!plan.valid)return false;
    void* workspace=nullptr;
    if(plan.algorithm.workspaceSize) {
        auto allocation=state.workspaces.find(stream);
        if(allocation==state.workspaces.end()) {
            if(capture!=hipStreamCaptureStatusNone)return false;
            // Keep one stable workspace per stream, including all captured plans.
            auto storage=empty({int64_t(workspace_bytes)},TensorOptions().device(Device{kCUDA,device}).dtype(kUInt8));
            allocation=state.workspaces.emplace(stream,std::move(storage)).first;
        }
        workspace=allocation->second.data_ptr();
    }
    const float alpha=1.f,beta=0.f;
    const auto status=hipblasLtMatmul(state.handle,plan.operation,&alpha,a,plan.a,b,plan.b,
        &beta,c,plan.c,c,plan.c,&plan.algorithm.algo,workspace,plan.algorithm.workspaceSize,stream);
    if(status==HIPBLAS_STATUS_NOT_SUPPORTED || status==HIPBLAS_STATUS_INVALID_VALUE)return false;
    check_lt(status);return true;
#else
    (void)a;(void)b;(void)c;(void)m;(void)n;(void)k;(void)transpose_a;(void)transpose_b;
    (void)lda;(void)ldb;(void)ldc;(void)stream;return false;
#endif
}
}
