#pragma once
template<int BD>__global__ void l2_norm_pair_kernel(const float* first,const float* second,
        float* out_first,float* out_second,int first_rows,int width,float eps) {
    const bool next=blockIdx.x>=unsigned(first_rows);
    const int row=next?blockIdx.x-first_rows:blockIdx.x;
    const auto* source=(next?second:first)+size_t(row)*width;
    auto* output=(next?out_second:out_first)+size_t(row)*width;
    float sum=0;
    for(int c=threadIdx.x;c<width;c+=BD) {const float x=source[c];sum+=x*x;}
    sum=block_sum<BD/32>(sum);
    const float inverse=1.f/fmaxf(sqrtf(sum),eps);
    for(int c=threadIdx.x;c<width;c+=BD)output[c]=source[c]*inverse;
}
std::vector<mfq_tensor_backend::Tensor> l2_norm_pair_cuda(mfq_tensor_backend::Tensor first,
        mfq_tensor_backend::Tensor second,double eps) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(first.is_cuda() && second.device()==first.device() &&
        first.is_contiguous() && second.is_contiguous() && first.scalar_type()==tb::kFloat32 &&
        second.scalar_type()==tb::kFloat32 && first.size(-1)==second.size(-1),
        "l2_norm_pair: expected contiguous FP32 GPU tensors with matching widths");
    auto a=tb::empty_like(first),b=tb::empty_like(second);
    const int width=int(first.size(-1)),n=int(first.numel()/width),m=int(second.numel()/width);
    l2_norm_pair_kernel<NORM_BD><<<n+m,NORM_BD,0,mfq_current_cuda_stream()>>>(
        first.data_ptr<float>(),second.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),n,width,float(eps));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return {a,b};
}
