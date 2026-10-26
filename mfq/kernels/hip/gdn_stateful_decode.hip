#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include "mfq_tensor_backend.h"
#include "mfq_hip_compat.h"
#include "reduce.cuh"
#include "gdn_stateful_decode.h"

template<int tiles,class Qkv,class Gate,class Output,int FixedWidth=0,int FixedTaps=0,bool InplaceConv=false,int BroadcastNorm=0,bool Snapshot=false>
__global__ void gdn_stateful_decode_kernel(const Qkv* qkv,const Gate* output_gate,
    const Gate* alpha,const Gate* beta,const float* convolution_state,const float* recurrent_state,
    const float* convolution_weight,const float* dt_bias,const float* a_log,const float* norm_weight,
    Output* output,float* convolution_out,float* recurrent_out,int key_heads,int value_heads,
    int width,int taps,int layout,float convolution_eps,double norm_eps,bool silu_gate,
    bool transposed_state,int64_t as0,int64_t as2,int64_t bs0,int64_t bs2,
    float* workspace,const int64_t* state_addresses,int64_t qkv_stride,int64_t gate_stride,bool warp_reductions,float* checkpoint_convolution,float* checkpoint_recurrent) {
    if constexpr(FixedWidth>0)width=FixedWidth;
    if constexpr(FixedTaps>0)taps=FixedTaps;
    const int bh=blockIdx.x/tiles,tile=blockIdx.x%tiles;
    const int batch=bh/value_heads,head=bh%value_heads;
    const int state_batch=state_addresses?0:batch;
    if(state_addresses) {
        convolution_state=reinterpret_cast<const float*>(state_addresses[3*batch]);
        recurrent_state=reinterpret_cast<const float*>(state_addresses[3*batch+1]);
        recurrent_out=reinterpret_cast<float*>(state_addresses[3*batch+1]);
        workspace=reinterpret_cast<float*>(state_addresses[3*batch+2]);
    }
    const int workspace_head=state_addresses?head:bh;
    const int key_head=head/(value_heads/key_heads),tid=threadIdx.x;
    const int qk_width=2*key_heads*width,channels=qk_width+value_heads*width;
    __shared__ float query[128],key[128],value[128],attended[128];
    __shared__ float qsum[256],ksum[256],norm_sum[128],decay_value,beta_value;
    __shared__ bool final_tile,final_key_group;
    const auto store_conv=[&](size_t index,float value) {
        convolution_out[index]=value;
        if constexpr(Snapshot)checkpoint_convolution[index]=value;
    };
    float q=0,k=0,v=0;
    if(tid<width) {
        const int qc=key_head*width+tid,kc=key_heads*width+qc,vc=qk_width+head*width+tid;
        const size_t state_base=size_t(batch)*(taps-1)*channels;
        const size_t input_base=size_t(state_batch)*(taps-1)*channels;
        for(int tap=0;tap<taps-1;++tap) {
            const size_t wq=layout==0?size_t(qc)*taps+tap:size_t(tap)*channels+qc;
            const size_t wk=layout==0?size_t(kc)*taps+tap:size_t(tap)*channels+kc;
            const size_t wv=layout==0?size_t(vc)*taps+tap:size_t(tap)*channels+vc;
            q+=convolution_state[input_base+size_t(tap)*channels+qc]*convolution_weight[wq];
            k+=convolution_state[input_base+size_t(tap)*channels+kc]*convolution_weight[wk];
            v+=convolution_state[input_base+size_t(tap)*channels+vc]*convolution_weight[wv];
        }
        const float q_current=static_cast<float>(qkv[size_t(batch)*qkv_stride+qc]);
        const float k_current=static_cast<float>(qkv[size_t(batch)*qkv_stride+kc]);
        const float v_current=static_cast<float>(qkv[size_t(batch)*qkv_stride+vc]);
        const size_t wq=layout==0?size_t(qc)*taps+taps-1:size_t(taps-1)*channels+qc;
        const size_t wk=layout==0?size_t(kc)*taps+taps-1:size_t(taps-1)*channels+kc;
        const size_t wv=layout==0?size_t(vc)*taps+taps-1:size_t(taps-1)*channels+vc;
        q+=q_current*convolution_weight[wq];
        k+=k_current*convolution_weight[wk];
        v+=v_current*convolution_weight[wv];
        q=q/(1.f+expf(-q));k=k/(1.f+expf(-k));v=v/(1.f+expf(-v));
        query[tid]=q;key[tid]=k;value[tid]=v;
        if constexpr(!InplaceConv) {
        for(int tap=0;tap<taps-2;++tap) {
            if(tile==0 && head%(value_heads/key_heads)==0) {
                store_conv(state_base+size_t(tap)*channels+qc,convolution_state[input_base+size_t(tap+1)*channels+qc]);
                store_conv(state_base+size_t(tap)*channels+kc,convolution_state[input_base+size_t(tap+1)*channels+kc]);
            }
            if(tile==0)store_conv(state_base+size_t(tap)*channels+vc,convolution_state[input_base+size_t(tap+1)*channels+vc]);
        }
        if(tile==0 && head%(value_heads/key_heads)==0) {
            store_conv(state_base+size_t(taps-2)*channels+qc,q_current);
            store_conv(state_base+size_t(taps-2)*channels+kc,k_current);
        }
        if(tile==0)store_conv(state_base+size_t(taps-2)*channels+vc,v_current);
        }
    }
    if(tid<256) {qsum[tid]=tid<width?q*q:0.f;ksum[tid]=tid<width?k*k:0.f;}
    if(tid==0) {
        const float a=static_cast<float>(alpha[int64_t(batch)*as0+int64_t(head)*as2])+dt_bias[head];
        const float sp=fmaxf(a,0.f)+log1pf(expf(-fabsf(a)));
        const float gate_log=sp*-expf(a_log[head]);
        decay_value=expf(gate_log);
        const float raw_beta=static_cast<float>(beta[int64_t(batch)*bs0+int64_t(head)*bs2]);
        beta_value=1.f/(1.f+expf(-raw_beta));
    }
    __syncthreads();
    if(!warp_reductions || width!=128)for(int stride=128;stride>=32;stride>>=1) {
        if(tid<stride) {qsum[tid]+=qsum[tid+stride];ksum[tid]+=ksum[tid+stride];}
        __syncthreads();
    }
    if(tid<32) {
        float qr=qsum[tid],kr=ksum[tid];
        if(warp_reductions && width==128) {
            // Reproduce the original 128,64,32 pair tree in registers.
            qr=(qr+qsum[tid+128])+(qsum[tid+64]+qsum[tid+192]);
            kr=(kr+ksum[tid+128])+(ksum[tid+64]+ksum[tid+192]);
            qr+=(qsum[tid+32]+qsum[tid+160])+(qsum[tid+96]+qsum[tid+224]);
            kr+=(ksum[tid+32]+ksum[tid+160])+(ksum[tid+96]+ksum[tid+224]);
        }
        #pragma unroll
        for(int stride=16;stride>0;stride>>=1) {
            const float qother=__shfl_down_sync(0xffffffff,qr,stride);
            const float kother=__shfl_down_sync(0xffffffff,kr,stride);
            if(tid<stride) {qr+=qother;kr+=kother;}
        }
        if(tid==0) {
            if constexpr(BroadcastNorm==1) {
                qsum[0]=1.f/fmaxf(sqrtf(qr),convolution_eps);
                ksum[0]=1.f/fmaxf(sqrtf(kr),convolution_eps);
            }else {qsum[0]=qr;ksum[0]=kr;}
        }
    }
    __syncthreads();
    if(tid<width) {
        if constexpr(BroadcastNorm==1) {query[tid]*=qsum[0];key[tid]*=ksum[0];}
        else if constexpr(BroadcastNorm==2 || BroadcastNorm==4) {
            float qi=0.f,ki=0.f;
            if((tid&31)==0) {
                qi=1.f/fmaxf(sqrtf(qsum[0]),convolution_eps);
                ki=1.f/fmaxf(sqrtf(ksum[0]),convolution_eps);
            }
            query[tid]*=__shfl_sync(0xffffffffu,qi,0);
            key[tid]*=__shfl_sync(0xffffffffu,ki,0);
        }
        else {
            query[tid]*=1.f/fmaxf(sqrtf(qsum[0]),convolution_eps);
            key[tid]*=1.f/fmaxf(sqrtf(ksum[0]),convolution_eps);
        }
    }
    __syncthreads();
    const int lane=tid&31,warp=tid>>5,warps=blockDim.x>>5;
    const int rows=(width+31)/32;
    const size_t state_head=(size_t(state_batch)*value_heads+head)*width*width;
    const float decay=decay_value,bt=beta_value,scale=1.f/sqrtf(float(width));
    const int columns_per_tile=(width+tiles-1)/tiles;
    const int column_end=min(width,(tile+1)*columns_per_tile);
    for(int column=tile*columns_per_tile+warp;column<column_end;column+=warps) {
        float states[4],keys[4],queries[4];
        #pragma unroll
        for(int row=0;row<4;++row) if(row<rows) {
            const int i=row*32+lane;
            const size_t index=transposed_state?size_t(column)*width+i:size_t(i)*width+column;
            states[row]=recurrent_state[state_head+index];
            keys[row]=key[i];queries[row]=query[i];
        }
        float projected=0.f;
        #pragma unroll
        for(int row=0;row<4;++row)if(row<rows)projected+=states[row]*keys[row];
        projected=warp_sum(projected);
        const float delta=(value[column]-decay*projected)*bt;
        float result=0.f;
        #pragma unroll
        for(int row=0;row<4;++row) if(row<rows) {
            const int i=row*32+lane;
            const size_t index=transposed_state?size_t(column)*width+i:size_t(i)*width+column;
            // Preserve the original warp-column kernel's contracted operand
            // order, including its one-row and multiple-row distinction.
            states[row]=rows==1
                ? __fmaf_rn(keys[row],delta,__fmul_rn(decay,states[row]))
                : __fmaf_rn(decay,states[row],__fmul_rn(keys[row],delta));
            result+=states[row]*queries[row];
            recurrent_out[state_head+index]=states[row];
            if constexpr(Snapshot)checkpoint_recurrent[state_head+index]=states[row];
        }
        result=warp_sum(result);
        if(lane==0) {
            const float computed=result*scale;
            if(tiles==1)attended[column]=computed;
            else workspace[size_t(workspace_head)*(width+1)+column]=computed;
        }
    }
    if(tiles>1) {
        // Only warp leaders publish attended columns into the cross-tile
        // workspace. Recurrent state is consumed after kernel completion;
        // the block barrier also completes every convolution-history read
        // before the completion counter can enable an in-place commit.
        if constexpr(BroadcastNorm==4) {
            if(lane==0)__threadfence();
        }else __threadfence();
        __syncthreads();
        if(tid==0) {
            auto* counter=reinterpret_cast<uint32_t*>(workspace+size_t(workspace_head)*(width+1)+width);
            const auto previous=__atomic_fetch_add(counter,1u,__ATOMIC_ACQ_REL);
            final_tile=(InplaceConv?(previous&255u):previous)==unsigned(tiles-1);
        }
        __syncthreads();
        if(!final_tile)return;
        if(tid<width)attended[tid]=workspace[size_t(workspace_head)*(width+1)+tid];
        if constexpr(InplaceConv)if(tid==0) {
            // The existing tile publication already follows every history
            // read. Publish group readiness before the output epilogue.
            const int group_size=value_heads/key_heads,first_head=key_head*group_size;
            auto* counter=reinterpret_cast<uint32_t*>(workspace+
                (size_t(batch)*value_heads+first_head)*(width+1)+width);
            const uint32_t bit=1u<<(8+head-first_head),mask=((1u<<group_size)-1u)<<8;
            const uint32_t previous=__atomic_fetch_or(counter,bit,__ATOMIC_ACQ_REL);
            final_key_group=((previous|bit)&mask)==mask;
        }
    }
    __syncthreads();
    if(tid<width)norm_sum[tid]=__fadd_rn(0.f,__fmul_rn(attended[tid],attended[tid]));
    __syncthreads();
    if(!warp_reductions || width!=128)for(int stride=width/2;stride>=32;stride>>=1) {
        if(tid<stride)norm_sum[tid]=__fadd_rn(norm_sum[tid],norm_sum[tid+stride]);
        __syncthreads();
    }
    if(tid<32) {
        float partial=norm_sum[tid];
        if(warp_reductions && width==128)partial=__fadd_rn(
            __fadd_rn(partial,norm_sum[tid+64]),__fadd_rn(norm_sum[tid+32],norm_sum[tid+96]));
        #pragma unroll
        for(int stride=16;stride>0;stride>>=1) {
            const float other=__shfl_down_sync(0xffffffff,partial,stride);
            if(tid<stride)partial=__fadd_rn(partial,other);
        }
        if(tid==0) {
            if constexpr(BroadcastNorm==1) {
                const float mean=__fdiv_rn(partial,float(width));
                const float variance=float(double(mean)+norm_eps);
                norm_sum[0]=__fdiv_rn(1.f,__fsqrt_rn(variance));
            }else norm_sum[0]=partial;
        }
    }
    __syncthreads();
    if constexpr(InplaceConv)if(tid>=width && tid<2*width) {
        // Threads unused by output normalization commit history in parallel.
        const int column=tid-width,vc=qk_width+head*width+column;
        const size_t base=size_t(batch)*(taps-1)*channels;
        for(int tap=0;tap<taps-2;++tap)
            store_conv(base+size_t(tap)*channels+vc,convolution_state[base+size_t(tap+1)*channels+vc]);
        store_conv(base+size_t(taps-2)*channels+vc,static_cast<float>(qkv[size_t(batch)*qkv_stride+vc]));
        if(final_key_group) {
            const int qc=key_head*width+column,kc=key_heads*width+qc;
            for(int tap=0;tap<taps-2;++tap) {
                store_conv(base+size_t(tap)*channels+qc,convolution_state[base+size_t(tap+1)*channels+qc]);
                store_conv(base+size_t(tap)*channels+kc,convolution_state[base+size_t(tap+1)*channels+kc]);
            }
            store_conv(base+size_t(taps-2)*channels+qc,static_cast<float>(qkv[size_t(batch)*qkv_stride+qc]));
            store_conv(base+size_t(taps-2)*channels+kc,static_cast<float>(qkv[size_t(batch)*qkv_stride+kc]));
        }
    }
    if(tid<width) {
        float inverse;
        if constexpr(BroadcastNorm==1)inverse=norm_sum[0];
        else if constexpr(BroadcastNorm>=2) {
            inverse=0.f;
            if((tid&31)==0) {
                const float mean=__fdiv_rn(norm_sum[0],float(width));
                const float variance=float(double(mean)+norm_eps);
                inverse=__fdiv_rn(1.f,__fsqrt_rn(variance));
            }
            inverse=__shfl_sync(0xffffffffu,inverse,0);
        }
        else {
            const float mean=__fdiv_rn(norm_sum[0],float(width));
            const float variance=float(double(mean)+norm_eps);
            inverse=__fdiv_rn(1.f,__fsqrt_rn(variance));
        }
        const size_t index=(size_t(batch)*value_heads+head)*width+tid;
        const float z=static_cast<float>(output_gate[size_t(batch)*gate_stride+size_t(head)*width+tid]);
        const float sigmoid=__fdiv_rn(1.f,__fadd_rn(1.f,expf(-z)));
        const float gate=silu_gate?__fmul_rn(z,sigmoid):sigmoid;
        const float scaled=__fmul_rn(__fmul_rn(attended[tid],inverse),norm_weight[tid]);
        output[index]=static_cast<Output>(__fmul_rn(scaled,gate));
    }
    if constexpr(InplaceConv) {
        // A subsequent replay starts only after this kernel completes. All
        // group heads have published readiness, so no counters remain in use.
        const int group_size=value_heads/key_heads,first_head=key_head*group_size;
        if(final_key_group && tid<group_size) {
                auto* counter=reinterpret_cast<uint32_t*>(workspace+
                    (size_t(batch)*value_heads+first_head+tid)*(width+1)+width);
                    __atomic_store_n(counter,0u,__ATOMIC_RELEASE);
        }
    }else if(tiles>1) {
        __syncthreads();
        if(tid==0) {
            auto* counter=reinterpret_cast<uint32_t*>(workspace+size_t(workspace_head)*(width+1)+width);
            __atomic_store_n(counter,0u,__ATOMIC_RELEASE);
        }
    }
}

__global__ void gdn_stateful_commit_convolution(const float* values,
    const int64_t* state_addresses,int64_t row_elements) {
    const int row=blockIdx.y;
    const int64_t column=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(column<row_elements)
        reinterpret_cast<float*>(state_addresses[3*row])[column]=values[int64_t(row)*row_elements+column];
}


std::vector<mfq_tensor_backend::Tensor> mfq_hip_gdn_stateful::decode(
    mfq_tensor_backend::Tensor qkv,mfq_tensor_backend::Tensor output_gate,
    mfq_tensor_backend::Tensor alpha,mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor convolution_state,mfq_tensor_backend::Tensor recurrent_state,
    mfq_tensor_backend::Tensor convolution_weight,mfq_tensor_backend::Tensor dt_bias,
    mfq_tensor_backend::Tensor a_log,mfq_tensor_backend::Tensor norm_weight,
    int64_t key_heads,int64_t value_heads,int64_t width,
    double convolution_eps,double norm_eps,bool silu_gate,bool output_half,
    bool transposed_state,mfq_tensor_backend::Tensor workspace,bool inplace_state,
    mfq_tensor_backend::Tensor state_addresses,
    mfq_tensor_backend::Tensor checkpoint_convolution,mfq_tensor_backend::Tensor checkpoint_recurrent) {
    constexpr int threads=256;
    const auto* tile_option=std::getenv("MFQ_HIP_GDN_STATE_TILES");
    const int tiles=tile_option?std::atoi(tile_option):1;
    MFQ_RUNTIME_CHECK(tiles==1 || tiles==2 || tiles==4,"HIP GDN state tiles must be 1, 2 or 4");
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(width==32 || width==64 || width==128,"GDN decode core width unsupported");
    MFQ_RUNTIME_CHECK(threads==256 || threads==512 || threads==1024,"GDN decode core block width unsupported");
    MFQ_RUNTIME_CHECK(qkv.dim()==3 && qkv.size(0)>0 && key_heads>0 && value_heads>0 &&
        value_heads%key_heads==0 && key_heads<=std::numeric_limits<int>::max()/width &&
        value_heads<=std::numeric_limits<int>::max()/width,
        "GDN decode core heads/batch disagree");
    const int64_t batch=qkv.size(0),channels=(2*key_heads+value_heads)*width,taps=convolution_state.size(1)+1;
    MFQ_RUNTIME_CHECK(channels<=std::numeric_limits<int>::max() && taps>1 &&
        taps<=std::numeric_limits<int>::max() && batch<=std::numeric_limits<int>::max()/value_heads/tiles &&
        std::isfinite(convolution_eps) && convolution_eps>0 && std::isfinite(norm_eps) && norm_eps>0,
        "GDN decode core geometry/epsilon exceeds supported bounds");
    MFQ_RUNTIME_CHECK(qkv.is_cuda() && qkv.stride(2)==1 && qkv.stride(0)>=channels && (qkv.scalar_type()==tb::kFloat16 || qkv.scalar_type()==tb::kFloat32) &&
        qkv.sizes().vec()==std::vector<int64_t>({batch,1,channels}),"GDN decode core qkv disagrees");
    const auto device=qkv.device();
    const auto real=[&](const tb::Tensor& t){return t.is_cuda() && t.device()==device && t.is_contiguous() && t.scalar_type()==tb::kFloat32;};
    const bool independent=state_addresses.defined();
    MFQ_RUNTIME_CHECK(!independent || (inplace_state && state_addresses.is_cuda() &&
        state_addresses.device()==device && state_addresses.is_contiguous() &&
        state_addresses.scalar_type()==tb::kInt64 &&
        state_addresses.sizes().vec()==std::vector<int64_t>({batch,3})),
        "GDN independent state addresses require inplace CUDA int64 [B,3]");
    MFQ_RUNTIME_CHECK(real(convolution_state) && real(recurrent_state) && real(convolution_weight) &&
        real(dt_bias) && real(a_log) && real(norm_weight),"GDN decode core parameters disagree");
    MFQ_RUNTIME_CHECK(checkpoint_convolution.defined()==checkpoint_recurrent.defined(),
        "GDN checkpoint buffers must be paired");
    MFQ_RUNTIME_CHECK(!checkpoint_convolution.defined() ||
        (!independent && real(checkpoint_convolution) && real(checkpoint_recurrent) &&
         checkpoint_convolution.sizes()==convolution_state.sizes() &&
         checkpoint_recurrent.sizes()==recurrent_state.sizes() &&
         checkpoint_convolution.data_ptr()!=convolution_state.data_ptr() &&
         checkpoint_recurrent.data_ptr()!=recurrent_state.data_ptr()),
        "GDN core checkpoint storage disagrees");
    const int64_t state_batch=independent?1:batch;
    MFQ_RUNTIME_CHECK(convolution_state.sizes().vec()==std::vector<int64_t>({state_batch,taps-1,channels}) &&
        recurrent_state.sizes().vec()==std::vector<int64_t>({state_batch,value_heads,width,width}),"GDN decode core state shape disagrees");
    const bool channels_first=convolution_weight.sizes().vec()==std::vector<int64_t>({channels,1,taps});
    MFQ_RUNTIME_CHECK(channels_first || convolution_weight.sizes().vec()==std::vector<int64_t>({taps,channels}),"GDN decode core conv weight disagrees");
    MFQ_RUNTIME_CHECK(output_gate.is_cuda() && alpha.is_cuda() && beta.is_cuda() &&
        output_gate.device()==device && alpha.device()==device && beta.device()==device &&
        output_gate.dim()==3 && output_gate.stride(2)==1 && output_gate.stride(0)>=value_heads*width &&
        output_gate.sizes().vec()==std::vector<int64_t>({batch,1,value_heads*width}) &&
        alpha.sizes().vec()==std::vector<int64_t>({batch,1,value_heads}) && beta.sizes()==alpha.sizes() &&
        output_gate.scalar_type()==alpha.scalar_type() && alpha.scalar_type()==beta.scalar_type() &&
        (alpha.scalar_type()==tb::kFloat16 || alpha.scalar_type()==tb::kFloat32),"GDN decode core gate inputs disagree");
    MFQ_RUNTIME_CHECK(dt_bias.numel()==value_heads && a_log.numel()==value_heads && norm_weight.numel()==width,
        "GDN decode core vector width disagrees");
    MfqCudaGuard guard(device);
    auto output=tb::empty({batch,1,value_heads*width},qkv.options().dtype(output_half?tb::kFloat16:tb::kFloat32));
    auto conv=tb::empty({batch,taps-1,channels},convolution_state.options());
    auto state=inplace_state?recurrent_state:tb::empty_like(recurrent_state);
    if(tiles>1 && !workspace.defined() && !independent)workspace=tb::zeros({batch,value_heads,width+1},convolution_state.options());
    MFQ_RUNTIME_CHECK(!workspace.defined() || (real(workspace) &&
        workspace.sizes().vec()==std::vector<int64_t>({state_batch,value_heads,width+1})),
        "GDN decode core workspace shape/device/dtype disagrees");
    const auto launch=[&]<int Tiles,class Qkv,class Gate,class Output,bool Snapshot,int FixedWidth=0,int FixedTaps=0>() {
        hipLaunchKernelGGL((gdn_stateful_decode_kernel<Tiles,Qkv,Gate,Output,FixedWidth,FixedTaps,false,0,Snapshot>),
            dim3(unsigned(batch*value_heads*Tiles)),dim3(256),0,mfq_current_cuda_stream(),
            qkv.data_ptr<Qkv>(),output_gate.data_ptr<Gate>(),alpha.data_ptr<Gate>(),beta.data_ptr<Gate>(),
            convolution_state.data_ptr<float>(),recurrent_state.data_ptr<float>(),convolution_weight.data_ptr<float>(),
            dt_bias.data_ptr<float>(),a_log.data_ptr<float>(),norm_weight.data_ptr<float>(),output.data_ptr<Output>(),
            conv.data_ptr<float>(),state.data_ptr<float>(),int(key_heads),int(value_heads),int(width),int(taps),channels_first?0:1,
            float(convolution_eps),norm_eps,silu_gate,transposed_state,alpha.stride(0),alpha.stride(2),beta.stride(0),beta.stride(2),
            workspace.defined()?workspace.data_ptr<float>():nullptr,
            independent?state_addresses.data_ptr<int64_t>():nullptr,qkv.stride(0),output_gate.stride(0),true,
            checkpoint_convolution.defined()?checkpoint_convolution.data_ptr<float>():nullptr,
            checkpoint_recurrent.defined()?checkpoint_recurrent.data_ptr<float>():nullptr);
    };
    const auto outputs=[&]<int Tiles,class Qkv,class Gate,bool Snapshot>() {
        const auto* fixed=std::getenv("MFQ_HIP_GDN_STATE_FIXED");
        if(width==128 && taps==4 && (!fixed || fixed[0]!='0')) {
            if(output_half)launch.template operator()<Tiles,Qkv,Gate,mfq_half,Snapshot,128,4>();
            else launch.template operator()<Tiles,Qkv,Gate,float,Snapshot,128,4>();
        } else {
            if(output_half)launch.template operator()<Tiles,Qkv,Gate,mfq_half,Snapshot>();
            else launch.template operator()<Tiles,Qkv,Gate,float,Snapshot>();
        }
    };
    const auto gates=[&]<int Tiles,class Qkv,bool Snapshot>() {
        if(alpha.scalar_type()==tb::kFloat16)outputs.template operator()<Tiles,Qkv,mfq_half,Snapshot>();
        else outputs.template operator()<Tiles,Qkv,float,Snapshot>();
    };
    const auto typed=[&]<int Tiles,bool Snapshot>() {
        if(qkv.scalar_type()==tb::kFloat16)gates.template operator()<Tiles,mfq_half,Snapshot>();
        else gates.template operator()<Tiles,float,Snapshot>();
    };
    const auto snapshots=[&]<int Tiles>() {
        if(checkpoint_convolution.defined())typed.template operator()<Tiles,true>();
        else typed.template operator()<Tiles,false>();
    };
    if(tiles==1)snapshots.template operator()<1>();
    else if(tiles==2)snapshots.template operator()<2>();
    else snapshots.template operator()<4>();
    MFQ_CUDA_CHECK(hipGetLastError());
    if(independent) {
        const int64_t elements=(taps-1)*channels;
        hipLaunchKernelGGL((gdn_stateful_commit_convolution),dim3(unsigned((elements+255)/256),unsigned(batch)),
            dim3(256),0,mfq_current_cuda_stream(),conv.data_ptr<float>(),state_addresses.data_ptr<int64_t>(),elements);
        MFQ_CUDA_CHECK(hipGetLastError());
    }
    return {output,conv,state};
}
