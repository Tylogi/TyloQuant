#pragma once
#include "mfe_ffn_runtime.h"

// CPU inference keeps FP32 activations, while GPU inference quantizes them.
// Select the arithmetic oracle from the observed adaptive plan for this call.
tb::Tensor moe_ffn_reference_parts(CudaExecutionContext& execution,const tb::Tensor& input,
        const tb::Tensor& ids,const std::array<MfeWeight,3>& baseline,
        const std::array<std::shared_ptr<MoeQuantRangeSource>,3>& sources,const mfq::MoeDispatchPlan& plan) {
    const int tokens=int(ids.size(0)),routes=int(ids.size(1));
    if(plan.kinds.size()!=size_t(tokens*routes))throw std::runtime_error("missing FFN reference dispatch");
    const auto route=build_moe_route_plan(ids,3);
    const auto g=baseline[0].forward(execution,input,route),u=baseline[1].forward(execution,input,route);
    auto expected=baseline[2].forward(execution,(g*tb::sigmoid(g))*u,route).cpu().to(tb::kFloat32).contiguous();
    const auto selected=ids.cpu().contiguous();
    for(int t=0;t<tokens;++t)for(int r=0;r<routes;++r) {
        const int pair=t*routes+r;
        const auto kind=plan.kinds[pair];
        if(kind==mfq::MoeDispatchKind::GpuResident || kind==mfq::MoeDispatchKind::GpuTransfer)continue;
        if(kind!=mfq::MoeDispatchKind::Cpu)throw std::runtime_error("invalid FFN reference dispatch");
        const int expert=selected.data_ptr<int32_t>()[pair];
        const auto x=input.narrow(0,t,1).cpu().to(tb::kFloat32).contiguous();
        const auto gate=linear(sources[0]->read_expert(expert),640,2560).forward(execution,x);
        const auto up=linear(sources[1]->read_expert(expert),640,2560).forward(execution,x);
        std::vector<float> activated(640);
        for(int col=0;col<640;++col) {
            const float gv=float(gate.data_ptr<mfq_half>()[col]),uv=float(up.data_ptr<mfq_half>()[col]);
            const float sigmoid=float(mfq_half(1.0/(1.0+std::exp(-double(gv)))));
            activated[col]=float(mfq_half(float(mfq_half(gv*sigmoid))*uv));
        }
        const auto hidden=tb::tensor(activated).reshape({1,640});
        const auto down=linear(sources[2]->read_expert(expert),2560,640)
            .forward(execution,hidden).to(tb::kFloat32).contiguous();
        std::memcpy(expected.data_ptr<float>()+int64_t(pair)*2560,down.data_ptr(),2560*sizeof(float));
    }
    return expected;
}

// The weighted shared path uses eight lanes per NVQ Down row; the ordinary
// projection uses sixteen. Validate each against a decoded FP64 dot.
struct MoeSharedOutputReference {
    std::array<std::shared_ptr<MixedMoeRuntime>,3> gpu;
    std::array<std::shared_ptr<MoeQuantRangeSource>,3> sources;
    NvqCpu up,down;
    NintCpu routed_gate;
    int64_t checks=0,quantized_rows=0,hidden_checks=0;
    double max_bound_ratio=0;
    bool reported=false;

    MoeSharedOutputReference(const std::filesystem::path& root,
            const std::array<std::string,3>& formats,CudaExecutionContext& execution) {
        for(int p=0;p<3;++p) {
            const auto path=root/(formats[p]+(p==2?"-2560-640.mfq":"-640-2560.mfq"));
            const auto model=mfq::open_model_source(path.string());
            const auto encoded=load_mfe_cpu(*model,"linear.weight");
            if(encoded.pools.size()!=1 || encoded.n_experts!=3)
                throw std::runtime_error("weighted shared reference requires one three-expert pool");
            gpu[p]=make_mixed_moe_runtime(encoded,true,execution.config);
            sources[p]=source(path);
            if(p==0)routed_gate=encoded.pools[0].weight;
            if(p==1)up=unpack_nvq(encoded.pools[0].payload,encoded.pools[0].dtype);
            if(p==2)down=unpack_nvq(encoded.pools[0].payload,encoded.pools[0].dtype);
        }
        if(down.format!=14 || up.format!=15 || down.gs!=24 || up.gs!=24)
            throw std::runtime_error("weighted shared FP64 reference requires canonical NVQ3J-L Up/NVQ2J-XL Down");
    }

    static NintDispatchError dot(const NvqCpu& weight,const Workspace& input,int pair,int row) {
        const auto bits=[](const std::vector<uint8_t>& data,int64_t bit,int width) {
            return cpu_load_packed_bits(data.data(),data.size(),bit,width);
        };
        const auto& down=weight;
        const bool d4=weight.format==15;
        const int index_bits=d4?10:12,bank_size=d4?1024*4:4096*8;
        mfq_half anchor;
        std::memcpy(&anchor,&down.neuron_scale_h[row],sizeof(anchor));
        const auto* codes=input.qx.data_ptr<int8_t>()+int64_t(pair)*down.ng*down.gs;
        const auto* scales=input.xscale.data_ptr<float>()+int64_t(pair)*down.ng;
        NintDispatchError result;
        for(int group=0;group<down.ng;++group) {
            const int state=bits(down.sub_scale_packed,(int64_t(row)*down.ng+group)*down.sub_bits,down.sub_bits);
            mfq_half multiplier;
            std::memcpy(&multiplier,down.codebook.data()+4+state*2,sizeof(multiplier));
            const int bank=uint8_t(down.codebook[36+state]);
            const double scale=double(float(anchor))*float(multiplier)*scales[group];
            int dot=0;
            for(int segment=0;segment<3 && group*3+segment<down.nsign;++segment) {
                const int vector=group*3+segment;
                const int first=vector*(d4?2:1);
                const int index=bits(down.indices_packed,(int64_t(row)*down.nvec+first)*index_bits,index_bits);
                const int signs=bits(down.aux_packed,(int64_t(row)*down.nsign+vector)*7,7);
                int parity=0;
                for(int bit=0;bit<7;++bit)parity^=(signs>>bit)&1;
                parity^=(down.sign_mode&1)?(index>>7)&1:0;
                int chunk=0;
                for(int i=0;i<8;++i) {
                    const int sign=i==7?parity:(signs>>i)&1;
                    const int code=d4 && i>=4?bits(down.indices_packed,
                        (int64_t(row)*down.nvec+first+1)*index_bits,index_bits):index;
                    const int value=down.codebook[64+bank*bank_size+code*(d4?4:8)+(d4?i%4:i)]*(sign?-1:1);
                    chunk+=value*codes[group*24+segment*8+i];
                }
                dot+=chunk;
                result.pool_mass+=std::abs(scale*chunk);
            }
            result.value+=scale*dot;
            result.grouped_mass+=std::abs(scale*dot);
        }
        return result;
    }

    static Workspace host_input(const MoeActivationWorkspace& input) {
        Workspace result;
        result.qx=input.qx.cpu().contiguous();
        result.xscale=input.xscale.cpu().contiguous();
        return result;
    }

    static double nint_bound(const NintCpu& weight,const NintDispatchError& reference) {
        const int groups_per_warp=32/((weight.gs+3)/4);
        // Dense NINT may add a second five-level tree across warp partials.
        return nint_dispatch_error_bound(reference.value,reference.pool_mass,
            (weight.ng+groups_per_warp-1)/groups_per_warp+5,weight.ng);
    }

    static void verify_hidden(double gate,double gate_bound,double up,double up_bound,float a,float b) {
        const double u=std::ldexp(1.0,-11),eta=std::ldexp(1.0,-25);
        const double silu=gate/(1+std::exp(-gate));
        // |SiLU'|<2. The Half sigmoid adds at most u+eta, followed by
        // Half gate*sigmoid and Half silu*up storage boundaries.
        double first_bound=2*gate_bound+(std::abs(gate)+gate_bound)*(u+eta);
        first_bound+=u*(std::abs(silu)+first_bound)+eta;
        const double reference=silu*up;
        double bound=std::abs(up)*first_bound+std::abs(silu)*up_bound+first_bound*up_bound;
        bound+=u*(std::abs(reference)+bound)+eta;
        if(!std::isfinite(reference) || !std::isfinite(bound))
            throw std::runtime_error("weighted shared SwiGLU reference is not finite");
        for(float value:{a,b})if(!std::isfinite(value) || std::abs(value-reference)>bound)
            throw std::runtime_error("weighted shared SwiGLU exceeds the FP64 propagated error bound");
    }

    void verify(CudaExecutionContext& execution,const tb::Tensor& input,const tb::Tensor& ids,
            const tb::Tensor& weights,const std::vector<int32_t>& selected,const mfq::MoeDispatchPlan& plan,
            const std::array<std::shared_ptr<const QuantLinear>,3>& shared,
            const std::array<NintCpu,3>& shared_cpu,const tb::Tensor& gate,
            const tb::Tensor& actual,const tb::Tensor& expected) {
        const int tokens=int(input.size(0)),routes=int(ids.size(1)),width=down.neuron_len;
        const int output=gpu[2]->out_per_expert;
        const auto route=build_moe_route_plan(ids,3);
        const auto project=[&](int p,const tb::Tensor& x) {
            return gpu[p]->forward(execution.config,execution.kl_mmq,false,false,x,route);
        };
        const auto g=project(0,input),u=project(1,input);
        const auto hidden=(g*tb::sigmoid(g))*u;
        const auto pairs=project(2,hidden).cpu().to(tb::kFloat32).contiguous();
        const auto workspace=gpu[2]->activation_workspace(hidden,tokens*routes,down.ng,down.gs,{});
        Workspace routed_input;
        routed_input.qx=workspace.qx.cpu().contiguous();
        routed_input.xscale=workspace.xscale.cpu().contiguous();
        const auto sg=shared[0]->forward(execution,input),su=shared[1]->forward(execution,input);
        const auto shared_hidden=(sg*tb::sigmoid(sg))*su;
        const auto shared_pairs=shared[2]->forward(execution,shared_hidden).cpu().to(tb::kFloat32).contiguous();
        const auto& shared_workspace=shared[2]->nint.workspace(tokens);
        Workspace shared_input;
        shared_input.qx=shared_workspace.qx.cpu().contiguous();
        shared_input.xscale=shared_workspace.xscale.cpu().contiguous();

        // Check the premise of a Down-only error bound using a separate runtime,
        // without adding calls or observations to either cache under test.
        const std::array<const NintWeight*,3> shared_views={&shared[0]->nint,&shared[1]->nint,&shared[2]->nint};
        MfeFfnRuntime fused({gpu[0].get(),gpu[1].get(),gpu[2].get()},input,ids,weights,
            moe_swiglu_sigmoid_table_cuda(),shared_views,gate,output,width);
        fused.run();
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        const auto fused_hidden=fused.hidden().cpu().contiguous();
        const auto original_hidden=hidden.cpu().contiguous(),original_shared_hidden=shared_hidden.cpu().contiguous();
        const auto& batch=fused.batch();
        std::vector<int8_t> codes(size_t(tokens)*(routes+1)*batch.quantized_stride);
        std::vector<float> scales(size_t(tokens)*(routes+1)*batch.scale_stride);
        MFQ_CUDA_CHECK(cudaMemcpy(codes.data(),batch.hidden_quantized,codes.size(),cudaMemcpyDeviceToHost));
        MFQ_CUDA_CHECK(cudaMemcpy(scales.data(),batch.hidden_scales,
            scales.size()*sizeof(float),cudaMemcpyDeviceToHost));
        Workspace candidate_input,candidate_shared_input;
        candidate_input.qx=tb::empty(routed_input.qx.sizes(),routed_input.qx.options());
        candidate_input.xscale=tb::empty(routed_input.xscale.sizes(),routed_input.xscale.options());
        candidate_shared_input.qx=tb::empty(shared_input.qx.sizes(),shared_input.qx.options());
        candidate_shared_input.xscale=tb::empty(shared_input.xscale.sizes(),shared_input.xscale.options());
        bool different_quantization=false;
        const auto gate_input=host_input(gpu[0]->activation_workspace(input,tokens,routed_gate.ng,routed_gate.gs,{}));
        const auto up_input=host_input(gpu[1]->activation_workspace(input,tokens,up.ng,up.gs,{}));
        std::array<Workspace,2> shared_inputs;
        for(int p=0;p<2;++p) {
            const auto& ws=shared[p]->nint.workspace(tokens);
            shared_inputs[p].qx=ws.qx.cpu().contiguous();
            shared_inputs[p].xscale=ws.xscale.cpu().contiguous();
        }
        for(int t=0;t<tokens;++t)for(int r=0;r<=routes;++r) {
            const bool is_shared=r==routes;
            const int slot=t*(routes+1)+r,pair=t*routes+r;
            const auto& quantized=is_shared?shared_input:routed_input;
            auto& candidate=is_shared?candidate_shared_input:candidate_input;
            const int index=is_shared?t:pair,groups=is_shared?shared_cpu[2].ng:down.ng;
            const int gs=is_shared?shared_cpu[2].gs:down.gs;
            std::memcpy(candidate.qx.data_ptr<int8_t>()+index*groups*gs,
                codes.data()+slot*batch.quantized_stride,groups*gs);
            std::memcpy(candidate.xscale.data_ptr<float>()+index*groups,
                scales.data()+slot*batch.scale_stride,groups*sizeof(float));
            if(!is_shared && plan.kinds[pair]==mfq::MoeDispatchKind::Cpu)continue;
            if(std::memcmp(codes.data()+slot*batch.quantized_stride,
                        quantized.qx.data_ptr<int8_t>()+index*groups*gs,groups*gs) ||
                    std::memcmp(scales.data()+slot*batch.scale_stride,
                        quantized.xscale.data_ptr<float>()+index*groups,groups*sizeof(float))) {
                different_quantization=true;
                ++quantized_rows;
                const auto& original=is_shared?original_shared_hidden:original_hidden;
                for(int col=0;col<width;++col) {
                    const int row=is_shared?col:selected[pair]*width+col;
                    const auto gr=nint_dispatch_reference(is_shared?shared_cpu[0]:routed_gate,
                        is_shared?shared_inputs[0]:gate_input,t,row);
                    const auto ur=is_shared?nint_dispatch_reference(shared_cpu[1],shared_inputs[1],t,row):
                        dot(up,up_input,t,row);
                    const double gb=nint_bound(is_shared?shared_cpu[0]:routed_gate,gr);
                    const double ub=is_shared?nint_bound(shared_cpu[1],ur):
                        nint_dispatch_error_bound(ur.value,ur.pool_mass,(3*up.ng+7)/8,up.ng);
                    verify_hidden(gr.value,gb,ur.value,ub,
                        float(original.data_ptr<mfq_half>()[index*width+col]),
                        float(fused_hidden.data_ptr<mfq_half>()[slot*batch.hidden_stride+col]));
                    hidden_checks+=2;
                }
            }
        }
        if(different_quantization) {
            // Re-quantize the candidate's validated Half hidden values through
            // the ordinary projection path before using its Q8 inputs as an oracle.
            const auto shaped=fused.hidden().reshape({tokens,routes+1,width});
            const auto routed_hidden=shaped.narrow(1,0,routes).contiguous();
            project(2,routed_hidden);
            const auto checked=host_input(gpu[2]->activation_workspace(
                routed_hidden,tokens*routes,down.ng,down.gs,{}));
            const auto sh=shaped.narrow(1,routes,1).contiguous().reshape({tokens,width});
            shared[2]->forward(execution,sh);
            const auto& sw=shared[2]->nint.workspace(tokens);
            const auto sq=sw.qx.cpu().contiguous(),ss=sw.xscale.cpu().contiguous();
            if(std::memcmp(checked.qx.data_ptr(),candidate_input.qx.data_ptr(),checked.qx.nbytes()) ||
                    std::memcmp(checked.xscale.data_ptr(),candidate_input.xscale.data_ptr(),
                        checked.xscale.nbytes()) ||
                    std::memcmp(sq.data_ptr(),candidate_shared_input.qx.data_ptr(),sq.nbytes()) ||
                    std::memcmp(ss.data_ptr(),candidate_shared_input.xscale.data_ptr(),ss.nbytes()))
                throw std::runtime_error("fused Down quantization differs from ordinary hidden quantization");
        }

        auto cpu_pairs=tb::zeros({tokens*routes,output},tb::TensorOptions().dtype(tb::kFloat32));
        for(int pair=0;pair<tokens*routes;++pair)if(plan.kinds[pair]==mfq::MoeDispatchKind::Cpu) {
            const int expert=selected[pair],token=pair/routes;
            const auto x=input.narrow(0,token,1).cpu().to(tb::kFloat32);
            const auto project_cpu=[&](int p,const tb::Tensor& value,int columns) {
                return linear(sources[p]->read_expert(expert),columns,p==2?width:int(input.size(1)))
                    .forward(execution,value).contiguous();
            };
            const auto cg=project_cpu(0,x,width),cu=project_cpu(1,x,width);
            std::vector<float> activated(width);
            for(int col=0;col<width;++col) {
                const float gv=float(cg.data_ptr<mfq_half>()[col]),uv=float(cu.data_ptr<mfq_half>()[col]);
                const float sigmoid=float(mfq_half(1.0/(1.0+std::exp(-double(gv)))));
                activated[col]=float(mfq_half(float(mfq_half(gv*sigmoid))*uv));
            }
            const auto down_value=project_cpu(2,tb::tensor(activated).reshape({1,width}),output).to(tb::kFloat32);
            std::memcpy(cpu_pairs.data_ptr<float>()+int64_t(pair)*output,down_value.data_ptr(),output*sizeof(float));
        }
        const auto a=actual.to(tb::kFloat32).contiguous(),e=expected.to(tb::kFloat32).contiguous();
        const auto router=weights.cpu().contiguous(),gates=gate.cpu().to(tb::kFloat32).contiguous();
        const double u16=std::ldexp(1.0,-11),u32=std::ldexp(1.0,-24),eta=std::ldexp(1.0,-25);
        const double u64=std::ldexp(1.0,-53),n=2*routes;
        const double gamma=n*u32/(1-n*u32)+n*u64/(1-n*u64);
        for(int t=0;t<tokens;++t)for(int col=0;col<output;++col)for(int mode=0;mode<2;++mode) {
            const int64_t index=int64_t(t)*output+col;
            if(a.data_ptr<float>()[index]==e.data_ptr<float>()[index])continue;
            double routed=0,mass=0,propagated=0;
            for(int r=0;r<routes;++r) {
                const int pair=t*routes+r;
                double value=cpu_pairs.data_ptr<float>()[int64_t(pair)*output+col],bound=0;
                if(plan.kinds[pair]!=mfq::MoeDispatchKind::Cpu) {
                    const auto reference=dot(down,mode?routed_input:candidate_input,pair,selected[pair]*output+col);
                    value=reference.value;
                    // Up to three chunk terms per group, eight lanes per row,
                    // scale products, FMA accumulation and the warp reduction.
                    bound=nint_dispatch_error_bound(value,reference.pool_mass,(3*down.ng+7)/8,down.ng);
                    if(mode && std::abs(pairs.data_ptr<float>()[int64_t(pair)*output+col]-value)>bound)
                        throw std::runtime_error("ordinary NVQ Down exceeds the FP64 error bound");
                }
                const double weight=router.data_ptr<float>()[pair];
                routed+=weight*value;
                mass+=std::abs(weight*value);
                propagated+=std::abs(weight)*bound;
            }
            const auto sd=nint_dispatch_reference(shared_cpu[2],mode?shared_input:candidate_shared_input,t,col);
            const int groups_per_warp=32/((shared_cpu[2].gs+3)/4);
            const double down_bound=nint_dispatch_error_bound(sd.value,sd.pool_mass,
                (shared_cpu[2].ng+groups_per_warp-1)/groups_per_warp,shared_cpu[2].ng);
            if(mode && std::abs(shared_pairs.data_ptr<float>()[int64_t(t)*output+col]-sd.value)>down_bound)
                throw std::runtime_error("ordinary shared Down exceeds the FP64 error bound");
            // R=Half(sum(w*D)), C=gate*shared_D, Y=R+C (with Half C/Y for Half gates).
            // Propagate each Down bound, gamma_(2*routes) weighted accumulation,
            // then each FP16/FP32 product, addition and storage boundary.
            const double routed_bound=u16*std::abs(routed)+(1+u16)*
                (propagated+gamma*(mass+propagated))+eta;
            const double gate_value=gates.data_ptr<float>()[t],contribution=gate_value*sd.value;
            const bool half_gate=gate.scalar_type()==tb::kFloat16;
            const double product_u=(half_gate?u16:u32)+u64;
            const double shared_bound=(1+product_u)*std::abs(gate_value)*down_bound+
                product_u*std::abs(contribution)+(half_gate?eta:0);
            const double reference=routed+contribution;
            double bound=routed_bound+shared_bound+(u32+u64)*
                (std::abs(routed)+std::abs(contribution)+routed_bound+shared_bound);
            if(half_gate)bound=u16*std::abs(reference)+(1+u16)*bound+eta;
            {
                const float value=mode?e.data_ptr<float>()[index]:a.data_ptr<float>()[index];
                const double error=std::abs(value-reference);
                if(!std::isfinite(value) || !std::isfinite(reference) || !std::isfinite(bound) || error>bound) {
                    std::ostringstream message;
                    message<<std::setprecision(12)<<"weighted shared output exceeds FP32/FP16 error bound"
                        <<" index="<<index<<" actual="<<value<<" reference="<<reference<<" bound="<<bound;
                    message<<" baseline="<<e.data_ptr<float>()[index]<<" candidate="<<a.data_ptr<float>()[index];
                    throw std::runtime_error(message.str());
                }
                max_bound_ratio=std::max(max_bound_ratio,error/bound);
                ++checks;
            }
            if(!reported && a.data_ptr<float>()[index]!=e.data_ptr<float>()[index]) {
                reported=true;
                std::cout<<std::setprecision(12)<<"weighted_shared_difference index="<<index
                    <<" actual="<<a.data_ptr<float>()[index]<<" original="<<e.data_ptr<float>()[index]
                    <<" reference="<<reference<<" bound="<<bound<<'\n';
            }
        }
    }
};
