#include "mfq_tensor_backend.h"
#include "mfq_hip_attention_ops.h"
#include "mfq_hip_sampling_ops.h"
#include "mfq_hip_norm_ops.h"
#include "mfq_hip_linear_attention_ops.h"
#include "gdn_decode.h"
#include "tq_input_gate.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <limits>
#include <vector>

namespace tb=mfq_tensor_backend;
static tb::Tensor floats(std::vector<float>& data,std::vector<int64_t> shape,tb::ScalarType type) {
    return tb::from_blob(data.data(),shape,tb::TensorOptions().dtype(tb::kFloat32))
        .clone().to(type).to(tb::Device(tb::kCUDA,0));
}
static void greedy() {
    for(auto type:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})
    for(int vocab:{9,16383,16384,32769,248320}) {
        constexpr int B=8;
        std::vector<float> data(size_t(B)*vocab);
        for(size_t i=0;i<data.size();++i)data[i]=std::sin(float(i)*.137f);
        for(int i=0;i<vocab;++i) {
            data[vocab+i]=-std::numeric_limits<float>::infinity();
            data[2*vocab+i]=std::numeric_limits<float>::quiet_NaN();
            data[3*vocab+i]=-FLT_MAX;
            data[4*vocab+i]=(i&1)?-0.f:0.f;
        }
        data[5*vocab+vocab-1]=data[5*vocab+3]=std::numeric_limits<float>::infinity();
        data[6*vocab+vocab-1]=data[6*vocab+2]=50.f;
        auto input=floats(data,{B,vocab},type);
        auto rounded=input.cpu().to(tb::kFloat32).contiguous();
        auto actual=sample_greedy_cuda(input).cpu().contiguous();
        for(int b=0;b<B;++b) {
            float best=-FLT_MAX;int index=0;
            for(int i=0;i<vocab;++i) {
                const float v=rounded.data_ptr<float>()[size_t(b)*vocab+i];
                if(v>best || (v==best && i<index)){best=v;index=i;}
            }
            if(actual.data_ptr<int64_t>()[b]!=index)
                throw std::runtime_error("greedy chunks changed argmax/tie/NaN semantics");
        }
    }
}
static void attention() {
    constexpr int B=2,Hq=4,Hk=2;
    for(auto type:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})
    for(int D:{64,128,256,512}) {
        const int maximum=D==256?8192:1024;
        std::vector<float> qdata(B*Hq*D),kdata(size_t(B)*Hk*maximum*D),vdata(kdata.size());
        for(size_t i=0;i<qdata.size();++i)qdata[i]=std::sin(float(i)*.37f)*.4f;
        for(size_t i=0;i<kdata.size();++i){kdata[i]=std::cos(float(i%7919)*.17f)*.3f;vdata[i]=std::sin(float(i%7873)*.19f)*.2f;}
        auto q=floats(qdata,{B,Hq,1,D},type),k=floats(kdata,{B,Hk,maximum,D},type),v=floats(vdata,{B,Hk,maximum,D},type);
        auto qr=q.cpu().to(tb::kFloat32).contiguous(),kr=k.cpu().to(tb::kFloat32).contiguous(),vr=v.cpu().to(tb::kFloat32).contiguous();
        auto options=q.options().dtype(tb::kFloat32);
        auto po=tb::empty({B*Hq,256,D},options),pm=tb::empty({B*Hq,256},options),pl=tb::empty({B*Hq,256},options);
        for(int length:{0,1,17,191,192,255,257,1024,8192}) {
            if(length>maximum)continue;
            std::vector<int64_t> lengths{length,std::max(0,length-13)};
            auto seq=tb::from_blob(lengths.data(),{B},tb::TensorOptions().dtype(tb::kInt64)).clone().to(tb::Device(tb::kCUDA,0));
            const double scale=1./std::sqrt(double(D));
            auto y=attention_cache_decode_dynamic_cuda(q,k,v,seq,scale,po,pm,pl,4).cpu().to(tb::kFloat32).contiguous();
            double error=0,norm=0;std::vector<double> reference(B*Hq*D);
            for(int b=0;b<B;++b)for(int h=0;h<Hq;++h) {
                const int n=int(lengths[b]),hk=h/(Hq/Hk);
                std::vector<double> scores(n),expected(D,0.);
                double maximum_score=-1e300,denom=0;
                for(int t=0;t<n;++t) {
                    double dot=0;
                    for(int c=0;c<D;++c)dot+=double(qr.data_ptr<float>()[(b*Hq+h)*D+c])*kr.data_ptr<float>()[((size_t(b)*Hk+hk)*maximum+t)*D+c];
                    maximum_score=std::max(maximum_score,scores[t]=dot*scale);
                }
                for(int t=0;t<n;++t) {
                    const double p=std::exp(scores[t]-maximum_score);denom+=p;
                    for(int c=0;c<D;++c)expected[c]+=p*vr.data_ptr<float>()[((size_t(b)*Hk+hk)*maximum+t)*D+c];
                }
                for(int c=0;c<D;++c) {
                    const double ref=expected[c]/(denom>0?denom:1),actual=y.data_ptr<float>()[(b*Hq+h)*D+c];
                    reference[(b*Hq+h)*D+c]=ref;
                    if(!std::isfinite(actual))throw std::runtime_error("attention produced non-finite output");
                    error+=(actual-ref)*(actual-ref);norm+=ref*ref;
                }
            }
            const double limit=type==tb::kFloat16?.001:type==tb::kBFloat16?.008:4e-6;
            // Correlated inputs can nearly cancel. Include a small absolute
            // tolerance rather than dividing only by a near-zero reference.
            const double atol=type==tb::kFloat32?2e-8:0.;
            if(std::sqrt(error/reference.size())>atol+limit*std::sqrt(norm/reference.size())) {
                auto old=attention_cache_decode_split_cuda(q,k,v,seq,scale,po,pm,pl,8).cpu().to(tb::kFloat32).contiguous();
                double old_error=0,max_abs=0;
                for(size_t i=0;i<reference.size();++i) {
                    old_error+=std::pow(double(old.data_ptr<float>()[i])-reference[i],2);
                    max_abs=std::max(max_abs,std::abs(double(y.data_ptr<float>()[i])-reference[i]));
                }
                std::cerr<<"legacy error="<<std::sqrt(old_error/std::max(norm,1e-30))<<" new max abs="<<max_abs<<'\n';
                std::cerr<<"D="<<D<<" length="<<length<<" error="<<std::sqrt(error/std::max(norm,1e-30))<<'\n';
                throw std::runtime_error("dynamic split attention differs from FP64 oracle");
            }
            for(int parts:{32,128}) {
                auto result=attention_cache_decode_dynamic_cuda(q,k,v,seq,scale,po,pm,pl,parts).cpu().to(tb::kFloat32).contiguous();
                double squared=0;
                for(size_t i=0;i<reference.size();++i) {
                    const double value=result.data_ptr<float>()[i];
                    if(!std::isfinite(value))throw std::runtime_error("parallel attention produced non-finite output");
                    squared+=std::pow(value-reference[i],2);
                }
                if(std::sqrt(squared/reference.size())>atol+limit*std::sqrt(norm/reference.size()))
                    throw std::runtime_error("parallel attention segments differ from FP64 oracle");
            }
        }
    }
}
static void cached_prefill() {
    constexpr int Hq=8,Hk=2,D=256;
    for(auto type:{tb::kFloat16,tb::kFloat32})
    for(auto shape:std::vector<std::vector<int>>{{1,17,0},{2,32,33},{1,129,257},{1,32,1031}}) {
        const int B=shape[0],T=shape[1],prefix=shape[2],capacity=prefix+T+7;
        std::vector<float> qdata(size_t(B)*Hq*T*D),kdata(size_t(B)*Hk*capacity*D),vdata(kdata.size());
        for(size_t i=0;i<qdata.size();++i)qdata[i]=std::sin(float(i%1877)*.173f)*.7f;
        for(size_t i=0;i<kdata.size();++i) {
            const int position=(i/D)%capacity;
            kdata[i]=position<prefix+T?std::cos(float(i%7919)*.17f)*.3f:std::numeric_limits<float>::quiet_NaN();
            vdata[i]=position<prefix+T?std::sin(float(i%7873)*.19f)*.2f:std::numeric_limits<float>::quiet_NaN();
        }
        auto q=floats(qdata,{B,Hq,T,D},type),k=floats(kdata,{B,Hk,capacity,D},tb::kFloat16),v=floats(vdata,{B,Hk,capacity,D},tb::kFloat16);
        auto qr=q.cpu().to(tb::kFloat32).contiguous(),kr=k.cpu().to(tb::kFloat32).contiguous(),vr=v.cpu().to(tb::kFloat32).contiguous();
        auto output=attention_cached_prefill_cuda(q,k,v,1./16.,prefix);
        if(output.scalar_type()!=type || output.sizes()!=q.sizes())
            throw std::runtime_error("cached prefill changed output type or layout");
        auto y=output.cpu().to(tb::kFloat32).contiguous();
        double squared=0,norm=0,max_error=0;
        for(int b=0;b<B;++b)for(int h=0;h<Hq;++h)for(int t=0;t<T;++t) {
            const int n=prefix+t+1;
            const size_t qi=(size_t(b)*Hq+h)*T*D+size_t(t)*D;
            const size_t ki=(size_t(b)*Hk+h/4)*capacity*D;
            std::vector<double> scores(n),expected(D,0.);
            double maximum=-1e300,denominator=0;
            for(int s=0;s<n;++s) {
                double score=0;
                for(int d=0;d<D;++d)score+=double(qr.data_ptr<float>()[qi+d])*kr.data_ptr<float>()[ki+size_t(s)*D+d];
                maximum=std::max(maximum,scores[s]=score/16.);
            }
            for(int s=0;s<n;++s) {
                const double p=std::exp(scores[s]-maximum);denominator+=p;
                for(int d=0;d<D;++d)expected[d]+=p*vr.data_ptr<float>()[ki+size_t(s)*D+d];
            }
            for(int d=0;d<D;++d) {
                const double reference=expected[d]/denominator,actual=y.data_ptr<float>()[qi+d];
                if(!std::isfinite(actual))throw std::runtime_error("cached prefill read invalid KV tail");
                squared+=std::pow(actual-reference,2);norm+=reference*reference;
                max_error=std::max(max_error,std::abs(actual-reference));
            }
        }
        // Same FP16 MFMA precision contract as the existing prefill test.
        if(std::sqrt(squared/norm)>.003 || max_error>.003)
            throw std::runtime_error("cached prefill differs from causal FP64 oracle");
    }
}
static void grouped_attention() {
    constexpr int B=2,Hq=8,Hk=2,D=256,capacity=65547;
    std::vector<float> qdata(B*Hq*D),kdata(size_t(B)*Hk*capacity*D),vdata(kdata.size());
    for(size_t i=0;i<qdata.size();++i)qdata[i]=std::sin(float(i%1877)*.37f)*.4f;
    for(size_t i=0;i<kdata.size();++i){kdata[i]=std::cos(float(i%7919)*.17f)*.3f;vdata[i]=std::sin(float(i%7873)*.19f)*.2f;}
    auto q=floats(qdata,{B,Hq,1,D},tb::kFloat16),k=floats(kdata,{B,Hk,capacity,D},tb::kFloat16),v=floats(vdata,{B,Hk,capacity,D},tb::kFloat16);
    auto qr=q.cpu().to(tb::kFloat32).contiguous(),kr=k.cpu().to(tb::kFloat32).contiguous(),vr=v.cpu().to(tb::kFloat32).contiguous();
    auto options=q.options().dtype(tb::kFloat32);
    auto po=tb::empty({B*Hq,256,D},options),pm=tb::empty({B*Hq,256},options),pl=tb::empty({B*Hq,256},options);
    for(int length:{0,1,65,8193,32768,65539}) {
        std::vector<int64_t> lengths{length,std::max(0,length-13)};
        auto seq=tb::from_blob(lengths.data(),{B},tb::TensorOptions().dtype(tb::kInt64)).clone().to(tb::Device(tb::kCUDA,0));
        auto y=attention_cache_decode_grouped_cuda(q,k,v,seq,1./16.,po,pm,pl,capacity).cpu().to(tb::kFloat32).contiguous();
        double squared=0,norm=0,max_error=0;
        for(int b=0;b<B;++b)for(int h=0;h<Hq;++h) {
            const int n=lengths[b];const size_t ki=(size_t(b)*Hk+h/4)*capacity*D;
            std::vector<double> scores(n),expected(D,0.);double maximum=-1e300,denominator=0;
            for(int s=0;s<n;++s) {
                double score=0;
                for(int d=0;d<D;++d)score+=double(qr.data_ptr<float>()[(b*Hq+h)*D+d])*kr.data_ptr<float>()[ki+size_t(s)*D+d];
                maximum=std::max(maximum,scores[s]=score/16.);
            }
            for(int s=0;s<n;++s) {
                const double p=std::exp(scores[s]-maximum);denominator+=p;
                for(int d=0;d<D;++d)expected[d]+=p*vr.data_ptr<float>()[ki+size_t(s)*D+d];
            }
            for(int d=0;d<D;++d) {
                const double reference=expected[d]/(denominator>0?denominator:1),actual=y.data_ptr<float>()[(b*Hq+h)*D+d];
                if(!std::isfinite(actual))throw std::runtime_error("grouped attention produced non-finite output");
                squared+=std::pow(actual-reference,2);norm+=reference*reference;max_error=std::max(max_error,std::abs(actual-reference));
            }
        }
        if(std::sqrt(squared/(B*Hq*D))>2e-5+.003*std::sqrt(norm/(B*Hq*D)) || max_error>.003)
            throw std::runtime_error("grouped attention differs from FP64 oracle");
    }
}
static void recurrent_prefill() {
    constexpr int B=2,Hq=2,Hv=4,D=128;
    for(bool transposed:{false,true})for(bool tiled:{false,true})for(int T:{1,17,129}) {
        std::vector<float> qd(B*Hq*T*D),kd(qd.size()),vd(B*Hv*T*D),gd(B*Hv*T),bd(gd.size()),sd(B*Hv*D*D);
        for(size_t i=0;i<qd.size();++i){qd[i]=std::sin(float(i%1877)*.37f)*.1f;kd[i]=std::cos(float(i%1879)*.19f)*.1f;}
        for(size_t i=0;i<vd.size();++i)vd[i]=std::sin(float(i%1973)*.23f)*.3f;
        for(size_t i=0;i<gd.size();++i){gd[i]=-.3f-.01f*float(i%7);bd[i]=.3f+.01f*float(i%11);}
        for(size_t i=0;i<sd.size();++i)sd[i]=std::cos(float(i%1993)*.17f)*.01f;
        auto q=floats(qd,{B,Hq,T,D},tb::kFloat32),k=floats(kd,{B,Hq,T,D},tb::kFloat32),v=floats(vd,{B,Hv,T,D},tb::kFloat32);
        auto g=floats(gd,{B,Hv,T},tb::kFloat32),beta=floats(bd,{B,Hv,T},tb::kFloat32),state=floats(sd,{B,Hv,D,D},tb::kFloat32);
        auto result=transposed
            ? (tiled?gdn_inplace_transposed_tiled_cuda(q,k,v,g,beta,state):gdn_inplace_transposed_cuda(q,k,v,g,beta,state))
            : (tiled?gdn_inplace_tiled_cuda(q,k,v,g,beta,state):gdn_inplace_cuda(q,k,v,g,beta,state));
        auto y=result[0].cpu().contiguous(),s=result[1].cpu().contiguous();
        double error=0,norm=0;
        for(int b=0;b<B;++b)for(int h=0;h<Hv;++h) {
            const int hq=tiled?h%Hq:h/(Hv/Hq),group=b*Hv+h;
            std::vector<double> reference(D*D);
            for(int c=0;c<D;++c)for(int r=0;r<D;++r)reference[c*D+r]=sd[size_t(group)*D*D+(transposed?c*D+r:r*D+c)];
            for(int t=0;t<T;++t) {
                const size_t qi=(size_t(b)*Hq+hq)*T*D+size_t(t)*D,vi=(size_t(group)*T+t)*D;
                const double decay=std::exp(double(gd[group*T+t]));
                for(int c=0;c<D;++c) {
                    double projection=0;
                    for(int r=0;r<D;++r)projection+=reference[c*D+r]*kd[qi+r];
                    const double delta=(vd[vi+c]-decay*projection)*bd[group*T+t];
                    double output=0;
                    for(int r=0;r<D;++r){auto& value=reference[c*D+r];value=decay*value+kd[qi+r]*delta;output+=value*qd[qi+r];}
                    const double expected=output/std::sqrt(double(D)),actual=y.data_ptr<float>()[vi+c];
                    error+=std::pow(actual-expected,2);norm+=expected*expected;
                }
            }
            for(int i=0;i<D*D;++i){const double actual=s.data_ptr<float>()[size_t(group)*D*D+(transposed?i:(i%D)*D+i/D)];error+=std::pow(actual-reference[i],2);norm+=reference[i]*reference[i];}
        }
        if(!std::isfinite(error) || std::sqrt(error/norm)>2e-5)throw std::runtime_error("GDN output/state differs from FP64 recurrence");
    }
}
static void convolution() {
    constexpr int B=2,C=129;
    for(int T:{1,17,65,129})for(int K:{3,4,5})for(int layout:{0,1,2})for(bool bias:{false,true}) {
        std::vector<float> xd(B*(T+K-1)*C),wd(C*K),bd(bias?C:0);
        for(size_t i=0;i<xd.size();++i)xd[i]=std::sin(float(i%1877)*.17f)*.3f;
        for(size_t i=0;i<wd.size();++i)wd[i]=std::cos(float(i%1879)*.11f)*.2f;
        for(int i=0;i<int(bd.size());++i)bd[i]=std::sin(float(i)*.13f)*.1f;
        auto x=floats(xd,{B,T+K-1,C},tb::kFloat32);
        const auto shape=layout==0?std::vector<int64_t>{C,1,K}:layout==1?std::vector<int64_t>{C,K}:std::vector<int64_t>{K,C};
        auto w=floats(wd,shape,tb::kFloat32),b=floats(bd,{int64_t(bd.size())},tb::kFloat32);
        auto y=ssm_conv_silu_cuda(x,w,b,T).cpu().contiguous();
        for(int n=0;n<B;++n)for(int t=0;t<T;++t)for(int c=0;c<C;++c) {
            double sum=0;
            for(int j=0;j<K;++j)sum+=double(xd[(size_t(n)*(T+K-1)+t+j)*C+c])*wd[layout==2?j*C+c:c*K+j];
            if(bias)sum+=bd[c];
            const double expected=sum/(1+std::exp(-sum)),actual=y.data_ptr<float>()[(size_t(n)*T+t)*C+c];
            if(!std::isfinite(actual) || std::abs(actual-expected)>2e-7)throw std::runtime_error("chunked convolution differs from FP64 oracle");
        }
    }
}
static void normalization() {
    for(auto type:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})
    for(int width:{32,128,256,2560,4096}) {
        std::vector<float> data(3*width),weights(width);
        for(size_t i=0;i<data.size();++i)data[i]=std::sin(float(i)*.23f)*.7f;
        for(int i=0;i<width;++i)weights[i]=.8f+std::cos(float(i)*.17f)*.2f;
        auto x=floats(data,{3,width},type),w=floats(weights,{width},tb::kFloat32);
        for(double offset:{0.,1.}) {
            auto old=rms_norm_offset_cuda(x.to(tb::kFloat32),w,1e-6,offset).cpu().contiguous();
            auto fused=rms_norm_offset_cuda(x,w,1e-6,offset).cpu().contiguous();
            for(size_t i=0;i<data.size();++i)
                if(old.data_ptr<float>()[i]!=fused.data_ptr<float>()[i])
                    throw std::runtime_error("typed RMSNorm changed FP32 rounding");
        }
        if(type==tb::kFloat32) {
            auto pair=l2_norm_pair_cuda(x,x,1e-6);
            auto old=l2_norm_cuda(x,1e-6).cpu().contiguous();
            for(auto t:pair) {
                auto result=t.cpu().contiguous();
                for(size_t i=0;i<data.size();++i)
                    if(result.data_ptr<float>()[i]!=old.data_ptr<float>()[i])
                        throw std::runtime_error("paired L2 norm changed reduction semantics");
            }
        }
    }
    for(int width:{32,64,128,256}) {
        const int rows=6;
        std::vector<float> data(rows*width),gates(data.size()),weights(width);
        for(size_t i=0;i<data.size();++i) {data[i]=std::sin(float(i)*.13f)*.7f;gates[i]=std::cos(float(i)*.31f)*12.f;}
        for(int i=0;i<width;++i)weights[i]=.9f+std::sin(float(i)*.37f)*.2f;
        auto x=floats(data,{2,3,1,width},tb::kFloat32),g=floats(gates,{2,1,3*width},tb::kFloat32),w=floats(weights,{width},tb::kFloat32);
        auto normalized=rms_norm_cuda(x.reshape({rows,width}),w,1e-6).reshape({2,1,3*width});
        auto old=(normalized*tb::silu(g)).cpu().contiguous();
        auto fused=gdn_rms_silu_decode_cuda(x,g,w,1e-6).cpu().contiguous();
        for(size_t i=0;i<data.size();++i)
            if(old.data_ptr<float>()[i]!=fused.data_ptr<float>()[i])
                throw std::runtime_error("fused GDN norm/SiLU changed separate FP32 operations");
    }
}
static void input_gate() {
    for(auto type:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})
    for(auto gate_type:{tb::kFloat16,tb::kBFloat16,tb::kFloat32})
    for(bool strided:{false,true}) {
        constexpr int width=29;
        std::vector<float> values(3*width),gates(values.size());
        for(size_t i=0;i<values.size();++i){values[i]=std::sin(float(i)*.31f)*2.f;gates[i]=std::cos(float(i)*.17f)*17.f;}
        auto x=floats(values,{3,width},type);
        auto gate=floats(gates,strided?std::vector<int64_t>{width,3}:std::vector<int64_t>{3,width},gate_type);
        if(strided)gate=gate.transpose(0,1);
        auto old_gate=gate.to(type);
        for(int mode:{1,2}) {
            auto expected=(x*(mode==1?tb::sigmoid(old_gate):tb::silu(old_gate))).cpu().to(tb::kFloat32).contiguous();
            auto actual=tq_gate_input(x,gate,mode).cpu().to(tb::kFloat32).contiguous();
            for(size_t i=0;i<values.size();++i)
                if(expected.data_ptr<float>()[i]!=actual.data_ptr<float>()[i]) {
                    std::cerr<<"input gate type="<<int(type)<<" gate="<<int(gate_type)<<" strided="<<strided<<" mode="<<mode<<'\n';
                    throw std::runtime_error("fused input gate changed original activation or multiplication rounding");
                }
        }
    }
}
int main() {
    if(hipSetDevice(0)!=hipSuccess)return 77;
    try {greedy();attention();cached_prefill();grouped_attention();recurrent_prefill();convolution();normalization();input_gate();std::cout<<"Greedy, attention/GDN/convolution FP64 oracles and exact norm/gate fusions passed\n";return 0;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
