#pragma once
namespace mfq_hip_attention {
template<int D,bool Split>__global__ __launch_bounds__(64) void decode_half2(
        const __half* q,const __half* k,const __half* v,const int64_t* lengths,
        __half* out,float* partial_o,float* partial_m,float* partial_l,
        int heads,int kv_heads,int capacity,int parts,int workspace_parts,float scale,int dynamic) {
    using H2=_Float16 __attribute__((ext_vector_type(2)));
    constexpr int Columns=16,Tokens=4,Pairs=D/(Columns*2);
    const int lane=threadIdx.x,column=lane%Columns,token_lane=lane/Columns;
    const int part=blockIdx.x%parts,row=blockIdx.x/parts,batch=row/heads,head=(row%heads)/(heads/kv_heads);
    const int length=max(0,min(int(lengths[batch]),capacity));
    const int active=attention_decode_active_parts(length,parts,dynamic);
    const size_t stat=size_t(row)*workspace_parts+part;
    if(part>=active) {
        if constexpr(Split) {
            #pragma unroll
            for(int i=0;i<Pairs;++i)if(token_lane==0) {
                partial_o[stat*D+2*(column+i*Columns)]=0.f;
                partial_o[stat*D+2*(column+i*Columns)+1]=0.f;
            }
            if(!lane){partial_m[stat]=-1e30f;partial_l[stat]=0.f;}
        }
        return;
    }
    const int begin=int(int64_t(length)*part/active),end=int(int64_t(length)*(part+1)/active);
    H2 query[Pairs];float2 result[Pairs];
    #pragma unroll
    for(int i=0;i<Pairs;++i) {
        query[i]=__builtin_bit_cast(H2,reinterpret_cast<const uint32_t*>(q)[size_t(row)*(D/2)+column+i*Columns]);
        result[i]={0.f,0.f};
    }
    const size_t base=size_t(batch*kv_heads+head)*capacity*(D/2);
    float maximum=-1e30f,denominator=0.f;
    for(int tile=begin;tile<end;tile+=Tokens) {
        const int token=tile+token_lane;
        const size_t index=base+size_t(min(token,end-1))*(D/2)+column;
        float dot=0;
        #pragma unroll
        for(int i=0;i<Pairs;++i) {
            const H2 value=__builtin_bit_cast(H2,reinterpret_cast<const uint32_t*>(k)[index+i*Columns]);
            dot=__builtin_amdgcn_fdot2(query[i],value,dot,false);
        }
        for(int d=8;d;d/=2)dot+=__shfl_xor(dot,d,Columns);
        const float score=token<end?dot*scale:-1e30f;
        float tile_max=score;
        for(int d=Columns;d<64;d*=2)tile_max=fmaxf(tile_max,__shfl_xor(tile_max,d,64));
        const float next=fmaxf(maximum,tile_max),correction=expf(maximum-next);
        const float probability=token<end?expf(score-next):0.f;
        float sum=probability;
        for(int d=Columns;d<64;d*=2)sum+=__shfl_xor(sum,d,64);
        #pragma unroll
        for(int i=0;i<Pairs;++i) {
            const auto packed=reinterpret_cast<const uint32_t*>(v)[index+i*Columns];
            const auto value=__half22float2(__builtin_bit_cast(__half2,packed));
            result[i].x=result[i].x*correction+probability*value.x;
            result[i].y=result[i].y*correction+probability*value.y;
        }
        maximum=next;denominator=denominator*correction+sum;
    }
    #pragma unroll
    for(int i=0;i<Pairs;++i) {
        float a=result[i].x,b=result[i].y;
        for(int d=Columns;d<64;d*=2){a+=__shfl_xor(a,d,64);b+=__shfl_xor(b,d,64);}
        if(!token_lane) {
            const int c=2*(column+i*Columns);
            if constexpr(Split){partial_o[stat*D+c]=a;partial_o[stat*D+c+1]=b;}
            else {
                const float inverse=1.f/(denominator>0?denominator:1.f);
                out[size_t(row)*D+c]=__float2half_rn(a*inverse);out[size_t(row)*D+c+1]=__float2half_rn(b*inverse);
            }
        }
    }
    if constexpr(Split)if(!lane){partial_m[stat]=maximum;partial_l[stat]=denominator;}
}
}
