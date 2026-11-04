#include "mfq_cuda_kv_hot_cache.h"
#include "mfq_cuda_resident_allocations.h"
#include "../ops/include/batch_selected_attention.h"
#include <algorithm>
#include <limits>
#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>

namespace mfq::cuda {
namespace tb=mfq_tensor_backend;
namespace {
using Stamp=unsigned long long;
constexpr Stamp busy=Stamp{1}<<63;
__global__ void reset_hot_cache(int32_t* mapping,int32_t* ids,Stamp* requested,Stamp* epochs,
    Stamp* control,int32_t* cold_mapping,int32_t* free_gpu,int32_t* cold_control,int maximum,int cells) {
    const int thread=blockIdx.x*blockDim.x+threadIdx.x;
    for(int i=thread;i<maximum;i+=blockDim.x*gridDim.x) {
        mapping[i]=-1;requested[i]=0;
        if(cold_mapping)cold_mapping[i]=-1;
        if(i<cells){ids[i]=-1;epochs[i]=0;if(free_gpu)free_gpu[i]=i;}
    }
    if(thread<7)control[thread]=0;
    if(cold_control && thread<3)cold_control[thread]=thread==2?cells:0;
}
__device__ int take_cold(int32_t* free_rows,int32_t* control,int capacity) {
    int n=atomicAdd(control+1,0);
    while(n>0) {const int previous=atomicCAS(control+1,n,n-1);if(previous==n)return free_rows[n-1];n=previous;}
    const int row=atomicAdd(control,1);assert(row<capacity);return row;
}
__global__ void discard_exclusive(const int64_t* positions,int count,int32_t* mapping,int32_t* ids,
        int32_t* cold_map,int32_t* free_rows,int32_t* free_gpu,int32_t* control) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        const int row=int(positions[i]),slot=mapping[row],cold=cold_map[row];
        if(slot>=0){ids[slot]=-1;free_gpu[atomicAdd(control+2,1)]=slot;}
        mapping[row]=-1;cold_map[row]=-1;
        if(cold>=0)free_rows[atomicAdd(control+1,1)]=cold;
    }
}
template<class Word> __global__ void evict_append_slots(Word* cold_keys,Word* cold_values,const Word* keys,const Word* values,
        const int64_t* positions,int count,int cells,int cold_cells,int heads,int kw,int vw,int32_t* mapping,int32_t* ids,
        int32_t* cold_map,int32_t* free_rows,int32_t* free_gpu,int32_t* control,Stamp* cursor) {
    const int lane=threadIdx.x%32,i=blockIdx.x*4+threadIdx.x/32;
    if(i>=min(count,cells))return;
    int slot=-1,row=-1,cold=-1;
    if(lane==0) {
        int n=atomicAdd(control+2,0);
        while(n>0){const int prior=atomicCAS(control+2,n,n-1);if(prior==n){slot=free_gpu[n-1];ids[slot]=-2;break;}n=prior;}
        while(slot<0){const int candidate=int(atomicAdd(cursor,Stamp{1})%cells),old=ids[candidate];
            if(old>=0 && atomicCAS(ids+candidate,old,-2)==old){slot=candidate;row=old;}}
        if(row>=0)cold=take_cold(free_rows,control,cold_cells);
    }
    slot=__shfl_sync(0xffffffff,slot,0);row=__shfl_sync(0xffffffff,row,0);cold=__shfl_sync(0xffffffff,cold,0);
    if(row>=0) {
        for(int x=lane;x<heads*kw;x+=32)cold_keys[(int64_t(x/kw)*cold_cells+cold)*kw+x%kw]=keys[(int64_t(x/kw)*cells+slot)*kw+x%kw];
        for(int x=lane;x<heads*vw;x+=32)cold_values[(int64_t(x/vw)*cold_cells+cold)*vw+x%vw]=values[(int64_t(x/vw)*cells+slot)*vw+x%vw];
    }
    __syncwarp();if(lane==0){__threadfence();if(row>=0){cold_map[row]=cold;mapping[row]=-1;}
        mapping[int(positions[count-min(count,cells)+i])]=slot;}
}
template<class Word> __global__ void append_exclusive(const Word* input_k,const Word* input_v,Word* keys,Word* values,
        Word* cold_keys,Word* cold_values,const int64_t* positions,int count,int cells,int cold_cells,int heads,int kw,int vw,
        int32_t* mapping,int32_t* ids,Stamp* epochs,int32_t* cold_map,int32_t* free_rows,int32_t* control) {
    const int lane=threadIdx.x%32;
    for(int i=blockIdx.x*4+threadIdx.x/32;i<count;i+=gridDim.x*4) {
        const int row=int(positions[i]);const bool resident=i>=count-cells;
        int slot=resident?mapping[row]:(lane==0?take_cold(free_rows,control,cold_cells):0);
        slot=__shfl_sync(0xffffffff,slot,0);
        const int stride=resident?cells:cold_cells;
        Word* out_k=resident?keys:cold_keys;Word* out_v=resident?values:cold_values;
        for(int x=lane;x<heads*kw;x+=32)out_k[(int64_t(x/kw)*stride+slot)*kw+x%kw]=input_k[(int64_t(x/kw)*count+i)*kw+x%kw];
        for(int x=lane;x<heads*vw;x+=32)out_v[(int64_t(x/vw)*stride+slot)*vw+x%vw]=input_v[(int64_t(x/vw)*count+i)*vw+x%vw];
        __syncwarp();if(lane==0){__threadfence();if(resident){mapping[row]=slot;ids[slot]=row;epochs[slot]=0;}else cold_map[row]=slot;}
    }
}
template<class Word> __global__ void gather_exclusive(Word* out,const Word* keys,const Word* cold,
        const int32_t* mapping,const int32_t* cold_map,int count,int heads,int width,int cells,int cold_cells) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(count)*heads*width;i+=int64_t(blockDim.x)*gridDim.x) {
        const int row=int(i/width)%count,head=int(i/(int64_t(count)*width)),col=int(i%width),slot=mapping[row];
        out[i]=slot>=0?keys[(int64_t(head)*cells+slot)*width+col]:cold[(int64_t(head)*cold_cells+cold_map[row])*width+col];
    }
}

template<class Word> __global__ void append_exclusive_decode(
        const void* input_k,const void* input_v,Word* keys,Word* values,Word* cold_keys,Word* cold_values,
        const int64_t* positions,int count,int cells,int cold_cells,int maximum,int heads,int kw,int vw,
        int64_t khs,int64_t kts,int64_t vhs,int64_t vts,int32_t* mapping,int32_t* ids,Stamp* epochs,
        int32_t* cold_map,int32_t* free_rows,int32_t* free_gpu,int32_t* control,Stamp* hot_control,
        const void* index_rows,void* index_storage,int index_width,int64_t index_stride,
        bool index_input_half,bool index_output_half,bool begin_prepare,bool float_keys,bool float_values,
        const int64_t* states,const int64_t* enabled,int rows,int field,int row) {
    if(states) {
        const bool batch = row < 0;
        if(batch) {
            row = int(blockIdx.x);
            positions += row;
            input_k = static_cast<const Word*>(input_k) + int64_t(row)*heads*kw;
            input_v = static_cast<const Word*>(input_v) + int64_t(row)*heads*vw;
        }
        if(!enabled[row])return;
        IndirectKvView cache{states, enabled, rows, field - 4, field};
        if(batch && (!cache.active(row) || positions[0] < 0 || positions[0] >= cache.capacity(row)))return;
        if(batch && !cache.exclusive(row)) {
            auto* target_k = cache.resolve<Word>(row, positions[0], 0, kw);
            auto* target_v = cache.resolve<Word>(row, positions[0], 0, vw, 1);
            if(!target_k || !target_v)return;
            for(int x=threadIdx.x;x<heads*kw;x+=blockDim.x)
                target_k[int64_t(x/kw)*cache.capacity(row)*kw+x%kw] = static_cast<const Word*>(input_k)[x];
            for(int x=threadIdx.x;x<heads*vw;x+=blockDim.x)
                target_v[int64_t(x/vw)*cache.capacity(row,1)*vw+x%vw] = static_cast<const Word*>(input_v)[x];
            return;
        }
        const auto at=[&](int offset){return 2*(int64_t(field+offset)*rows+row);};
        keys=reinterpret_cast<Word*>(states[at(HotKvStateFields::Keys)]);
        values=reinterpret_cast<Word*>(states[at(HotKvStateFields::Values)]);
        cold_keys=reinterpret_cast<Word*>(states[at(HotKvStateFields::ColdKeys)]);
        cold_values=reinterpret_cast<Word*>(states[at(HotKvStateFields::ColdValues)]);
        cells=int(states[at(HotKvStateFields::Keys)+1]);
        cold_cells=int(states[at(HotKvStateFields::ColdKeys)+1]);
        maximum=int(states[at(HotKvStateFields::Mapping)+1]);
        mapping=reinterpret_cast<int32_t*>(states[at(HotKvStateFields::Mapping)]);
        ids=reinterpret_cast<int32_t*>(states[at(HotKvStateFields::SlotIds)]);
        epochs=reinterpret_cast<Stamp*>(states[at(HotKvStateFields::Epochs)]);
        cold_map=reinterpret_cast<int32_t*>(states[at(HotKvStateFields::ColdMapping)]);
        free_rows=reinterpret_cast<int32_t*>(states[at(HotKvStateFields::FreeCold)]);
        free_gpu=reinterpret_cast<int32_t*>(states[at(HotKvStateFields::FreeGpu)]);
        control=reinterpret_cast<int32_t*>(states[at(HotKvStateFields::ColdControl)]);
        hot_control=reinterpret_cast<Stamp*>(states[at(HotKvStateFields::Control)]);
    }
    __shared__ int slots[32],old_rows[32],cold_rows[32];
    const int tid=threadIdx.x;
    const int team=count==1?128:count==2?64:32;
    const int lane=tid&(team-1),worker=tid/team;
    // The graph serializes users of this cache. Plan all rewritten rows first,
    // then reserve disjoint slots; no per-row global atomic or grid barrier.
    if(tid==0) {
        int free_count=control[1],gpu_count=control[2],next_cold=control[0];
        Stamp cursor=hot_control[1];
        for(int i=0;i<count;++i) {
            const int row=int(positions[i]);assert(row>=0 && row<maximum);
            const int slot=mapping[row],cold=cold_map[row];
            if(slot>=0){ids[slot]=-1;free_gpu[gpu_count++]=slot;}
            if(cold>=0)free_rows[free_count++]=cold;
            mapping[row]=-1;cold_map[row]=-1;
        }
        for(int i=0;i<count;++i) {
            int slot=-1,row=-1,cold=-1;
            if(gpu_count>0)slot=free_gpu[--gpu_count];
            else {
                for(int attempt=0;attempt<cells;++attempt) {
                    const int candidate=int(cursor++%Stamp(cells));
                    if(ids[candidate]>=0){slot=candidate;row=ids[slot];break;}
                }
                assert(slot>=0);
            }
            ids[slot]=-2;
            if(row>=0) {
                cold=free_count>0?free_rows[--free_count]:next_cold++;
                assert(cold<cold_cells);
            }
            slots[i]=slot;old_rows[i]=row;cold_rows[i]=cold;
        }
        control[0]=next_cold;control[1]=free_count;control[2]=gpu_count;hot_control[1]=cursor;
        if(begin_prepare){++hot_control[0];hot_control[2]=0;hot_control[6]=positions[count-1]<cells;}
    }
    __syncthreads();
    for(int i=worker;i<count;i+=blockDim.x/team) {
        const int slot=slots[i],cold=cold_rows[i];
        for(int x=lane;x<heads*kw;x+=team) {
            const int head=x/kw,col=x%kw;const auto at=(int64_t(head)*cells+slot)*kw+col;
            if(old_rows[i]>=0)cold_keys[(int64_t(head)*cold_cells+cold)*kw+col]=keys[at];
            const auto from=head*khs+i*kts+col;
            keys[at]=float_keys?Word(__half_as_ushort(__float2half_rn(static_cast<const float*>(input_k)[from]))):
                static_cast<const Word*>(input_k)[from];
        }
        for(int x=lane;x<heads*vw;x+=team) {
            const int head=x/vw,col=x%vw;const auto at=(int64_t(head)*cells+slot)*vw+col;
            if(old_rows[i]>=0)cold_values[(int64_t(head)*cold_cells+cold)*vw+col]=values[at];
            const auto from=head*vhs+i*vts+col;
            values[at]=float_values?Word(__half_as_ushort(__float2half_rn(static_cast<const float*>(input_v)[from]))):
                static_cast<const Word*>(input_v)[from];
        }
        for(int col=lane;col<index_width;col+=team) {
            const auto from=i*index_stride+col,to=positions[i]*index_width+col;
            const float value=index_input_half?__half2float(static_cast<const __half*>(index_rows)[from]):
                static_cast<const float*>(index_rows)[from];
            if(index_output_half)static_cast<__half*>(index_storage)[to]=__float2half_rn(value);
            else static_cast<float*>(index_storage)[to]=value;
        }
    }
    __syncthreads();
    if(tid==0)for(int i=0;i<count;++i) {
        if(old_rows[i]>=0){cold_map[old_rows[i]]=cold_rows[i];mapping[old_rows[i]]=-1;}
        const int row=int(positions[i]),slot=slots[i];mapping[row]=slot;ids[slot]=row;epochs[slot]=0;
    }
}
__global__ void begin(Stamp* control,const int64_t* positions,int tokens,int cells,bool skip_small) {
    if(!threadIdx.x){++control[0];control[2]=0;control[6]=skip_small && positions && positions[tokens-1]<cells;}
}
template<class Index> __global__ void invalidate_rows(const Index* positions,int64_t count,
        int32_t* mapping,int32_t* ids,Stamp* epochs,int maximum) {
    for(int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;at<count;at+=int64_t(gridDim.x)*blockDim.x) {
        const auto row=int64_t(positions[at]);if(row<0 || row>=maximum)continue;
        const auto slot=atomicExch(mapping+row,-1);
        if(slot>=0 && ids[slot]==row){ids[slot]=-1;epochs[slot]=0;}
    }
}
template<class Index,class Position> __global__ void mark_rows(const Index* selected,const Position* positions,
        int64_t count,int columns,int tokens,int maximum,int32_t* mapping,int32_t* slot_ids,
        Stamp* requested,Stamp* epochs,int32_t* queue,Stamp* control) {
    if(control[6])return;
    const auto epoch=control[0];
    for(int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;at<count;at+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t row=selected?int64_t(selected[at]):at;
        if(row<0 || row>=maximum || (!selected && positions && row>int64_t(positions[tokens-1])))continue;
        if(atomicExch(requested+row,epoch)==epoch)continue;
        queue[atomicAdd(control+2,Stamp{1})]=int32_t(row);
        const int slot=mapping[row];
        if(slot>=0 && slot_ids[slot]==row){epochs[slot]=epoch;atomicAdd(control+3,Stamp{1});}
    }
}
template<class Word> __global__ void resolve_rows(const Word* keys,const Word* values,Word* hot_keys,Word* hot_values,
        int32_t* mapping,int32_t* slot_ids,Stamp* epochs,const int32_t* queue,Stamp* control,
        int maximum,int cells,int heads,int kw,int vw,int32_t* cold_map,int cold_cells,int32_t* free_cold,int32_t* cold_control) {
    const int lane=threadIdx.x&31;
    for(auto work=int64_t(blockIdx.x)*4+threadIdx.x/32;Stamp(work)<control[2];work+=int64_t(gridDim.x)*4) {
    int row=-1,slot=-1,old=-1,cold=-1;
    if(lane==0 && Stamp(work)<control[2]) {
        row=queue[work];
        if(mapping[row]>=0)row=-1;
        else {
            mapping[row]=-2;
            const auto epoch=control[0];
            // A full query union can exceed the resident window. Bounded
            // probing leaves those rows in RAM rather than stalling a graph.
            for(int attempt=0;attempt<128;++attempt) {
                const int candidate=int(atomicAdd(control+1,Stamp{1})%Stamp(cells));
                const auto prior=atomicCAS(epochs+candidate,Stamp{0},Stamp{0});
                if(prior==epoch || (prior&busy))continue;
                if(atomicCAS(epochs+candidate,prior,epoch|busy)!=prior)continue;
                slot=candidate;old=slot_ids[slot];
                if(old>=0)atomicCAS(mapping+old,slot,-1);
                slot_ids[slot]=-1;break;
            }
            if(slot<0){mapping[row]=-1;atomicAdd(control+5,Stamp{1});}
        }
    }
    row=__shfl_sync(0xffffffff,row,0);slot=__shfl_sync(0xffffffff,slot,0);old=__shfl_sync(0xffffffff,old,0);
    if(row<0 || slot<0)continue;
    cold=cold_map?cold_map[row]:row;
    const int source_cells=cold_map?cold_cells:maximum;
    for(int at=lane;at<heads*kw;at+=32) {
        const int head=at/kw,col=at%kw;
        const auto to=(int64_t(head)*cells+slot)*kw+col,from=(int64_t(head)*source_cells+cold)*kw+col;
        const Word value=keys[from],prior=hot_keys[to];hot_keys[to]=value;
        if(cold_map && old>=0)const_cast<Word*>(keys)[from]=prior;
    }
    for(int at=lane;at<heads*vw;at+=32) {
        const int head=at/vw,col=at%vw;
        const auto to=(int64_t(head)*cells+slot)*vw+col,from=(int64_t(head)*source_cells+cold)*vw+col;
        const Word value=values[from],prior=hot_values[to];hot_values[to]=value;
        if(cold_map && old>=0)const_cast<Word*>(values)[from]=prior;
    }
    __syncwarp();
    if(lane==0){__threadfence();
        if(cold_map){cold_map[row]=-1;if(old>=0)cold_map[old]=cold;else free_cold[atomicAdd(cold_control+1,1)]=cold;}
        slot_ids[slot]=row;mapping[row]=slot;epochs[slot]=control[0];atomicAdd(control+4,Stamp{1});}
    }
}
template<class Word> __global__ void publish_rows(const Word* keys,const Word* values,Word* hot_keys,Word* hot_values,
        const int64_t* positions,int count,int maximum,int cells,int heads,int kw,int vw,
        int32_t* mapping,int32_t* slot_ids,Stamp* epochs,Stamp* control) {
    const int lane=threadIdx.x&31;
    const int at=blockIdx.x*4+threadIdx.x/32;
    if(at>=min(count,cells))return;
    const auto first=positions[0],last=positions[count-1];
    // Assign one warp to each slot and retain its last row in this append.
    const int slot=int((first+at)%cells);
    const int64_t row=last-(last-slot)%cells;
    if(row<first || row<0 || row>=maximum)return;
    if(lane==0) {const int old=slot_ids[slot];if(old>=0)atomicCAS(mapping+old,slot,-1);}
    for(int col=lane;col<heads*kw;col+=32) {
        const int head=col/kw,d=col%kw;
        hot_keys[(int64_t(head)*cells+slot)*kw+d]=keys[(int64_t(head)*maximum+row)*kw+d];
    }
    for(int col=lane;col<heads*vw;col+=32) {
        const int head=col/vw,d=col%vw;
        hot_values[(int64_t(head)*cells+slot)*vw+d]=values[(int64_t(head)*maximum+row)*vw+d];
    }
    __syncwarp();
    if(lane==0){__threadfence();slot_ids[slot]=int(row);mapping[row]=slot;epochs[slot]=0;atomicAdd(control+4,Stamp{1});}
}

}
HotKvCache HotKvCache::allocate(int64_t heads,int64_t maximum,int64_t cells,int64_t kw,int64_t vw,
        const tb::TensorOptions& options) {
    MFQ_RUNTIME_CHECK(heads>0 && maximum>0 && maximum<=INT_MAX && cells>0 && cells<=maximum,
        "invalid resident KV geometry");
    ResidentAllocationScope resident;
    HotKvCache result;
    result.keys=tb::empty({1,heads,cells,kw},options);result.values=tb::empty({1,heads,cells,vw},options);
    const auto integers=options.dtype(tb::kInt32),stamps=options.dtype(tb::kInt64);
    result.mapping=tb::full({maximum},-1,integers);result.slot_ids=tb::full({cells},-1,integers);
    result.requested=tb::zeros({maximum},stamps);result.epochs=tb::zeros({cells},stamps);
    result.queue=tb::empty({maximum},integers);result.control=tb::zeros({7},stamps);
    return result;
}
HotKvView HotKvCache::view() const {
    if(!defined())return {};
    return {keys.data_ptr(),values.data_ptr(),mapping.data_ptr<int32_t>(),keys.size(2),
        exclusive()?cold_mapping.data_ptr<int32_t>():nullptr,exclusive()?cold_keys.size(2):0};
}
void HotKvCache::set_cold_storage(const tb::Tensor& k,const tb::Tensor& v) {
    MFQ_RUNTIME_CHECK(defined() && k.size(2)==mapping.numel()-keys.size(2) && k.size(2)>0,"invalid exclusive KV capacity");
    cold_keys=k;cold_values=v;
    cold_mapping=tb::full({mapping.numel()},-1,mapping.options());
    free_cold=tb::empty({k.size(2)},mapping.options());free_gpu=tb::arange(keys.size(2),mapping.options());
    cold_control=tb::zeros({3},mapping.options());cold_control.narrow(0,2,1).fill_(keys.size(2));
}
void HotKvCache::append(const tb::Tensor& k,const tb::Tensor& v,const tb::Tensor& positions) {
    MFQ_RUNTIME_CHECK(exclusive() && k.size(2)==positions.numel() && v.size(2)==positions.numel(),"exclusive KV append geometry mismatch");
    if(append_decode(k,v,positions))return;
    auto source_k=k.contiguous(),source_v=v.contiguous(),pos=positions.to(tb::kInt64).contiguous();
    const int count=int(pos.numel()),cells=int(keys.size(2)),cold_cells=int(cold_keys.size(2)),heads=int(keys.size(1)),kw=int(keys.size(3)),vw=int(values.size(3));
    if(!count)return;
    auto* map=mapping.data_ptr<int32_t>();auto* ids=slot_ids.data_ptr<int32_t>();auto* cold=cold_mapping.data_ptr<int32_t>();
    auto* free=free_cold.data_ptr<int32_t>();auto* meta=cold_control.data_ptr<int32_t>();
    auto* gpu_free=free_gpu.data_ptr<int32_t>();
    discard_exclusive<<<std::min(65535,(count+255)/256),256,0,mfq_current_cuda_stream()>>>(pos.data_ptr<int64_t>(),count,map,ids,cold,free,gpu_free,meta);
    #define MFQ_APPEND_EXCLUSIVE(W) \
        evict_append_slots<<<(std::min(count,cells)+3)/4,128,0,mfq_current_cuda_stream()>>>( \
            reinterpret_cast<W*>(cold_keys.data_ptr()),reinterpret_cast<W*>(cold_values.data_ptr()), \
            reinterpret_cast<const W*>(keys.data_ptr()),reinterpret_cast<const W*>(values.data_ptr()),pos.data_ptr<int64_t>(), \
            count,cells,cold_cells,heads,kw,vw,map,ids,cold,free,gpu_free,meta,reinterpret_cast<Stamp*>(control.data_ptr())+1); \
        append_exclusive<<<std::min(65535,(count+3)/4),128,0,mfq_current_cuda_stream()>>>( \
            reinterpret_cast<const W*>(source_k.data_ptr()),reinterpret_cast<const W*>(source_v.data_ptr()), \
            reinterpret_cast<W*>(keys.data_ptr()),reinterpret_cast<W*>(values.data_ptr()), \
            reinterpret_cast<W*>(cold_keys.data_ptr()),reinterpret_cast<W*>(cold_values.data_ptr()),pos.data_ptr<int64_t>(), \
            count,cells,cold_cells,heads,kw,vw,map,ids,reinterpret_cast<Stamp*>(epochs.data_ptr()),cold,free,meta)
    if(keys.element_size()==2){MFQ_APPEND_EXCLUSIVE(uint16_t);}else{MFQ_APPEND_EXCLUSIVE(uint32_t);}
    #undef MFQ_APPEND_EXCLUSIVE
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}
bool HotKvCache::append_decode(const tb::Tensor& k,const tb::Tensor& v,const tb::Tensor& positions,
        const tb::Tensor& index_rows,const tb::Tensor& index_storage,bool begin_prepare,
        const BatchDecodeInputs* inputs,int64_t field,int64_t row) {
    const auto* option=std::getenv("MFQ_KV_APPEND_FUSED");
    if((!inputs && option && option[0]=='0') || !exclusive() || positions.numel()<1 || positions.numel()>32 ||
        positions.numel()>keys.size(2) || positions.scalar_type()!=tb::kInt64 || !positions.is_contiguous())return false;
    if(!inputs && !option && positions.numel()>8)return false;
    MFQ_RUNTIME_CHECK(!inputs || (!index_rows.defined() && !index_storage.defined() && !begin_prepare),
        "indirect KV append requires separate index updates");
    const int count=int(positions.numel()),cells=int(keys.size(2)),cold_cells=int(cold_keys.size(2)),heads=int(keys.size(1)),
        kw=int(keys.size(3)),vw=int(values.size(3));
    const auto compatible=[&](const tb::Tensor& source,const tb::Tensor& target,int width) {
        return source.is_cuda() && source.device()==target.device() && source.dim()==4 && source.size(0)==1 &&
            source.size(1)==heads && source.size(2)==count && source.size(3)==width && source.stride(3)==1 &&
            (source.scalar_type()==target.scalar_type() ||
                (source.scalar_type()==tb::kFloat32 && target.scalar_type()==tb::kFloat16));
    };
    if(!compatible(k,keys,kw) || !compatible(v,values,vw) || positions.device()!=keys.device())return false;
    int index_width=0;
    if(index_rows.defined() || index_storage.defined()) {
        const auto numeric=[](const tb::Tensor& t){return t.scalar_type()==tb::kFloat16 || t.scalar_type()==tb::kFloat32;};
        if(!index_rows.defined() || !index_storage.defined() || !numeric(index_rows) || !numeric(index_storage) ||
            index_rows.device()!=keys.device() || index_storage.device()!=keys.device() || index_rows.dim()!=3 ||
            index_storage.dim()!=3 || index_rows.size(0)!=1 || index_storage.size(0)!=1 || index_rows.size(1)!=count ||
            index_storage.size(1)!=mapping.numel() || index_rows.size(2)!=index_storage.size(2) ||
            index_rows.stride(2)!=1 || !index_storage.is_contiguous())return false;
        index_width=int(index_rows.size(2));
    }
    const auto launch=[&](auto word) {
        using Word=decltype(word);
        append_exclusive_decode<<<1,128,0,mfq_current_cuda_stream()>>>(
            k.data_ptr(),v.data_ptr(),
            static_cast<Word*>(keys.data_ptr()),static_cast<Word*>(values.data_ptr()),
            static_cast<Word*>(cold_keys.data_ptr()),static_cast<Word*>(cold_values.data_ptr()),positions.data_ptr<int64_t>(),
            count,cells,cold_cells,int(mapping.numel()),heads,kw,vw,k.stride(1),k.stride(2),v.stride(1),v.stride(2),
            mapping.data_ptr<int32_t>(),slot_ids.data_ptr<int32_t>(),reinterpret_cast<Stamp*>(epochs.data_ptr()),
            cold_mapping.data_ptr<int32_t>(),free_cold.data_ptr<int32_t>(),free_gpu.data_ptr<int32_t>(),
            cold_control.data_ptr<int32_t>(),reinterpret_cast<Stamp*>(control.data_ptr()),
            index_width?index_rows.data_ptr():nullptr,index_width?index_storage.data_ptr():nullptr,index_width,
            index_width?index_rows.stride(1):0,index_width && index_rows.scalar_type()==tb::kFloat16,
            index_width && index_storage.scalar_type()==tb::kFloat16,begin_prepare,
            k.scalar_type()==tb::kFloat32 && keys.scalar_type()==tb::kFloat16,
            v.scalar_type()==tb::kFloat32 && values.scalar_type()==tb::kFloat16,
            inputs?inputs->states().data_ptr<int64_t>():nullptr,
            inputs?inputs->enabled().data_ptr<int64_t>():nullptr,inputs?int(inputs->rows()):0,int(field),int(row));
    };
    if(keys.element_size()==2)launch(uint16_t{});else if(keys.element_size()==4)launch(uint32_t{});else return false;
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return true;
}
void batch_attention_store(const BatchDecodeInputs& inputs, int64_t field,
        const tb::Tensor& keys, const tb::Tensor& values) {
    MFQ_RUNTIME_CHECK(keys.dim() == 4 && values.dim() == 4 && keys.size(0) == inputs.rows() &&
        values.size(0) == inputs.rows() && keys.size(1) == 1 && values.size(1) == 1 &&
        keys.size(2) == values.size(2) && keys.scalar_type() == values.scalar_type(),
        "invalid independent KV update geometry");
    auto key = keys.contiguous(), value = values.contiguous();
    const int heads = int(keys.size(2)), kw = int(keys.size(3)), vw = int(values.size(3));
    const auto launch = [&](auto word) {
        using Word = decltype(word);
        append_exclusive_decode<<<unsigned(inputs.rows()), 128, 0, mfq_current_cuda_stream()>>>(
            key.data_ptr(), value.data_ptr(), static_cast<Word*>(nullptr), static_cast<Word*>(nullptr),
            static_cast<Word*>(nullptr), static_cast<Word*>(nullptr), inputs.positions().data_ptr<int64_t>(),
            1, 0, 0, 0, heads, kw, vw, kw, heads*kw, vw, heads*vw,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, 0, 0, false, false, false, false, false,
            inputs.states().data_ptr<int64_t>(), inputs.enabled().data_ptr<int64_t>(),
            int(inputs.rows()), int(field + 4), -1);
    };
    if(key.scalar_type() == tb::kFloat16)
        launch(uint16_t{});
    else {
        MFQ_RUNTIME_CHECK(key.scalar_type() == tb::kInt32, "independent KV requires F16 or packed I32");
        launch(uint32_t{});
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}
void HotKvCache::bind_decode_state(BatchDecodeInputs& inputs,int64_t field,int64_t row) const {
    MFQ_RUNTIME_CHECK(exclusive(),"indirect KV state requires exclusive tiers");
    const tb::Tensor* tensors[]={&keys,&values,&cold_keys,&cold_values,&mapping,&slot_ids,&epochs,
        &cold_mapping,&free_cold,&free_gpu,&cold_control,&control};
    for(int offset=0;offset<HotKvStateFields::Count;++offset)
        inputs.state(row,field+offset,*tensors[offset],offset<HotKvStateFields::Mapping?2:-1);
}
std::pair<tb::Tensor,tb::Tensor> HotKvCache::gather(int64_t count) const {
    MFQ_RUNTIME_CHECK(exclusive() && count>0 && count<=mapping.numel(),"invalid exclusive KV read");
    auto read=[&](const tb::Tensor& gpu,const tb::Tensor& ram){
        auto out=tb::empty({1,gpu.size(1),count,gpu.size(3)},gpu.options());
        const int blocks=int(std::min<int64_t>(65535,(out.numel()+255)/256));
        #define MFQ_GATHER_EXCLUSIVE(W) gather_exclusive<<<blocks,256,0,mfq_current_cuda_stream()>>>( \
            reinterpret_cast<W*>(out.data_ptr()),reinterpret_cast<const W*>(gpu.data_ptr()),reinterpret_cast<const W*>(ram.data_ptr()), \
            mapping.data_ptr<int32_t>(),cold_mapping.data_ptr<int32_t>(),int(count),int(gpu.size(1)),int(gpu.size(3)),int(gpu.size(2)),int(ram.size(2)))
        if(gpu.element_size()==2){MFQ_GATHER_EXCLUSIVE(uint16_t);}else{MFQ_GATHER_EXCLUSIVE(uint32_t);}
        #undef MFQ_GATHER_EXCLUSIVE
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();return out;
    };
    return {read(keys,cold_keys),read(values,cold_values)};
}
std::size_t HotKvCache::bytes() const {
    std::size_t bytes=0;
    for(const auto* tensor:{&keys,&values,&mapping,&slot_ids,&requested,&epochs,&queue,&control,&cold_mapping,&free_cold,&free_gpu,&cold_control})
        if(tensor->defined())bytes+=tensor->numel()*tensor->element_size();
    return bytes;
}
void HotKvCache::reset() {
    if(!defined())return;
    // Reuse every preallocated table, including the GPU free-slot list.
    // Reset must work at the memory ceiling without a temporary arange.
    MfqCudaGuard guard(mapping.device());
    reset_hot_cache<<<std::min<int64_t>(65535,(mapping.numel()+255)/256),256,0,mfq_current_cuda_stream()>>>(
        mapping.data_ptr<int32_t>(),slot_ids.data_ptr<int32_t>(),
        reinterpret_cast<Stamp*>(requested.data_ptr()),reinterpret_cast<Stamp*>(epochs.data_ptr()),
        reinterpret_cast<Stamp*>(control.data_ptr()),exclusive()?cold_mapping.data_ptr<int32_t>():nullptr,
        exclusive()?free_gpu.data_ptr<int32_t>():nullptr,exclusive()?cold_control.data_ptr<int32_t>():nullptr,
        int(mapping.numel()),int(keys.size(2)));
    MFQ_CUDA_CHECK(cudaGetLastError());
}
void HotKvCache::invalidate(const tb::Tensor& positions) {
    if(!defined() || !positions.numel())return;
    MFQ_RUNTIME_CHECK(positions.is_cuda() && positions.is_contiguous() &&
        (positions.scalar_type()==tb::kInt32 || positions.scalar_type()==tb::kInt64),"invalid KV update positions");
    const int blocks=int(std::min<int64_t>(65535,(positions.numel()+255)/256));
    if(positions.scalar_type()==tb::kInt64)invalidate_rows<<<blocks,256,0,mfq_current_cuda_stream()>>>(
        positions.data_ptr<int64_t>(),positions.numel(),mapping.data_ptr<int32_t>(),slot_ids.data_ptr<int32_t>(),
        reinterpret_cast<Stamp*>(epochs.data_ptr()),int(mapping.numel()));
    else invalidate_rows<<<blocks,256,0,mfq_current_cuda_stream()>>>(positions.data_ptr<int32_t>(),positions.numel(),
        mapping.data_ptr<int32_t>(),slot_ids.data_ptr<int32_t>(),reinterpret_cast<Stamp*>(epochs.data_ptr()),int(mapping.numel()));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}
void HotKvCache::publish(const tb::Tensor& source_keys,const tb::Tensor& source_values,const tb::Tensor& positions) {
    if(!defined() || !positions.numel())return;
    MFQ_RUNTIME_CHECK(source_keys.is_contiguous() && source_values.is_contiguous(),"resident KV publication source must be contiguous");
    auto indices=positions.to(tb::kInt64).contiguous();
    invalidate(indices);
    const int grid=int((std::min<int64_t>(indices.numel(),keys.size(2))+3)/4);
    #define MFQ_PUBLISH_KV(Word) publish_rows<<<grid,128,0,mfq_current_cuda_stream()>>>( \
        reinterpret_cast<const Word*>(source_keys.data_ptr()),reinterpret_cast<const Word*>(source_values.data_ptr()), \
        reinterpret_cast<Word*>(keys.data_ptr()),reinterpret_cast<Word*>(values.data_ptr()),indices.data_ptr<int64_t>(), \
        int(indices.numel()),int(mapping.numel()),int(keys.size(2)),int(keys.size(1)),int(keys.size(3)),int(values.size(3)), \
        mapping.data_ptr<int32_t>(),slot_ids.data_ptr<int32_t>(),reinterpret_cast<Stamp*>(epochs.data_ptr()), \
        reinterpret_cast<Stamp*>(control.data_ptr()))
    if(keys.element_size()==2){MFQ_PUBLISH_KV(uint16_t);}else{MFQ_PUBLISH_KV(uint32_t);}
    #undef MFQ_PUBLISH_KV
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

void HotKvCache::prepare(const tb::Tensor& source_keys,const tb::Tensor& source_values,
        const tb::Tensor& selected,const tb::Tensor& positions,int64_t columns,bool skip_small_prefix,bool epoch_ready) {
    if(!defined())return;
    MFQ_RUNTIME_CHECK(source_keys.is_contiguous() && source_values.is_contiguous() && source_keys.size(0)==1 &&
        source_keys.size(2)==mapping.numel() && source_keys.scalar_type()==keys.scalar_type() &&
        source_values.scalar_type()==values.scalar_type(),"resident KV source geometry mismatch");
    auto indices=selected.defined()?selected.to(tb::kInt32).contiguous():tb::Tensor{};
    auto boundaries=positions.defined()?positions.to(tb::kInt64).contiguous():tb::Tensor{};
    const auto count=indices.defined()?indices.numel():columns;
    MFQ_RUNTIME_CHECK(count>0 && columns<=mapping.numel() &&
        (!indices.defined() || (indices.size(0)==1 && indices.dim()==3)),"invalid resident KV selection");
    auto* meta=reinterpret_cast<Stamp*>(control.data_ptr());
    const int blocks=int(std::min<int64_t>(65535,(count+255)/256));
    if(!epoch_ready)begin<<<1,1,0,mfq_current_cuda_stream()>>>(meta,boundaries.defined()?boundaries.data_ptr<int64_t>():nullptr,
        boundaries.defined()?int(boundaries.numel()):0,int(keys.size(2)),skip_small_prefix);
    mark_rows<int32_t,int64_t><<<blocks,256,0,mfq_current_cuda_stream()>>>(
        indices.defined()?indices.data_ptr<int32_t>():nullptr,boundaries.defined()?boundaries.data_ptr<int64_t>():nullptr,
        count,int(keys.size(2)),boundaries.defined()?int(boundaries.numel()):0,int(mapping.numel()),mapping.data_ptr<int32_t>(),
        slot_ids.data_ptr<int32_t>(),reinterpret_cast<Stamp*>(requested.data_ptr()),reinterpret_cast<Stamp*>(epochs.data_ptr()),
        queue.data_ptr<int32_t>(),meta);
    const int grid=int(std::min<int64_t>(128,(std::min<int64_t>(count,mapping.numel())+3)/4));
    #define MFQ_RESOLVE_KV(Word) resolve_rows<<<grid,128,0,mfq_current_cuda_stream()>>>( \
        reinterpret_cast<const Word*>(source_keys.data_ptr()),reinterpret_cast<const Word*>(source_values.data_ptr()), \
        reinterpret_cast<Word*>(keys.data_ptr()),reinterpret_cast<Word*>(values.data_ptr()),mapping.data_ptr<int32_t>(), \
        slot_ids.data_ptr<int32_t>(),reinterpret_cast<Stamp*>(epochs.data_ptr()),queue.data_ptr<int32_t>(),meta, \
        int(mapping.numel()),int(keys.size(2)),int(keys.size(1)),int(keys.size(3)),int(values.size(3)), \
        exclusive()?cold_mapping.data_ptr<int32_t>():nullptr,exclusive()?int(cold_keys.size(2)):0, \
        exclusive()?free_cold.data_ptr<int32_t>():nullptr,exclusive()?cold_control.data_ptr<int32_t>():nullptr)
    if(keys.element_size()==2){MFQ_RESOLVE_KV(uint16_t);}else{MFQ_RESOLVE_KV(uint32_t);}
    #undef MFQ_RESOLVE_KV
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}
}
