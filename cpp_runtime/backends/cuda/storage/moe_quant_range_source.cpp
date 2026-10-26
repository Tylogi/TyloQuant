#include "../runtime/execution_options.h"
#include "moe_quant_range_source.h"
#include "moe_pinned_expert_arena.h"
#include "cpu_projection_rows.h"
#include "mfq/cpu_expert_pool.h"
#include "mfq/host_copy.h"
#include "quant_linear.h"
#include "format.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace tb=mfq_tensor_backend;
using mfq::cuda::quant_format::cpu_u8_tensor;
using mfq::cuda::quant_format::cpu_f16_to_f32_tensor;

namespace {
void copy_neuron_anchors(std::vector<uint16_t>& anchors,const tb::Tensor& target) {
    const auto count=int64_t(anchors.size());
    auto output=tb::from_blob(target.data_ptr(),{count},
        tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
    output.copy_(tb::from_blob(anchors.data(),{count},
        tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat16)));
}
#ifdef MFQ_NATIVE_CUDA_RUNTIME
struct NvqFieldOwner {
    mfq::NvqRows::Fields fields;
    explicit NvqFieldOwner(mfq::NvqRows::Fields value):fields(std::move(value)) {
        mfq::cuda::charge_tensor_host_bytes(fields.nbytes());
    }
    ~NvqFieldOwner(){mfq::cuda::tensor_host_bytes.fetch_sub(fields.nbytes());}
};
tb::Tensor nvq_field_view(const std::shared_ptr<NvqFieldOwner>& owner,std::vector<uint8_t>& bytes) {
    if(bytes.empty())return tb::empty({0},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
    auto storage=std::make_shared<tb::TensorStorage>();
    const auto device=tb::Device{tb::kCPU,0};
    storage->owner=owner;storage->base=bytes.data();storage->bytes=bytes.size();storage->device=device;
    const std::array<int64_t,1> shape{int64_t(bytes.size())};
    return tb::Tensor(storage,tb::make_contiguous_view(bytes.data(),shape,tb::kUInt8,device));
}
#endif
std::vector<tb::Tensor> expert_field_tensors(const MixedMoePool& pool) {
    std::vector<tb::Tensor> fields;
    if (pool.family==MixedMoeFamily::Nint)
        fields={pool.nint.q_packed,pool.nint.row_q_bits,pool.nint.row_q_bit_offsets,
            pool.nint.sub_scale,pool.nint.sub_min,pool.nint.neuron_scale,pool.nint.neuron_min};
    else if(pool.family==MixedMoeFamily::Mxfp4)fields={pool.mxfp4.values,pool.mxfp4.scales};
    else fields={pool.nvq.indices_packed,pool.nvq.aux_packed,pool.nvq.sub_scale_packed,pool.nvq.neuron_scale};
    return fields;
}
std::size_t expert_fields_bytes(const MixedMoePool& pool) {
    std::size_t bytes=0;
    for (const auto& field:expert_field_tensors(pool)) bytes+=field.numel()*field.element_size();
    return bytes;
}
}

MoeQuantRangeSource::MoeQuantRangeSource(std::shared_ptr<mfq::MfeQuantExpertStore> store)
    :store_(std::move(store)),metadata_(std::make_shared<MixedMoeRuntime>()) {
    if (!store_) throw std::invalid_argument("missing quantized expert range source");
    const auto* profile=std::getenv("MFQ_PROFILE_SSD_EXPERTS");
    profile_ssd_=profile && profile[0]=='1';
    const auto dense=mfq::cuda::runtime_options::moe_nvq_dense();
    const auto preload=mfq::cuda::runtime_options::moe_preload_all();
    const auto sealed=mfq::cuda::runtime_options::moe_assert_resident();
    const bool complete=preload && *preload && sealed && *sealed;
    dense_groups_=dense ? *dense : complete;
    if(dense_groups_ && !complete)
        throw std::invalid_argument("compact NVQ requires complete sealed preload");
    metadata_->n_experts=store_->num_experts();
    metadata_->out_per_expert=store_->out_per_expert();
    metadata_->neuron_len=store_->neuron_len();
    q_strides_.assign(store_->pool_count(),0);
    field_bytes_.assign(store_->pool_count(),0);
    field_layouts_.resize(store_->pool_count());arena_bytes_.resize(store_->pool_count());
    gpu_resident_=std::make_unique<std::atomic_bool[]>(store_->num_experts());
    for (int expert=0; expert<store_->num_experts(); ++expert) gpu_resident_[expert].store(false);
    for (std::size_t index=0; index<store_->pool_count(); ++index) {
        const auto& ids=store_->pool_expert_ids(index);
        if (store_->expert_dtype(ids.front())=="NINT") {
            std::uint64_t maximum=0;
            for (int expert:ids) maximum=std::max(maximum,store_->expert_values_bits(expert));
            const auto bytes=(maximum+7)/8+8;
            if (bytes>std::numeric_limits<std::int64_t>::max()) throw std::overflow_error("NINT expert stride overflow");
            q_strides_[index]=static_cast<std::int64_t>(bytes);
        }
        auto pool=read_expert(ids.front());
        field_bytes_[index]=expert_fields_bytes(pool);
        std::size_t packed=0;
        for(const auto& field:expert_field_tensors(pool)) {
            packed=(packed+15)&~std::size_t(15);
            const auto bytes=std::size_t(field.numel()*field.element_size());
            field_layouts_[index].push_back({field.sizes().vec(),field.scalar_type(),packed,bytes});
            packed+=bytes;
        }
        arena_bytes_[index]=(packed+15)&~std::size_t(15);
        pool.local_experts=static_cast<int>(ids.size());
        std::vector<std::int32_t> local(store_->num_experts(),-1);
        for (std::size_t j=0; j<ids.size(); ++j) local[ids[j]]=static_cast<std::int32_t>(j);
        pool.expert_local=tb::tensor(local,tb::TensorOptions().dtype(tb::kInt32));
        metadata_->pools.push_back(std::move(pool));
    }
}

MixedMoePool MoeQuantRangeSource::read_expert(int expert) const {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    if(pinned_arena_) {
        auto pool=allocate_expert(expert);
        copy_expert_fields(expert,expert_field_tensors(pool));
        return pool;
    }
#endif
    return read_expert_canonical(expert);
}

MixedMoePool MoeQuantRangeSource::allocate_expert(int expert) const {
    const auto index=store_->expert_pool(expert);
    std::vector<tb::Tensor> fields;
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    {
        auto block=pinned_arena_?pinned_arena_->allocate(arena_bytes_.at(index)):nullptr;
        const auto& layouts=field_layouts_.at(index);
        std::vector<std::shared_ptr<MoePinnedExpertArena::Block>> pinned_fields;
        std::vector<std::size_t> pageable_offsets;
        std::shared_ptr<tb::Tensor> pageable;
        if(!block) {
            // Large expert images can fail despite smaller free intervals.
            // Retain any fields that fit there; only the remaining fields
            // need pageable storage and the later DMA staging copy.
            pinned_fields.resize(layouts.size());pageable_offsets.resize(layouts.size());
            std::size_t bytes=0,pinned_bytes=0;
            for(std::size_t f=0;f<layouts.size();++f) {
                if(pinned_arena_)pinned_fields[f]=pinned_arena_->allocate(layouts[f].bytes);
                if(pinned_fields[f])pinned_bytes+=layouts[f].bytes;
                else {
                    bytes=(bytes+15)&~std::size_t(15);pageable_offsets[f]=bytes;bytes+=layouts[f].bytes;
                }
            }
            pageable=std::make_shared<tb::Tensor>(tb::empty({std::int64_t(bytes)},
                tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8)));
            if(pinned_bytes) {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                    "moe_partial_pinned_expert scope=first_fragmented_allocation enabled=1 pinned_bytes=%zu pageable_bytes=%zu\n",
                    pinned_bytes,bytes);
            }
        }
        for(std::size_t f=0;f<layouts.size();++f) {
            const auto& layout=layouts[f];
            auto storage=std::make_shared<tb::TensorStorage>();
            const auto device=tb::Device{tb::kCPU,0};
            std::uint8_t* address=nullptr;
            if(block){storage->owner=block;address=block->data()+layout.offset;}
            else if(pinned_fields[f]){storage->owner=pinned_fields[f];address=pinned_fields[f]->data();}
            else {
                storage->owner=pageable;
                if(layout.bytes)address=static_cast<std::uint8_t*>(pageable->data_ptr())+pageable_offsets[f];
            }
            storage->base=address;storage->bytes=layout.bytes;
            storage->device=device;storage->pinned=bool(block) || bool(pinned_fields[f]);
            fields.emplace_back(storage,tb::make_contiguous_view(address,layout.shape,layout.dtype,device));
        }
    }
#else
    for(const auto& layout:field_layouts_.at(index))
        fields.push_back(tb::empty(layout.shape,tb::TensorOptions().device(tb::kCPU).dtype(layout.dtype)));
#endif
    auto pool=metadata_->pools.at(index);pool.local_experts=1;pool.expert_local={};
    if(pool.family==MixedMoeFamily::Nvq) {
        auto& w=pool.nvq;w.workspaces.clear();
        w.indices_packed=fields[0];w.aux_packed=fields[1];w.sub_scale_packed=fields[2];w.neuron_scale=fields[3];
    } else if(pool.family==MixedMoeFamily::Mxfp4) {
        pool.mxfp4.values=fields[0];pool.mxfp4.scales=fields[1];
    } else {
        auto& w=pool.nint;w.workspaces.clear();
        w.q_packed=fields[0];w.row_q_bits=fields[1];w.row_q_bit_offsets=fields[2];
        w.sub_scale=fields[3];w.sub_min=fields[4];w.neuron_scale=fields[5];w.neuron_min=fields[6];
    }
    return pool;
}

MixedMoePool MoeQuantRangeSource::read_expert_canonical(int expert) const {
    if (disk_sealed_.load()) {
        ++sealed_disk_attempts_;
        throw std::runtime_error("expert disk read forbidden after complete residency preload");
    }
    using Clock=std::chrono::steady_clock;
    const bool profile=profile_ssd_ && preloaded_.load();
    const auto begin=profile?Clock::now():Clock::time_point{};
    const auto index=store_->expert_pool(expert);
    auto fields=!dense_groups_ && std::size_t(index)<metadata_->pools.size()?store_->read_nvq_fields(expert):std::nullopt;
    const auto encoded=fields?mfq::MfeQuantExpert{}:store_->read_expert(expert);
    const auto source_bytes=fields?fields->source_nbytes:encoded.payload.size();
    const auto read_done=profile?Clock::now():Clock::time_point{};
    if (preloaded_.load()) {
        ++disk_reads_after_preload_;
        disk_bytes_after_preload_.fetch_add(fields?fields->source_nbytes:encoded.payload.size(),std::memory_order_relaxed);
    }
    MixedMoePool result;
    if(fields) {
        result=metadata_->pools[index];result.local_experts=1;result.expert_local={};
        auto& w=result.nvq;w.workspaces.clear();
        w.neuron_scale=cpu_f16_to_f32_tensor(fields->neuron_scale,store_->out_per_expert());
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        // Tensor owners retain the actual read buffers through every DMA
        // consumer. Avoid copying each packed stream into another CPU tensor.
        fields->neuron_scale=std::vector<uint16_t>{};
        auto owner=std::make_shared<NvqFieldOwner>(std::move(*fields));
        w.indices_packed=nvq_field_view(owner,owner->fields.indices);
        w.aux_packed=nvq_field_view(owner,owner->fields.aux);
        w.sub_scale_packed=nvq_field_view(owner,owner->fields.states);
#else
        w.indices_packed=cpu_u8_tensor(fields->indices,{int64_t(fields->indices.size())});
        w.aux_packed=cpu_u8_tensor(fields->aux,{int64_t(fields->aux.size())});
        w.sub_scale_packed=cpu_u8_tensor(fields->states,{int64_t(fields->states.size())});
#endif
        ++materializations_;
    } else result=decode_expert(expert,encoded);
    if(profile) {
        const auto encoded_time=std::chrono::duration_cast<std::chrono::nanoseconds>(read_done-begin).count();
        encoded_ns_.fetch_add(encoded_time,std::memory_order_relaxed);
        const auto materialize=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-read_done).count();
        materialize_ns_.fetch_add(materialize,std::memory_order_relaxed);
        if(result.family!=MixedMoeFamily::Mxfp4)
            (result.family==MixedMoeFamily::Nint?nint_materialize_ns_:nvq_materialize_ns_).fetch_add(materialize,std::memory_order_relaxed);
        record_format_io(result.family==MixedMoeFamily::Nint?0:result.family==MixedMoeFamily::Nvq?result.nvq.kernel_format:-1,
            source_bytes,encoded_time,materialize);
    }
    return result;
}

void MoeQuantRangeSource::copy_expert_fields(int expert,const std::vector<tb::Tensor>& targets,bool touch_cache) const {
    const auto index=store_->expert_pool(expert);
    std::size_t bytes=0;
    for(const auto& target:targets) {
        if(!target.is_cpu())throw std::invalid_argument("SSD field copy requires host targets");
        bytes+=target.numel()*target.element_size();
    }
    if(bytes!=field_bytes_.at(index))throw std::invalid_argument("SSD field copy layout differs");
    const auto copy=[&](const MixedMoePool& pool) {
        const auto fields=expert_field_tensors(pool);
        if(fields.size()!=targets.size())throw std::invalid_argument("SSD field copy count differs");
        for(std::size_t f=0;f<fields.size();++f) {
            const auto size=fields[f].numel()*fields[f].element_size();
            if(size!=targets[f].numel()*targets[f].element_size())
                throw std::invalid_argument("SSD field copy shape differs");
            mfq::copy_host_range({targets[f].data_ptr(),fields[f].data_ptr(),size});
        }
    };
    if(host_cache_)if(auto cached=host_cache_->find(host_key(expert),touch_cache)) {copy(cached->weights);return;}
    if(host_cache_ && !touch_cache && !gpu_resident(expert) && !disk_sealed_.load())
        if(auto cached=host_cache_->acquire_scan(host_key(expert),bytes,[&]{return read_expert(expert);})) {
            copy(cached->weights);return;
        }
    if(disk_sealed_.load()) {
        ++sealed_disk_attempts_;
        throw std::runtime_error("expert disk copy forbidden after complete residency preload");
    }
    const auto& prototype=metadata_->pools.at(index);
    if(prototype.family==MixedMoeFamily::Mxfp4 && targets.size()==2) {
        using Clock=std::chrono::steady_clock;
        const bool profile=profile_ssd_ && preloaded_.load();
        const auto begin=profile?Clock::now():Clock::time_point{};
        std::array<mfq::QuantFieldTarget,2> fields;
        for(std::size_t f=0;f<2;++f)fields[f]={static_cast<std::uint8_t*>(targets[f].data_ptr()),
            std::size_t(targets[f].nbytes())};
        std::size_t source_bytes=0;
        if(!store_->copy_mxfp4_fields(expert,fields,&source_bytes))
            throw std::logic_error("MXFP4 range source differs from prototype");
        ++direct_field_reads_;++materializations_;
        if(preloaded_.load()) {
            ++disk_reads_after_preload_;
            disk_bytes_after_preload_.fetch_add(source_bytes,std::memory_order_relaxed);
        }
        if(profile)encoded_ns_.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count(),std::memory_order_relaxed);
        return;
    }
    if(prototype.family==MixedMoeFamily::Nint && targets.size()==7) {
        using Clock=std::chrono::steady_clock;
        const bool profile=profile_ssd_ && preloaded_.load();
        const auto begin=profile?Clock::now():Clock::time_point{};
        const auto rows=std::size_t(store_->out_per_expert());
        std::array<std::vector<std::uint16_t>,2> anchors;
        for(auto& anchor:anchors)anchor.resize(rows);
        std::array<mfq::QuantFieldTarget,7> fields;
        for(std::size_t f=0;f<5;++f)fields[f]={static_cast<std::uint8_t*>(targets[f].data_ptr()),
            std::size_t(targets[f].numel()*targets[f].element_size())};
        for(std::size_t f=5;f<7;++f) {
            if(targets[f].numel()*targets[f].element_size()!=rows*4)
                throw std::invalid_argument("NINT neuron target size differs");
            fields[f]={reinterpret_cast<std::uint8_t*>(anchors[f-5].data()),rows*2};
        }
        std::size_t source_bytes=0;
        if(!store_->copy_nint_fields(expert,fields,&source_bytes))
            throw std::logic_error("NINT range source differs from prototype");
        const auto read_done=profile?Clock::now():Clock::time_point{};
        for(std::size_t f=5;f<7;++f) {
            copy_neuron_anchors(anchors[f-5],targets[f]);
        }
        ++direct_field_reads_;++materializations_;
        if(preloaded_.load()) {
            ++disk_reads_after_preload_;
            disk_bytes_after_preload_.fetch_add(source_bytes,std::memory_order_relaxed);
        }
        if(profile) {
            const auto encoded_time=std::chrono::duration_cast<std::chrono::nanoseconds>(read_done-begin).count();
            const auto materialize=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-read_done).count();
            encoded_ns_.fetch_add(encoded_time,std::memory_order_relaxed);
            materialize_ns_.fetch_add(materialize,std::memory_order_relaxed);
            nint_materialize_ns_.fetch_add(materialize,std::memory_order_relaxed);
            record_format_io(0,source_bytes,encoded_time,materialize);
        }
        return;
    }
    if(!dense_groups_ && prototype.family==MixedMoeFamily::Nvq && targets.size()==4) {
        using Clock=std::chrono::steady_clock;
        const bool profile=profile_ssd_ && preloaded_.load();
        const auto begin=profile?Clock::now():Clock::time_point{};
        std::vector<std::uint16_t> anchors(store_->out_per_expert());
        const auto target=[&](std::size_t f) {
            return mfq::NvqRows::FieldTarget{static_cast<std::uint8_t*>(targets[f].data_ptr()),
                std::size_t(targets[f].numel()*targets[f].element_size())};
        };
        const std::array<mfq::NvqRows::FieldTarget,4> fields={
            mfq::NvqRows::FieldTarget{reinterpret_cast<std::uint8_t*>(anchors.data()),anchors.size()*2},
            target(2),target(0),target(1)};
        std::size_t source_bytes=0;
        if(store_->copy_nvq_fields(expert,fields,&source_bytes)) {
            const auto read_done=profile?Clock::now():Clock::time_point{};
            if(targets[3].numel()*targets[3].element_size()!=anchors.size()*4)
                throw std::invalid_argument("SSD neuron scale target differs");
            copy_neuron_anchors(anchors,targets[3]);
            ++direct_field_reads_;++materializations_;
            if(preloaded_.load()) {
                ++disk_reads_after_preload_;
                disk_bytes_after_preload_.fetch_add(source_bytes,std::memory_order_relaxed);
            }
            if(profile) {
                const auto encoded_time=std::chrono::duration_cast<std::chrono::nanoseconds>(read_done-begin).count();
                encoded_ns_.fetch_add(encoded_time,std::memory_order_relaxed);
                const auto materialize=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-read_done).count();
                materialize_ns_.fetch_add(materialize,std::memory_order_relaxed);
                nvq_materialize_ns_.fetch_add(materialize,std::memory_order_relaxed);
                record_format_io(prototype.nvq.kernel_format,source_bytes,encoded_time,materialize);
            }
            return;
        }
    }
    copy(read_expert_canonical(expert));
}

bool MoeQuantRangeSource::copy_nvq_expert_fields(const std::vector<int>& experts,
        const std::array<tb::Tensor,4>& targets) const {
    if(experts.size()<2 || dense_groups_)return false;
    if(disk_sealed_.load())throw std::runtime_error("bulk expert disk read forbidden after complete residency preload");
    const auto index=store_->expert_pool(experts.front());
    const auto& prototype=metadata_->pools.at(index);
    if(prototype.family!=MixedMoeFamily::Nvq)return false;
    for(const auto& target:targets)if(!target.is_cpu() || !target.is_contiguous())
        throw std::invalid_argument("bulk NVQ fields require contiguous host targets");
    using Clock=std::chrono::steady_clock;
    const bool profile=profile_ssd_ && preloaded_.load();
    const auto begin=profile?Clock::now():Clock::time_point{};
    const auto rows=std::size_t(store_->out_per_expert())*experts.size();
    if(targets[3].nbytes()!=rows*4)throw std::invalid_argument("bulk NVQ neuron target differs");
    std::vector<std::uint16_t> anchors(rows);
    const auto field=[&](std::size_t f) {
        return mfq::NvqRows::FieldTarget{static_cast<std::uint8_t*>(targets[f].data_ptr()),std::size_t(targets[f].nbytes())};
    };
    const std::array<mfq::NvqRows::FieldTarget,4> fields={
        mfq::NvqRows::FieldTarget{reinterpret_cast<std::uint8_t*>(anchors.data()),rows*2},field(2),field(0),field(1)};
    std::size_t source_bytes=0;
    if(!store_->copy_nvq_fields(experts,fields,&source_bytes))return false;
    const auto read_done=profile?Clock::now():Clock::time_point{};
    copy_neuron_anchors(anchors,targets[3]);
    direct_field_reads_.fetch_add(experts.size(),std::memory_order_relaxed);
    materializations_.fetch_add(experts.size(),std::memory_order_relaxed);
    ++bulk_field_reads_;
    if(preloaded_.load()) {
        disk_reads_after_preload_.fetch_add(experts.size(),std::memory_order_relaxed);
        disk_bytes_after_preload_.fetch_add(source_bytes,std::memory_order_relaxed);
    }
    if(profile) {
        const auto encoded=std::chrono::duration_cast<std::chrono::nanoseconds>(read_done-begin).count();
        const auto materialize=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-read_done).count();
        encoded_ns_.fetch_add(encoded,std::memory_order_relaxed);
        materialize_ns_.fetch_add(materialize,std::memory_order_relaxed);
        nvq_materialize_ns_.fetch_add(materialize,std::memory_order_relaxed);
        record_format_io(prototype.nvq.kernel_format,source_bytes,encoded,materialize,experts.size());
    }
    return true;
}

void MoeQuantRangeSource::record_format_io(int format,std::uint64_t bytes,
        std::uint64_t encoded_ns,std::uint64_t materialize_ns,std::uint64_t reads) const {
    if(format<0 || std::size_t(format)>=format_io_.size())return;
    const std::array<std::uint64_t,4> values{reads,bytes,encoded_ns,materialize_ns};
    for(std::size_t i=0;i<values.size();++i)
        format_io_[format][i].fetch_add(values[i],std::memory_order_relaxed);
}

MoeQuantRangeSource::FormatIoStats MoeQuantRangeSource::format_io_stats() const {
    FormatIoStats result{};
    for(std::size_t format=0;format<result.size();++format)
        for(std::size_t i=0;i<result[format].size();++i)
            result[format][i]=format_io_[format][i].load(std::memory_order_relaxed);
    return result;
}

MixedMoePool MoeQuantRangeSource::decode_expert(int expert,const mfq::MfeQuantExpert& encoded) const {
    const auto index=store_->expert_pool(expert);
    MixedMoePool pool;
    pool.local_experts=1;
    if (encoded.dtype=="NINT") {
        pool.family=MixedMoeFamily::Nint;
        pool.nint=to_device_mfe_nint(unpack_nint(encoded.payload),1,store_->out_per_expert(),false);
        const auto stride=q_strides_.at(index);
        if (pool.nint.q_expert_stride>stride) throw std::runtime_error("NINT expert exceeded metadata stride");
        if (pool.nint.q_expert_stride!=stride) {
            auto padded=tb::zeros({1,stride},tb::TensorOptions().dtype(tb::kUInt8));
            std::memcpy(padded.data_ptr(),pool.nint.q_packed.data_ptr(),pool.nint.q_packed.numel());
            pool.nint.q_packed=std::move(padded);
            pool.nint.q_expert_stride=stride;
        }
    } else if(encoded.dtype=="MXFP4") {
        pool.family=MixedMoeFamily::Mxfp4;
        pool.mxfp4=to_device_mxfp4(unpack_mxfp4(encoded.payload),false);
    } else if (encoded.dtype == "NVQ" || encoded.dtype == "NPQ") {
        pool.family=MixedMoeFamily::Nvq;
        pool.nvq=to_cpu_nvq(unpack_nvq(encoded.payload,encoded.dtype));
        if(dense_groups_ && prepare_nvq_dense_groups(pool.nvq))++dense_materializations_;
        if (static_cast<std::size_t>(index)<metadata_->pools.size()) {
            pool.nvq.codebook=metadata_->pools[index].nvq.codebook;
            pool.nvq.bank_map=metadata_->pools[index].nvq.bank_map;
        }
    } else {
        throw std::runtime_error("unsupported CUDA expert range dtype=" + encoded.dtype +
            "; use the full MFE reader for native cohorts, or quantize experts with NINT/NVQ or MXFP4");
    }
    ++materializations_;
    return pool;
}

void MoeQuantRangeSource::bind_host_cache(std::shared_ptr<MoeHostExpertCache> cache,int source_id) {
    if (!cache || source_id<0 || host_cache_) throw std::logic_error("invalid or repeated expert host cache binding");
    host_cache_=std::move(cache);
    source_id_=source_id;
}

mfq::MoeCacheKey MoeQuantRangeSource::host_key(int expert) const {
    return {source_id_,store_->expert_pool(expert),expert};
}

std::size_t MoeQuantRangeSource::expert_field_bytes(int expert) const {
    return field_bytes_.at(store_->expert_pool(expert));
}

bool MoeQuantRangeSource::gpu_resident(int expert) const {
    (void)store_->expert_pool(expert);
    return gpu_resident_[expert].load();
}

MoeHostExpertCache::Lease MoeQuantRangeSource::acquire_expert(int expert,bool touch) const {
    const auto bytes=expert_field_bytes(expert);
    if (host_cache_) return host_cache_->acquire(host_key(expert),bytes,[this,expert] {
        auto pool=read_expert(expert);
        if (expert_fields_bytes(pool)!=expert_field_bytes(expert))
            throw std::runtime_error("decoded expert exceeded registered RAM layout");
        return pool;
    },!gpu_resident(expert),touch);
    auto value=std::make_shared<MoeHostExpert>();
    value->weights=read_expert(expert);
    value->bytes=bytes;
    return value;
}

bool MoeQuantRangeSource::demote_expert(int expert,const MoeHostExpertCache::Load& copy) {
    mark_gpu_resident(expert,false);
    return host_cache_ && host_cache_->demote(host_key(expert),expert_field_bytes(expert),[&] {
        auto pool=copy();
        if (expert_fields_bytes(pool)!=expert_field_bytes(expert))
            throw std::runtime_error("GPU eviction exceeded registered RAM layout");
        return pool;
    });
}

void MoeQuantRangeSource::mark_gpu_resident(int expert,bool resident) {
    (void)store_->expert_pool(expert);
    gpu_resident_[expert].store(resident);
    if (resident && host_cache_) host_cache_->promote(host_key(expert));
}

tb::Tensor MoeQuantRangeSource::forward_cpu(CudaExecutionContext& execution,tb::Tensor input,
        const MoeRoutePlan& route,const std::unordered_map<int,MoeHostExpertCache::Lease>* prepared) const {
    const int tokens=static_cast<int>(route.ids.size(0)),routes=static_cast<int>(route.ids.size(1));
    if (route.n_experts!=store_->num_experts() || input.size(0)!=tokens || input.size(-1)!=store_->neuron_len() ||
        (input.dim()!=2 && input.dim()!=3) || (input.dim()==3 && input.size(1)!=routes))
        throw std::runtime_error("CPU cold expert input/route shape mismatch");
    const auto device=input.device();
    auto ids=route.ids.to(tb::kCPU).to(tb::kInt32).contiguous();
    auto x=input.to(tb::kCPU).to(tb::kFloat32).contiguous();
    const int width=store_->neuron_len(),output=store_->out_per_expert();
    auto result=tb::empty({tokens,routes,output},tb::TensorOptions().dtype(tb::kFloat16));
    std::vector<std::vector<int>> selected(store_->num_experts());
    for (int position=0; position<tokens*routes; ++position) {
        const auto expert=ids.data_ptr<std::int32_t>()[position];
        if (expert<0 || expert>=store_->num_experts()) throw std::out_of_range("CPU cold expert route ID");
        selected[expert].push_back(position);
    }
    struct Job {
        MoeHostExpertCache::Lease lease;
        CpuProjectionRows projection;
        int expert=0, offset=0, count=0;
    };
    std::vector<Job> jobs;
    auto packed=tb::empty({tokens*routes,width},tb::TensorOptions().dtype(tb::kFloat32));
    auto values=tb::empty({tokens*routes,output},tb::TensorOptions().dtype(tb::kFloat32));
    int offset=0;
    for (int expert=0; expert<store_->num_experts(); ++expert) {
        const auto& positions=selected[expert];
        if(positions.empty()) continue;
        auto lease=prepared ? prepared->at(expert) : acquire_expert(expert);
        const auto& pool=lease->weights;
        auto projection=pool.family==MixedMoeFamily::Nint ? make_cpu_projection_rows(pool.nint) :
            pool.family==MixedMoeFamily::Mxfp4 ? make_cpu_projection_rows(pool.mxfp4) : make_cpu_projection_rows(pool.nvq);
        for(std::size_t row=0;row<positions.size();++row) {
            const int source=input.dim()==2 ? positions[row]/routes : positions[row];
            std::memcpy(packed.data_ptr<float>()+std::size_t(offset+row)*width,
                x.data_ptr<float>()+std::size_t(source)*width,width*sizeof(float));
        }
        jobs.push_back({std::move(lease),std::move(projection),expert,offset,static_cast<int>(positions.size())});
        offset+=static_cast<int>(positions.size());
    }
    mfq::cpu::expert_pool().rows(static_cast<int64_t>(jobs.size())*output,[&](int64_t begin,int64_t end) {
        while(begin<end) {
            const auto index=begin/output;
            const int first=static_cast<int>(begin%output);
            const int last=static_cast<int>(std::min<int64_t>(output,first+end-begin));
            const auto& job=jobs[static_cast<std::size_t>(index)];
            job.projection.run(packed.data_ptr<float>()+std::size_t(job.offset)*width,job.count,width,
                values.data_ptr<float>()+std::size_t(job.offset)*output,output,first,last);
            begin+=last-first;
        }
    });
    for(const auto& job:jobs) {
        const auto& positions=selected[job.expert];
        for(int row=0;row<job.count;++row) {
            auto* dst=result.data_ptr<mfq_half>()+std::size_t(positions[row])*output;
            const auto* src=values.data_ptr<float>()+std::size_t(job.offset+row)*output;
            for(int column=0;column<output;++column) dst[column]=mfq_half(src[column]);
        }
    }
    cpu_projections_.fetch_add(jobs.size());
    return result.to(device);
}
