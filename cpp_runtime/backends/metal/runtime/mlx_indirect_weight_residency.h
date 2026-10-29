#pragma once
#include "mlx_weight_residency.h"
#include "mlx_kernel_prepare.h"
#include "mlx_cached_kernel.h"
#include <mlx/fast.h>
#include <mlx/primitives.h>
#include <unordered_map>
#include <atomic>

namespace mfq::metal {
struct MlxIndirectWeightRegistry {
    struct Entry {
        std::weak_ptr<mlx::core::array::Data> allocation;
        std::weak_ptr<MlxWeightResidency> residency;
    };
    std::mutex mutex;
    std::atomic<bool> active{false};
    std::unordered_map<const void*,Entry> entries;
};
inline MlxIndirectWeightRegistry& indirect_weight_registry() {
    static MlxIndirectWeightRegistry registry;
    return registry;
}
inline void register_indirect_weight_pool(const mlx::core::array& words,
        const std::shared_ptr<MlxWeightResidency>& residency) {
    auto& registry=indirect_weight_registry();
    std::lock_guard lock(registry.mutex);
    std::erase_if(registry.entries,[](const auto& item){return item.second.allocation.expired()||item.second.residency.expired();});
    registry.entries[words.data_shared_ptr().get()]={words.data_shared_ptr(),residency};
    registry.active.store(true, std::memory_order_release);
}
class MlxIndirectWeightKernel final:public mlx::core::Primitive,public MlxPreparableKernel {
public:
    MlxIndirectWeightKernel(std::shared_ptr<mlx::core::Primitive> kernel,
            std::vector<std::shared_ptr<MlxWeightResidency>> residency,
            std::vector<std::shared_ptr<mlx::core::array::Data>> handles)
        :Primitive(kernel->stream()),kernel_(std::move(kernel)),residency_(std::move(residency)),
            handles_(std::move(handles)){}
    const char* name() const override{return "MlxIndirectWeightKernel";}
    std::string preparation_key() const override{return mlx_kernel_preparation_key(*kernel_);}
    void prepare_gpu() override {prepare_mlx_kernel_gpu(*kernel_);}
    void eval_cpu(const std::vector<mlx::core::array>&,std::vector<mlx::core::array>&) override {
        throw std::runtime_error("Indirect weight kernels require GPU");
    }
    void eval_gpu(const std::vector<mlx::core::array>& inputs,std::vector<mlx::core::array>& outputs) override {
        auto* buffer=mlx::core::metal::get_command_encoder(stream()).get_command_buffer();
        for(auto& residency:residency_)residency->use_command_buffer(buffer);
        buffer->addCompletedHandler([residency=residency_,handles=handles_](MTL::CommandBuffer*) {});
        kernel_->eval_gpu(inputs,outputs);
    }
private:
    std::shared_ptr<mlx::core::Primitive> kernel_;
    std::vector<std::shared_ptr<MlxWeightResidency>> residency_;
    std::vector<std::shared_ptr<mlx::core::array::Data>> handles_;
};
inline mlx::core::fast::CustomKernelFunction indirect_weight_metal_kernel(
        const std::string& name,const std::vector<std::string>& inputs,const std::vector<std::string>& outputs,
        const std::string& source,const std::string& header="",bool contiguous=true,bool atomic=false,
        const mlx::core::CompileOptions& options={}) {
    auto function=cached_metal_kernel(name,inputs,outputs,source,header,contiguous,atomic,options);
    return [function=std::move(function)](auto&&... arguments) {
        auto values=function(std::forward<decltype(arguments)>(arguments)...);
        auto& registry = indirect_weight_registry();
        if (!registry.active.load(std::memory_order_acquire)) return values;
        std::vector<std::shared_ptr<MlxWeightResidency>> residency;
        std::vector<std::shared_ptr<mlx::core::array::Data>> handles;
        {
            std::lock_guard lock(registry.mutex);
            for(const auto& input:values.front().inputs()) {
                const auto found=registry.entries.find(input.data_shared_ptr().get());
                if(found==registry.entries.end()||found->second.allocation.expired())continue;
                auto owner=found->second.residency.lock();
                if(!owner)continue;
                handles.push_back(input.data_shared_ptr());
                if(std::find(residency.begin(),residency.end(),owner)==residency.end())residency.push_back(std::move(owner));
            }
        }
        if(residency.empty())return values;
        auto original=values.front().primitive_ptr();
        if(std::string_view(original->name())!="CustomKernel"&&!dynamic_cast<MlxPreparableKernel*>(original.get()))
            throw std::runtime_error("Indirect weight kernel does not support preparation");
        auto primitive=std::make_shared<MlxIndirectWeightKernel>(std::move(original),std::move(residency),std::move(handles));
        for (auto& value : values) value.primitive_ptr() = primitive;
        return values;
    };
}
}
