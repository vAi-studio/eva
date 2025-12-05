#include <array>
#include <vector>
#include <spirv_reflect.h>
#include <memory>
#include "eva-runtime.h"
#include "./eva-error.h"

using namespace eva;


void* createReflectShaderModule(const SpvBlob& spvBlob)
{
    SpvReflectShaderModule* pModule = new SpvReflectShaderModule;
    SpvReflectResult result = spvReflectCreateShaderModule(
        spvBlob.sizeInBytes,
        spvBlob.data.get(),
        pModule
    );

    if (result != SPV_REFLECT_RESULT_SUCCESS) {
        printf("Failed to create SPIR-V Reflect Shader Module: %d\n", result);
        return {};
    }

    return pModule;
}

void destroyReflectShaderModule(void* pModule)
{
    spvReflectDestroyShaderModule((SpvReflectShaderModule*)pModule);
}

std::array<uint32_t, 3> extractWorkGroupSize(const void* pModule)
{
    const SpvReflectShaderModule& module = *(const SpvReflectShaderModule*)pModule;
    std::array<uint32_t, 3> workGroupSize = { 1, 1, 1 };

    if (module.spirv_execution_model == SpvExecutionModelGLCompute) 
    {
        ASSERT_(module.entry_point_count == 1);

        const auto& entryPoint = module.entry_points[0];
        workGroupSize[0] = entryPoint.local_size.x;
        workGroupSize[1] = entryPoint.local_size.y;
        workGroupSize[2] = entryPoint.local_size.z;
    }

    return workGroupSize;
}

PipelineLayoutDesc extractPipelineLayoutDesc(const void* pModule)
{
    PipelineLayoutDesc desc;
    const SpvReflectShaderModule& module = *(const SpvReflectShaderModule*)pModule;
    
    std::vector<SpvReflectDescriptorSet*> srcSetLayouts;
    {
        uint32_t count = 0;
        spvReflectEnumerateDescriptorSets(&module, &count, nullptr);
        srcSetLayouts.resize(count);
        spvReflectEnumerateDescriptorSets(&module, &count, srcSetLayouts.data());
    }

    for (const auto* pSrcSetLayout : srcSetLayouts) 
    {
        const SpvReflectDescriptorSet& srcSetLayout = *pSrcSetLayout;
        const uint32_t numBindings = srcSetLayout.binding_count;
        const uint32_t setId = srcSetLayout.set;

        DescriptorSetLayoutDesc dstSrcLayout;

        for (uint32_t j = 0; j < numBindings; ++j) 
        {
            const SpvReflectDescriptorBinding& srcBinding = *srcSetLayout.bindings[j];

            dstSrcLayout.bindings.try_emplace(
                srcBinding.binding,
                srcBinding.binding,
                (DESCRIPTOR_TYPE)(uint32_t)srcBinding.descriptor_type,
                [&] {
                    uint32_t count = 1;
                    for (uint32_t k = 0; k < srcBinding.array.dims_count; ++k)
                        count *= srcBinding.array.dims[k];
                    return count;
                }(),
                (SHADER_STAGE)(uint32_t)module.shader_stage
            );
        }

        desc.setLayouts.try_emplace(setId, std::move(dstSrcLayout));
    }

    SpvReflectResult result;
    const SpvReflectBlockVariable* pcBlock = spvReflectGetEntryPointPushConstantBlock(&module, "main", &result);
    if (result == SPV_REFLECT_RESULT_SUCCESS && pcBlock) 
    {
        // desc.pushConstants.push_back(
        //     PushConstantRange(
        //         pcBlock->offset, 
        //         pcBlock->size - pcBlock->offset  // The reflect library returns the total size, but we need the size from offset to end
        //     ) |= module.shader_stage
        // );

        desc.pushConstant = std::make_unique<PushConstantRange>(
            pcBlock->offset, 
            pcBlock->size,                          // Just use the total size
            (SHADER_STAGE)(uint32_t)module.shader_stage
	    );
    }

    return desc;
}
