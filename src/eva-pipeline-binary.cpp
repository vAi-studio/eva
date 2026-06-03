#include "eva-pipeline-binary.h"

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sstream>

namespace eva {

namespace {

std::atomic<uint64_t> gPipelineBinaryDumpCounter{0};

const char* pipelineBinaryDumpDir()
{
    const char* dir = std::getenv("EVA_PIPELINE_BINARY_DUMP_DIR");
    return (dir && dir[0] != '\0') ? dir : nullptr;
}

std::string sanitizeFileStem(std::string text)
{
    if (text.empty())
        text = "pipeline";
    for (char& ch : text)
    {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (!std::isalnum(c) && ch != '-' && ch != '_' && ch != '.')
            ch = '_';
    }
    return text;
}

uint64_t fnv1a64(const uint8_t* data, size_t size)
{
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i)
    {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

void dumpCapturedPipelineBinaryIfRequested(
    const ComputePipeline& pipeline,
    const ComputePipelineCreateInfo& info,
    uint32_t workgroupSizeX,
    uint32_t workgroupSizeY,
    uint32_t workgroupSizeZ)
{
#ifdef VK_KHR_PIPELINE_BINARY_EXTENSION_NAME
    if (const char* dumpDir = pipelineBinaryDumpDir())
    {
        try
        {
            std::filesystem::path outDir(dumpDir);
            std::filesystem::create_directories(outDir);

            const uint64_t index = gPipelineBinaryDumpCounter.fetch_add(1);
            std::ostringstream stem;
            stem << "pipeline_" << std::setw(5) << std::setfill('0') << index
                 << "_" << sanitizeFileStem(info.debugName);

            auto binaries = pipeline.capturedPipelineBinaryData();
            for (size_t i = 0; i < binaries.size(); ++i)
            {
                std::filesystem::path path = outDir / (stem.str() + "_" + std::to_string(i) + ".bin");
                std::ofstream file(path, std::ios::binary);
                file.write(
                    reinterpret_cast<const char*>(binaries[i].data()),
                    static_cast<std::streamsize>(binaries[i].size()));
            }

            std::filesystem::path metaPath = outDir / (stem.str() + ".meta.txt");
            std::ofstream meta(metaPath);
            meta << "debugName=" << info.debugName << "\n";
            meta << "workgroup=" << workgroupSizeX << "," << workgroupSizeY << "," << workgroupSizeZ << "\n";
            meta << "pushConstantSize=" << pipeline.pushConstantSize() << "\n";
            meta << "binaryCount=" << binaries.size() << "\n";
            if (const auto* blob = std::get_if<SpvBlob>(&*info.csStage.shader))
            {
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(blob->data.get());
                meta << "spvSizeBytes=" << blob->sizeInBytes << "\n";
                meta << "spvFnv1a64=0x" << std::hex << fnv1a64(bytes, blob->sizeInBytes) << std::dec << "\n";
            }
            else
            {
                meta << "spvSizeBytes=unavailable\n";
                meta << "spvFnv1a64=unavailable\n";
            }
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "[EVA] pipeline binary dump failed for '%s': %s\n", info.debugName.c_str(), e.what());
        }
    }
#else
    (void)pipeline;
    (void)info;
    (void)workgroupSizeX;
    (void)workgroupSizeY;
    (void)workgroupSizeZ;
#endif
}

std::vector<std::vector<uint8_t>> ComputePipeline::capturedPipelineBinaryData() const
{
#ifdef VK_KHR_PIPELINE_BINARY_EXTENSION_NAME
    auto fpCreatePipelineBinariesKHR =
        reinterpret_cast<PFN_vkCreatePipelineBinariesKHR>(
            vkGetDeviceProcAddr(impl().vkDevice, "vkCreatePipelineBinariesKHR"));
    auto fpDestroyPipelineBinaryKHR =
        reinterpret_cast<PFN_vkDestroyPipelineBinaryKHR>(
            vkGetDeviceProcAddr(impl().vkDevice, "vkDestroyPipelineBinaryKHR"));
    auto fpGetPipelineBinaryDataKHR =
        reinterpret_cast<PFN_vkGetPipelineBinaryDataKHR>(
            vkGetDeviceProcAddr(impl().vkDevice, "vkGetPipelineBinaryDataKHR"));
    auto fpReleaseCapturedPipelineDataKHR =
        reinterpret_cast<PFN_vkReleaseCapturedPipelineDataKHR>(
            vkGetDeviceProcAddr(impl().vkDevice, "vkReleaseCapturedPipelineDataKHR"));

    if (!fpCreatePipelineBinariesKHR ||
        !fpDestroyPipelineBinaryKHR ||
        !fpGetPipelineBinaryDataKHR ||
        !fpReleaseCapturedPipelineDataKHR)
    {
        throw std::runtime_error("VK_KHR_pipeline_binary function pointers are unavailable");
    }

    VkPipelineBinaryCreateInfoKHR binaryCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
        .pipeline = impl().vkPipeline,
    };
    VkPipelineBinaryHandlesInfoKHR handlesInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
    };

    VkResult result = fpCreatePipelineBinariesKHR(impl().vkDevice, &binaryCreateInfo, nullptr, &handlesInfo);
    if (result != VK_SUCCESS && result != VK_INCOMPLETE)
        throw std::runtime_error(std::string("vkCreatePipelineBinariesKHR(count) failed: ") + vkResult2String(result));
    if (handlesInfo.pipelineBinaryCount == 0)
        throw std::runtime_error("vkCreatePipelineBinariesKHR returned zero pipeline binaries");

    std::vector<VkPipelineBinaryKHR> handles(handlesInfo.pipelineBinaryCount, VK_NULL_HANDLE);
    handlesInfo.pPipelineBinaries = handles.data();
    result = fpCreatePipelineBinariesKHR(impl().vkDevice, &binaryCreateInfo, nullptr, &handlesInfo);
    if (result != VK_SUCCESS && result != VK_INCOMPLETE)
        throw std::runtime_error(std::string("vkCreatePipelineBinariesKHR(handles) failed: ") + vkResult2String(result));

    std::vector<std::vector<uint8_t>> binaries;
    binaries.reserve(handlesInfo.pipelineBinaryCount);
    try
    {
        for (uint32_t i = 0; i < handlesInfo.pipelineBinaryCount; ++i)
        {
            VkPipelineBinaryDataInfoKHR dataInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR,
                .pipelineBinary = handles[i],
            };
            VkPipelineBinaryKeyKHR key{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR,
            };
            size_t dataSize = 0;
            result = fpGetPipelineBinaryDataKHR(impl().vkDevice, &dataInfo, &key, &dataSize, nullptr);
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::string("vkGetPipelineBinaryDataKHR(size) failed: ") + vkResult2String(result));

            std::vector<uint8_t> data(dataSize);
            result = fpGetPipelineBinaryDataKHR(impl().vkDevice, &dataInfo, &key, &dataSize, data.data());
            if (result != VK_SUCCESS)
                throw std::runtime_error(std::string("vkGetPipelineBinaryDataKHR(data) failed: ") + vkResult2String(result));
            data.resize(dataSize);
            binaries.emplace_back(std::move(data));
        }

        VkReleaseCapturedPipelineDataInfoKHR releaseInfo{
            .sType = VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR,
            .pipeline = impl().vkPipeline,
        };
        result = fpReleaseCapturedPipelineDataKHR(impl().vkDevice, &releaseInfo, nullptr);
        if (result != VK_SUCCESS)
            throw std::runtime_error(std::string("vkReleaseCapturedPipelineDataKHR failed: ") + vkResult2String(result));
    }
    catch (...)
    {
        for (VkPipelineBinaryKHR handle : handles)
            if (handle != VK_NULL_HANDLE)
                fpDestroyPipelineBinaryKHR(impl().vkDevice, handle, nullptr);
        throw;
    }

    for (VkPipelineBinaryKHR handle : handles)
        if (handle != VK_NULL_HANDLE)
            fpDestroyPipelineBinaryKHR(impl().vkDevice, handle, nullptr);

    return binaries;
#else
    throw std::runtime_error("VK_KHR_pipeline_binary is unavailable in current Vulkan headers");
#endif
}

} // namespace eva
