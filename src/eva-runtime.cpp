#include <vulkan/vulkan_core.h>
#ifdef EVA_ENABLE_WINDOW
    #define GLFW_INCLUDE_VULKAN
    #include <GLFW/glfw3.h>
#endif
#include <array>
#include <deque>
#include <map>
#include <unordered_map>
#include <set>
#include <algorithm>// std::all_of, std::any_of
#include <fstream>
#include "eva-native-factory.h"
#include "eva-runtime.h"

// #define USE_DEBUG_PRINTF 1


#if defined(_WIN32) || defined(_WIN64)
    #define EVA_PLATFORM_WINDOWS
#elif defined(__ANDROID__)
    #define EVA_PLATFORM_ANDROID
#elif defined(__linux__)
    #if defined(WAYLAND_DISPLAY) || defined(EVA_USE_WAYLAND)
        #define EVA_PLATFORM_WAYLAND
    #else
        #define EVA_PLATFORM_XLIB
    #endif
#elif defined(__APPLE__)
    #include <TargetConditionals.h>
    #if TARGET_OS_IPHONE
        #define EVA_PLATFORM_IOS
    #else
        #define EVA_PLATFORM_MACOS
    #endif
#endif

namespace eva {
    static bool gCoopMat2Supported = false;
    bool isCoopMat2Supported() { return gCoopMat2Supported; }

    static bool gCudaKernelLaunchSupported = false;
    bool isCudaKernelLaunchSupported() { return gCudaKernelLaunchSupported; }
}

std::vector<const char*> getRequiredInstanceExtensions()
{
#ifdef EVA_ENABLE_WINDOW
    return { 
        "VK_KHR_surface" 

    #ifdef EVA_PLATFORM_WINDOWS
        , "VK_KHR_win32_surface"
    #elif defined(EVA_PLATFORM_ANDROID)
        , "VK_KHR_android_surface"
    #elif defined(EVA_PLATFORM_XLIB)
        , "VK_KHR_xlib_surface"
    #elif defined(EVA_PLATFORM_WAYLAND)
        , "VK_KHR_wayland_surface"
    #elif defined(EVA_PLATFORM_MACOS) || defined(EVA_PLATFORM_IOS)
        , "VK_EXT_metal_surface"
        , "VK_KHR_portability_enumeration"
    #endif

    };
    
#else
    return {};
#endif
}


void* createReflectShaderModule(const eva::SpvBlob& spvBlob);
void destroyReflectShaderModule(void* pModule);
eva::PipelineLayoutDesc extractPipelineLayoutDesc(const void* pModule);
std::array<uint32_t, 3> extractWorkGroupSize(const void* pModule);

eva::SpvBlob eva::SpvBlob::readFrom(const char* filepath)
{
    FILE* file = fopen(filepath, "rb");
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // uint32_t[]로 직접 할당 (정렬 보장)
    size_t uint32Count = (fileSize + 3) / 4; // 4바이트 단위로 올림
    std::shared_ptr<uint32_t[]> data(new uint32_t[uint32Count]);
    fread(data.get(), 1, fileSize, file);
    fclose(file);
    
    return { data, fileSize };
}


PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR_ = nullptr;
PFN_vkGetPipelineExecutablePropertiesKHR vkGetPipelineExecutablePropertiesKHR_ = nullptr;
PFN_vkGetPipelineExecutableInternalRepresentationsKHR vkGetPipelineExecutableInternalRepresentationsKHR_ = nullptr;
PFN_vkGetPipelineExecutableStatisticsKHR vkGetPipelineExecutableStatisticsKHR_ = nullptr;
#ifdef EVA_ENABLE_RAYTRACING
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_ = nullptr;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_ = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR_ = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR_ = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR_ = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR_ = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR_ = nullptr;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR_ = nullptr;
#endif



inline static uint64_t hashCombine(uint64_t h1, uint64_t h2)
{
	return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
}

using namespace eva;


static void printQueueFamily(uint32_t qfIndex, uint32_t qCount, VkQueueFlags qFlags)
{
    printf("[Queue Family %u] Queue Count: %u, Queue Flags: %s%s%s%s%s\n", qfIndex, qCount,
        (qFlags & VK_QUEUE_GRAPHICS_BIT) ? "Graphics " : "",
        (qFlags & VK_QUEUE_COMPUTE_BIT) ? "Compute " : "",
        (qFlags & VK_QUEUE_TRANSFER_BIT) ? "Transfer " : "",
        (qFlags & VK_QUEUE_SPARSE_BINDING_BIT) ? "SparseBinding " : "",
        (qFlags & VK_QUEUE_PROTECTED_BIT) ? "Protected " : "");
}

static VkInstance createVkInstance()
{
    static bool first = true;
    EVA_ASSERT(first);
    first = false;

    std::vector<const char*> extensions = getRequiredInstanceExtensions();
    std::vector<const char*> requiredLayers;
#ifndef NDEBUG
    requiredLayers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    auto allLayers = arrayFrom(vkEnumerateInstanceLayerProperties);
    bool ok = std::all_of(requiredLayers.begin(), requiredLayers.end(), [&](const char* layer) {
        return std::any_of(allLayers.begin(), allLayers.end(), [&](const auto& porps) {
            return strcmp(layer, porps.layerName) == 0;
        });
    });
    EVA_ASSERT(ok);
    
    VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Vulkan App",
        .apiVersion = VK_API_VERSION_1_3
    };

    VkValidationFeatureEnableEXT enables[] = {
        VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
    };
    VkValidationFeaturesEXT features = {
        .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
        .enabledValidationFeatureCount = 1,
        .pEnabledValidationFeatures = enables,
    };

    return create<VkInstance>({
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
#if USE_DEBUG_PRINTF
        .pNext = requiredLayers.empty() ? nullptr : &features,
#endif
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = (uint32_t)requiredLayers.size(),
        .ppEnabledLayerNames = requiredLayers.data(),
        .enabledExtensionCount = (uint32_t)extensions.size(),
        .ppEnabledExtensionNames = extensions.data(),
    });
}

static void printGpuInfo(uint32_t order, VkPhysicalDevice physicalDevice)
{
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);

    printf("[GPU %d] %s\n", order, props.deviceName);
    printf("        API Version: %d.%d.%d\n",
        VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));
    printf("        Driver Version: %d.%d.%d\n",
        VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(props.driverVersion), VK_VERSION_PATCH(props.driverVersion));
    printf("        Device Type: %s\n",
        props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "Integrated GPU" :
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "Discrete GPU" :
                props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? "Virtual GPU" :
                    props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? "CPU" : "Other");

    // Device Local Memory 정보
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            printf("        Device Local Memory (VRAM): %.2f GB\n",
                memProps.memoryHeaps[i].size / (1024.0 * 1024.0 * 1024.0));
            break;
        }
    }

    // Compute Shader 정보
    printf("        Max Workgroup Size:  [%u, %u, %u]\n",
        props.limits.maxComputeWorkGroupSize[0],
        props.limits.maxComputeWorkGroupSize[1],
        props.limits.maxComputeWorkGroupSize[2]);
    printf("        Max Workgroup Invocations: %u\n", props.limits.maxComputeWorkGroupInvocations);
    printf("        Max Shared Memory: %u bytes (%.1f KB)\n",
        props.limits.maxComputeSharedMemorySize,
        props.limits.maxComputeSharedMemorySize / 1024.0f);

    // Subgroup 정보 (Vulkan 1.1+)
    VkPhysicalDeviceSubgroupProperties subgroupProps = {};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroupProps.pNext = nullptr;

    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;

    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

    printf("        Subgroup Size: %u\n", subgroupProps.subgroupSize);
    printf("        Subgroup Supported Operations: 0x%x", subgroupProps.supportedOperations);
    if (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) printf(" BASIC");
    if (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) printf(" VOTE");
    if (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) printf(" ARITHMETIC");
    if (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) printf(" BALLOT");
    if (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) printf(" SHUFFLE");
    if (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) printf(" SHUFFLE_REL");
    if (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) printf(" QUAD");
    printf("\n");
}

static bool deviceSupportsExtensions(
    VkPhysicalDevice physicalDevice, 
    const std::vector<const char*>& extensions)
{
    auto deviceExtensions = arrayFrom(vkEnumerateDeviceExtensionProperties, physicalDevice, nullptr);

    return std::all_of(extensions.begin(), extensions.end(), [&](const char* reqExtension) {
        return std::any_of(deviceExtensions.begin(), deviceExtensions.end(), [&](const auto& props) {
            return strcmp(props.extensionName, reqExtension) == 0;
        });
    });
}

static inline VkPhysicalDeviceMemoryProperties getMemorySpec(VkPhysicalDevice physicalDevice)
{
    VkPhysicalDeviceMemoryProperties spec;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &spec);
    return spec;
}

template <typename VkResource>
static std::pair<VkMemoryAllocateInfo, VkMemoryPropertyFlags> getMemoryAllocInfo(
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    VkResource resource,
    VkMemoryPropertyFlags reqMemProps)
{
    static VkPhysicalDevice cached = VK_NULL_HANDLE;
    static VkPhysicalDeviceMemoryProperties spec;
    if (cached != physicalDevice) {
        spec = getMemorySpec(physicalDevice);
        cached = physicalDevice;
    }

    VkMemoryRequirements memRequirements;
    if constexpr (std::is_same_v<VkResource, VkBuffer>) 
        vkGetBufferMemoryRequirements(device, resource, &memRequirements);  // It should be removed for performance!
    else if constexpr (std::is_same_v<VkResource, VkImage>)
        vkGetImageMemoryRequirements(device, resource, &memRequirements);
    else
        static_assert(sizeof(VkResource) == 0, "Invalid VkResource type"); // TODO: Linux Clang 호환성 - dependent false

    /*
    In Vulkan specification, the memoryTypes array is ordered by the following rules:
        For each pair of elements X and Y returned in memoryTypes, X must be placed at a lower index
        position than Y if:
            • the set of bit flags returned in the propertyFlags member of X is a strict subset of the set of bit
            flags returned in the propertyFlags member of Y; or
    */
    uint32_t i = 0;
    for ( ; i < spec.memoryTypeCount; ++i) {
        if ((memRequirements.memoryTypeBits & (1<<i)) != 0
            && (spec.memoryTypes[i].propertyFlags & reqMemProps) == reqMemProps) 
            break;
    }
    EVA_ASSERT(i != spec.memoryTypeCount);

    return { 
        {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = i,
        }, 
        spec.memoryTypes[i].propertyFlags
    };
}


/////////////////////////////////////////////////////////////////////////////////////////
// Impl classes
/////////////////////////////////////////////////////////////////////////////////////////
struct Runtime::Impl {
    const VkInstance instance;
    std::vector<Device> devices;
#ifdef EVA_ENABLE_WINDOW
    std::vector<Window> windows;
#endif
    Impl(VkInstance instance) : instance(instance) {}
};


struct Device::Impl {
    const uint32_t qfIndex[queue_max];    // uint32_t(-1) if and only if DeviceSettings does not require the queue type
    const uint32_t qCount[queue_max];
    const std::vector<std::vector<Queue>> queues;
    const VkPhysicalDevice vkPhysicalDevice;
    const VkDevice vkDevice;
    // const VkInstance vkInstance;
    const Runtime& parent;
    const DeviceSettings settings;
    const DeviceCapabilities capabilities;

    std::set<CommandPool::Impl**> cmdPools;
    std::set<Fence::Impl**> fences;
    std::set<Semaphore::Impl**> semaphores;
    std::set<ShaderModule::Impl**> shaderModules;
    std::set<ComputePipeline::Impl**> computePipelines;

    std::set<Buffer::Impl**> buffers;
    std::set<Image::Impl**> images;
    std::set<Sampler::Impl**> samplers;

    std::set<DescriptorSetLayout::Impl**> descSetLayouts;
    std::set<PipelineLayout::Impl**> pipelineLayouts;
    std::set<DescriptorPool::Impl**> descPools;
    std::set<QueryPool::Impl**> queryPools;

#ifdef EVA_ENABLE_RAYTRACING
    struct {
        const uint32_t shaderGroupHandleSize = portable::shaderGroupHandleSize;
        uint32_t shaderGroupHandleAlignment;
        uint32_t shaderGroupBaseAlignment;
        uint32_t minAccelerationStructureScratchOffsetAlignment;
        const uint32_t asBufferOffsetAlignment = 256;
    } rtProps;

    std::set<RaytracingPipeline::Impl**> raytracingPipelines;
    std::set<AccelerationStructure::Impl**> accelerationStructures;
#endif


    CommandPool defaultCmdPool[queue_max][8] = {};

    Impl(VkPhysicalDevice vkPhysicalDevice,
        VkDevice vkDevice,
        const Runtime& parent,
        const DeviceSettings& settings,
        const DeviceCapabilities& capabilities,
        uint32_t graphicsQfIndex,
        uint32_t computeQfIndex,
        uint32_t transferQfIndex,
        std::vector<std::vector<Queue>>&& queues)
    : vkPhysicalDevice(vkPhysicalDevice)
    , vkDevice(vkDevice)
    , parent(parent)
    , settings(settings)
    , capabilities(capabilities)
    , qfIndex{
        graphicsQfIndex,
        computeQfIndex,
        transferQfIndex}
        , qCount{
            graphicsQfIndex != uint32_t(-1) ? (uint32_t) queues[graphicsQfIndex].size() : 0,
            computeQfIndex != uint32_t(-1) ? (uint32_t) queues[computeQfIndex].size() : 0,
            transferQfIndex != uint32_t(-1) ? (uint32_t) queues[transferQfIndex].size() : 0}
    , queues(std::move(queues))
    {}
    ~Impl();                               
};


struct Queue::Impl {
    const VkQueue vkQueue;
    const uint32_t qfIndex;
    const uint32_t index;
    const float priority;

    Impl(
        VkQueue vkQueue,  
        uint32_t qfIndex, 
        uint32_t index,
        float priority)
    : vkQueue(vkQueue)
    , qfIndex(qfIndex)
    , index(index)
    , priority(priority) {}
};


struct CommandPool::Impl {
    const VkDevice vkDevice;
    const Device device;
    const VkCommandPool vkCmdPool;
    const COMMAND_POOL_CREATE flags;
    const uint32_t qfIndex;
    const QueueType type;
    std::deque<CommandBuffer> cmdBuffers;

    Impl(VkDevice vkDevice,
        Device device,
        VkCommandPool vkCmdPool, 
        COMMAND_POOL_CREATE flags,
        uint32_t qfIndex,
        QueueType type)
    : vkDevice(vkDevice)
    , device(device)
    , vkCmdPool(vkCmdPool)
    , flags(flags)
    , qfIndex(qfIndex)
    , type(type) {} 
    ~Impl();
};


struct CommandBuffer::Impl {
    const VkCommandBuffer vkCmdBuffer;
    const Device device;
    const uint32_t qfIndex;
    const QueueType type;
    Pipeline boundPipeline;
    Queue lastSubmittedQueue;

    Impl( 
        VkCommandBuffer vkCmdBuffer, 
        Device device,
        uint32_t qfIndex,
        QueueType type)
    : vkCmdBuffer(vkCmdBuffer)
    , device(device)
    , qfIndex(qfIndex)
    , type(type)
    {}
};


CommandPool::Impl::~Impl()
{
    for (auto& cmdBuffer : cmdBuffers) 
    {
        if (*cmdBuffer.ppImpl) 
            delete *cmdBuffer.ppImpl;
        delete cmdBuffer.ppImpl;
    }
    vkDestroyCommandPool(vkDevice, vkCmdPool, nullptr); 
}


struct Fence::Impl {
    const VkDevice vkDevice;
    const VkFence vkFence;

    Impl(VkDevice vkDevice, 
        VkFence vkFence) 
    : vkDevice(vkDevice)
    , vkFence(vkFence) {}
    ~Impl() {
        vkDestroyFence(vkDevice, vkFence, nullptr);
    }
};


struct Semaphore::Impl {
    const VkDevice vkDevice;
    const VkSemaphore vkSemaphore;

    Impl(VkDevice vkDevice, 
        VkSemaphore vkSemaphore) 
    : vkDevice(vkDevice)
    , vkSemaphore(vkSemaphore) {}
    ~Impl() {
        vkDestroySemaphore(vkDevice, vkSemaphore, nullptr);
    }
};


struct ShaderModule::Impl {
    const VkDevice vkDevice;
    const VkShaderModule vkModule;
    const SHADER_STAGE stage;
    void* pModule; // for SPIRV-Reflect

    Impl(VkDevice vkDevice, 
        VkShaderModule vkModule,
        SHADER_STAGE stage,
        void* pModule)
    : vkDevice(vkDevice)
    , vkModule(vkModule)
    , stage(stage)
    , pModule(pModule) {}
    ~Impl() {
        discardReflect();
        vkDestroyShaderModule(vkDevice, vkModule, nullptr);
    }

    void discardReflect() {
        if (pModule)
        {
            destroyReflectShaderModule(pModule);
            pModule = nullptr;
        }
    }
};


struct ComputePipeline::Impl {
    const VkDevice vkDevice;
    const VkPipeline vkPipeline;
    const PipelineLayout layout;
    const std::array<uint32_t, 3> workGroupSize;

    Impl(VkDevice vkDevice,             
        VkPipeline vkPipeline, 
        PipelineLayout layout,
        uint32_t sizeX,
        uint32_t sizeY,
        uint32_t sizeZ) 
    : vkDevice(vkDevice)
    , vkPipeline(vkPipeline)
    , layout(layout)
    , workGroupSize{sizeX, sizeY, sizeZ} {}
    ~Impl() {                           
        vkDestroyPipeline(vkDevice, vkPipeline, nullptr); 
    }
};


struct Buffer::Impl {
    const VkDevice vkDevice;
    const VkBuffer vkBuffer;
    const VkDeviceMemory vkMemory;
    const uint64_t size;
    const BUFFER_USAGE usage;
    const MEMORY_PROPERTY reqMemProps;
    const MEMORY_PROPERTY memProps;
    uint8_t* mapped = nullptr;
    uint64_t mappedOffset = 0;  // used for debug
    uint64_t mappedSize = 0;    // used for debug
    DeviceAddress deviceAddress = 0;

    Impl(VkDevice vkDevice, 
        VkBuffer vkBuffer, 
        VkDeviceMemory vkMemory, 
        uint64_t size, 
        BUFFER_USAGE usage,
        MEMORY_PROPERTY reqMemProps,
        MEMORY_PROPERTY memProps)
    : vkDevice(vkDevice)
    , vkBuffer(vkBuffer)
    , vkMemory(vkMemory)
    , size(size)
    , usage(usage)
    , reqMemProps(reqMemProps)
    , memProps(memProps) {}
    ~Impl() {         
        if (mapped) vkUnmapMemory(vkDevice, vkMemory);
        vkDestroyBuffer(vkDevice, vkBuffer, nullptr); 
        vkFreeMemory(vkDevice, vkMemory, nullptr); 
    }

    VkMappedMemoryRange getRange(uint64_t offset, uint64_t size) const;

};


struct Image::Impl {
    const VkDevice vkDevice;
    const VkImage vkImage;
    const VkDeviceMemory vkMemory;
    const FORMAT format;
    const uint32_t width;
    const uint32_t height;
    const uint32_t depth;
    const uint32_t arrayLayers;
    const IMAGE_USAGE usage;
    // IMAGE_LAYOUT currentLayout;
    const bool ownedBySwapchain;

    struct ImageViewDescHash {
        uint64_t operator()(const ImageViewDesc& d) const noexcept {
            uint64_t h = std::hash<int>{}(static_cast<int>(d.viewType));
            h = hashCombine(h, (uint64_t) d.format);
            h = hashCombine(h, (uint64_t) d.components.r);
            h = hashCombine(h, (uint64_t) d.components.g);
            h = hashCombine(h, (uint64_t) d.components.b);
            h = hashCombine(h, (uint64_t) d.components.a); 
            return h;
        }
    };
    mutable std::unordered_map<ImageViewDesc, ImageView, ImageViewDescHash> imageViews;

    Impl(VkDevice vkDevice, 
        VkImage vkImage, 
        VkDeviceMemory vkMemory, 
        FORMAT format,
        uint32_t width,
        uint32_t height,
        uint32_t depth,
        uint32_t arrayLayers,
        IMAGE_USAGE usage,
        // IMAGE_LAYOUT initialLayout,
        bool ownedBySwapchain)
    : vkDevice(vkDevice)
    , vkImage(vkImage)
    , vkMemory(vkMemory)
    , format(format)
    , width(width)
    , height(height)
    , depth(depth)
    , arrayLayers(arrayLayers)
    , usage(usage)
    // , currentLayout(initialLayout) 
    , ownedBySwapchain(ownedBySwapchain) {}
    ~Impl();
};


struct ImageView::Impl {
    const VkImageView vkImageView;

    Impl(VkImageView vkImageView)
    : vkImageView(vkImageView) {}
};


Image::Impl::~Impl() 
{                           
    for (auto& [_, view] : imageViews)
    {
        if(*view.ppImpl) 
        {
            vkDestroyImageView(vkDevice, view.impl().vkImageView, nullptr);
            delete *view.ppImpl;
        }
        delete view.ppImpl;
    } 

    if (!ownedBySwapchain)
    {
        vkDestroyImage(vkDevice, vkImage, nullptr); 
        vkFreeMemory(vkDevice, vkMemory, nullptr); 
    }
}


struct Sampler::Impl {
    const VkDevice vkDevice;
    const VkSampler vkSampler;

    Impl(VkDevice vkDevice, 
        VkSampler vkSampler)
    : vkDevice(vkDevice)
    , vkSampler(vkSampler) {}
    ~Impl() {
        vkDestroySampler(vkDevice, vkSampler, nullptr);
    }
};


struct DescriptorSetLayout::Impl {
    const VkDevice vkDevice;
    const VkDescriptorSetLayout vkSetLayout;
    const DescriptorSetLayoutDesc desc;

    Impl(VkDevice vkDevice,                    
        VkDescriptorSetLayout vkSetLayout,
        DescriptorSetLayoutDesc desc) 
    : vkDevice(vkDevice)
    , vkSetLayout(vkSetLayout)
    , desc(std::move(desc))
    {}
    ~Impl() {                                  
        vkDestroyDescriptorSetLayout(vkDevice, vkSetLayout, nullptr); 
    }
};


struct PipelineLayout::Impl {
    const VkDevice vkDevice;
    const VkPipelineLayout vkPipeLayout;
    const std::map<uint32_t, DescriptorSetLayout> setLayouts;
    // const std::vector<PushConstantRange> pushConstants;
    // const PushConstantRange uniquePushConstant;
    const std::unique_ptr<PushConstantRange> uniquePushConstant;

    Impl(VkDevice vkDevice,           
        VkPipelineLayout vkPipeLayout, 
        std::map<uint32_t, DescriptorSetLayout>&& setLayouts,
        std::unique_ptr<PushConstantRange> pushConstant)
    : vkDevice(vkDevice)
    , vkPipeLayout(vkPipeLayout)
    , setLayouts(std::move(setLayouts))
    , uniquePushConstant(std::move(pushConstant)) {}
    ~Impl() {
        vkDestroyPipelineLayout(vkDevice, vkPipeLayout, nullptr); 
    }
};


struct DescriptorPool::Impl {
    const VkDevice vkDevice;
    const VkDescriptorPool vkDescPool;
    std::deque<DescriptorSet> descSets;

    Impl(VkDevice vkDevice,             
        VkDescriptorPool vkDescPool) 
    : vkDevice(vkDevice)
    , vkDescPool(vkDescPool) {}
    ~Impl();
};


struct DescriptorSet::Impl {
    const VkDevice vkDevice;
    const VkDescriptorSet vkDescSet;
    const DescriptorSetLayout layout;

    Impl(VkDevice vkDevice,            
        VkDescriptorSet vkDescSet,
        DescriptorSetLayout layout) 
    : vkDevice(vkDevice)
    , vkDescSet(vkDescSet)
    , layout(layout) {}
};


DescriptorPool::Impl::~Impl()
{
    for (auto& descSet : descSets) 
    {
        if (*descSet.ppImpl) 
            delete *descSet.ppImpl;
        delete descSet.ppImpl;
    }
    vkDestroyDescriptorPool(vkDevice, vkDescPool, nullptr); 
}


#ifdef EVA_ENABLE_RAYTRACING
struct RaytracingPipeline::Impl {
    const VkDevice vkDevice;
    const VkPipeline vkPipeline;
    const PipelineLayout layout;

    std::vector<ShaderGroupHandle> hitGroupHandles;

    struct Sbt {
        Buffer buffer;
        VkStridedDeviceAddressRegionKHR rgen{};
        VkStridedDeviceAddressRegionKHR miss{};
        VkStridedDeviceAddressRegionKHR hitGroup{};
    } sbt;

    Impl(VkDevice vkDevice,
        VkPipeline vkPipeline,
        PipelineLayout layout)
    : vkDevice(vkDevice)
    , vkPipeline(vkPipeline)
    , layout(layout) {}
    ~Impl() {
        vkDestroyPipeline(vkDevice, vkPipeline, nullptr);
    }
};


struct AccelerationStructure::Impl {
    VkDevice vkDevice;
    VkAccelerationStructureKHR vkAccelStruct;
    DeviceAddress deviceAddress;

    Impl(
        VkDevice vkDevice,
        VkAccelerationStructureKHR vkAccelStruct)
    : vkDevice(vkDevice)
    , vkAccelStruct(vkAccelStruct)
    {}
    ~Impl() {
        vkDestroyAccelerationStructureKHR_(vkDevice, vkAccelStruct, nullptr);
    }
};
#endif


struct QueryPool::Impl {
    VkDevice vkDevice;
    VkQueryPool vkQueryPool;
    uint32_t queryCount;
    float timestampPeriod;  // nanoseconds per tick

    Impl(VkDevice vkDevice, VkQueryPool vkQueryPool, uint32_t queryCount, float timestampPeriod)
    : vkDevice(vkDevice)
    , vkQueryPool(vkQueryPool)
    , queryCount(queryCount)
    , timestampPeriod(timestampPeriod)
    {}

    ~Impl() {
        vkDestroyQueryPool(vkDevice, vkQueryPool, nullptr);
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Runtime
/////////////////////////////////////////////////////////////////////////////////////////
Runtime& Runtime::get()
{
    static Runtime singleton;
    return singleton;
}

Runtime::Runtime()
{
#ifdef EVA_ENABLE_WINDOW
    glfwInit();
#endif
    pImpl = new Impl(createVkInstance());
}

Runtime::~Runtime()
{
#ifdef EVA_ENABLE_WINDOW
    for (auto& window : impl().windows)
    {
        window.destroy();
        delete window.ppImpl;
    }
#endif

    for (auto& device : impl().devices)
    {
        device.destroy();
        delete device.ppImpl;
    }

    vkDestroyInstance(impl().instance, nullptr);
    delete pImpl;

#ifdef EVA_ENABLE_WINDOW
    // glfwTerminate();
#endif
}

void* Runtime::nativeInstance() const { return impl().instance; }

uint32_t Runtime::deviceCount() const
{
    return (uint32_t)impl().devices.size();
}

Device Runtime::device(int gpuIndex)
{
    if (gpuIndex < 0)
        gpuIndex += (int)impl().devices.size();
        
    EVA_ASSERT(gpuIndex >= 0 && gpuIndex < (int)impl().devices.size());
    return impl().devices[gpuIndex];
}

Device Runtime::device(DeviceSettings settings)
{
    for (auto& device : impl().devices) 
    {
        if (device && settings <= device.impl().settings)
            return device;
    }
    return createDevice(settings);
}


struct PNextChain {
    void* head = nullptr;
    void* tail = nullptr;

    struct Block {
        void* ptr{};
        std::size_t size{};
    };
    std::vector<Block> blocks;

    ~PNextChain() { reset(); }

    void reset() {
        for (auto& b : blocks) {
            if (b.ptr) ::operator delete(b.ptr); // 정렬 태그 없이 delete
        }
        blocks.clear();
        head = nullptr;
        tail = nullptr;
    }

    void* operator&() { return head; }

    template<class T>
    T& add(T&& s) {
        using U = std::decay_t<T>;
        void* mem = ::operator new(sizeof(U)); // 일반 new: max_align_t 정렬 보장
        std::memcpy(mem, &s, sizeof(U));
        blocks.push_back({mem, sizeof(U)});

        auto* p = reinterpret_cast<U*>(mem);
        // VkBaseOutStructure(또는 VkBaseInStructure)로 캐스팅해 pNext 연결
        auto* base = reinterpret_cast<VkBaseOutStructure*>(p);
        base->pNext = nullptr;

        if (tail) {
            reinterpret_cast<VkBaseOutStructure*>(tail)->pNext = base;
        } else {
            head = p;
        }
        tail = p;
        return *p;
    }
};

Device Runtime::createDevice(const DeviceSettings& settings)
{
    auto physicalDevices = arrayFrom(vkEnumeratePhysicalDevices, impl().instance);
    
    printf("**************************************************************\n");
    printf("Detected %zu physical devices:\n", physicalDevices.size());
    for (uint32_t i = 0; i < physicalDevices.size(); ++i) 
        printGpuInfo(i, physicalDevices[i]);
    fflush(stdout);

    int selected = 0;

    if (physicalDevices.size() > 1)
    {
        while(true)
        {
            printf("Select a physical device (0-%d): ", (uint32_t)physicalDevices.size() - 1);
            fflush(stdout);
            scanf("%d", &selected);
            if (selected < 0 || selected >= (int)physicalDevices.size())
            {            
                fprintf(stderr, "Invalid index %d! Please select again.\n", selected);
                continue;
            }
            break;
        }
    }

    printf("[GPU %d] is Selected.\n", selected);
    printf("**************************************************************\n\n");
    fflush(stdout);

    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////
    VkPhysicalDevice pd = physicalDevices[selected];
    auto qfProps = arrayFrom(vkGetPhysicalDeviceQueueFamilyProperties, pd);

    // Query optional atomic float feature support
    bool hasAtomicFloatExt = deviceSupportsExtensions(pd, {VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME});
    bool supportsAtomicFloat = false;
    if (hasAtomicFloatExt) {
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
        atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &atomicFloatFeatures;
        vkGetPhysicalDeviceFeatures2(pd, &features2);

        supportsAtomicFloat = atomicFloatFeatures.shaderBufferFloat32AtomicAdd;
    }
    printf("        Atomic Float (shaderBufferFloat32AtomicAdd): %s\n", supportsAtomicFloat ? "Supported" : "Not Supported");
    fflush(stdout);

    // LocalSizeId in SPIR-V requires maintenance4 feature enabled.
    VkPhysicalDeviceMaintenance4Features maintenance4Features{};
    maintenance4Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    {
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &maintenance4Features;
        vkGetPhysicalDeviceFeatures2(pd, &features2);
    }
    if (!maintenance4Features.maintenance4) {
        throw std::runtime_error("The selected physical device does not support maintenance4 (required by LocalSizeId shaders).");
    }

    // Required extensions
    std::vector<const char*> reqExtentions = {
        VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME
    };

    // Optional: shader atomic float
    if (supportsAtomicFloat)
    {
        reqExtentions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    }
    else
    {
        printf("[EVA] VK_EXT_shader_atomic_float is not supported on this device. Atomic float path disabled.\n");
    }

#ifdef EVA_ENABLE_WINDOW
    if (settings.enableWindow)
    {
        bool presentSupport = false;
        for (uint32_t i = 0; i < qfProps.size(); ++i)
            presentSupport |= (bool) glfwGetPhysicalDevicePresentationSupport(impl().instance, pd, i);

        if (!presentSupport)
            throw std::runtime_error("The selected physical device does not support presentation.");

        reqExtentions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
#endif

#ifdef EVA_ENABLE_RAYTRACING
    if (settings.enableRaytracing)
    {
        reqExtentions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        reqExtentions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        reqExtentions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        // From here, extensions required by the above extensions (extension hierarchy)
        reqExtentions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        reqExtentions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        reqExtentions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    }
#endif

    // VK_KHR_cooperative_matrix for Tensor Core GEMM
    // VK_NV_cooperative_matrix2 for tensor operations (llama.cpp style)
    bool cooperative_matrix_supported = false;
    bool coopmat2_supported = false;
    if (settings.enableCooperativeMatrix)
    {
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
        reqExtentions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        cooperative_matrix_supported = true;
#else
        printf("[EVA] VK_KHR_cooperative_matrix is not available in this Vulkan SDK/headers. Cooperative matrix disabled.\n");
#endif

        // Check if VK_NV_cooperative_matrix2 extension is available
        auto deviceExtensions = arrayFrom(vkEnumerateDeviceExtensionProperties, pd, nullptr);
        bool coopmat2_ext_available = std::any_of(deviceExtensions.begin(), deviceExtensions.end(),
            [](const auto& props) {
                return strcmp(props.extensionName, "VK_NV_cooperative_matrix2") == 0;
            });

#ifdef VK_NV_COOPERATIVE_MATRIX_2_SPEC_VERSION
        if (coopmat2_ext_available)
        {
            // Query coopmat2 features (llama.cpp style)
            VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features{};
            coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
            coopmat2_features.pNext = nullptr;

            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &coopmat2_features;
            vkGetPhysicalDeviceFeatures2(pd, &features2);

            // Check all required features (llama.cpp checks these)
            if (coopmat2_features.cooperativeMatrixWorkgroupScope &&
                coopmat2_features.cooperativeMatrixFlexibleDimensions &&
                coopmat2_features.cooperativeMatrixReductions &&
                coopmat2_features.cooperativeMatrixConversions &&
                coopmat2_features.cooperativeMatrixPerElementOperations &&
                coopmat2_features.cooperativeMatrixTensorAddressing &&
                coopmat2_features.cooperativeMatrixBlockLoads)
            {
                reqExtentions.push_back("VK_NV_cooperative_matrix2");
                coopmat2_supported = true;
                printf("[EVA] VK_NV_cooperative_matrix2 enabled (all features supported)\n");
            }
            else
            {
                printf("[EVA] VK_NV_cooperative_matrix2 available but missing features:\n");
                printf("  WorkgroupScope=%d FlexibleDims=%d Reductions=%d Conversions=%d\n",
                    coopmat2_features.cooperativeMatrixWorkgroupScope,
                    coopmat2_features.cooperativeMatrixFlexibleDimensions,
                    coopmat2_features.cooperativeMatrixReductions,
                    coopmat2_features.cooperativeMatrixConversions);
                printf("  PerElementOps=%d TensorAddr=%d BlockLoads=%d\n",
                    coopmat2_features.cooperativeMatrixPerElementOperations,
                    coopmat2_features.cooperativeMatrixTensorAddressing,
                    coopmat2_features.cooperativeMatrixBlockLoads);
            }
        }
#else
        if (coopmat2_ext_available)
        {
            printf("[EVA] VK_NV_cooperative_matrix2 extension is present but feature structs are unavailable in current Vulkan headers. coopmat2 disabled.\n");
        }
#endif
    }

    // VK_KHR_pipeline_executable_properties for SASS dump
    if (settings.enablePipelineExecutableInfo)
    {
        reqExtentions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        printf("[EVA] VK_KHR_pipeline_executable_properties enabled\n");
    }

    // VK_NV_cuda_kernel_launch for CUDA PTX kernels in Vulkan command buffers
    bool cuda_kernel_launch_supported = false;
    if (settings.enableCudaKernelLaunch)
    {
        // Check if extension is available
        auto deviceExtensions_cuda = arrayFrom(vkEnumerateDeviceExtensionProperties, pd, nullptr);
        bool cuda_ext_available = std::any_of(deviceExtensions_cuda.begin(), deviceExtensions_cuda.end(),
            [](const auto& props) {
                return strcmp(props.extensionName, "VK_NV_cuda_kernel_launch") == 0;
            });

        if (cuda_ext_available)
        {
            // Query feature support
            // Define struct manually to avoid vulkan_beta.h dependency
            struct VkPhysicalDeviceCudaKernelLaunchFeaturesNV_t {
                VkStructureType sType;
                void* pNext;
                VkBool32 cudaKernelLaunchFeatures;
            };
            // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV = 1000307003
            VkPhysicalDeviceCudaKernelLaunchFeaturesNV_t cuda_features{};
            cuda_features.sType = (VkStructureType)1000307003;
            cuda_features.pNext = nullptr;

            VkPhysicalDeviceFeatures2 features2_cuda{};
            features2_cuda.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2_cuda.pNext = &cuda_features;
            vkGetPhysicalDeviceFeatures2(pd, &features2_cuda);

            if (cuda_features.cudaKernelLaunchFeatures)
            {
                reqExtentions.push_back("VK_NV_cuda_kernel_launch");
                cuda_kernel_launch_supported = true;
                printf("[EVA] VK_NV_cuda_kernel_launch enabled (cudaKernelLaunchFeatures supported)\n");
            }
            else
            {
                printf("[EVA] VK_NV_cuda_kernel_launch available but cudaKernelLaunchFeatures not supported\n");
            }
        }
        else
        {
            printf("[EVA] VK_NV_cuda_kernel_launch not available on this device\n");
        }
    }

    if (!deviceSupportsExtensions(pd, reqExtentions))
    {
        throw std::runtime_error("The selected physical device does not support the required extensions.");
    }
    
    std::vector<uint32_t> qfIndices[queue_max];
    for (uint32_t i = 0; i < qfProps.size(); i++) 
    {
        if (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT 
            && qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) 
            qfIndices[queue_graphics].push_back(i);
        
        else if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) 
            qfIndices[queue_compute].push_back(i);
        
        else if (qfProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) 
            qfIndices[queue_transfer].push_back(i);
    }

    if(qfIndices[queue_graphics].size() > 1) 
    {
        printf("[Note] Multiple queue families supporting Graphics (and Compute) operations were detected:\n");
        for (uint32_t i : qfIndices[queue_graphics])
            printQueueFamily(i, qfProps[i].queueCount, qfProps[i].queueFlags);
        printf("The first available queue family will be selected.\n");
    }

    if(qfIndices[queue_compute].size() > 1) 
    {
        printf("[Note] Multiple queue families supporting Compute operations were detected:\n");
        for (uint32_t i : qfIndices[queue_compute])
            printQueueFamily(i, qfProps[i].queueCount, qfProps[i].queueFlags);
        printf("The first available queue family will be selected.\n");
    }

    if(qfIndices[queue_transfer].size() > 1) 
    {
        printf("[Note] Multiple queue families supporting Transfer operations were detected:\n");
        for (uint32_t i : qfIndices[queue_transfer])
            printQueueFamily(i, qfProps[i].queueCount, qfProps[i].queueFlags);
        printf("The first available queue family will be selected.\n");
    }
    fflush(stdout);

    uint32_t qfIndex[queue_max] = { uint32_t(-1), uint32_t(-1), uint32_t(-1) };

    if (settings.enableGraphicsQueues) 
    {
        if (!qfIndices[queue_graphics].empty()) 
        {
            qfIndex[queue_graphics] = qfIndices[queue_graphics][0];
        } 
        else 
        {
            throw std::runtime_error("No queue family with Graphics support found.");
        }
    } 
    EVA_ASSERT(!settings.enableGraphicsQueues || qfIndex[queue_graphics] != uint32_t(-1));

    if (settings.enableComputeQueues) 
    {
        if (!qfIndices[queue_compute].empty()) 
        {
            qfIndex[queue_compute] = qfIndices[queue_compute][0];
        } 
        else
        {
            if (qfIndex[queue_graphics] != uint32_t(-1)) 
            {
                qfIndex[queue_compute] = qfIndex[queue_graphics];
            } 
            else // settings.enableGraphicsQueues == false
            {
                if (!qfIndices[queue_graphics].empty())
                {
                    qfIndex[queue_compute] = qfIndices[queue_graphics][0];
                } 
                else // qfIndices[queue_compute].empty() && qfIndices[queue_graphics].empty()
                {
                    throw std::runtime_error("No queue family with Compute support found.");
                }
            }
        }
    } 
    EVA_ASSERT(!settings.enableComputeQueues || qfIndex[queue_compute] != uint32_t(-1));

    if (settings.enableTransferQueues) 
    {
        if (!qfIndices[queue_transfer].empty()) 
        {
            qfIndex[queue_transfer] = qfIndices[queue_transfer][0];
        } 
        /*
        The rule of fallback selection:
        - Prefer an unused queue type over one already in use.
        - Prefer a graphics queue over a compute queue.
        */
        else
        {
            if (!settings.enableGraphicsQueues && !qfIndices[queue_graphics].empty())
            {
                qfIndex[queue_transfer] = qfIndices[queue_graphics][0];
            }
            else if (!settings.enableComputeQueues && !qfIndices[queue_compute].empty()) 
            {
                qfIndex[queue_transfer] = qfIndices[queue_compute][0];
            } 

            else if (qfIndex[queue_graphics] != uint32_t(-1)) 
            {
                qfIndex[queue_transfer] = qfIndex[queue_graphics];
            } 
            else if (qfIndex[queue_compute] != uint32_t(-1)) 
            {
                qfIndex[queue_transfer] = qfIndex[queue_compute];
            } 
            else // qfIndices[queue_transfer].empty() && qfIndices[queue_graphics].empty() && qfIndices[queue_compute].empty()
            {
                throw std::runtime_error("No queue family with Transfer support found.");
            }
        }
    }
    EVA_ASSERT(!settings.enableTransferQueues || qfIndex[queue_transfer] != uint32_t(-1));
    
    std::set<uint32_t> uniqueQfIndices = {
        qfIndex[queue_graphics],
        qfIndex[queue_compute],
        qfIndex[queue_transfer]
    };
    
    std::vector<VkDeviceQueueCreateInfo> queueFamilyInfos;
    std::vector<std::vector<float>> priorities(qfProps.size());
    for (auto qfIndex : uniqueQfIndices) 
    {
        if (qfIndex == uint32_t(-1)) 
            continue;

        priorities[qfIndex].resize(qfProps[qfIndex].queueCount, 0.5f);

        queueFamilyInfos.push_back(VkDeviceQueueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VkDeviceQueueCreateFlags(0),
            .queueFamilyIndex = qfIndex,
            .queueCount = qfProps[qfIndex].queueCount,
            .pQueuePriorities = priorities[qfIndex].data()     // TODO: Set queue priorities (How to set accross different types but same family?)
        }); // TODO: Linux Clang 호환성 - emplace_back → push_back + designated initializer
    }

    PNextChain chain;
    {
        chain.add(VkPhysicalDeviceFeatures2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        });

        // Required features
        chain.add(VkPhysicalDeviceSynchronization2Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
            .synchronization2 = VK_TRUE,
        });

        chain.add(VkPhysicalDeviceMaintenance4Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES,
            .maintenance4 = VK_TRUE,
        });

        chain.add(VkPhysicalDeviceVulkan11Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .storageBuffer16BitAccess = VK_TRUE,
        });
        chain.add(VkPhysicalDeviceShaderFloat16Int8Features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
            .shaderFloat16 = VK_TRUE,
        });
        chain.add(VkPhysicalDeviceBufferDeviceAddressFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
            .bufferDeviceAddress = VK_TRUE,
        });

        // chain.add(VkPhysicalDeviceRobustness2FeaturesEXT{
        //     .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
        //     .nullDescriptor = VK_TRUE,
        // });

        // chain.add(VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT{
        //     .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
        //     .graphicsPipelineLibrary = VK_TRUE,
        // });
        // Optional features
        if (supportsAtomicFloat) {
            chain.add(VkPhysicalDeviceShaderAtomicFloatFeaturesEXT{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
                .shaderBufferFloat32AtomicAdd = VK_TRUE,
            });
        }

#ifdef EVA_ENABLE_RAYTRACING
        if (settings.enableRaytracing)
        {
            chain.add(VkPhysicalDeviceBufferDeviceAddressFeatures{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
                .bufferDeviceAddress = VK_TRUE,
            });
            chain.add(VkPhysicalDeviceAccelerationStructureFeaturesKHR{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
                .accelerationStructure = VK_TRUE,
            });
            chain.add(VkPhysicalDeviceRayTracingPipelineFeaturesKHR{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
                .rayTracingPipeline = VK_TRUE,
            });
        }
#endif

        // VK_KHR_cooperative_matrix for Tensor Core GEMM
        if (settings.enableCooperativeMatrix && cooperative_matrix_supported)
        {
#ifdef VK_KHR_cooperative_matrix
            chain.add(VkPhysicalDeviceCooperativeMatrixFeaturesKHR{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
                .cooperativeMatrix = VK_TRUE,
            });
#else
            printf("[EVA] VkPhysicalDeviceCooperativeMatrixFeaturesKHR is unavailable in current Vulkan headers. Cooperative matrix disabled.\n");
#endif

            // VK_NV_cooperative_matrix2 for tensor operations (coopMatLoadTensorNV etc.)
            // Enable all features if supported (detected in extension query phase)
            if (coopmat2_supported)
            {
#ifdef VK_NV_COOPERATIVE_MATRIX_2_SPEC_VERSION
                chain.add(VkPhysicalDeviceCooperativeMatrix2FeaturesNV{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
                    .cooperativeMatrixWorkgroupScope = VK_TRUE,
                    .cooperativeMatrixFlexibleDimensions = VK_TRUE,
                    .cooperativeMatrixReductions = VK_TRUE,
                    .cooperativeMatrixConversions = VK_TRUE,
                    .cooperativeMatrixPerElementOperations = VK_TRUE,
                    .cooperativeMatrixTensorAddressing = VK_TRUE,
                    .cooperativeMatrixBlockLoads = VK_TRUE,
                });

                // Set global flag for runtime query
                eva::gCoopMat2Supported = true;
#else
                printf("[EVA] coopmat2 features are unavailable in current Vulkan headers. coopmat2 disabled.\n");
#endif
            }
        }

        // VK_KHR_pipeline_executable_properties for SASS dump
        if (settings.enablePipelineExecutableInfo)
        {
            chain.add(VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
                .pipelineExecutableInfo = VK_TRUE,
            });
        }

        // VK_NV_cuda_kernel_launch for CUDA PTX kernels
        if (cuda_kernel_launch_supported)
        {
            // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV = 1000307003
            struct CudaKernelLaunchFeatures {
                VkStructureType sType;
                void* pNext;
                VkBool32 cudaKernelLaunchFeatures;
            };
            CudaKernelLaunchFeatures cuda_feat{};
            cuda_feat.sType = (VkStructureType)1000307003;
            cuda_feat.pNext = nullptr;
            cuda_feat.cudaKernelLaunchFeatures = VK_TRUE;
            chain.add(cuda_feat);

            eva::gCudaKernelLaunchSupported = true;
        }
    }

    VkDeviceCreateInfo deviceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &chain,
        .queueCreateInfoCount = (uint32_t)queueFamilyInfos.size(),
        .pQueueCreateInfos = queueFamilyInfos.data(),
        .enabledExtensionCount = (uint32_t)reqExtentions.size(),
        .ppEnabledExtensionNames = reqExtentions.data(),
    };
    
    VkDevice vkDevice = create<VkDevice>(pd, deviceCreateInfo);
    
    std::vector<std::vector<Queue>> queues(qfProps.size());
    for (auto qfIndex : uniqueQfIndices) 
    {
        if (qfIndex == uint32_t(-1)) 
            continue;

        queues[qfIndex].resize(qfProps[qfIndex].queueCount);

        for (uint32_t j = 0; j < qfProps[qfIndex].queueCount; ++j) 
        {
            VkQueue vkQueue;
            vkGetDeviceQueue(vkDevice, qfIndex, j, &vkQueue);
            auto pImpl = new Queue::Impl(
                vkQueue,
                qfIndex,
                j,
                priorities[qfIndex][j]);

            queues[qfIndex][j].ppImpl = new Queue::Impl*(pImpl);
        }
    }

    // Populate DeviceCapabilities
    DeviceCapabilities caps;
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);

        caps.deviceName = props.deviceName;
        caps.isDiscreteGpu = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
        caps.maxWorkgroupSize[0] = props.limits.maxComputeWorkGroupSize[0];
        caps.maxWorkgroupSize[1] = props.limits.maxComputeWorkGroupSize[1];
        caps.maxWorkgroupSize[2] = props.limits.maxComputeWorkGroupSize[2];
        caps.maxWorkgroupInvocations = props.limits.maxComputeWorkGroupInvocations;
        caps.maxSharedMemorySize = props.limits.maxComputeSharedMemorySize;

        // Subgroup properties
        VkPhysicalDeviceSubgroupProperties subgroupProps{};
        subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroupProps;
        vkGetPhysicalDeviceProperties2(pd, &props2);

        caps.subgroupSize = subgroupProps.subgroupSize;
        caps.subgroupOperations = subgroupProps.supportedOperations;

        // Optional features
        caps.supportsAtomicFloat = supportsAtomicFloat;
        caps.supportsAtomicFloatShared = false;  // TODO: query if needed

        // Device local memory
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(pd, &memProps);
        caps.deviceLocalMemorySize = 0;
        for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
            if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                caps.deviceLocalMemorySize = memProps.memoryHeaps[i].size;
                break;
            }
        }
    }

    auto pImpl = new Device::Impl(
        pd,
        vkDevice,
        *this,
        settings,
        caps,
        qfIndex[queue_graphics],
        qfIndex[queue_compute],
        qfIndex[queue_transfer],
        std::move(queues)
    );

#ifdef EVA_ENABLE_RAYTRACING
    if (settings.enableRaytracing)
    {
        vkGetBufferDeviceAddressKHR_ = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(vkDevice, "vkGetBufferDeviceAddressKHR");
        vkCreateAccelerationStructureKHR_ = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(vkDevice, "vkCreateAccelerationStructureKHR");
        vkDestroyAccelerationStructureKHR_ = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(vkDevice, "vkDestroyAccelerationStructureKHR");
        vkGetAccelerationStructureBuildSizesKHR_ = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(vkDevice, "vkGetAccelerationStructureBuildSizesKHR");
        vkGetAccelerationStructureDeviceAddressKHR_ = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(vkDevice, "vkGetAccelerationStructureDeviceAddressKHR");
        vkCmdBuildAccelerationStructuresKHR_ = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(vkDevice, "vkCmdBuildAccelerationStructuresKHR");
        vkCreateRayTracingPipelinesKHR_ = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(vkDevice, "vkCreateRayTracingPipelinesKHR");
        vkGetRayTracingShaderGroupHandlesKHR_ = (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(vkDevice, "vkGetRayTracingShaderGroupHandlesKHR");
        vkCmdTraceRaysKHR_ = (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(vkDevice, "vkCmdTraceRaysKHR");

        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProps{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
        };
        VkPhysicalDeviceAccelerationStructurePropertiesKHR accelStructProps{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
            .pNext = &rayTracingPipelineProps,
        };
        VkPhysicalDeviceProperties2 deviceProps2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &accelStructProps,
        };

        vkGetPhysicalDeviceProperties2(pd, &deviceProps2);
        pImpl->rtProps.shaderGroupHandleAlignment = rayTracingPipelineProps.shaderGroupHandleAlignment;
        pImpl->rtProps.shaderGroupBaseAlignment = rayTracingPipelineProps.shaderGroupBaseAlignment;
        pImpl->rtProps.minAccelerationStructureScratchOffsetAlignment = accelStructProps.minAccelerationStructureScratchOffsetAlignment;
    }
#endif

    // @chay116 - Load vkGetBufferDeviceAddress (always, BDA used by all GEMM paths)
    // Note: In Vulkan 1.2+, bufferDeviceAddress is a core feature (no KHR suffix)
    if (vkGetBufferDeviceAddressKHR_ == nullptr)
    {
        // Try core Vulkan 1.2 version first
        vkGetBufferDeviceAddressKHR_ = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(vkDevice, "vkGetBufferDeviceAddress");
        // Fallback to KHR version if core not available
        if (vkGetBufferDeviceAddressKHR_ == nullptr)
            vkGetBufferDeviceAddressKHR_ = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(vkDevice, "vkGetBufferDeviceAddressKHR");

        if (vkGetBufferDeviceAddressKHR_)
            printf("[EVA] vkGetBufferDeviceAddress loaded successfully!\n");
        else
            printf("[EVA] ERROR: Failed to load vkGetBufferDeviceAddress!\n");
    }

    // Load pipeline executable properties function pointers
    if (settings.enablePipelineExecutableInfo)
    {
        vkGetPipelineExecutablePropertiesKHR_ = (PFN_vkGetPipelineExecutablePropertiesKHR)
            vkGetDeviceProcAddr(vkDevice, "vkGetPipelineExecutablePropertiesKHR");
        vkGetPipelineExecutableInternalRepresentationsKHR_ = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)
            vkGetDeviceProcAddr(vkDevice, "vkGetPipelineExecutableInternalRepresentationsKHR");
        vkGetPipelineExecutableStatisticsKHR_ = (PFN_vkGetPipelineExecutableStatisticsKHR)
            vkGetDeviceProcAddr(vkDevice, "vkGetPipelineExecutableStatisticsKHR");

        if (vkGetPipelineExecutablePropertiesKHR_ && vkGetPipelineExecutableInternalRepresentationsKHR_)
            printf("[EVA] Pipeline executable properties loaded (SASS dump available)\n");
        else
            printf("[EVA] ERROR: Failed to load pipeline executable properties functions!\n");
    }

    return impl().devices.emplace_back(new Device::Impl*(pImpl));
}


/////////////////////////////////////////////////////////////////////////////////////////
// Device
/////////////////////////////////////////////////////////////////////////////////////////
Device::Impl::~Impl() 
{
    vkDeviceWaitIdle(vkDevice);

    for (auto qs : queues) 
    {
        for (auto q : qs) 
        {
            if (*q.ppImpl) delete *q.ppImpl;
            delete q.ppImpl;
        }
    }

    auto deleter = [](auto& ppImpls) 
    {
        for (auto ppImpl : ppImpls) 
        {
            if (*ppImpl) delete *ppImpl;
            delete ppImpl;
        }
    };

    deleter(cmdPools);
    
    deleter(fences);
    deleter(semaphores);

    deleter(shaderModules);
    deleter(computePipelines);
    
    deleter(buffers);
    deleter(images);
    deleter(samplers);
    
    deleter(descSetLayouts);
    deleter(pipelineLayouts);
    deleter(descPools);
    deleter(queryPools);

#ifdef EVA_ENABLE_RAYTRACING
    deleter(raytracingPipelines);
    deleter(accelerationStructures);
#endif

    vkDestroyDevice(vkDevice, nullptr);
}

void* Device::nativeDevice() const { return impl().vkDevice; }
void* Device::nativePhysicalDevice() const { return impl().vkPhysicalDevice; }
bool Device::hasShaderAtomicFloat() const
{
    auto deviceExtensions = arrayFrom(vkEnumerateDeviceExtensionProperties, impl().vkPhysicalDevice, nullptr);
    return std::any_of(deviceExtensions.begin(), deviceExtensions.end(), [](const auto& props) {
        return strcmp(props.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0;
    });
}

void Device::reportGPUQueueFamilies() const
{
    auto qfProps = arrayFrom(vkGetPhysicalDeviceQueueFamilyProperties, impl().vkPhysicalDevice);  
    printf("The device's total queue families:\n");
    for (uint32_t i = 0; i < qfProps.size(); ++i) 
        printQueueFamily(i, qfProps[i].queueCount, qfProps[i].queueFlags);
    fflush(stdout);
}

void Device::reportAssignedQueues() const
{
    printf("Every assigned queues for the logical device:\n");
    auto reportQfs = [&](const std::vector<Queue>& qs) {
        uint32_t i = 0;
        for (auto q : qs) 
        printf("  %u => Family Index: %u, Priority: %.2f\n", 
            i++, 
            q.queueFamilyIndex(),
            q.priority());
    };
    printf("***Graphics Queues***\n");
    reportQfs(impl().queues[impl().qfIndex[queue_graphics]]);
    printf("***Compute Queues***\n");
    reportQfs(impl().queues[impl().qfIndex[queue_compute]]);
    printf("***Transfer Queues***\n");
    reportQfs(impl().queues[impl().qfIndex[queue_transfer]]);
    fflush(stdout);
}

const DeviceCapabilities& Device::capabilities() const
{
    return impl().capabilities;
}

uint32_t Device::queueCount(QueueType type) const 
{ 
    return impl().qCount[type];  // 0 if and only if impl().qfIndex[type] == uint32_t(-1)
}

// bool Device::supportPresent(QueueType type) const 
// { 
//     return impl().qfIndex[type] != uint32_t(-1) ? false : 
//         impl().parent.impl().qfSupportPresent.at(impl().vkPhysicalDevice)[impl().qfIndex[type]];
// }

Queue Device::queue(QueueType type, uint32_t index) const 
{ 
    EVA_ASSERT(impl().qfIndex[type] != uint32_t(-1));
    Queue q = impl().queues[impl().qfIndex[type]] [index % impl().qCount[type]];
    q._type = type;
    return q;
}

QueueSelector Device::queue(uint32_t index) const 
{ 
    return QueueSelector(*this, index);
}

CommandPool Device::setDefalutCommandPool(QueueType type, CommandPool cmdPool)
{
    EVA_ASSERT(impl().qfIndex[type] != uint32_t(-1));
    impl().defaultCmdPool[type][(uint32_t)cmdPool.impl().flags] = cmdPool;
    return cmdPool;
}

CommandBuffer Device::newCommandBuffer(QueueType type, COMMAND_POOL_CREATE poolFlags)
{
    return newCommandBuffers(1, type, poolFlags)[0];
}

std::vector<CommandBuffer> Device::newCommandBuffers(uint32_t count, QueueType type, COMMAND_POOL_CREATE poolFlags)
{
    EVA_ASSERT(impl().qfIndex[type] != uint32_t(-1));
    if (!impl().defaultCmdPool[type][(uint32_t)poolFlags])
        impl().defaultCmdPool[type][(uint32_t)poolFlags] = createCommandPool(type, poolFlags);

    return impl().defaultCmdPool[type][(uint32_t)poolFlags].newCommandBuffers(count);
}


/////////////////////////////////////////////////////////////////////////////////////////
// Queue
/////////////////////////////////////////////////////////////////////////////////////////
QueueType Queue::type() const
{
    return _type;
}

uint32_t Queue::queueFamilyIndex() const 
{ 
    return impl().qfIndex; 
}

uint32_t Queue::index() const 
{ 
    return impl().index; 
}

float Queue::priority() const 
{ 
    return impl().priority; 
}

Queue Queue::submit(CommandBuffer cmdBuffer)
{
    // EVA_ASSERT(cmdBuffer.queueFamilyIndex() == impl().qfIndex); // VUID-vkQueueSubmit-pCommandBuffers-00074
    EVA_ASSERT(_type == cmdBuffer.type()); // VUID-vkQueueSubmit-pCommandBuffers-00074
    cmdBuffer.impl().lastSubmittedQueue = *this;

    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuffer.impl().vkCmdBuffer,
    };
    ASSERT_SUCCESS(vkQueueSubmit(impl().vkQueue, 1, &submitInfo, VK_NULL_HANDLE));
    return *this;
}

Queue Queue::submit(std::vector<CommandBuffer> cmdBuffers)
{
    for (auto& cmdBuffer : cmdBuffers) {
        // EVA_ASSERT(cmdBuffer.queueFamilyIndex() == impl().qfIndex); // VUID-vkQueueSubmit-pCommandBuffers-00074
        EVA_ASSERT(_type == cmdBuffer.type()); // VUID-vkQueueSubmit-pCommandBuffers-00074
        cmdBuffer.impl().lastSubmittedQueue = *this;
    }

    std::vector<VkCommandBuffer> vkCmdBuffers(cmdBuffers.size());
    for (uint32_t i = 0; i < cmdBuffers.size(); ++i) 
        vkCmdBuffers[i] = cmdBuffers[i].impl().vkCmdBuffer;

    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = (uint32_t)vkCmdBuffers.size(),
        .pCommandBuffers = vkCmdBuffers.data(),
    };
    ASSERT_SUCCESS(vkQueueSubmit(impl().vkQueue, 1, &submitInfo, VK_NULL_HANDLE));
    return *this;
}

Queue Queue::submit(std::vector<SubmissionBatchInfo>&& batches, std::optional<Fence> fence)
{
    size_t batchCount = batches.size();
    size_t waitSemCount = 0;
    size_t cmdBuffCount = 0;
    size_t signalSemCount = 0;
    for (uint32_t i=0; i<batchCount; ++i) 
    {
        waitSemCount += std::get<0>(batches[i]).size();
        cmdBuffCount += std::get<1>(batches[i]).size();
        signalSemCount += std::get<2>(batches[i]).size();
    }

    uint32_t waitSemOffset = 0;
    uint32_t cmdBufferOffset = 0;
    uint32_t signalSemOffset = 0;
    
#ifdef VULKAN_VERSION_1_3
    std::vector<VkSubmitInfo2> submitInfos(batchCount, {VK_STRUCTURE_TYPE_SUBMIT_INFO_2});

    std::vector<VkSemaphoreSubmitInfo> waitSems; waitSems.reserve(waitSemCount);
    std::vector<VkCommandBufferSubmitInfo> cmdBuffers; cmdBuffers.reserve(cmdBuffCount);
    std::vector<VkSemaphoreSubmitInfo> signalSems; signalSems.reserve(signalSemCount);  
    
    for (uint32_t i=0; i<batchCount; ++i) 
    {
        auto& [inWaitSems, inCmdBuffers, inSignalSems] = batches[i];
        VkSubmitInfo2& info = submitInfos[i];

        info.waitSemaphoreInfoCount = (uint32_t) inWaitSems.size();
        info.pWaitSemaphoreInfos = waitSems.data() + waitSemOffset;

        for (auto& inWaitSem : inWaitSems)
        {
            waitSems.push_back(VkSemaphoreSubmitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .semaphore = inWaitSem.sem.impl().vkSemaphore,
                .value = 0,
                .stageMask = (VkPipelineStageFlags2)(uint64_t)inWaitSem.stage,
                .deviceIndex = 0
            }); // TODO: Linux Clang 호환성 - emplace_back → push_back + designated initializer
        }
        waitSemOffset += info.waitSemaphoreInfoCount;

        info.commandBufferInfoCount = (uint32_t) inCmdBuffers.size();
        info.pCommandBufferInfos = cmdBuffers.data() + cmdBufferOffset;
        for (auto& inCmdBuffer : inCmdBuffers) 
        {
            // EVA_ASSERT(inCmdBuffer.queueFamilyIndex() == impl().qfIndex); // VUID-vkQueueSubmit2-pCommandBuffers-00074
            EVA_ASSERT(_type == inCmdBuffer.type()); // VUID-vkQueueSubmit2-pCommandBuffers-00074
            inCmdBuffer.impl().lastSubmittedQueue = *this;
            cmdBuffers.push_back(VkCommandBufferSubmitInfo{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                .pNext = nullptr,
                .commandBuffer = inCmdBuffer.impl().vkCmdBuffer,
                .deviceMask = 0
            }); // TODO: Linux Clang 호환성 - emplace_back → push_back + designated initializer
        }
        cmdBufferOffset += info.commandBufferInfoCount;

        info.signalSemaphoreInfoCount = (uint32_t) inSignalSems.size();
        info.pSignalSemaphoreInfos = signalSems.data() + signalSemOffset;
        for (auto& inSignalSem : inSignalSems)
        {
            signalSems.push_back(VkSemaphoreSubmitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .semaphore = inSignalSem.sem.impl().vkSemaphore,
                .value = 0,
                .stageMask = (VkPipelineStageFlags2)(uint64_t)inSignalSem.stage,
                .deviceIndex = 0
            }); // TODO: Linux Clang 호환성 - emplace_back → push_back + designated initializer
        }
        signalSemOffset += info.signalSemaphoreInfoCount;
    }

    ASSERT_SUCCESS(vkQueueSubmit2(impl().vkQueue, batchCount, submitInfos.data(),
        fence ? fence->impl().vkFence : VK_NULL_HANDLE));

#else
    std::vector<VkSubmitInfo> submitInfos(batchCount, {VK_STRUCTURE_TYPE_SUBMIT_INFO});

    std::vector<VkSemaphore> waitSems; waitSems.reserve(waitSemCount);
    std::vector<VkPipelineStageFlags> waitStages; waitStages.reserve(waitSemCount);
    std::vector<VkCommandBuffer> cmdBuffers; cmdBuffers.reserve(cmdBuffCount);
    std::vector<VkSemaphore> signalSems; signalSems.reserve(signalSemCount);
    
    for (uint32_t i=0; i<batchCount; ++i) 
    {
        const auto& [inWaitSems, inCmdBuffers, inSignalSems] = batches[i];
        VkSubmitInfo& info = submitInfos[i];

        info.waitSemaphoreCount = (uint32_t) inWaitSems.size();
        info.pWaitSemaphores = waitSems.data() + waitSemOffset;
        info.pWaitDstStageMask = waitStages.data() + waitSemOffset;

        for (auto& inWaitSem : inWaitSems) 
        {
            waitSems.push_back(inWaitSem.sem.impl().vkSemaphore);
            waitStages.push_back((VkPipelineStageFlags)(uint32_t)(uint64_t)inWaitSem.stage);
        }
        waitSemOffset += info.waitSemaphoreCount;

        info.commandBufferCount = (uint32_t) inCmdBuffers.size();
        info.pCommandBuffers = cmdBuffers.data() + cmdBufferOffset;
        for (auto& inCmdBuffer : inCmdBuffers) 
        {
            // EVA_ASSERT(inCmdBuffer.queueFamilyIndex() == impl().qfIndex); // VUID-vkQueueSubmit-pCommandBuffers-00074
            EVA_ASSERT(_type == inCmdBuffer.type()); // VUID-vkQueueSubmit-pCommandBuffers-00074
            inCmdBuffer.impl().lastSubmittedQueue = *this;
            cmdBuffers.push_back(inCmdBuffer.impl().vkCmdBuffer);
        }
        cmdBufferOffset += info.commandBufferCount;

        info.signalSemaphoreCount = (uint32_t) inSignalSems.size();
        info.pSignalSemaphores = signalSems.data() + signalSemOffset;
        for (auto& inSignalSem : inSignalSems) 
        {
            signalSems.push_back(inSignalSem.sem.impl().vkSemaphore);
        }
        signalSemOffset += info.signalSemaphoreCount;
    }

    ASSERT_SUCCESS(vkQueueSubmit(impl().vkQueue, batchCount, submitInfos.data(),
        fence ? fence->impl().vkFence : VK_NULL_HANDLE));
#endif

    return *this;
}

Queue Queue::submit(std::vector<SubmissionBatchInfo>&& batches)
{
    return submit(std::move(batches), std::nullopt);
}

Queue Queue::waitIdle()
{
    ASSERT_SUCCESS(vkQueueWaitIdle(impl().vkQueue));
    return *this;
}


/////////////////////////////////////////////////////////////////////////////////////////
// CommandPool
/////////////////////////////////////////////////////////////////////////////////////////
CommandPool Device::createCommandPool(QueueType type, COMMAND_POOL_CREATE flags)
{
    uint32_t qfIndex = impl().qfIndex[type];
    EVA_ASSERT(qfIndex != uint32_t(-1));

    auto vkHandle = create<VkCommandPool>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = (VkCommandPoolCreateFlags)(uint32_t)flags,
        .queueFamilyIndex = qfIndex,
    });

    auto pImpl = new CommandPool::Impl(
        impl().vkDevice, 
        *this,
        vkHandle, 
        flags,
        qfIndex,
        type);

    return *impl().cmdPools.insert(new CommandPool::Impl*(pImpl)).first;
}

QueueType CommandPool::type() const
{
    return impl().type;
}

std::vector<CommandBuffer> CommandPool::newCommandBuffers(uint32_t count)
{
    std::vector<VkCommandBuffer> vkCmdBuffers = allocate<VkCommandBuffer>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = impl().vkCmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,           // TODO: support VK_COMMAND_BUFFER_LEVEL_SECONDARY
        .commandBufferCount = count,
    });

    std::vector<CommandBuffer> cmdBuffers(count);
    for (uint32_t i = 0; i < count; ++i) 
    {
        auto pImpl = new CommandBuffer::Impl(
            vkCmdBuffers[i], 
            impl().device,
            impl().qfIndex,
            impl().type
        );

        cmdBuffers[i] = impl().cmdBuffers.emplace_back(new CommandBuffer::Impl*(pImpl));
    }

    return cmdBuffers;
}

CommandBuffer CommandPool::newCommandBuffer()
{
    return newCommandBuffers(1)[0];
}


/////////////////////////////////////////////////////////////////////////////////////////
// CommandBuffer
/////////////////////////////////////////////////////////////////////////////////////////
void* CommandBuffer::nativeCommandBuffer() const { return impl().vkCmdBuffer; }

QueueType CommandBuffer::type() const
{
    return impl().type;
}

uint32_t CommandBuffer::queueFamilyIndex() const
{
    return impl().qfIndex;
}

CommandBuffer CommandBuffer::submit(uint32_t index) const
{
    impl().device.queue(impl().type, index).submit(*this);
    return *this;
}

Queue CommandBuffer::lastSubmittedQueue() const
{
    return impl().lastSubmittedQueue;
}

CommandBuffer CommandBuffer::reset(bool keepCapacity)
{
    VkCommandBufferResetFlags flag = keepCapacity ? 
        VkCommandBufferResetFlags(0) :
        VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT;
    ASSERT_SUCCESS(vkResetCommandBuffer(impl().vkCmdBuffer, flag));
    return *this;
}

CommandBuffer CommandBuffer::begin(COMMAND_BUFFER_USAGE flags)
{
    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = (VkCommandBufferUsageFlags)(uint32_t)flags,
    };
    ASSERT_SUCCESS(vkBeginCommandBuffer(impl().vkCmdBuffer, &beginInfo));
    return *this;
}

CommandBuffer CommandBuffer::end()
{
    ASSERT_SUCCESS(vkEndCommandBuffer(impl().vkCmdBuffer));
    return *this;
}

CommandBuffer CommandBuffer::bindPipeline(Pipeline pipeline)
{
    impl().boundPipeline = pipeline;

    VkPipelineBindPoint bindPoint;
    VkPipeline vkPipeline;

    std::visit([&](auto&& pipeline)
    {
        using T = std::decay_t<decltype(pipeline)>;
        if constexpr (std::is_same_v<T, ComputePipeline>)
        {
            EVA_ASSERT(type() <= queue_compute);  // VUID-vkCmdBindPipeline-pipelineBindPoint-00777
            bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
            vkPipeline = pipeline.impl().vkPipeline;
        }
#ifdef EVA_ENABLE_RAYTRACING
        else if constexpr (std::is_same_v<T, RaytracingPipeline>)
        {
            bindPoint = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
            vkPipeline = pipeline.impl().vkPipeline;
        }
#endif
        else
            EVA_ASSERT(false);
    }, pipeline);
    
    vkCmdBindPipeline(impl().vkCmdBuffer, bindPoint, vkPipeline);
    return *this;
}

CommandBuffer CommandBuffer::bindDescSets(
    PipelineLayout layout,
    PIPELINE_BIND_POINT bindPoint,
    std::vector<DescriptorSet> descSets,
    uint32_t firstSet)
{
    std::vector<VkDescriptorSet> vkDescSets(descSets.size());
    for (uint32_t i = 0; i < descSets.size(); ++i)
        vkDescSets[i] = descSets[i] ? descSets[i].impl().vkDescSet : VK_NULL_HANDLE;

    vkCmdBindDescriptorSets(
        impl().vkCmdBuffer, (VkPipelineBindPoint)(uint32_t)bindPoint,
        layout.impl().vkPipeLayout, firstSet,
        (uint32_t)vkDescSets.size(), vkDescSets.data(),
        0, nullptr);
    return *this;
}

CommandBuffer CommandBuffer::bindDescSets(
    std::vector<DescriptorSet> descSets,
    uint32_t firstSet)
{
    std::vector<VkDescriptorSet> vkDescSets(descSets.size());
    for (uint32_t i = 0; i < descSets.size(); ++i)
        vkDescSets[i] = descSets[i] ? descSets[i].impl().vkDescSet : VK_NULL_HANDLE;

    auto index = impl().boundPipeline.index();
    VkPipelineBindPoint bindPoint;
    VkPipelineLayout layout;
    if (index == 0) {
        bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
        layout = std::get<ComputePipeline>(impl().boundPipeline).impl().layout.impl().vkPipeLayout;
    }
#ifdef EVA_ENABLE_RAYTRACING
    else if (index == 2) {
        bindPoint = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
        layout = std::get<RaytracingPipeline>(impl().boundPipeline).impl().layout.impl().vkPipeLayout;
    }
#endif
    else
        EVA_ASSERT(false);

    vkCmdBindDescriptorSets(
        impl().vkCmdBuffer, bindPoint,
        layout, firstSet,
        (uint32_t)vkDescSets.size(), vkDescSets.data(),
        0, nullptr);
    return *this;
}

CommandBuffer CommandBuffer::setPushConstants(
    PipelineLayout layout, 
    SHADER_STAGE stageFlags, 
    uint32_t offset, 
    uint32_t size,
    const void* values)
{
    vkCmdPushConstants(
        impl().vkCmdBuffer, 
        layout.impl().vkPipeLayout, 
        (VkShaderStageFlags)(uint32_t)stageFlags, 
        offset, 
        size, 
        values);
    return *this;
}

CommandBuffer CommandBuffer::setPushConstants(
    uint32_t offset,
    uint32_t size,
    const void* data)
{
    auto index = impl().boundPipeline.index();
    PipelineLayout layout;
    if (index == 0) {
        layout = std::get<ComputePipeline>(impl().boundPipeline).impl().layout;
    }
#ifdef EVA_ENABLE_RAYTRACING
    else if (index == 2) {
        layout = std::get<RaytracingPipeline>(impl().boundPipeline).impl().layout;
    }
#endif
    else
        EVA_ASSERT(false);

    // safe from VUID-vkCmdPushConstants-offset-01796
    // VkShaderStageFlags stageFlags = 0;
    // for (auto& range: layout.impl().pushConstants)
    // {
    //     if (offset < range.offset + range.size && range.offset < offset + size) 
    //         stageFlags |= (uint32_t) range.stageFlags;  // Check!!
    // }

    EVA_ASSERT(layout.impl().uniquePushConstant != nullptr);

    vkCmdPushConstants(
        impl().vkCmdBuffer, 
        layout.impl().vkPipeLayout, 
        (VkShaderStageFlags)(uint32_t)layout.impl().uniquePushConstant->stageFlags, 
        offset, 
        size, 
        data);
    return *this;
}

CommandBuffer CommandBuffer::barrier(
    std::vector<BarrierInfo> barrierInfos)
{
#ifdef VULKAN_VERSION_1_3
    std::vector<VkMemoryBarrier2> memoryBarriers;
    std::vector<VkBufferMemoryBarrier2> bufferBarriers;
    std::vector<VkImageMemoryBarrier2> imageBarriers;
    bufferBarriers.reserve(barrierInfos.size());
    imageBarriers.reserve(barrierInfos.size());

    for (auto& barrierInfo : barrierInfos) 
    {
        std::visit([&](auto&& barrier) {
            using T = std::decay_t<decltype(barrier)>;

            uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            if constexpr (std::is_same_v<T, BufferMemoryBarrier> 
                          || std::is_same_v<T, ImageMemoryBarrier>) 
            {
                // Safe from VUID-vkCmdPipelineBarrier2-srcQueueFamilyIndex-10387 
                if (barrier.opType == OwnershipTransferOpType::release)
                {
                    srcQueueFamilyIndex = queueFamilyIndex();
                    dstQueueFamilyIndex = impl().device.impl().qfIndex[barrier.pairedQueue];
                }
                else if (barrier.opType == OwnershipTransferOpType::acquire)
                {
                    srcQueueFamilyIndex = impl().device.impl().qfIndex[barrier.pairedQueue];
                    dstQueueFamilyIndex = queueFamilyIndex();
                }
                else EVA_ASSERT(barrier.opType == OwnershipTransferOpType::none);
            }
            
            if constexpr (std::is_same_v<T, MemoryBarrier>)
            {
                memoryBarriers.push_back(VkMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = (VkPipelineStageFlags2) barrier.srcMask.scope.stage,
                    .srcAccessMask = (VkAccessFlags2) barrier.srcMask.scope.access,
                    .dstStageMask = (VkPipelineStageFlags2) barrier.dstMask.scope.stage,
                    .dstAccessMask = (VkAccessFlags2) barrier.dstMask.scope.access
                }); // TODO: Linux Clang 호환성 - emplace_back → push_back + designated initializer
            }
            else if constexpr (std::is_same_v<T, BufferMemoryBarrier>)
            {
                bufferBarriers.push_back(VkBufferMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = (VkPipelineStageFlags2) barrier.srcMask.scope.stage,
                    .srcAccessMask = (VkAccessFlags2) barrier.srcMask.scope.access,
                    .dstStageMask = (VkPipelineStageFlags2) barrier.dstMask.scope.stage,
                    .dstAccessMask = (VkAccessFlags2) barrier.dstMask.scope.access,
                    .srcQueueFamilyIndex = srcQueueFamilyIndex,
                    .dstQueueFamilyIndex = dstQueueFamilyIndex,
                    .buffer = barrier.buffer.buffer.impl().vkBuffer,
                    .offset = barrier.buffer.offset,
                    .size = barrier.buffer.size
                }); // TODO: Linux Clang 호환성 - emplace_back → push_back + designated initializer
            }
            else if constexpr (std::is_same_v<T, ImageMemoryBarrier>) 
            {
                static std::map<SYNC_SCOPE::T, IMAGE_LAYOUT> defaultLayouts = {
                    {SYNC_SCOPE::NONE, IMAGE_LAYOUT::UNDEFINED},
                    {SYNC_SCOPE::TRANSFER_SRC, IMAGE_LAYOUT::TRANSFER_SRC},
                    {SYNC_SCOPE::TRANSFER_DST, IMAGE_LAYOUT::TRANSFER_DST},
                    {SYNC_SCOPE::COMPUTE_READ, IMAGE_LAYOUT::SHADER_READ_ONLY},
                    {SYNC_SCOPE::COMPUTE_WRITE, IMAGE_LAYOUT::GENERAL},
                    {SYNC_SCOPE::RAYTRACING_READ, IMAGE_LAYOUT::SHADER_READ_ONLY},
                    {SYNC_SCOPE::RAYTRACING_WRITE, IMAGE_LAYOUT::GENERAL},
                    {SYNC_SCOPE::PRESENT_SRC, IMAGE_LAYOUT::PRESENT_SRC},
                };

                if (barrier.oldLayout == IMAGE_LAYOUT::UNDEFINED && 
                    barrier.newLayout == IMAGE_LAYOUT::UNDEFINED) 
                {
                    barrier.oldLayout = defaultLayouts.at(barrier.srcMask.scope);
                    barrier.newLayout = defaultLayouts.at(barrier.dstMask.scope);

                    // Storage-only images must use GENERAL layout (not SHADER_READ_ONLY)
                    // But if SAMPLED_BIT is also set, SHADER_READ_ONLY is valid for sampling
                    VkImageUsageFlags usage = (VkImageUsageFlags)(uint32_t) barrier.image.impl().usage;
                    bool isStorageOnly = (usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0 &&
                                         (usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0;
                    if (isStorageOnly)
                    {
                        if (barrier.oldLayout == IMAGE_LAYOUT::SHADER_READ_ONLY)
                            barrier.oldLayout = IMAGE_LAYOUT::GENERAL;
                        if (barrier.newLayout == IMAGE_LAYOUT::SHADER_READ_ONLY)
                            barrier.newLayout = IMAGE_LAYOUT::GENERAL;
                    }

                    if (barrier.oldLayout == IMAGE_LAYOUT::PRESENT_SRC) 
                    {
                        barrier.srcMask = SYNC_SCOPE::NONE;
                    }

                    if (barrier.newLayout == IMAGE_LAYOUT::PRESENT_SRC) 
                    {
                        barrier.dstMask = SYNC_SCOPE::NONE;
                    }
                }

                imageBarriers.push_back(VkImageMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = (VkPipelineStageFlags2) barrier.srcMask.scope.stage,
                    .srcAccessMask = (VkAccessFlags2) barrier.srcMask.scope.access,
                    .dstStageMask = (VkPipelineStageFlags2) barrier.dstMask.scope.stage,
                    .dstAccessMask = (VkAccessFlags2) barrier.dstMask.scope.access,
                    .oldLayout = (VkImageLayout)(uint32_t) barrier.oldLayout,
                    .newLayout = (VkImageLayout)(uint32_t) barrier.newLayout,
                    .srcQueueFamilyIndex = srcQueueFamilyIndex,
                    .dstQueueFamilyIndex = dstQueueFamilyIndex,
                    .image = barrier.image.impl().vkImage,
                    .subresourceRange = VkImageSubresourceRange{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,    // TODO: support depth/stencil
                        .levelCount = VK_REMAINING_MIP_LEVELS,      // TODO: support level control
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,    // TODO: support layer control
                    }
                }); // TODO: Linux Clang 호환성 - emplace_back → push_back + designated initializer

                // barrier.image.impl().currentLayout = barrier.newLayout; // update current layout
            }
        }, barrierInfo);
    }
    
    VkDependencyInfo depInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = (uint32_t)memoryBarriers.size(),
        .pMemoryBarriers = memoryBarriers.data(),
        .bufferMemoryBarrierCount = (uint32_t)bufferBarriers.size(),
        .pBufferMemoryBarriers = bufferBarriers.data(),
        .imageMemoryBarrierCount = (uint32_t)imageBarriers.size(),
        .pImageMemoryBarriers = imageBarriers.data(),
    };
    vkCmdPipelineBarrier2(impl().vkCmdBuffer, &depInfo);

#else
    std::vector<VkMemoryBarrier> memoryBarriers;
    std::vector<VkBufferMemoryBarrier> bufferBarriers;
    std::vector<VkImageMemoryBarrier> imageBarriers;
    bufferBarriers.reserve(barrierInfos.size());
    imageBarriers.reserve(barrierInfos.size());

    VkPipelineStageFlags srcStageMask = 0;
    VkPipelineStageFlags dstStageMask = 0;

    for (auto& barrierInfo : barrierInfos) 
    {
        std::visit([&](auto&& barrier) {
            using T = std::decay_t<decltype(barrier)>;

            uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

            if constexpr (std::is_same_v<T, BufferMemoryBarrier>) 
            {
                // Safe from VUID-vkCmdPipelineBarrier-srcQueueFamilyIndex-00187
                if (barrier.opType == OwnershipTransferOpType::release)
                {
                    srcQueueFamilyIndex = queueFamilyIndex();
                    dstQueueFamilyIndex = impl().device.impl().qfIndex[barrier.pairedQueue];
                }
                else if (barrier.opType == OwnershipTransferOpType::acquire)
                {
                    srcQueueFamilyIndex = impl().device.impl().qfIndex[barrier.pairedQueue];
                    dstQueueFamilyIndex = queueFamilyIndex();
                }
                else EVA_ASSERT(barrier.opType == OwnershipTransferOpType::none);
            }
            
            if constexpr (std::is_same_v<T, MemoryBarrier>)
            {
                srcStageMask |= (VkPipelineStageFlags) barrier.srcMask.stage;
                dstStageMask |= (VkPipelineStageFlags) barrier.dstMask.stage;
                memoryBarriers.push_back(VkMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = (VkAccessFlags) barrier.srcMask.access,
                    .dstAccessMask = (VkAccessFlags) barrier.dstMask.access
                }); // TODO: Linux Clang 호환성 - emplace_back → push_back + designated initializer
            }
            else if constexpr (std::is_same_v<T, BufferMemoryBarrier>) {
                srcStageMask |= (VkPipelineStageFlags) barrier.srcMask.stage;
                dstStageMask |= (VkPipelineStageFlags) barrier.dstMask.stage;
                bufferBarriers.push_back(VkBufferMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = (VkAccessFlags) barrier.srcMask.access,
                    .dstAccessMask = (VkAccessFlags) barrier.dstMask.access,
                    .srcQueueFamilyIndex = srcQueueFamilyIndex,
                    .dstQueueFamilyIndex = dstQueueFamilyIndex,
                    .buffer = barrier.buffer.impl().vkBuffer,
                    .offset = barrier.offset,
                    .size = barrier.size
                }); // TODO: Linux Clang 호환성 - emplace_back → push_back + designated initializer
            }
            else if constexpr (std::is_same_v<T, ImageMemoryBarrier>) {
                // TODO: Support image & image barrier
            }
        }, barrierInfo);
    }
    
    vkCmdPipelineBarrier(
        impl().vkCmdBuffer,
        srcStageMask, dstStageMask,
        0,
        (uint32_t)memoryBarriers.size(), memoryBarriers.data(),
        (uint32_t)bufferBarriers.size(), bufferBarriers.data(),
        (uint32_t)imageBarriers.size(), imageBarriers.data());
#endif

    return *this;
}

CommandBuffer CommandBuffer::barrier(BarrierInfo barrierInfo)
{
    return barrier(std::vector<BarrierInfo>{barrierInfo});
}

CommandBuffer CommandBuffer::copyBuffer(
    Buffer src, 
    Buffer dst, 
    uint64_t srcOffset,  
    uint64_t dstOffset,  
    uint64_t size)
{
	EVA_ASSERT((uint32_t)src.impl().usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT); 
	EVA_ASSERT((uint32_t)dst.impl().usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    EVA_ASSERT(srcOffset < src.size()); // VUID-vkCmdCopyBuffer-srcOffset-00113
    EVA_ASSERT(dstOffset < dst.size()); // VUID-vkCmdCopyBuffer-dstOffset-00114

    if (size == EVA_WHOLE_SIZE)
        size = std::min(src.size() - srcOffset, dst.size() - dstOffset);  
    else 
    {
        EVA_ASSERT(srcOffset + size <= src.size()); // VUID-vkCmdCopyBuffer-size-00115
        EVA_ASSERT(dstOffset + size <= dst.size()); // VUID-vkCmdCopyBuffer-size-00116
    }

	VkBufferCopy copyRegion{  
		.srcOffset = srcOffset,
		.dstOffset = dstOffset,
		.size = size, 
	};

	vkCmdCopyBuffer(
		impl().vkCmdBuffer,
		src.impl().vkBuffer,
		dst.impl().vkBuffer,
		1, &copyRegion);
	return *this;
}

CommandBuffer CommandBuffer::copyBuffer(BufferRange src, BufferRange dst)
{
    return copyBuffer(
        src.buffer, dst.buffer,
        src.offset, dst.offset,
        std::min(src.size, dst.size));
}

CommandBuffer CommandBuffer::fillBuffer(
    Buffer dst,
    uint32_t data,
    uint64_t dstOffset,
    uint64_t size)
{
    EVA_ASSERT((uint32_t)dst.impl().usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    EVA_ASSERT(dstOffset < dst.size());

    if (size == EVA_WHOLE_SIZE)
        size = dst.size() - dstOffset;
    else
        EVA_ASSERT(dstOffset + size <= dst.size());

    // vkCmdFillBuffer requires size to be a multiple of 4
    EVA_ASSERT(size % 4 == 0);
    EVA_ASSERT(dstOffset % 4 == 0);

    vkCmdFillBuffer(
        impl().vkCmdBuffer,
        dst.impl().vkBuffer,
        dstOffset,
        size,
        data);

    return *this;
}

CommandBuffer CommandBuffer::copyImage(
    Image src, Image dst, std::vector<CopyRegion> regions)
{
	EVA_ASSERT((uint32_t)src.impl().usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	EVA_ASSERT((uint32_t)dst.impl().usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (regions.empty()) 
        regions.push_back({});
    
    std::vector<VkImageCopy> copyRegions;
    copyRegions.reserve(regions.size());
    
    for (const auto& region : regions) 
    {
        EVA_ASSERT(region.offsetX + region.width <= src.impl().width);
        EVA_ASSERT(region.offsetY + region.height <= src.impl().height);
        EVA_ASSERT(region.offsetZ + region.depth <= src.impl().depth);
        EVA_ASSERT(region.baseLayer + region.layerCount <= src.impl().arrayLayers);
        
        copyRegions.emplace_back(VkImageCopy{
            .srcSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,    // TODO: support depth/stencil
                .mipLevel = 0,                              // TODO: support mipmap levels
                .baseArrayLayer = region.baseLayer,
                .layerCount = region.layerCount ? region.layerCount : src.impl().arrayLayers - region.baseLayer,
            },
            .srcOffset = {(int)region.offsetX, (int)region.offsetY, (int)region.offsetZ},
            .dstSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,    // TODO: support depth/stencil
                .mipLevel = 0,                              // TODO: support mipmap levels
                .baseArrayLayer = region.baseLayer,
                .layerCount = region.layerCount ? region.layerCount : dst.impl().arrayLayers - region.baseLayer,
            },
            .dstOffset = {(int)region.offsetX, (int)region.offsetY, (int)region.offsetZ},
            .extent = {
                region.width ? region.width : src.impl().width - region.offsetX, 
                region.height ? region.height : src.impl().height - region.offsetY, 
                region.depth ? region.depth : src.impl().depth - region.offsetZ
            },
        });
    }
    
    vkCmdCopyImage(
        impl().vkCmdBuffer,
        src.impl().vkImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, // Image must be in correct layout
        dst.impl().vkImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, // Image must be in correct layout
        (uint32_t)copyRegions.size(),
        copyRegions.data()
    );
    
    return *this;
}

CommandBuffer CommandBuffer::copyBufferToImage(
    BufferRange src, Image dst, std::vector<CopyRegion> regions)
{
	EVA_ASSERT((uint32_t)src.usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	EVA_ASSERT((uint32_t)dst.impl().usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (regions.empty()) 
        regions.push_back({});
    
    std::vector<VkBufferImageCopy> copyRegions;
    copyRegions.reserve(regions.size());
    
    for (const auto& region : regions) 
    {
        EVA_ASSERT(region.offsetX + region.width <= dst.impl().width);
        EVA_ASSERT(region.offsetY + region.height <= dst.impl().height);
        EVA_ASSERT(region.offsetZ + region.depth <= dst.impl().depth);
        EVA_ASSERT(region.baseLayer + region.layerCount <= dst.impl().arrayLayers);
        
        copyRegions.emplace_back(VkBufferImageCopy{
            .bufferOffset = region.bufferOffset + src.offset,
            .bufferRowLength = region.bufferRowLength,      // Tightly packed
            .bufferImageHeight = region.bufferImageHeight,  // Tightly packed
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,    // TODO: support depth/stencil
                .mipLevel = 0,                              // TODO: support mipmap levels
                .baseArrayLayer = region.baseLayer,
                .layerCount = region.layerCount ? region.layerCount : dst.impl().arrayLayers - region.baseLayer,
            },
            .imageOffset = {(int)region.offsetX, (int)region.offsetY, (int)region.offsetZ},
            .imageExtent = {
                region.width ? region.width : dst.impl().width - region.offsetX, 
                region.height ? region.height : dst.impl().height - region.offsetY, 
                region.depth ? region.depth : dst.impl().depth - region.offsetZ
            },
        });
    }
    
    vkCmdCopyBufferToImage(
        impl().vkCmdBuffer,
        src.buffer.impl().vkBuffer,
        dst.impl().vkImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        (uint32_t)copyRegions.size(),
        copyRegions.data()
    );
    
    return *this;
}

CommandBuffer CommandBuffer::copyImageToBuffer(
    Image src, BufferRange dst, std::vector<CopyRegion> regions)
{
	EVA_ASSERT((uint32_t)src.impl().usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	EVA_ASSERT((uint32_t)dst.usage() & VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (regions.empty()) 
        regions.push_back({});
    
    std::vector<VkBufferImageCopy> copyRegions;
    copyRegions.reserve(regions.size());
    
    for (const auto& region : regions) 
    {
        EVA_ASSERT(region.offsetX + region.width <= src.impl().width);
        EVA_ASSERT(region.offsetY + region.height <= src.impl().height);
        EVA_ASSERT(region.offsetZ + region.depth <= src.impl().depth);
        EVA_ASSERT(region.baseLayer + region.layerCount <= src.impl().arrayLayers);
        
        copyRegions.emplace_back(VkBufferImageCopy{
            .bufferOffset = region.bufferOffset + dst.offset,
            .bufferRowLength = region.bufferRowLength,      // Tightly packed
            .bufferImageHeight = region.bufferImageHeight,  // Tightly packed
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,    // TODO: support depth/stencil
                .mipLevel = 0,                              // TODO: support mipmap levels
                .baseArrayLayer = region.baseLayer,
                .layerCount = region.layerCount ? region.layerCount : src.impl().arrayLayers - region.baseLayer,
            },
            .imageOffset = {(int)region.offsetX, (int)region.offsetY, (int)region.offsetZ},
            .imageExtent = {
                region.width ? region.width : src.impl().width - region.offsetX, 
                region.height ? region.height : src.impl().height - region.offsetY, 
                region.depth ? region.depth : src.impl().depth - region.offsetZ
            },
        });
    }
    
    vkCmdCopyImageToBuffer(
        impl().vkCmdBuffer,
        src.impl().vkImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, // Image must be in correct layout
        dst.buffer.impl().vkBuffer,
        (uint32_t)copyRegions.size(),
        copyRegions.data()
    );
    
    return *this;
}

CommandBuffer CommandBuffer::dispatch2(uint32_t numThreadsInX, uint32_t numThreadsInY, uint32_t numThreadsInZ)
{
    EVA_ASSERT(type() <= queue_compute);  // VUID-vkCmdDispatch-commandBuffer-cmdpool (Implicit)  
    auto pipeline = std::get_if<ComputePipeline>(&impl().boundPipeline);
    EVA_ASSERT(pipeline != nullptr);

    auto [groupSizeInX, groupSizeInY, groupSizeInZ] = pipeline->impl().workGroupSize;

    vkCmdDispatch(
        impl().vkCmdBuffer, 
        (numThreadsInX + groupSizeInX - 1) / groupSizeInX,
        (numThreadsInY + groupSizeInY - 1) / groupSizeInY,
        (numThreadsInZ + groupSizeInZ - 1) / groupSizeInZ);
    return *this;
}

CommandBuffer CommandBuffer::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    EVA_ASSERT(type() <= queue_compute);  // VUID-vkCmdDispatch-commandBuffer-cmdpool (Implicit)
    vkCmdDispatch(impl().vkCmdBuffer, groupCountX, groupCountY, groupCountZ);
    return *this;
}


CommandBuffer CommandBuffer::resetQueryPool(QueryPool pool, uint32_t firstQuery, uint32_t queryCount)
{
    if (queryCount == 0) queryCount = pool.queryCount() - firstQuery;
    vkCmdResetQueryPool(impl().vkCmdBuffer, pool.impl().vkQueryPool, firstQuery, queryCount);
    return *this;
}


CommandBuffer CommandBuffer::writeTimestamp(PIPELINE_STAGE stage, QueryPool pool, uint32_t query)
{
    vkCmdWriteTimestamp(
        impl().vkCmdBuffer,
        (VkPipelineStageFlagBits)(uint64_t)stage,
        pool.impl().vkQueryPool,
        query);
    return *this;
}


#ifdef EVA_ENABLE_RAYTRACING
CommandBuffer CommandBuffer::traceRays(
    ShaderBindingTable hitGroupSbt, uint32_t width, uint32_t height, uint32_t depth)
{
    auto* rtp = std::get_if<RaytracingPipeline>(&impl().boundPipeline);
    EVA_ASSERT(rtp != nullptr);

    static VkStridedDeviceAddressRegionKHR callable = {}; // Not used
    const auto& sbt = rtp->impl().sbt;

    VkStridedDeviceAddressRegionKHR hitGpSbt = {
        .deviceAddress = hitGroupSbt.buffer.deviceAddress(),
        .stride = (DeviceAddress) hitGroupSbt.recordSize,
        .size = (DeviceAddress) hitGroupSbt.recordSize * hitGroupSbt.numRecords,
    };

    vkCmdTraceRaysKHR_(
        impl().vkCmdBuffer,
        &sbt.rgen,
        &sbt.miss,
        &hitGpSbt,
        &callable,
        width, height, depth
    );

    return *this;
}

CommandBuffer CommandBuffer::traceRays(uint32_t width, uint32_t height, uint32_t depth)
{
    auto* rtp = std::get_if<RaytracingPipeline>(&impl().boundPipeline);
    EVA_ASSERT(rtp != nullptr);

    static VkStridedDeviceAddressRegionKHR callable = {}; // Not used
    const auto& sbt = rtp->impl().sbt;

    vkCmdTraceRaysKHR_(
        impl().vkCmdBuffer,
        &sbt.rgen,
        &sbt.miss,
        &sbt.hitGroup,
        &callable,
        width, height, depth
    );

    return *this;
}
#endif


/////////////////////////////////////////////////////////////////////////////////////////
// Fence
/////////////////////////////////////////////////////////////////////////////////////////
Fence Device::createFence(bool signaled)
{
    auto vkHandle = create<VkFence>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = (VkFenceCreateFlags) (signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0),
    });

    auto pImpl = new Fence::Impl(
        impl().vkDevice, 
        vkHandle);
    
    return *impl().fences.insert(new Fence::Impl*(pImpl)).first;
}

Result Device::waitFences(std::vector<Fence> fences, bool waitAll, uint64_t timeout)
{
    std::vector<VkFence> vkFences(fences.size());
    for (uint32_t i = 0; i < fences.size(); ++i) 
        vkFences[i] = fences[i].impl().vkFence;

    return (Result) vkWaitForFences(impl().vkDevice, (uint32_t)fences.size(), vkFences.data(), waitAll, timeout);
}

void Device::resetFences(std::vector<Fence> fences)
{
    std::vector<VkFence> vkFences(fences.size());
    for (uint32_t i = 0; i < fences.size(); ++i) 
        vkFences[i] = fences[i].impl().vkFence;

    ASSERT_SUCCESS(vkResetFences(impl().vkDevice, (uint32_t)fences.size(), vkFences.data()));
}

Result Fence::wait(bool autoReset, uint64_t timeout) const
{
    VkResult res = vkWaitForFences(impl().vkDevice, 1, &impl().vkFence, VK_TRUE, timeout);
    if(res==VK_SUCCESS && autoReset)
        ASSERT_SUCCESS(vkResetFences(impl().vkDevice, 1, &impl().vkFence));
    return (Result) res;
}

void Fence::reset() const
{
    ASSERT_SUCCESS(vkResetFences(impl().vkDevice, 1, &impl().vkFence));
}

bool Fence::isSignaled() const
{
    return vkGetFenceStatus(impl().vkDevice, impl().vkFence) == VK_SUCCESS;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Semaphore
/////////////////////////////////////////////////////////////////////////////////////////
Semaphore Device::createSemaphore()
{
    auto vkHandle = create<VkSemaphore>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    });

    auto pImpl = new Semaphore::Impl(
        impl().vkDevice, 
        vkHandle);

    return *impl().semaphores.insert(new Semaphore::Impl*(pImpl)).first;
}


/////////////////////////////////////////////////////////////////////////////////////////
// ShaderModule
/////////////////////////////////////////////////////////////////////////////////////////
ShaderModule Device::createShaderModule(const ShaderModuleCreateInfo& info)
{
    VkShaderModule vkHandle = create<VkShaderModule>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = info.spv.sizeInBytes,
        .pCode = info.spv.data.get(),
    });

    auto pImpl = new ShaderModule::Impl(
        impl().vkDevice,
        vkHandle,
        info.stage,
        info.withSpirvReflect ? createReflectShaderModule(info.spv) : nullptr
    );

    return *impl().shaderModules.insert(new ShaderModule::Impl*(pImpl)).first;
}

bool ShaderModule::hasReflect() const
{
    return impl().pModule != nullptr;
}

void ShaderModule::discardReflect() 
{
    impl().discardReflect();
}

PipelineLayoutDesc ShaderModule::extractPipelineLayoutDesc() const
{
    EVA_ASSERT(impl().pModule != nullptr);
    return ::extractPipelineLayoutDesc(impl().pModule);
}

ShaderModule::operator uint64_t() const
{
    return (uint64_t) *ppImpl;
}

static inline uint64_t to_uint64(const ShaderInput& shader) noexcept
{
    if (auto* m = std::get_if<ShaderModule>(&shader))
        return (uint64_t) *m;

    const SpvBlob& blob = std::get<SpvBlob>(shader);
    return (uint64_t) blob.data.get();
}

static inline bool operator==(const ShaderInput& a, const ShaderInput& b) noexcept
{
    return to_uint64(a) == to_uint64(b);
}

static inline uint64_t hashShaderInput(const ShaderInput& shader) noexcept 
{
    return std::hash<uint64_t>{}(to_uint64(shader));
}


/////////////////////////////////////////////////////////////////////////////////////////
// SpecializationConstant
/////////////////////////////////////////////////////////////////////////////////////////
void SpecializationConstant::buildCache() const 
{
    if (cachedMapEntries.has_value()) 
        return;
    
    size_t totalSize = 0;
    for (auto& [_, bytes] : orderedConstants) 
        totalSize += bytes.size();
    
    cachedMapEntries = std::vector<SpecializationMapEntry>{};
    cachedMapEntries->reserve(orderedConstants.size());
    cachedData = std::vector<uint8_t>{};
    cachedData->reserve(totalSize);
    
    for (auto& [id, bytes] : orderedConstants) 
    {
        uint32_t offset = static_cast<uint32_t>(cachedData->size());
        cachedMapEntries->push_back({
            .constantID = id,
            .offset = offset,
            .size = static_cast<uint32_t>(bytes.size())
        });
        cachedData->insert(cachedData->end(), bytes.begin(), bytes.end());
    }
    
    cachedSpecInfo = SpecializationInfo{
        .mapEntryCount = static_cast<uint32_t>(cachedMapEntries->size()),
        .pMapEntries = cachedMapEntries->data(),
        .dataSize = cachedData->size(),
        .pData = cachedData->data()
    };
}

uint64_t SpecializationConstant::hash() const noexcept
{
    auto mix64 = [](uint64_t x) noexcept {
        x ^= x >> 33;  x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;  x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    };

    uint64_t h = 0x9e3779b97f4a7c15ULL;

    for (const auto& [id, bytes] : orderedConstants) 
    {
        // (1) 메타( ID와 길이 )를 아주 가볍게 주입
        h ^= mix64((uint64_t)id | (bytes.size() << 32));

        // (2) 값 바이트들을 64비트 청크로 빠르게 섞기
        const uint8_t* p = bytes.data();
        size_t n = bytes.size();
        while (n >= 8) 
        {
            // h ^= mix64(*(uint64_t*)p); // unaligned access?
            uint64_t chunk; std::memcpy(&chunk, p, 8);
            h ^= mix64(chunk);

            p += 8;
            n -= 8;
        }

        if (n) // 남은 0~7바이트를 한 번만 섞기
        { 
            uint64_t chunk = 0;
            std::memcpy(&chunk, p, n);
            h ^= mix64(chunk ^ ((uint64_t)n << 56));
        }
    }

    // (3) 최종 마무리
    return mix64(h);
}


bool ShaderStage::operator==(const ShaderStage& other) const noexcept
{
    if (!shader.has_value() && !other.shader.has_value())
        return true;
    return *shader == *other.shader && specialization == other.specialization;
}


size_t std::hash<eva::ShaderStage>::operator()(const eva::ShaderStage& stage) const noexcept 
{
    constexpr uint64_t kNullTag = 0x9E3779B97F4A7C15ull;
    uint64_t h = stage.shader ? hashShaderInput(*stage.shader) : kNullTag;
    return hashCombine(h, stage.specialization.hash());
}


/////////////////////////////////////////////////////////////////////////////////////////
// ComputePipeline
/////////////////////////////////////////////////////////////////////////////////////////
ComputePipeline Device::createComputePipeline(const ComputePipelineCreateInfo& info)
{
    // Skip SPIRV-Reflect entirely if both layout AND workgroupSize are provided
    // This is required for coopmat2 shaders which SPIRV-Reflect cannot parse
    const bool skipReflect = info.layout.has_value() && info.workgroupSize.has_value();
    const bool useReflect = !skipReflect;

    ShaderModule csModule;
    bool createTempModule = false;

    if (auto* mod = std::get_if<ShaderModule>(&*info.csStage.shader))
    {
        csModule = *mod;
    }
    else
    {
        csModule = createShaderModule({
            .stage = SHADER_STAGE::COMPUTE,
            .spv = std::get<SpvBlob>(*info.csStage.shader),
            .withSpirvReflect = useReflect,
        });
        createTempModule = true;
    }

    uint32_t sizeX, sizeY, sizeZ;
    PipelineLayout layout;

    if (skipReflect) {
        // Use provided workgroup size and layout (coopmat2 path)
        auto& wgSize = info.workgroupSize.value();
        sizeX = wgSize[0];
        sizeY = wgSize[1];
        sizeZ = wgSize[2];
        layout = info.layout.value();
    } else {
        // Use SPIRV-Reflect for workgroup size and optionally layout
        EVA_ASSERT(csModule.hasReflect());
        auto wgSize = extractWorkGroupSize(csModule.impl().pModule);
        sizeX = wgSize[0];
        sizeY = wgSize[1];
        sizeZ = wgSize[2];

        if (info.layout.has_value())
        {
            layout = info.layout.value();
        }
        else
        {
            auto layoutDesc = csModule.extractPipelineLayoutDesc();
            if (info.autoLayoutAllowAllStages)
            {
                for (auto& [setId, setLayout] : layoutDesc.setLayouts)
                    for (auto& [bindId, binding] : setLayout.bindings)
                        binding.stageFlags = SHADER_STAGE::ALL;
                if (layoutDesc.pushConstant)
                    layoutDesc.pushConstant->stageFlags = SHADER_STAGE::ALL;
            }
            layout = createPipelineLayout(std::move(layoutDesc));
        }
    }

    const auto& spec = info.csStage.specialization;

    VkPipelineShaderStageCreateInfo stageInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = (VkShaderStageFlagBits)(uint32_t) csModule.impl().stage,
        .module = csModule.impl().vkModule,
        .pName = "main",
    };

    stageInfo.pSpecializationInfo = (VkSpecializationInfo*) spec.getInfo();

    VkComputePipelineCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .flags = (VkPipelineCreateFlags)(
            vkGetPipelineExecutablePropertiesKHR_
                ? (VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR
                   | VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR)
                : 0),
        .stage = stageInfo,
        .layout = layout.impl().vkPipeLayout,
    };

    VkPipeline vkHandle;
    ASSERT_SUCCESS(vkCreateComputePipelines(
        impl().vkDevice, VK_NULL_HANDLE,
        1, &createInfo,
        nullptr, &vkHandle));

    auto pImpl = new ComputePipeline::Impl(
        impl().vkDevice,
        vkHandle,
        layout,
        sizeX, sizeY, sizeZ);

    if (createTempModule)
        csModule.destroy();

    return *impl().computePipelines.insert(new ComputePipeline::Impl*(pImpl)).first;
}

PipelineLayout ComputePipeline::layout() const
{
    return impl().layout;
}

DescriptorSetLayout ComputePipeline::descSetLayout(uint32_t setId) const
{
    return impl().layout.descSetLayout(setId);
}

uint32_t ComputePipeline::pushConstantSize() const
{
    return impl().layout.pushConstantSize();
}


/////////////////////////////////////////////////////////////////////////////////////////
// RaytracingPipeline
/////////////////////////////////////////////////////////////////////////////////////////
#ifdef EVA_ENABLE_RAYTRACING
uint32_t Device::shaderGroupHandleSize() const
{
    return impl().rtProps.shaderGroupHandleSize;
}

uint32_t Device::shaderGroupHandleAlignment() const
{
    return impl().rtProps.shaderGroupHandleAlignment;
}

uint32_t Device::shaderGroupBaseAlignment() const
{
    return impl().rtProps.shaderGroupBaseAlignment;
}

uint32_t Device::asBufferOffsetAlignment() const
{
    return impl().rtProps.asBufferOffsetAlignment;
}

uint32_t Device::minAccelerationStructureScratchOffsetAlignment() const
{
    return impl().rtProps.minAccelerationStructureScratchOffsetAlignment;
}

RaytracingPipeline Device::createRaytracingPipeline(const RaytracingPipelineCreateInfo& info)
{
    uint32_t maxSize = 1 + (uint32_t)info.missStages.size() + (uint32_t)info.hitGroups.size()*3;

    std::unordered_map<ShaderStage, uint32_t> stageIdxCache(maxSize);
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    std::vector<ShaderModule> modules;
    std::vector<bool> isTempModule;
    stages.reserve(maxSize);
    modules.reserve(maxSize);
    isTempModule.reserve(maxSize);

    auto makeModule = [&](SHADER_STAGE stage,
                          const ShaderInput& src,
                          bool reflect) -> std::pair<ShaderModule, bool>
    {
        if (auto* mod = std::get_if<ShaderModule>(&src))
            return {*mod, false};

        ShaderModule m = createShaderModule({
            .stage = stage,
            .spv = std::get<SpvBlob>(src),
            .withSpirvReflect = reflect,
        });
        return {m, true};
    };

    auto addStage = [&](SHADER_STAGE stage,
                        const ShaderStage& stageInfo,
                        bool reflect) -> uint32_t
    {
        auto [it, inserted] = stageIdxCache.try_emplace(stageInfo, 0);
        if (!inserted)
            return it->second;

        try
        {
            auto [m, isTemp] = makeModule(stage, *stageInfo.shader, reflect);

            VkPipelineShaderStageCreateInfo ci = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = (VkShaderStageFlagBits)(uint32_t) stage,
                .module = m.impl().vkModule,
                .pName = "main",
            };
            ci.pSpecializationInfo = (VkSpecializationInfo*)stageInfo.specialization.getInfo();
            stages.push_back(ci);
            modules.push_back(std::move(m));
            isTempModule.push_back(isTemp);

            it->second = static_cast<uint32_t>(stages.size() - 1);
            return it->second;
        }
        catch (...)
        {
            stageIdxCache.erase(it);
            throw;
        }
    };

    const bool wantReflect = !info.layout.has_value();

    uint32_t rgenIdx;
    std::vector<uint32_t> missIdx(info.missStages.size()); 
    std::vector<std::tuple<int, int, int>> hgIdx(info.hitGroups.size(), {-1, -1, -1});

    rgenIdx = addStage(SHADER_STAGE::RAYGEN, info.rgenStage, wantReflect);

    for (size_t i = 0; i < info.missStages.size(); ++i) 
        missIdx[i] = addStage(SHADER_STAGE::MISS, info.missStages[i], wantReflect);

    for (size_t i = 0; i < info.hitGroups.size(); ++i) 
    {
        auto& [chitIx, ahitIx, isecIx] = hgIdx[i];
        const auto& [chitStage, ahitStage, isecStage] = info.hitGroups[i];
        if (chitStage.shader) chitIx = (int) addStage(SHADER_STAGE::CLOSEST_HIT, chitStage, wantReflect);
        if (ahitStage.shader) ahitIx = (int) addStage(SHADER_STAGE::ANY_HIT, ahitStage, wantReflect);
        if (isecStage.shader) isecIx = (int) addStage(SHADER_STAGE::INTERSECTION, isecStage, wantReflect);
    }

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(1 + missIdx.size() + hgIdx.size(), {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
        .generalShader      = VK_SHADER_UNUSED_KHR,
        .closestHitShader   = VK_SHADER_UNUSED_KHR,
        .anyHitShader       = VK_SHADER_UNUSED_KHR,
        .intersectionShader = VK_SHADER_UNUSED_KHR
    }); 
    {
        groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[0].generalShader = rgenIdx;

        for (uint32_t i=0; i < missIdx.size(); ++i) 
        {
            groups[1 + i].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            groups[1 + i].generalShader = missIdx[i];
        }

        for (uint32_t i=0; i < hgIdx.size(); ++i)
        {
            auto& [chitIx, ahitIx, isecIx] = hgIdx[i];
            auto& g = groups[1 + (uint32_t)missIdx.size() + i];

            g.type = (isecIx >= 0)
                ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
            g.closestHitShader   = (chitIx >= 0) ? static_cast<uint32_t>(chitIx) : VK_SHADER_UNUSED_KHR;
            g.anyHitShader       = (ahitIx >= 0) ? static_cast<uint32_t>(ahitIx) : VK_SHADER_UNUSED_KHR;
            g.intersectionShader = (isecIx >= 0) ? static_cast<uint32_t>(isecIx) : VK_SHADER_UNUSED_KHR;
        }
    }

    PipelineLayout layout;
    if (info.layout.has_value()) 
    {
        layout = info.layout.value();
    } 
    else 
    {
        PipelineLayoutDesc layoutDesc;
        for (auto& m : modules)
            layoutDesc |= m.extractPipelineLayoutDesc();
            
        if (info.autoLayoutAllowAllStages) 
        {
            for (auto& [setId, setLayout] : layoutDesc.setLayouts) 
                for (auto& [bindId, binding] : setLayout.bindings) 
                    binding.stageFlags = SHADER_STAGE::ALL;
            if (layoutDesc.pushConstant)
                layoutDesc.pushConstant->stageFlags = SHADER_STAGE::ALL;
        }
        layout = createPipelineLayout(std::move(layoutDesc));
    }

    VkRayTracingPipelineCreateInfoKHR rtci{ 
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount = (uint32_t) stages.size(),
        .pStages = stages.data(),
        .groupCount = (uint32_t) groups.size(),
        .pGroups = groups.data(),
        .maxPipelineRayRecursionDepth = info.maxRecursionDepth,
        .layout = layout.impl().vkPipeLayout
    };

    VkPipeline vkHandle;
    ASSERT_SUCCESS(vkCreateRayTracingPipelinesKHR_(
        impl().vkDevice, VK_NULL_HANDLE, VK_NULL_HANDLE,
        1, &rtci, nullptr, &vkHandle));

    auto pImpl = new RaytracingPipeline::Impl(
        impl().vkDevice,
        vkHandle,
        layout);
    
    {
        Buffer& sbtBuffer = pImpl->sbt.buffer;
        const uint32_t handleSize = impl().rtProps.shaderGroupHandleSize;
        const uint32_t recordAlign = impl().rtProps.shaderGroupHandleAlignment;
        const uint32_t tableAlign = impl().rtProps.shaderGroupBaseAlignment;

        /*
        raygen과 miss 쉐이더에서 SBT를 통한 커스텀 데이터는 사용하지 않는다는 가정. (굳이 쓸 이유 없음, 유니폼 버퍼가 더 효율적)
        */
        const uint32_t recordStride = alignTo(handleSize, recordAlign);
        const uint64_t missOffset = alignTo(recordStride, tableAlign);
        const uint64_t bufferSize = missOffset + (uint64_t)recordStride * info.missStages.size();

        sbtBuffer = createBuffer({
            .size = bufferSize,
            .usage = BUFFER_USAGE::SHADER_BINDING_TABLE | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
            .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
        }); // TODO : support device local memory

        uint32_t numHandles = groups.size();
        std::vector<ShaderGroupHandle> handles(numHandles);
        vkGetRayTracingShaderGroupHandlesKHR_(
            impl().vkDevice, vkHandle, 0, numHandles, 
            sizeof(ShaderGroupHandle) * numHandles, handles.data());

        uint8_t* pMap = sbtBuffer.map();
        {
            std::memcpy(pMap, &handles[0], sizeof(ShaderGroupHandle));
            pMap += missOffset;
            for (uint32_t i = 0; i < info.missStages.size(); ++i) 
            {
                std::memcpy(pMap, &handles[1 + i], sizeof(ShaderGroupHandle));
                pMap += recordStride;
            }
            
            pImpl->hitGroupHandles.resize(info.hitGroups.size());
            for (uint32_t i = 0; i < info.hitGroups.size(); ++i)
                pImpl->hitGroupHandles[i] = handles[1 + (uint32_t)info.missStages.size() + i];
            
            sbtBuffer.unmap();
        }

        auto sbtAddress = sbtBuffer.deviceAddress();

        pImpl->sbt.rgen = {
            .deviceAddress = sbtAddress,
            .stride = recordStride,
            .size = recordStride,
        };

        pImpl->sbt.miss = {
            .deviceAddress = sbtAddress + missOffset,
            .stride = recordStride,
            .size = recordStride * (uint32_t)missIdx.size(),
        };
    }

    for (size_t i = 0; i < modules.size(); ++i)
    {
        if (isTempModule[i])
            modules[i].destroy();
    }

    return *impl().raytracingPipelines.insert(new RaytracingPipeline::Impl*(pImpl)).first;
}

ShaderGroupHandle RaytracingPipeline::getHitGroupHandle(uint32_t groupIndex) const
{
    return impl().hitGroupHandles[groupIndex];
}

void RaytracingPipeline::setHitGroupSbt(ShaderBindingTable sbt)
{
    impl().sbt.hitGroup = {
        .deviceAddress = sbt.buffer.deviceAddress(),
        .stride = (VkDeviceSize) sbt.recordSize,
        .size = (VkDeviceSize) sbt.recordSize * sbt.numRecords,
    };
}

PipelineLayout RaytracingPipeline::layout() const
{
    return impl().layout;
}

DescriptorSetLayout RaytracingPipeline::descSetLayout(uint32_t setId) const
{
    return impl().layout.descSetLayout(setId);
}
#endif // EVA_ENABLE_RAYTRACING


/////////////////////////////////////////////////////////////////////////////////////////
// Buffer
/////////////////////////////////////////////////////////////////////////////////////////
Buffer Device::createBuffer(const BufferCreateInfo& info) 
{
    auto vkHandle = create<VkBuffer>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = info.size,
        .usage = (VkBufferUsageFlags)(uint32_t)info.usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,  // TODO: support VK_SHARING_MODE_CONCURRENT
    });

    auto memInfo = getMemoryAllocInfo(
        impl().vkPhysicalDevice, impl().vkDevice, vkHandle, (VkMemoryPropertyFlags)(uint32_t)info.reqMemProps);

    static VkMemoryAllocateFlagsInfo flagsInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    };

    if ((uint32_t)info.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
        memInfo.first.pNext = &flagsInfo;

    VkDeviceMemory memory = allocate<VkDeviceMemory>(impl().vkDevice, memInfo.first);
    ASSERT_SUCCESS(vkBindBufferMemory(impl().vkDevice, vkHandle, memory, 0));

    
    auto pImpl = new Buffer::Impl(
        impl().vkDevice,
        vkHandle,
        memory,
        info.size,
        info.usage,
        info.reqMemProps,
        (MEMORY_PROPERTY)(uint32_t)memInfo.second);

    if ((uint32_t)info.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
    {
        // @chay116: Ensure function pointer is loaded (coopmat2 or raytracing)
        if (vkGetBufferDeviceAddressKHR_ == nullptr)
        {
            fprintf(stderr, "[EVA ERROR] vkGetBufferDeviceAddressKHR_ is null! "
                    "SHADER_DEVICE_ADDRESS requires coopmat2 or raytracing support.\n");
            EVA_ASSERT(false);
        }
        VkBufferDeviceAddressInfo addressInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = vkHandle
        };
        pImpl->deviceAddress = vkGetBufferDeviceAddressKHR_(impl().vkDevice, &addressInfo);
    }

    Buffer result = *impl().buffers.insert(new Buffer::Impl*(pImpl)).first;
    // @chay116 - BEGIN
    // Set cached values for inline accessors
    result._cachedSize = info.size;
    result._cachedUsage = info.usage;
    // @chay116 - END
    return result;
}

uint8_t* Buffer::map(uint64_t offset, uint64_t size)
{
    EVA_ASSERT(*this);
    EVA_ASSERT((uint32_t)impl().memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);  // VUID-vkMapMemory-memory-00682
    EVA_ASSERT(offset < impl().size);                                   // VUID-vkMapMemory-offset-00679
    
    if (size == EVA_WHOLE_SIZE)
        size = impl().size - offset;
    else
        EVA_ASSERT(offset + size <= impl().size);  // VUID-vkMapMemory-size-00681

    if (impl().mapped)
    {
        if (impl().mappedOffset != offset || impl().mappedSize != size)
            vkUnmapMemory(impl().vkDevice, impl().vkMemory);
        else
            return impl().mapped;
    }

    ASSERT_SUCCESS(vkMapMemory(impl().vkDevice, impl().vkMemory, offset, size, 0, (void**)&impl().mapped));
    impl().mappedOffset = offset;
    impl().mappedSize = size;
    return impl().mapped;
}

VkMappedMemoryRange Buffer::Impl::getRange(uint64_t offset, uint64_t size) const
{
    EVA_ASSERT(mapped);                                                                    // VUID-VkMappedMemoryRange-memory-00684
    if (size == EVA_WHOLE_SIZE) 
        EVA_ASSERT(mappedOffset <= offset && offset <= mappedOffset + mappedSize);         // VUID-VkMappedMemoryRange-memory-00686
    else 
        EVA_ASSERT(mappedOffset <= offset && offset + size <= mappedOffset + mappedSize);  // VUID-VkMappedMemoryRange-memory-00685

    return {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = vkMemory,
        .offset = offset,
        .size = size,
    };
}

void Buffer::flush(uint64_t offset, uint64_t size) const
{
    VkMappedMemoryRange range = impl().getRange(offset, size);
    if ((uint32_t)impl().memProps & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) 
        return;
    vkFlushMappedMemoryRanges(impl().vkDevice, 1, &range);
}

void Buffer::invalidate(uint64_t offset, uint64_t size) const
{
    VkMappedMemoryRange range = impl().getRange(offset, size);
    if ((uint32_t)impl().memProps & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) 
        return;
    vkInvalidateMappedMemoryRanges(impl().vkDevice, 1, &range);
}

void Buffer::unmap()
{
    if (!impl().mapped)  // avoid VUID-vkUnmapMemory-memory-00689
        return;

    EVA_ASSERT((uint32_t)impl().memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    vkUnmapMemory(impl().vkDevice, impl().vkMemory);
    impl().mapped = nullptr;
    impl().mappedOffset = 0;
    impl().mappedSize = 0;
}

// Note: Buffer::size() and Buffer::usage() are inline in eva-runtime.h (cached values)

MEMORY_PROPERTY Buffer::memoryProperties() const
{
    return impl().memProps;
}

void* Buffer::nativeBuffer() const
{
    return impl().vkBuffer;
}

void* Buffer::nativeMemory() const
{
    return impl().vkMemory;
}

DeviceAddress Buffer::deviceAddress() const
{
    EVA_ASSERT((uint32_t)impl().usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    return impl().deviceAddress;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Image
/////////////////////////////////////////////////////////////////////////////////////////
Image Device::createImage(const ImageCreateInfo& info)
{
    IMAGE_LAYOUT initialLayout = info.preInitialized ? IMAGE_LAYOUT::PREINITIALIZED : IMAGE_LAYOUT::UNDEFINED;

    auto vkHandle = create<VkImage>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = (VkImageCreateFlags)(uint32_t)info.flags,
        .imageType = info.extent.depth > 1 ? VK_IMAGE_TYPE_3D : 
                     info.extent.height > 1 ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_1D,
        .format = (VkFormat)(uint32_t)info.format,
        .extent = {
            info.extent.width,
            info.extent.height,
            info.extent.depth
        },
        .mipLevels = 1,                             // TODO: support mipmap 
        .arrayLayers = info.arrayLayers,
        .samples = VK_SAMPLE_COUNT_1_BIT,           // TODO: support multi-sampling
        .tiling = VK_IMAGE_TILING_OPTIMAL,          // TODO: support VK_IMAGE_TILING_LINEAR
        .usage = (VkImageUsageFlags)(uint32_t)info.usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,   // TODO: support VK_SHARING_MODE_CONCURRENT
        .initialLayout = (VkImageLayout)(uint32_t) initialLayout,
    });

    auto memInfo = getMemoryAllocInfo(
        impl().vkPhysicalDevice, impl().vkDevice, vkHandle, (VkMemoryPropertyFlags)(uint32_t)info.reqMemProps);

    VkDeviceMemory memory = allocate<VkDeviceMemory>(impl().vkDevice, memInfo.first);
    ASSERT_SUCCESS(vkBindImageMemory(impl().vkDevice, vkHandle, memory, 0));

    auto pImpl = new Image::Impl(
        impl().vkDevice, 
        vkHandle, 
        memory, 
        info.format,
        info.extent.width, 
        info.extent.height, 
        info.extent.depth,
        info.arrayLayers,
        info.usage,
        // initialLayout,
        false);

    return *impl().images.insert(new Image::Impl*(pImpl)).first;
}

ImageView Image::view(ImageViewDesc&& desc) const
{   
    if (desc.viewType == IMAGE_VIEW_TYPE::MAX_ENUM)
    {
        // if (this->operator->()->depth > 1)  // OK
        // if ((*this).operator->()->depth > 1)  // OK
        // if ((*this)->depth > 1)
        if (impl().depth > 1)
        // if (impl().depth > 1)
        {
            EVA_ASSERT(impl().arrayLayers == 1); // VUID-VkImageCreateInfo-imageType-00961
            desc.viewType = IMAGE_VIEW_TYPE::_3D;
        }
        else
        {
            if (impl().height == 1)
                desc.viewType = impl().arrayLayers == 1 ? IMAGE_VIEW_TYPE::_1D : IMAGE_VIEW_TYPE::_1D_ARRAY;
            else 
                desc.viewType = impl().arrayLayers == 1 ? IMAGE_VIEW_TYPE::_2D : IMAGE_VIEW_TYPE::_2D_ARRAY;
        }
    }
    
    if (desc.format == FORMAT::MAX_ENUM)
        desc.format = impl().format;

    if (auto it = impl().imageViews.find(desc); it != impl().imageViews.end())
        return it->second;

    VkComponentMapping components = {
        .r = (VkComponentSwizzle)(uint32_t)desc.components.r,
        .g = (VkComponentSwizzle)(uint32_t)desc.components.g,
        .b = (VkComponentSwizzle)(uint32_t)desc.components.b,
        .a = (VkComponentSwizzle)(uint32_t)desc.components.a,
    };

    auto vkHandle = create<VkImageView>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = impl().vkImage,
        .viewType = (VkImageViewType)(uint32_t)desc.viewType,
        .format = (VkFormat)(uint32_t)desc.format,
        .components = components,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,    // TODO: support depth/stencil
            .levelCount = VK_REMAINING_MIP_LEVELS,      // TODO: support mipmap
            .layerCount = VK_REMAINING_ARRAY_LAYERS,    // TODO: support layer control
        },
    });

    auto pImpl = new ImageView::Impl(vkHandle);
    return impl().imageViews.emplace(std::move(desc), new ImageView::Impl*(pImpl)).first->second;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Sampler
/////////////////////////////////////////////////////////////////////////////////////////
Sampler Device::createSampler(const SamplerCreateInfo& info)
{
    auto vkHandle = create<VkSampler>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = (VkFilter)(uint32_t)info.magFilter,
        .minFilter = (VkFilter)(uint32_t)info.minFilter,
        .mipmapMode = (VkSamplerMipmapMode)(uint32_t)info.mipmapMode,
        .addressModeU = (VkSamplerAddressMode)(uint32_t)info.addressModeU,
        .addressModeV = (VkSamplerAddressMode)(uint32_t)info.addressModeV,
        .addressModeW = (VkSamplerAddressMode)(uint32_t)info.addressModeW,
        .mipLodBias = info.mipLodBias,
        .anisotropyEnable = info.anisotropyEnable,
        .maxAnisotropy = info.maxAnisotropy,
        .compareEnable = info.compareEnable,
        .compareOp = (VkCompareOp)(uint32_t)info.compareOp,
        .minLod = info.minLod,
        .maxLod = info.maxLod,
        .borderColor = (VkBorderColor)(uint32_t)info.borderColor,
        .unnormalizedCoordinates = info.unnormalizedCoordinates,
    });

    auto pImpl = new Sampler::Impl(
        impl().vkDevice,
        vkHandle);
    
    return *impl().samplers.insert(new Sampler::Impl*(pImpl)).first;
}


/////////////////////////////////////////////////////////////////////////////////////////
// DescriptorSetLayout
/////////////////////////////////////////////////////////////////////////////////////////
DescriptorSetLayout Device::createDescriptorSetLayout(DescriptorSetLayoutDesc desc)
{
    std::vector<VkDescriptorSetLayoutBinding> vkBindings; vkBindings.reserve(desc.bindings.size());

    for (auto& [_, bindingInfo] : desc.bindings) 
    {
        vkBindings.push_back(VkDescriptorSetLayoutBinding{
            .binding = bindingInfo.binding,
            .descriptorType = (VkDescriptorType)(uint32_t)bindingInfo.descriptorType,
            .descriptorCount = bindingInfo.descriptorCount,
            .stageFlags = bindingInfo.stageFlags == SHADER_STAGE::NONE ?
                VK_SHADER_STAGE_ALL :
                (VkShaderStageFlags)(uint32_t)bindingInfo.stageFlags,
            .pImmutableSamplers = nullptr
        }); // TODO: Clang 호환성 - emplace_back → push_back + designated initializer
    }

    auto vkHandle = create<VkDescriptorSetLayout>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = (uint32_t)vkBindings.size(),
        .pBindings = vkBindings.data(),
    });

    auto pImpl = new DescriptorSetLayout::Impl(
        impl().vkDevice,
        vkHandle,
        std::move(desc)
    );

    return *impl().descSetLayouts.insert(new DescriptorSetLayout::Impl*(pImpl)).first;
}


/////////////////////////////////////////////////////////////////////////////////////////
// PipelineLayout
/////////////////////////////////////////////////////////////////////////////////////////
PipelineLayout Device::createPipelineLayout(PipelineLayoutDesc desc)
{
    static VkDescriptorSetLayout emptyVkSetLayout = createDescriptorSetLayout({}).impl().vkSetLayout;
    uint32_t numSetLayouts = desc.setLayouts.size();
    uint32_t numPaddedSetLayouts = numSetLayouts > 0 ? desc.setLayouts.rbegin()->first + 1 : 0;
    
    std::map<uint32_t, DescriptorSetLayoutDesc>&& setLayoutDescs = std::move(desc.setLayouts);
    std::map<uint32_t, DescriptorSetLayout> setLayouts;
    std::vector<VkDescriptorSetLayout> vkSetLayouts(numPaddedSetLayouts, emptyVkSetLayout);

    for (auto& [setId, setLayoutDesc] : setLayoutDescs) 
        vkSetLayouts[setId] 
            = (setLayouts[setId] = createDescriptorSetLayout(std::move(setLayoutDesc)))
                .impl().vkSetLayout;
    
    // std::vector<VkPushConstantRange> vkPushConstants; 
    // vkPushConstants.reserve(desc.pushConstants.size());
    // for (auto& pushConstant : desc.pushConstants) 
    // {
    //     vkPushConstants.emplace_back(
    //         pushConstant.stageFlags == SHADER_STAGE::NONE ? 
    //             VK_SHADER_STAGE_ALL : 
    //             (VkShaderStageFlags)(uint32_t)pushConstant.stageFlags,
    //         pushConstant.offset,
    //         pushConstant.size
    //     );
    // }

    VkPushConstantRange vkPushConstant = {};
    if (desc.pushConstant) 
    {
        vkPushConstant.stageFlags = (desc.pushConstant->stageFlags == SHADER_STAGE::NONE) ? 
            VK_SHADER_STAGE_ALL : (VkShaderStageFlags)(uint32_t)desc.pushConstant->stageFlags;
        vkPushConstant.offset = desc.pushConstant->offset;
        vkPushConstant.size = desc.pushConstant->size;
    }
    
    auto vkHandle = create<VkPipelineLayout>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = (uint32_t)vkSetLayouts.size(),
        .pSetLayouts = vkSetLayouts.data(),
        .pushConstantRangeCount = desc.pushConstant ? 1u : 0u,
        .pPushConstantRanges = &vkPushConstant,
    });

    auto pImpl = new PipelineLayout::Impl(
        impl().vkDevice,
        vkHandle,
        std::move(setLayouts),
        std::move(desc.pushConstant)
    );

    return *impl().pipelineLayouts.insert(new PipelineLayout::Impl*(pImpl)).first;
}

DescriptorSetLayout PipelineLayout::descSetLayout(uint32_t setId) const
{
    return impl().setLayouts.at(setId);
}

uint32_t PipelineLayout::pushConstantSize() const
{
    return impl().uniquePushConstant ? impl().uniquePushConstant->size : 0;
}


/////////////////////////////////////////////////////////////////////////////////////////
// DescriptorPool
/////////////////////////////////////////////////////////////////////////////////////////
DescriptorPool Device::createDescriptorPool(const DescriptorPoolCreateInfo& info)
{
    std::vector<VkDescriptorPoolSize> vkPoolSizes;
    vkPoolSizes.reserve(info.maxTypes.size());
    for (const auto& poolSize : info.maxTypes)
    {
        vkPoolSizes.push_back({
            .type = (VkDescriptorType)(uint32_t)poolSize.type,
            .descriptorCount = poolSize.descriptorCount,
        });
    }

    auto vkHandle = create<VkDescriptorPool>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = info.maxSets,
        .poolSizeCount = (uint32_t)vkPoolSizes.size(),
        .pPoolSizes = vkPoolSizes.data(),
    });

    auto pImpl = new DescriptorPool::Impl(
        impl().vkDevice,
        vkHandle);

    return *impl().descPools.insert(new DescriptorPool::Impl*(pImpl)).first;
}


QueryPool Device::createQueryPool(uint32_t queryCount)
{
    VkQueryPoolCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = queryCount,
    };

    VkQueryPool vkQueryPool;
    ASSERT_SUCCESS(vkCreateQueryPool(impl().vkDevice, &createInfo, nullptr, &vkQueryPool));

    // Get timestamp period from physical device
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(impl().vkPhysicalDevice, &props);

    auto pImpl = new QueryPool::Impl(
        impl().vkDevice,
        vkQueryPool,
        queryCount,
        props.limits.timestampPeriod);

    return *impl().queryPools.insert(new QueryPool::Impl*(pImpl)).first;
}


bool Device::supportsTimestampQueries() const
{
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(impl().vkPhysicalDevice, &props);
    return props.limits.timestampComputeAndGraphics == VK_TRUE;
}


std::vector<uint64_t> QueryPool::getResults(uint32_t firstQuery, uint32_t queryCount)
{
    if (queryCount == 0) queryCount = impl().queryCount - firstQuery;

    std::vector<uint64_t> results(queryCount);
    vkGetQueryPoolResults(
        impl().vkDevice,
        impl().vkQueryPool,
        firstQuery,
        queryCount,
        queryCount * sizeof(uint64_t),
        results.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    );
    return results;
}


double QueryPool::getElapsedMs(uint32_t startQuery, uint32_t endQuery)
{
    auto results = getResults(startQuery, endQuery - startQuery + 1);
    if (results.size() < 2) return 0.0;

    uint64_t elapsed = results.back() - results.front();
    // timestampPeriod is in nanoseconds, convert to milliseconds
    return static_cast<double>(elapsed) * impl().timestampPeriod / 1e6;
}


uint32_t QueryPool::queryCount() const
{
    return impl().queryCount;
}


std::vector<DescriptorSet> DescriptorPool::operator()(std::vector<DescriptorSetLayout> setLayouts)
{
    std::vector<VkDescriptorSetLayout> vkSetLayouts(setLayouts.size());
    for (uint32_t i = 0; i < setLayouts.size(); ++i) 
        vkSetLayouts[i] = setLayouts[i].impl().vkSetLayout;

    std::vector<VkDescriptorSet> vkDescSets = allocate<VkDescriptorSet>(impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = impl().vkDescPool,
        .descriptorSetCount = (uint32_t)vkSetLayouts.size(),
        .pSetLayouts = vkSetLayouts.data(),
    });

    std::vector<DescriptorSet> descSets;
    descSets.reserve(setLayouts.size()); // TODO: Linux Clang 호환성 - vector(size) → reserve + push_back
    for (uint32_t i = 0; i < setLayouts.size(); ++i)
    {
        auto pImpl = new DescriptorSet::Impl(
            impl().vkDevice,
            vkDescSets[i],
            setLayouts[i]);

        descSets.push_back(impl().descSets.emplace_back(new DescriptorSet::Impl*(pImpl)));
    }

    return descSets;
}


/////////////////////////////////////////////////////////////////////////////////////////
// DescriptorSet
/////////////////////////////////////////////////////////////////////////////////////////
/*
* If the nullDescriptor feature is enabled, the buffer, acceleration structure, imageView, or
* bufferView can be VK_NULL_HANDLE. Loads from a null descriptor return zero values and stores
* and atomics to a null descriptor are discarded. A null acceleration structure descriptor results in
* the miss shader being invoked.
*/
DescriptorSet DescriptorSet::write(
    std::vector<Descriptor> descriptors, 
    uint32_t startBindingId, 
    uint32_t startArrayOffset)
{
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkDescriptorImageInfo> imageInfos;
    writes.reserve(descriptors.size());
    bufferInfos.reserve(descriptors.size());
    imageInfos.reserve(descriptors.size());

#ifdef EVA_ENABLE_RAYTRACING
    std::vector<VkWriteDescriptorSetAccelerationStructureKHR> asInfos;
    std::vector<VkAccelerationStructureKHR> asArray;
    asInfos.reserve(descriptors.size());
    asArray.reserve(descriptors.size());
#endif

    auto& bindingInfos = impl().layout.impl().desc.bindings;
    uint32_t consumedDescriptors = 0;
    
    auto iter = bindingInfos.lower_bound(startBindingId);   // It avoids VUID-VkWriteDescriptorSet-dstBinding-00316
    EVA_ASSERT(startArrayOffset == 0 || 
        (iter->first == startBindingId && startArrayOffset < iter->second.descriptorCount)); // Undefined behavior, not even mentioned in the Vulkan spec.

    auto resourceTypeOf = [](VkDescriptorType t) {
        switch (t) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            return 0;   // buffer
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        // case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            return 1;  // image
#ifdef EVA_ENABLE_RAYTRACING
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            return 2;  // acceleration structure
#endif
        default: return -1;
        }
    };

    while(iter != bindingInfos.end()) 
    {
        const auto& [dstBinding, bindingInfo] = *iter;

        uint32_t consecutiveDescCount = bindingInfo.descriptorCount - startArrayOffset;
        auto descriptorType = bindingInfo.descriptorType;
        auto stageFlags = bindingInfo.stageFlags;

        while (++iter != bindingInfos.end()        // See "Consecutive Binding Updates" in Vulkan spec.
            && iter->second.stageFlags == stageFlags
            && iter->second.descriptorType == descriptorType) 
        {
            consecutiveDescCount += iter->second.descriptorCount;
        } 
        consecutiveDescCount = std::min(consecutiveDescCount, (uint32_t)descriptors.size() - consumedDescriptors);

        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = impl().vkDescSet,
            .dstBinding = dstBinding,
            .dstArrayElement = startArrayOffset,
            .descriptorCount = consecutiveDescCount,
            .descriptorType = (VkDescriptorType)(uint32_t)descriptorType,
        });

        if (int t = resourceTypeOf((VkDescriptorType)(uint32_t)descriptorType); t == 0)
        {
            writes.back().pBufferInfo = bufferInfos.data() + bufferInfos.size();

            for (uint32_t i = 0; i < consecutiveDescCount; ++i) 
            {
                // bufferInfos.push_back(std::get<BufferDescriptor>(descriptors[consumedDescriptors + i]).descInfo());
                auto& desc = std::get<BufferDescriptor>(descriptors[consumedDescriptors + i]);
                bufferInfos.emplace_back(
                    desc.buffer ? desc.buffer.impl().vkBuffer : VK_NULL_HANDLE,
                    desc.offset,
                    desc.size
                );
            }
        }
        
        else if (t == 1)
        {
            IMAGE_LAYOUT defaultLayout = (descriptorType == DESCRIPTOR_TYPE::STORAGE_IMAGE) 
            ? IMAGE_LAYOUT::GENERAL 
            : IMAGE_LAYOUT::SHADER_READ_ONLY;

            writes.back().pImageInfo = imageInfos.data() + imageInfos.size();

            for (uint32_t i = 0; i < consecutiveDescCount; ++i) 
            {
                // imageInfos.push_back(std::get<ImageDescriptor>(descriptors[consumedDescriptors + i]).descInfo());
                auto& desc = std::get<ImageDescriptor>(descriptors[consumedDescriptors + i]);
                imageInfos.push_back(VkDescriptorImageInfo{
                    .sampler = desc.sampler ? desc.sampler->impl().vkSampler : VK_NULL_HANDLE,
                    .imageView = desc.imageView ? desc.imageView->impl().vkImageView : VK_NULL_HANDLE,
                    .imageLayout = (VkImageLayout)(uint32_t) (desc.imageLayout != IMAGE_LAYOUT::MAX_ENUM
                                                ? desc.imageLayout : defaultLayout)
                }); // TODO: Clang 호환성 - emplace_back → push_back + designated initializer
            }
        }
        
#ifdef EVA_ENABLE_RAYTRACING
        else if (t == 2)
        {
            writes.back().pNext = &asInfos.emplace_back(
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                nullptr,
                consecutiveDescCount,
                asArray.data() + asArray.size()
            );

            for (uint32_t i = 0; i < consecutiveDescCount; ++i)
            {
                auto& as = std::get<AccelerationStructure>(descriptors[consumedDescriptors + i]);
                asArray.push_back(as ? as.impl().vkAccelStruct : VK_NULL_HANDLE);
            }
        }
#endif

        else
            EVA_ASSERT(false); 

        consumedDescriptors += consecutiveDescCount;
        startArrayOffset = 0;

        if (consumedDescriptors == descriptors.size()) // Normal exit condition
            break;
    }
    EVA_ASSERT(consumedDescriptors == descriptors.size()); // If given shader data has not been fully consumed, it is considered as an error.
    
    vkUpdateDescriptorSets(impl().vkDevice, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    return *this;
}

DescriptorSet DescriptorSet::operator=(std::vector<DescriptorSet>&& data)
{
    EVA_ASSERT(data.size() == 1);
    *this = data[0];
    return *this;
}



/////////////////////////////////////////////////////////////////////////////////////////
// Window
/////////////////////////////////////////////////////////////////////////////////////////
#ifdef EVA_ENABLE_WINDOW
/*
* TODO: separate Swapchain from Window
* 지금대로라면, window가 파괴되기 전에 (swapchain을 소유한)device가 파괴되어서는 안된다.
*/
struct Window::Impl {
    const GLFWwindow* pWindow;
    const VkSurfaceKHR vkSurface;
    uint32_t width = 0;
    uint32_t height = 0;

    VkInstance vkInstance;
    VkDevice vkDevice;
    VkSwapchainKHR vkSwapchain;
    std::vector<Image> swapchainImages;
    mutable uint32_t presentingImgIdx = uint32_t(-1);
    FORMAT swapchainImageFormat;

    // Input callbacks
    void (*mouseButtonCallback)(int button, int action, double xpos, double ypos) = nullptr;
    void (*keyCallback)(int key, int action, int mods) = nullptr;
    void (*cursorPosCallback)(double xpos, double ypos) = nullptr;
    void (*scrollCallback)(double xoffset, double yoffset) = nullptr;

    // Pre-present command buffers (one per swapchain image)
    std::vector<CommandBuffer> prePresentCommandBuffers;
    std::vector<Semaphore> presentableSemaphores;

    Impl(
        const GLFWwindow* pWindow,
        const VkSurfaceKHR vkSurface,
        uint32_t width,
        uint32_t height,
        VkInstance vkInstance,
        VkDevice vkDevice,
        VkSwapchainKHR vkSwapchain,
        std::vector<Image>&& swapchainImages,
        FORMAT swapchainImageFormat)
        : pWindow(pWindow)
        , vkSurface(vkSurface)
        , width(width)
        , height(height)
        , vkInstance(vkInstance)
        , vkDevice(vkDevice)
        , vkSwapchain(vkSwapchain)
        , swapchainImages(std::move(swapchainImages))
        , swapchainImageFormat(swapchainImageFormat)
    {
    }
    ~Impl() {
        for (auto image : swapchainImages)
        {
            image.destroy();
        }
        vkDestroySwapchainKHR(vkDevice, vkSwapchain, nullptr);
        vkDestroySurfaceKHR(vkInstance, vkSurface, nullptr);
        glfwDestroyWindow((GLFWwindow*)pWindow);
    }

};

Window Runtime::createWindow(WindowCreateInfo info)
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, info.hidden ? GLFW_FALSE : GLFW_TRUE);
    GLFWwindow* pWindow = glfwCreateWindow(info.width, info.height, info.title, nullptr, nullptr);
    if (!pWindow) 
        throw std::runtime_error("Failed to create GLFW window.");

    VkSurfaceKHR vkSurface;
    ASSERT_SUCCESS(glfwCreateWindowSurface(impl().instance, pWindow, nullptr, &vkSurface));

    /*
    * From here, we create a swapchain for the window.
    */
    VkPhysicalDevice pd = info.device.impl().vkPhysicalDevice;
    
    VkSurfaceCapabilitiesKHR capabilities;
    ASSERT_SUCCESS(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, vkSurface, &capabilities));
    EVA_ASSERT(capabilities.currentExtent.width == info.width      // in almost platforms
        || capabilities.currentExtent.width == UINT32_MAX);     // in Wayland, etc.

    EVA_ASSERT( 
        ([&]() { 
        for (auto& f : arrayFrom(vkGetPhysicalDeviceSurfaceFormatsKHR, pd, vkSurface)) 
            if (f.format == (uint32_t)info.swapChainImageFormat && f.colorSpace == (uint32_t)info.swapChainImageColorSpace) 
                return true; 
            return false; 
        })() 
    ); // we do not support fallback formats for simplicity

    /*
    vkGetPhysicalDeviceSurfaceFormatsKHR 은 통과하였으나
    해당 포맷이 OPTIMAL_TILING + usage 조합에 지원되지 않을 수도 있다.
    */
    EVA_ASSERT(
        ([&]() { 
            VkFormatProperties formatProps;
            vkGetPhysicalDeviceFormatProperties(pd, (VkFormat)(uint32_t)info.swapChainImageFormat, &formatProps);
            if (((uint32_t)info.swapChainImageUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
                && !(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
                return false;
            if (((uint32_t)info.swapChainImageUsage & VK_IMAGE_USAGE_STORAGE_BIT)
                && !(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
                return false;
            if (((uint32_t)info.swapChainImageUsage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                && !(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT))
                return false;
            return true;
        })()
    );

    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR; // guaranteed to be available
    {
        auto presentModes = arrayFrom(vkGetPhysicalDeviceSurfacePresentModesKHR, pd, vkSurface);
        for (auto mode : presentModes) 
        {
            if (mode == (VkPresentModeKHR)(uint32_t)info.preferredPresentMode) 
            {
                presentMode = mode; 
                break; 
            }
        }
    }
    
    uint32_t minSwapChainImages = info.minSwapChainImages > 0 ? info.minSwapChainImages : capabilities.minImageCount + 1;
    if (0 < capabilities.maxImageCount && capabilities.maxImageCount < minSwapChainImages) 
        minSwapChainImages = capabilities.maxImageCount;

    VkSwapchainKHR vkSwapchain = create<VkSwapchainKHR>(info.device.impl().vkDevice, {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = vkSurface,
        .minImageCount = minSwapChainImages,
        .imageFormat = (VkFormat)(uint32_t)info.swapChainImageFormat,
        .imageColorSpace = (VkColorSpaceKHR)(uint32_t)info.swapChainImageColorSpace,
        .imageExtent = { info.width, info.height }, 
        .imageArrayLayers = 1,
        .imageUsage = (VkImageUsageFlags)(uint32_t)info.swapChainImageUsage,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,          // TODO: support VK_SHARING_MODE_CONCURRENT
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,    // do not support other composite modes for simplicity
        .presentMode = presentMode,                             
        .clipped = VK_TRUE,                                     // it allow to skip fragment shader of final render pass for pixels that are not visible 
    });

    std::vector<Image> swapchainImages;
    {
        auto vkImages = arrayFrom(vkGetSwapchainImagesKHR, info.device.impl().vkDevice, vkSwapchain);
        swapchainImages.reserve(vkImages.size());
        for (auto vkImage : vkImages)
        {
            auto pImpl = new Image::Impl(
                info.device.impl().vkDevice,
                vkImage,
                VK_NULL_HANDLE,    // memory is owned by the swapchain
                info.swapChainImageFormat,
                info.width,
                info.height,
                1,                  // depth
                1,                  // arrayLayers
                info.swapChainImageUsage,
                // IMAGE_LAYOUT::UNDEFINED,
                true);              // owned by swapchain

            swapchainImages.push_back(*info.device.impl().images.insert(new Image::Impl*(pImpl)).first);
        }
    }

    auto pImpl = new Window::Impl(
        pWindow,
        vkSurface,
        info.width,
        info.height,
        impl().instance,
        info.device.impl().vkDevice,
        vkSwapchain,
        std::move(swapchainImages),
        info.swapChainImageFormat
    );

    auto numImages = pImpl->swapchainImages.size();

    pImpl->presentableSemaphores.resize(numImages);
    for (uint32_t i = 0; i < numImages; ++i)
        pImpl->presentableSemaphores[i] = info.device.createSemaphore();

    if (info.prePresentCommandPool)
        pImpl->prePresentCommandBuffers = info.prePresentCommandPool.newCommandBuffers(numImages);
    else
        pImpl->prePresentCommandBuffers = info.device.newCommandBuffers(numImages, info.prePresentCommandPoolType, info.prePresentCommandPoolFlags);

    return impl().windows.emplace_back(new Window::Impl*(pImpl));
}

const std::vector<Image>& Window::swapChainImages() const
{
    return impl().swapchainImages;
}

void Window::recordPrePresentCommands(std::function<void(CommandBuffer, Image)> recordFunc)
{
    for (size_t i = 0; i < impl().swapchainImages.size(); ++i)
        recordFunc(impl().prePresentCommandBuffers[i], impl().swapchainImages[i]);
}

/*
Let S be the number of images in swapchain. If swapchain is created with
VkSwapchainPresentModesCreateInfoEXT, let M be the maximum of the values in
VkSurfaceCapabilitiesKHR::minImageCount when queried with each present mode in
VkSwapchainPresentModesCreateInfoEXT::pPresentModes in VkSurfacePresentModeEXT. Otherwise,
let M be the value of VkSurfaceCapabilitiesKHR::minImageCount without a
VkSurfacePresentModeEXT as part of the query input.
vkAcquireNextImageKHR should not be called if the number of images that the application has
currently acquired is greater than S-M. If vkAcquireNextImageKHR is called when the number of
images that the application has currently acquired is less than or equal to S-M,
vkAcquireNextImageKHR must return in finite time with an allowed VkResult code.
*/
uint32_t Window::acquireNextImageIndex(Semaphore onNextScImageWritable) const
{
    uint32_t imageIndex = 0;
    ASSERT_SUCCESS(vkAcquireNextImageKHR(
        impl().vkDevice,
        impl().vkSwapchain,
        UINT64_MAX,         // no timeout for simplicity
        onNextScImageWritable.impl().vkSemaphore,
        VK_NULL_HANDLE,     // no fence for simplicity
        &imageIndex));

    return imageIndex;
}

void Window::present(Queue queue, std::vector<Semaphore> waitSemaphores, uint32_t imageIndex) const
{
    std::vector<VkSemaphore> vkWaitSemaphores(waitSemaphores.size());
    for (size_t i = 0; i < waitSemaphores.size(); i++)
        vkWaitSemaphores[i] = waitSemaphores[i].impl().vkSemaphore;

    VkPresentInfoKHR presentInfo = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
        .pWaitSemaphores = vkWaitSemaphores.data(),
        .swapchainCount = 1,
        .pSwapchains = &impl().vkSwapchain,
        .pImageIndices = &imageIndex
    };
    ASSERT_SUCCESS(vkQueuePresentKHR(queue.impl().vkQueue, &presentInfo));
}

std::pair<CommandBuffer, Semaphore> Window::getNextPresentingContext(Semaphore onNextScImageWritable) const
{
    uint32_t imageIndex = acquireNextImageIndex(onNextScImageWritable);
    EVA_ASSERT(impl().presentingImgIdx == uint32_t(-1));
    impl().presentingImgIdx = imageIndex;

    return {
        impl().prePresentCommandBuffers[imageIndex],
        impl().presentableSemaphores[imageIndex]
    };
}

void Window::present(Queue queue) const
{
    uint32_t imageIndex = impl().presentingImgIdx;
    EVA_ASSERT(imageIndex != uint32_t(-1));
    impl().presentingImgIdx = uint32_t(-1);

    present(queue, {impl().presentableSemaphores[imageIndex]}, imageIndex);
}

bool Window::shouldClose() const
{
    return glfwWindowShouldClose((GLFWwindow*)impl().pWindow);
}

void Window::pollEvents() const
{
    glfwPollEvents();
}

void Window::focus() const
{
    GLFWwindow* window = (GLFWwindow*)impl().pWindow;
    if (window) {
        glfwFocusWindow(window);
        glfwRequestWindowAttention(window);
    }
}

void Window::setTitle(const char* title) const
{
    glfwSetWindowTitle((GLFWwindow*)impl().pWindow, title);
}

void Window::setMouseButtonCallback(void (*callback)(int button, int action, double xpos, double ypos))
{
    impl().mouseButtonCallback = callback;

    auto glfwCallback = [](GLFWwindow* window, int button, int action, int mods) {
        // Get Window::Impl from user pointer
        Window::Impl* pImpl = (Window::Impl*)glfwGetWindowUserPointer(window);
        if (pImpl && pImpl->mouseButtonCallback) {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            pImpl->mouseButtonCallback(button, action, xpos, ypos);
        }
    };

    // Set user pointer to Window::Impl
    glfwSetWindowUserPointer((GLFWwindow*)impl().pWindow, const_cast<Window::Impl*>(&impl()));
    glfwSetMouseButtonCallback((GLFWwindow*)impl().pWindow, glfwCallback);
}

void Window::setKeyCallback(void (*callback)(int key, int action, int mods))
{
    impl().keyCallback = callback;

    auto glfwCallback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        // Get Window::Impl from user pointer
        Window::Impl* pImpl = (Window::Impl*)glfwGetWindowUserPointer(window);
        if (pImpl && pImpl->keyCallback) {
            pImpl->keyCallback(key, action, mods);
        }
    };

    // Set user pointer to Window::Impl
    glfwSetWindowUserPointer((GLFWwindow*)impl().pWindow, const_cast<Window::Impl*>(&impl()));
    glfwSetKeyCallback((GLFWwindow*)impl().pWindow, glfwCallback);
}

void Window::setCursorPosCallback(void (*callback)(double xpos, double ypos))
{
    impl().cursorPosCallback = callback;

    auto glfwCallback = [](GLFWwindow* window, double xpos, double ypos) {
        // Get Window::Impl from user pointer
        Window::Impl* pImpl = (Window::Impl*)glfwGetWindowUserPointer(window);
        if (pImpl && pImpl->cursorPosCallback) {
            pImpl->cursorPosCallback(xpos, ypos);
        }
    };

    // Set user pointer to Window::Impl
    glfwSetWindowUserPointer((GLFWwindow*)impl().pWindow, const_cast<Window::Impl*>(&impl()));
    glfwSetCursorPosCallback((GLFWwindow*)impl().pWindow, glfwCallback);
}

void Window::setScrollCallback(void (*callback)(double xoffset, double yoffset))
{
    impl().scrollCallback = callback;

    auto glfwCallback = [](GLFWwindow* window, double xoffset, double yoffset) {
        // Get Window::Impl from user pointer
        Window::Impl* pImpl = (Window::Impl*)glfwGetWindowUserPointer(window);
        if (pImpl && pImpl->scrollCallback) {
            pImpl->scrollCallback(xoffset, yoffset);
        }
    };

    // Set user pointer to Window::Impl
    glfwSetWindowUserPointer((GLFWwindow*)impl().pWindow, const_cast<Window::Impl*>(&impl()));
    glfwSetScrollCallback((GLFWwindow*)impl().pWindow, glfwCallback);
}
#endif // EVA_ENABLE_WINDOW


/////////////////////////////////////////////////////////////////////////////////////////
// AccelerationStructure
/////////////////////////////////////////////////////////////////////////////////////////
#ifdef EVA_ENABLE_RAYTRACING
AccelerationStructure Device::createAccelerationStructure(const AsCreateInfo& info)
{
    EVA_ASSERT((uint32_t)info.internalBuffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);  // VUID-VkAccelerationStructureCreateInfoKHR-buffer-03614
    EVA_ASSERT((uint32_t)info.internalBuffer.memoryProperties() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);          // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03707
    EVA_ASSERT(info.internalBuffer.offset % asBufferOffsetAlignment() == 0);
    EVA_ASSERT(info.size != 0);
    
    VkAccelerationStructureKHR vkAccelStruct;
    {
        VkAccelerationStructureCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = info.internalBuffer.buffer.impl().vkBuffer,
            .offset = (VkDeviceSize)info.internalBuffer.offset,
            .size = (VkDeviceSize) info.size,
            .type = (VkAccelerationStructureTypeKHR)(uint32_t)info.asType,
        };
        ASSERT_SUCCESS(vkCreateAccelerationStructureKHR_(impl().vkDevice, &createInfo, nullptr, &vkAccelStruct));
    }

    auto pImpl = new AccelerationStructure::Impl(
        impl().vkDevice, 
        vkAccelStruct
    );

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = vkAccelStruct,
    };

    EVA_ASSERT((uint32_t)info.internalBuffer.usage() & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    pImpl->deviceAddress = vkGetAccelerationStructureDeviceAddressKHR_(impl().vkDevice, &addressInfo);

    return *impl().accelerationStructures.insert(new AccelerationStructure::Impl*(pImpl)).first;
}

DeviceAddress AccelerationStructure::deviceAddress() const
{
    // EVA_ASSERT(impl().usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    return impl().deviceAddress;
}


// VkAccelerationStructureBuildSizesInfoKHR Device::getBuildSizesInfo(const AsBuildInfo& info) const
// {
//     VkAccelerationStructureGeometryKHR base{
//         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
//     };

//     VkAccelerationStructureTypeKHR asType;
//     VkBuildAccelerationStructureFlagsKHR buildFlags;
//     size_t geometryCount;
//     std::vector<uint32_t> primitiveCounts;

//     std::visit([&](auto&& asInfo){
//         using T = std::decay_t<decltype(asInfo)>;
//         geometryCount = asInfo.geometries.size();
//         buildFlags = asInfo.buildFlags;
//         primitiveCounts.reserve(geometryCount);

//         if constexpr (std::is_same_v<T, AsBuildInfoTriangles>) 
//         {
//             asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
//             base.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
//             base.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
//             for (const auto& geom : asInfo.geometries)
//                 primitiveCounts.push_back(geom.triangleCount);
//         } 
//         else if constexpr (std::is_same_v<T, AsBuildInfoAabbs>) 
//         {
//             asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
//             base.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
//             base.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
//             for (const auto& geom : asInfo.geometries)
//                 primitiveCounts.push_back(geom.aabbCount);
//         } 
//         else 
//         {
//             EVA_ASSERT(asInfo.geometries.size() == 1);
//             asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
//             base.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
//             base.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
//             primitiveCounts.push_back(asInfo.geometries[0].instanceCount);
//         };
//     }, info);

//     std::vector<VkAccelerationStructureGeometryKHR> geometries(geometryCount, base);

//     VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
//         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
//         .type = asType,
//         .flags = buildFlags,
//         .geometryCount = (uint32_t) geometryCount,
//         .pGeometries = geometries.data(),
//     };

//     VkAccelerationStructureBuildSizesInfoKHR sizeInfo {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
//     vkGetAccelerationStructureBuildSizesKHR_(
//         impl().vkDevice,
//         VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, // not support HOST build
//         &buildInfo,
//         primitiveCounts.data(),
//         &sizeInfo
//     );

//     return sizeInfo;
// }


AsBuildSizesInfo Device::getBuildSizesInfo(const AsBuildInfo& info) const
{
    VkAccelerationStructureGeometryKHR base{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = (VkGeometryTypeKHR)(uint32_t)info.geometryType,
    };

    if (info.geometryType == GEOMETRY_TYPE::TRIANGLES) 
        base.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    else if (info.geometryType == GEOMETRY_TYPE::AABBS) 
        base.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    else if (info.geometryType == GEOMETRY_TYPE::INSTANCES)
        base.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    else 
        EVA_ASSERT(false);

    std::vector<VkAccelerationStructureGeometryKHR> geometries(info.primitiveCounts.size(), base);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = info.geometryType == GEOMETRY_TYPE::INSTANCES
            ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
            : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .flags = (VkBuildAccelerationStructureFlagsKHR)(uint32_t)info.buildFlags,        
        .geometryCount = (uint32_t) geometries.size(),
        .pGeometries = geometries.data(),
    };

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR_(
        impl().vkDevice,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        info.primitiveCounts.data(),
        &sizeInfo
    );

    return {
        sizeInfo.accelerationStructureSize,
        sizeInfo.updateScratchSize,
        sizeInfo.buildScratchSize
    };
}

// CommandBuffer CommandBuffer::buildAccelerationStructures(const AsBuildInfoTriangles& info)
// {
//     EVA_ASSERT(info.scratchBuffer.deviceAddress() % impl().device.minAccelerationStructureScratchOffsetAlignment() == 0);  // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03710
//     EVA_ASSERT(info.scratchBuffer.memoryProperties() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);    // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03707
//     EVA_ASSERT(info.scratchBuffer.usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);                // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03674

//     // Either all geometries use the common buffer, or each geometry has its own buffer.
//     bool useCommonBuffer = info.common.vertexInput.buffer;
//     EVA_ASSERT(
//         (useCommonBuffer && !info.geometries[0].vertexInput.buffer) ||
//         (!useCommonBuffer && info.geometries[0].vertexInput.buffer)
//     );

//     std::vector<VkAccelerationStructureGeometryKHR> geometries(info.geometries.size(), {
//         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
//         .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
//     });

//     std::vector<VkAccelerationStructureBuildRangeInfoKHR> rangeInfo(geometries.size(), {});
    
//     uint32_t accVtxCount = 0;
//     uint32_t accIdxOffset = 0;
//     for (size_t i = 0; i < geometries.size(); ++i) 
//     {
//         auto& srcGeometry = info.geometries[i];

//         geometries[i].flags = srcGeometry.flags;
//         rangeInfo[i].primitiveCount = srcGeometry.triangleCount;

//         auto& dst = geometries[i].geometry.triangles;
        
//         auto& [vtxBuffer, vtxStride] = useCommonBuffer ? info.common.vertexInput : srcGeometry.vertexInput;
//         EVA_ASSERT(vtxBuffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
//         EVA_ASSERT(vtxBuffer.deviceAddress() % 4 == 0);   // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03711
//         EVA_ASSERT(vtxStride % 4 == 0);                   // VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexStride-03735
//         dst.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
//         dst.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;              // This format is the only one supported
//         dst.vertexData.deviceAddress = vtxBuffer.deviceAddress();
//         dst.vertexStride = (VkDeviceSize) vtxStride;
//         dst.maxVertex = srcGeometry.vertexCount - 1;
//         if (useCommonBuffer)
//         {
//             rangeInfo[i].firstVertex = accVtxCount;
//             accVtxCount += srcGeometry.vertexCount;
//         }
        
//         auto& [idxBuffer, idxStride] = useCommonBuffer ? info.common.indexInput : srcGeometry.indexInput;
//         /*
//         ◦ If the geometry uses indices, primitiveCount × 3 indices are consumed from
//         VkAccelerationStructureGeometryTrianglesDataKHR::indexData, starting at an offset of
//         primitiveOffset. The value of firstVertex is added to the index values before fetching
//         vertices.
//         ◦ If the geometry does not use indices, primitiveCount × 3 vertices are consumed from
//         VkAccelerationStructureGeometryTrianglesDataKHR::vertexData, starting at an offset of
//         primitiveOffset + VkAccelerationStructureGeometryTrianglesDataKHR::vertexStride ×
//         firstVertex.
//         ◦ If VkAccelerationStructureGeometryTrianglesDataKHR::transformData is not NULL, a single
//         VkTransformMatrixKHR structure is consumed from
//         VkAccelerationStructureGeometryTrianglesDataKHR::transformData, at an offset of
//         transformOffset. This matrix describes a transformation from the space in which the
//         vertices for all triangles in this geometry are described to the space in which the
//         acceleration structure is defined.
//         */
//         if (idxBuffer)
//         {
//             EVA_ASSERT(idxBuffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
//             EVA_ASSERT(idxStride == 2 || idxStride == 4);
//             EVA_ASSERT((idxStride == 2 && idxBuffer.deviceAddress() % 2 == 0) ||
//                 (idxStride == 4 && idxBuffer.deviceAddress() % 4 == 0) );  // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03712
//             dst.indexType = idxStride == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
//             dst.indexData.deviceAddress = idxBuffer.deviceAddress();
//             if (useCommonBuffer)
//             {
//                 rangeInfo[i].primitiveOffset = accIdxOffset;
//                 accIdxOffset += srcGeometry.triangleCount * 3 * idxStride;
//             }
//         }
//         else
//         {
//             dst.indexType = VK_INDEX_TYPE_NONE_KHR;             // VUID-VkAccelerationStructureGeometryTrianglesDataKHR-indexType-03798
//             /*
//             primitiveOffset is 0 because the role of the offset is included in firstVertex. 
//             */
//             EVA_ASSERT(srcGeometry.vertexCount == srcGeometry.triangleCount * 3); 
//             rangeInfo[i].primitiveOffset = 0;
//         }
        
//         if (srcGeometry.transformBuffer)
//         {
//             EVA_ASSERT(srcGeometry.transformBuffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
//             dst.transformData.deviceAddress = srcGeometry.transformBuffer.deviceAddress();
//             rangeInfo[i].transformOffset = 0;
//             EVA_ASSERT(dst.transformData.deviceAddress % 16 == 0); // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03810
//             EVA_ASSERT(rangeInfo[i].transformOffset % 16 == 0);    // VUID-VkAccelerationStructureBuildRangeInfoKHR-transformOffset-03658
//         }
//     }

//     VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
//         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
//         .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
//         .flags = info.buildFlags,
//         .mode = info.srcAs ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
//         .srcAccelerationStructure = info.srcAs ? info.srcAs.impl().vkAccelStruct : VK_NULL_HANDLE,
//         .dstAccelerationStructure = info.dstAs.impl().vkAccelStruct,
//         .geometryCount = (uint32_t)geometries.size(),
//         .pGeometries = geometries.data(),
//         .scratchData = {.deviceAddress = info.scratchBuffer.deviceAddress()},
//     };

//     VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos[] = {rangeInfo.data()};
//     vkCmdBuildAccelerationStructuresKHR_(
//         impl().vkCmdBuffer,
//         1,
//         &buildInfo,
//         pRangeInfos
//     );

//     return *this;
// }

// CommandBuffer CommandBuffer::buildAccelerationStructures(const AsBuildInfoAabbs& info)
// {
//     EVA_ASSERT(info.scratchBuffer.deviceAddress() % impl().device.minAccelerationStructureScratchOffsetAlignment() == 0); // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03710
//     EVA_ASSERT(info.scratchBuffer.memoryProperties() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);           // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03707
//     EVA_ASSERT(info.scratchBuffer.usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);                       // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03674

//     // Either all geometries use the common AABB buffer, or each geometry has its own AABB buffer.
//     bool useCommonAabbBuffer = info.common.aabbInput.buffer;
//     EVA_ASSERT(
//         (useCommonAabbBuffer && !info.geometries[0].aabbInput.buffer) ||
//         (!useCommonAabbBuffer && info.geometries[0].aabbInput.buffer)
//     ); 

//     std::vector<VkAccelerationStructureGeometryKHR> geometries(info.geometries.size(), {
//         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
//         .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
//     });

//     std::vector<VkAccelerationStructureBuildRangeInfoKHR> rangeInfo(geometries.size(), {});
//     uint32_t accCount = 0;

//     for (size_t i = 0; i < geometries.size(); ++i) 
//     {
//         auto& srcGeometry = info.geometries[i];

//         geometries[i].flags = srcGeometry.flags;
//         rangeInfo[i].primitiveCount = srcGeometry.aabbCount;

//         auto& dst = geometries[i].geometry.aabbs;
//         auto& src = useCommonAabbBuffer ? info.common.aabbInput : srcGeometry.aabbInput;
//         EVA_ASSERT(src.buffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
//         EVA_ASSERT(src.buffer.deviceAddress() % 8 == 0);   // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03714
//         EVA_ASSERT(src.stride % 8 == 0);                   // VUID-VkAccelerationStructureGeometryAabbsDataKHR-stride-03545

//         dst.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
//         dst.data.deviceAddress = src.buffer.deviceAddress();
//         dst.stride = (VkDeviceSize) src.stride;
//         if (useCommonAabbBuffer)
//         {
//             rangeInfo[i].primitiveOffset = accCount * src.stride;
//             accCount += srcGeometry.aabbCount;
//         }
//     }

//     VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
//         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
//         .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
//         .flags = info.buildFlags,
//         .mode = info.srcAs ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
//         .srcAccelerationStructure = info.srcAs ? info.srcAs.impl().vkAccelStruct : VK_NULL_HANDLE,
//         .dstAccelerationStructure = info.dstAs.impl().vkAccelStruct,
//         .geometryCount = (uint32_t)geometries.size(),
//         .pGeometries = geometries.data(),
//         .scratchData = {.deviceAddress = info.scratchBuffer.deviceAddress()},
//     };

//     VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos[] = {rangeInfo.data()};
//     vkCmdBuildAccelerationStructuresKHR_(
//         impl().vkCmdBuffer,
//         1,
//         &buildInfo,
//         pRangeInfos
//     );

//     return *this;
// }
// CommandBuffer CommandBuffer::buildAccelerationStructures(const AsBuildInfoInstances& info) 
// {
//     VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
//         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
//         .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
//         .flags = info.buildFlags,
//         .mode = info.srcAs ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
//         .srcAccelerationStructure = info.srcAs ? info.srcAs.impl().vkAccelStruct : VK_NULL_HANDLE,
//         .dstAccelerationStructure = info.dstAs.impl().vkAccelStruct,
//         .scratchData = {.deviceAddress = info.scratchBuffer.deviceAddress()},
//     };

//     EVA_ASSERT(info.geometries.size() == 1); 
//     std::vector<VkAccelerationStructureGeometryKHR> geometries(1, {
//         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
//         .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
//     });

//     VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {
//         .primitiveCount = info.geometries[0].instanceCount
//     };

//     buildInfo.geometryCount = 1;
//     buildInfo.pGeometries = geometries.data();

//     auto& dst = geometries[0].geometry.instances;
//     auto& src = info.geometries[0].instanceInput;

//     EVA_ASSERT((uint32_t)src.buffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
//     EVA_ASSERT(src.buffer.deviceAddress() % 8 == 0);   // VUID
//     dst.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
//     dst.arrayOfPointers = VK_FALSE;
//     dst.data.deviceAddress = src.buffer.deviceAddress();
    
//     VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos[] = {&rangeInfo};
//     vkCmdBuildAccelerationStructuresKHR_(
//         impl().vkCmdBuffer,
//         1,
//         &buildInfo,
//         pRangeInfos
//     );

//     return *this;
// }

CommandBuffer CommandBuffer::buildAccelerationStructures(const AsBuildInfo& info) 
{
    return buildAccelerationStructures(std::vector<AsBuildInfo>{info});
}

// CommandBuffer CommandBuffer::buildAccelerationStructures(const std::vector<AsBuildInfo>& infos)
// {
//     std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(infos.size(), {
//         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
//     });

//     std::vector<std::vector<VkAccelerationStructureGeometryKHR>> geometriesArray(infos.size());
//     std::vector<std::vector<VkAccelerationStructureBuildRangeInfoKHR>> rangeInfosArray(infos.size());

//     for (size_t i = 0; i < infos.size(); ++i) 
//     {
//         std::visit([&](auto&& srcInfo) 
//         {
//             using T = std::decay_t<decltype(srcInfo)>;
//             auto& dstInfo = buildInfos[i];
//             auto& dstGeometries = geometriesArray[i];
//             auto& rangeInfos = rangeInfosArray[i];
//             size_t geometryCount = srcInfo.geometries.size();
//             dstGeometries.resize(geometryCount, {
//                 .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
//             });
//             rangeInfos.resize(geometryCount);

//             dstInfo.flags = srcInfo.buildFlags;
//             dstInfo.mode = srcInfo.srcAs ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
//             dstInfo.srcAccelerationStructure = srcInfo.srcAs ? srcInfo.srcAs.impl().vkAccelStruct : VK_NULL_HANDLE;
//             dstInfo.dstAccelerationStructure = srcInfo.dstAs.impl().vkAccelStruct;
//             dstInfo.geometryCount = (uint32_t) geometryCount;
//             dstInfo.pGeometries = dstGeometries.data();
//             dstInfo.scratchData = {.deviceAddress = srcInfo.scratchBuffer.deviceAddress()};
            
//             bool useCommonBuffer = false;
//             if constexpr (std::is_same_v<T, AsBuildInfoTriangles>) 
//             {
//                 dstInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
//                 useCommonBuffer = (bool) srcInfo.common.vertexInput.buffer;
//             }
//             else if constexpr (std::is_same_v<T, AsBuildInfoAabbs>) 
//             {
//                 dstInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
//                 useCommonBuffer = (bool) srcInfo.common.aabbInput.buffer;
//             }

//             uint32_t accVtxCount = 0;
//             uint32_t accIdxOffset = 0;
//             uint32_t accAabbCount = 0;
//             for (size_t j = 0; j < geometryCount; ++j) 
//             {
//                 auto& srcGeometry = srcInfo.geometries[j];
//                 auto& dstGeometry = dstGeometries[j];
//                 auto& rangeInfo = rangeInfos[j];
//                 dstGeometry.flags = srcGeometry.flags;
                
//                 if constexpr (std::is_same_v<T, AsBuildInfoTriangles>) 
//                 {
//                     dstGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
//                     rangeInfo.primitiveCount = srcGeometry.triangleCount;

//                     auto& dstTriangles = dstGeometry.geometry.triangles;
//                     auto& [vtxBuffer, vtxStride] = useCommonBuffer ? srcInfo.common.vertexInput : srcInfo.geometries[j].vertexInput;
//                     EVA_ASSERT(vtxBuffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
//                     EVA_ASSERT(vtxBuffer.deviceAddress() % 4 == 0);   // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03711
//                     EVA_ASSERT(vtxStride % 4 == 0);                   // VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexStride-03735
//                     dstTriangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
//                     dstTriangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;              // This format is the only one supported
//                     dstTriangles.vertexData.deviceAddress = vtxBuffer.deviceAddress();
//                     dstTriangles.vertexStride = (VkDeviceSize) vtxStride;
//                     dstTriangles.maxVertex = srcGeometry.vertexCount - 1;
//                     if (useCommonBuffer)
//                     {
//                         rangeInfo.firstVertex = accVtxCount;
//                         accVtxCount += srcGeometry.vertexCount;
//                     }

//                     /*
//                     ◦ If the geometry uses indices, primitiveCount × 3 indices are consumed from
//                     VkAccelerationStructureGeometryTrianglesDataKHR::indexData, starting at an offset of
//                     primitiveOffset. The value of firstVertex is added to the index values before fetching
//                     vertices.
//                     ◦ If the geometry does not use indices, primitiveCount × 3 vertices are consumed from
//                     VkAccelerationStructureGeometryTrianglesDataKHR::vertexData, starting at an offset of
//                     primitiveOffset + VkAccelerationStructureGeometryTrianglesDataKHR::vertexStride ×
//                     firstVertex.
//                     ◦ If VkAccelerationStructureGeometryTrianglesDataKHR::transformData is not NULL, a single
//                     VkTransformMatrixKHR structure is consumed from
//                     VkAccelerationStructureGeometryTrianglesDataKHR::transformData, at an offset of
//                     transformOffset. This matrix describes a transformation from the space in which the
//                     vertices for all triangles in this geometry are described to the space in which the
//                     acceleration structure is defined.
//                     */
//                     auto& [idxBuffer, idxStride] = useCommonBuffer ? srcInfo.common.indexInput : srcInfo.geometries[j].indexInput;
//                     if (idxBuffer)
//                     {
//                         EVA_ASSERT(idxBuffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
//                         EVA_ASSERT(idxStride == 2 || idxStride == 4);
//                         EVA_ASSERT((idxStride == 2 && idxBuffer.deviceAddress() % 2 == 0) ||
//                             (idxStride == 4 && idxBuffer.deviceAddress() % 4 == 0) );  // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03712
//                         dstTriangles.indexType = idxStride == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
//                         dstTriangles.indexData.deviceAddress = idxBuffer.deviceAddress();
//                         if (useCommonBuffer)
//                         {
//                             rangeInfo.primitiveOffset = accIdxOffset;
//                             accIdxOffset += srcGeometry.triangleCount * 3 * idxStride;
//                         }
//                     }
//                     else
//                     {
//                         dstTriangles.indexType = VK_INDEX_TYPE_NONE_KHR;             // VUID-VkAccelerationStructureGeometryTrianglesDataKHR-indexType-03798
//                         /*
//                         primitiveOffset is 0 because the role of the offset is included in firstVertex. 
//                         */
//                         EVA_ASSERT(srcGeometry.vertexCount == srcGeometry.triangleCount * 3); 
//                         rangeInfo.primitiveOffset = 0;
//                     }

//                     if (srcGeometry.transformBuffer)
//                     {
//                         EVA_ASSERT(srcGeometry.transformBuffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
//                         dstTriangles.transformData.deviceAddress = srcGeometry.transformBuffer.deviceAddress();
//                         rangeInfo.transformOffset = 0;
//                         EVA_ASSERT(dstTriangles.transformData.deviceAddress % 16 == 0); // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03810
//                         EVA_ASSERT(rangeInfo.transformOffset % 16 == 0);    // VUID-VkAccelerationStructureBuildRangeInfoKHR-transformOffset-03658
//                     }
//                 } 
//                 else if constexpr (std::is_same_v<T, AsBuildInfoAabbs>) 
//                 {
//                     dstGeometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
//                     rangeInfo.primitiveCount = srcGeometry.aabbCount;

//                     auto& dstAabbs = dstGeometry.geometry.aabbs;
//                     auto& srcAabbs = useCommonBuffer ? srcInfo.common.aabbInput : srcInfo.geometries[j].aabbInput;
//                     EVA_ASSERT(srcAabbs.buffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
//                     EVA_ASSERT(srcAabbs.buffer.deviceAddress() % 8 == 0);   // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03714
//                     EVA_ASSERT(srcAabbs.stride % 8 == 0);                   // VUID-VkAccelerationStructureGeometryAabbsDataKHR-stride-03545

//                     dstAabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
//                     dstAabbs.data.deviceAddress = srcAabbs.buffer.deviceAddress();
//                     dstAabbs.stride = (VkDeviceSize) srcAabbs.stride;
//                     if (useCommonBuffer)
//                     {
//                         rangeInfo.primitiveOffset = accAabbCount * srcAabbs.stride;
//                         accAabbCount += srcGeometry.aabbCount;
//                     }
//                 } 
//                 else EVA_ASSERT(false);
//             }

//         }, infos[i]);
//     }

//     std::vector<VkAccelerationStructureBuildRangeInfoKHR*> pRangeInfos(infos.size());
//     for (size_t i = 0; i < infos.size(); ++i)
//         pRangeInfos[i] = rangeInfosArray[i].data();

//     vkCmdBuildAccelerationStructuresKHR_(
//         impl().vkCmdBuffer,
//         (uint32_t)buildInfos.size(),
//         buildInfos.data(),
//         pRangeInfos.data()
//     );

//     return *this;
// }


CommandBuffer CommandBuffer::buildAccelerationStructures(const std::vector<AsBuildInfo>& infos)
{
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(infos.size(), {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    });

    std::vector<std::vector<VkAccelerationStructureGeometryKHR>> geometriesArray(infos.size());
    std::vector<std::vector<VkAccelerationStructureBuildRangeInfoKHR>> rangeInfosArray(infos.size());

    for (size_t i = 0; i < infos.size(); ++i) 
    {
        auto& srcInfo = infos[i];
        auto& dstInfo = buildInfos[i];
        auto& dstGeometries = geometriesArray[i];
        auto& rangeInfos = rangeInfosArray[i];

        size_t geometryCount = srcInfo.primitiveCounts.size();
        EVA_ASSERT(geometryCount > 0);
        EVA_ASSERT(geometryCount == 1 || srcInfo.geometryType != GEOMETRY_TYPE::INSTANCES);

        dstGeometries.resize(geometryCount, {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = (VkGeometryTypeKHR)(uint32_t)srcInfo.geometryType,
        });
        rangeInfos.resize(geometryCount);

        dstInfo.type = srcInfo.geometryType == GEOMETRY_TYPE::INSTANCES
            ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
            : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        dstInfo.flags = (VkBuildAccelerationStructureFlagsKHR)(uint32_t)srcInfo.buildFlags;
        dstInfo.mode = srcInfo.srcAs ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        dstInfo.srcAccelerationStructure = srcInfo.srcAs ? srcInfo.srcAs.impl().vkAccelStruct : VK_NULL_HANDLE;
        dstInfo.dstAccelerationStructure = srcInfo.dstAs.impl().vkAccelStruct;
        dstInfo.geometryCount = (uint32_t) geometryCount;
        dstInfo.pGeometries = dstGeometries.data();
        dstInfo.scratchData = {.deviceAddress = srcInfo.scratchBuffer.deviceAddress()};

        std::visit([&](auto&& inputs)
        {
            using T = std::decay_t<decltype(inputs)>;
            
            bool useCommonInput = false;
            if constexpr (std::is_same_v<T, AsBuildInfo::Triangles>) 
            {
                useCommonInput = (bool) inputs.vertexInput.buffer;
                EVA_ASSERT(inputs.eachGeometry.empty() || inputs.eachGeometry.size() == geometryCount);
            }
            else if constexpr (std::is_same_v<T, AsBuildInfo::Aabbs>) 
            {
                useCommonInput = (bool) inputs.aabbInput.buffer;
                EVA_ASSERT(inputs.eachGeometry.empty() || inputs.eachGeometry.size() == geometryCount);
            }

            uint32_t accVtxCount = 0;
            uint32_t accIdxOffset = 0;
            uint32_t accAabbCount = 0;
            for (size_t j = 0; j < geometryCount; ++j) 
            {
                // auto& srcGeometry = srcInfo.geometries[j];
                // dstGeometry.flags = srcGeometry.flags;
                
                auto& rangeInfo = rangeInfos[j];
                rangeInfo.primitiveCount = srcInfo.primitiveCounts[j];

                if constexpr (std::is_same_v<T, AsBuildInfo::Triangles>) 
                {
                    dstGeometries[j].flags = (VkGeometryFlagsKHR)(inputs.eachGeometry.empty() ? 0u : (uint32_t)inputs.eachGeometry[j].flags);

                    auto& dstTriangles = dstGeometries[j].geometry.triangles;
                    auto& [vtxBuffer, vtxStride] = useCommonInput ? inputs.vertexInput : inputs.eachGeometry[j].vertexInput;
                    EVA_ASSERT((uint32_t)vtxBuffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
                    EVA_ASSERT(vtxBuffer.deviceAddress() % 4 == 0);   // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03711
                    EVA_ASSERT(vtxStride % 4 == 0);                   // VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexStride-03735
                    
                    dstTriangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                    dstTriangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;              // This format is the only one supported
                    dstTriangles.vertexData.deviceAddress = vtxBuffer.deviceAddress();
                    dstTriangles.vertexStride = (VkDeviceSize) vtxStride;
                    dstTriangles.maxVertex = inputs.vertexCounts[j] - 1;
                    if (useCommonInput)
                    {
                        rangeInfo.firstVertex = accVtxCount;
                        accVtxCount += inputs.vertexCounts[j];
                    }

                    /*
                    ◦ If the geometry uses indices, primitiveCount × 3 indices are consumed from
                    VkAccelerationStructureGeometryTrianglesDataKHR::indexData, starting at an offset of
                    primitiveOffset. The value of firstVertex is added to the index values before fetching
                    vertices.
                    ◦ If the geometry does not use indices, primitiveCount × 3 vertices are consumed from
                    VkAccelerationStructureGeometryTrianglesDataKHR::vertexData, starting at an offset of
                    primitiveOffset + VkAccelerationStructureGeometryTrianglesDataKHR::vertexStride ×
                    firstVertex.
                    ◦ If VkAccelerationStructureGeometryTrianglesDataKHR::transformData is not NULL, a single
                    VkTransformMatrixKHR structure is consumed from
                    VkAccelerationStructureGeometryTrianglesDataKHR::transformData, at an offset of
                    transformOffset. This matrix describes a transformation from the space in which the
                    vertices for all triangles in this geometry are described to the space in which the
                    acceleration structure is defined.
                    */
                    auto& [idxBuffer, idxStride] = useCommonInput ? inputs.indexInput : inputs.eachGeometry[j].indexInput;
                    if (idxBuffer)
                    {
                        EVA_ASSERT((uint32_t)idxBuffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
                        EVA_ASSERT(idxStride == 2 || idxStride == 4);
                        EVA_ASSERT((idxStride == 2 && idxBuffer.deviceAddress() % 2 == 0) ||
                            (idxStride == 4 && idxBuffer.deviceAddress() % 4 == 0) );  // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03712
                        dstTriangles.indexType = idxStride == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
                        dstTriangles.indexData.deviceAddress = idxBuffer.deviceAddress();
                        if (useCommonInput)
                        {
                            rangeInfo.primitiveOffset = accIdxOffset;
                            accIdxOffset += srcInfo.primitiveCounts[j] * 3 * idxStride;
                        }
                    }
                    else
                    {
                        dstTriangles.indexType = VK_INDEX_TYPE_NONE_KHR;             // VUID-VkAccelerationStructureGeometryTrianglesDataKHR-indexType-03798
                        EVA_ASSERT(inputs.vertexCounts[j] == srcInfo.primitiveCounts[j] * 3); 
                        rangeInfo.primitiveOffset = 0;                              // primitiveOffset is 0 because the role of the offset is included in firstVertex.
                    }

                    if (!inputs.eachGeometry.empty() && inputs.eachGeometry[j].transformBuffer)
                    {
                        EVA_ASSERT((uint32_t)inputs.eachGeometry[j].transformBuffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
                        dstTriangles.transformData.deviceAddress = inputs.eachGeometry[j].transformBuffer.deviceAddress();
                        rangeInfo.transformOffset = 0;
                        EVA_ASSERT(dstTriangles.transformData.deviceAddress % 16 == 0); // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03810
                        EVA_ASSERT(rangeInfo.transformOffset % 16 == 0);    // VUID-VkAccelerationStructureBuildRangeInfoKHR-transformOffset-03658
                    }
                } 
                else if constexpr (std::is_same_v<T, AsBuildInfo::Aabbs>) 
                {
                    dstGeometries[j].flags = (VkGeometryFlagsKHR)(inputs.eachGeometry.empty() ? 0u : (uint32_t)inputs.eachGeometry[j].flags);

                    auto& dstAabbs = dstGeometries[j].geometry.aabbs;
                    auto& srcAabbs = useCommonInput ? inputs.aabbInput : inputs.eachGeometry[j].aabbInput;
                    EVA_ASSERT((uint32_t)srcAabbs.buffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
                    EVA_ASSERT(srcAabbs.buffer.deviceAddress() % 8 == 0);   // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03714
                    EVA_ASSERT(srcAabbs.stride % 8 == 0);                   // VUID-VkAccelerationStructureGeometryAabbsDataKHR-stride-03545
                    
                    dstAabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
                    dstAabbs.data.deviceAddress = srcAabbs.buffer.deviceAddress();
                    dstAabbs.stride = (VkDeviceSize) srcAabbs.stride;
                    if (useCommonInput)
                    {
                        rangeInfo.primitiveOffset = accAabbCount * srcAabbs.stride;
                        accAabbCount += srcInfo.primitiveCounts[j];
                    }
                } 
                else if constexpr (std::is_same_v<T, AsBuildInfo::Instances>) 
                {
                    auto& dstInstances = dstGeometries[j].geometry.instances;
                    auto& srcInstances = inputs.instanceInput;
                    EVA_ASSERT((uint32_t)srcInstances.buffer.usage() & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR); // VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673
                    EVA_ASSERT(srcInstances.buffer.deviceAddress() % 16 == 0);   // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03715
                    
                    dstInstances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
                    dstInstances.arrayOfPointers = VK_FALSE;
                    dstInstances.data.deviceAddress = srcInstances.buffer.deviceAddress();
                } 
                else EVA_ASSERT(false);
            }
            
        }, srcInfo.inputs);
    }

    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> pRangeInfos(infos.size());
    for (size_t i = 0; i < infos.size(); ++i)
        pRangeInfos[i] = rangeInfosArray[i].data();

    vkCmdBuildAccelerationStructuresKHR_(
        impl().vkCmdBuffer,
        (uint32_t)buildInfos.size(),
        buildInfos.data(),
        pRangeInfos.data()
    );

    return *this;
}
#endif // EVA_ENABLE_RAYTRACING



#define DESTROY_MACRO(class_name)  \
void class_name::destroy() { \
    if (!*this) return; \
    delete *ppImpl; \
    *ppImpl = nullptr; \
} \

DESTROY_MACRO(Device)
DESTROY_MACRO(Queue)
DESTROY_MACRO(CommandPool)
DESTROY_MACRO(CommandBuffer)
DESTROY_MACRO(Fence)
DESTROY_MACRO(Semaphore)
DESTROY_MACRO(ShaderModule)
DESTROY_MACRO(ComputePipeline)
DESTROY_MACRO(Buffer)
DESTROY_MACRO(Image)
DESTROY_MACRO(Sampler)
DESTROY_MACRO(DescriptorSetLayout)
DESTROY_MACRO(PipelineLayout)
DESTROY_MACRO(DescriptorPool)
DESTROY_MACRO(DescriptorSet)
DESTROY_MACRO(QueryPool)
#ifdef EVA_ENABLE_WINDOW
    DESTROY_MACRO(Window)
#endif
#ifdef EVA_ENABLE_RAYTRACING
    DESTROY_MACRO(RaytracingPipeline)
    DESTROY_MACRO(AccelerationStructure)
#endif
