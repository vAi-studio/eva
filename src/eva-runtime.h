#ifndef EVA_RUNTIME_H
#define EVA_RUNTIME_H

#include <vector>
#include <map>
#include <string>
#include <variant>
#include <optional>
#include <memory>
#include <tuple>
#include <utility>
#include <cstring>
#include <functional>

#define VULKAN_VERSION_1_3  // TODO: whether to use this or not depends on the system

#include "eva-error.h"
#include "eva-enums.h"


#define EVA_ATTACHMENT_UNUSED              (~0U)
#define EVA_FALSE                          0U
#define EVA_LOD_CLAMP_NONE                 1000.0F
#define EVA_QUEUE_FAMILY_IGNORED           (~0U)
#define EVA_REMAINING_ARRAY_LAYERS         (~0U)
#define EVA_REMAINING_MIP_LEVELS           (~0U)
#define EVA_SUBPASS_EXTERNAL               (~0U)
#define EVA_TRUE                           1U
#define EVA_WHOLE_SIZE                     (~0ULL)
#define EVA_MAX_MEMORY_TYPES               32U
#define EVA_MAX_PHYSICAL_DEVICE_NAME_SIZE  256U
#define EVA_UUID_SIZE                      16U
#define EVA_MAX_EXTENSION_NAME_SIZE        256U
#define EVA_MAX_DESCRIPTION_SIZE           256U
#define EVA_MAX_MEMORY_HEAPS               16U

namespace eva {

typedef uint32_t Bool32;
typedef uint64_t DeviceAddress;
typedef uint64_t DeviceSize;
typedef uint32_t Flags;
typedef uint32_t SampleMask;


class Runtime;
class Device;
class Queue;
class CommandPool;
class CommandBuffer;
class Fence;
class Semaphore;

class ShaderModule;
class ComputePipeline;
class GraphicsPipeline;

class Buffer;
class Image;
class ImageView;
class Sampler;

class DescriptorSetLayout;
class PipelineLayout;
class DescriptorPool;
class DescriptorSet;

class Window;
class RaytracingPipeline;
class AccelerationStructure;


#define VULKAN_FRIENDS \
    friend class Runtime; \
    friend class Device; \
    friend class Queue; \
    friend class CommandPool; \
    friend class CommandBuffer; \
    friend class Fence; \
    friend class Semaphore; \
    friend class ShaderModule; \
    friend class ComputePipeline; \
    friend class GraphicsPipeline; \
    friend class RaytracingPipeline; \
    friend class Buffer; \
    friend class Image; \
    friend class ImageView; \
    friend class DescriptorSetLayout; \
    friend class PipelineLayout; \
    friend class DescriptorPool; \
    friend class DescriptorSet; \
    friend class Window; \
    friend class AccelerationStructure; \
    friend class Submitting;



#define VULKAN_CLASS_COMMON \
    VULKAN_FRIENDS \
    struct Impl; Impl* pImpl; \
    Impl& impl() { return *pImpl; } \
    const Impl& impl() const { return *pImpl; } \


#define VULKAN_CLASS_COMMON2(class_name) \
    VULKAN_FRIENDS \
    struct Impl; \
    Impl** ppImpl; \
public: \
    class_name(class_name::Impl** ppImpl=nullptr) : ppImpl(ppImpl) {} \
    operator bool() const { return ppImpl && *ppImpl; } \
    bool operator==(const class_name&) const = default; \
    void destroy(); \
private: \
    Impl& impl() { return **ppImpl; } \
    const Impl& impl() const { return **ppImpl; } \



struct ShaderModuleCreateInfo;
struct ComputePipelineCreateInfo;
struct RaytracingPipelineCreateInfo;
struct BufferCreateInfo;
struct ImageCreateInfo;
struct ImageViewDesc;
struct SamplerCreateInfo;
struct DescriptorPoolCreateInfo;
struct BufferRange;
struct SemaphoreStage;
struct QueueSelector;
struct BindingInfo;
struct DescriptorSetLayoutDesc;
struct PipelineLayoutDesc;

struct MemoryBarrier;
struct BufferMemoryBarrier;
struct ImageMemoryBarrier;

struct CopyRegion;
struct WindowCreateInfo;



using BarrierInfo = std::variant<MemoryBarrier, BufferMemoryBarrier, ImageMemoryBarrier>;
using Pipeline = std::variant<ComputePipeline, GraphicsPipeline, RaytracingPipeline>;
using Resource = std::variant<Buffer, Image>;
using SubmissionBatchInfo = std::tuple<
    std::vector<SemaphoreStage>, 
    std::vector<CommandBuffer>, 
    std::vector<SemaphoreStage>
>; 


struct BufferDescriptor;
struct ImageDescriptor;
using Descriptor = std::variant<BufferDescriptor, ImageDescriptor, AccelerationStructure>;





/*
* From [Table 69. Required Limits] in the spec:
*/
namespace portable {
    constexpr uint32_t minMemoryMapAlignment = 64;      // min, vkMapMemory() minimum alignment
#ifdef EVA_ENABLE_RAYTRACING
    constexpr uint32_t shaderGroupHandleSize = 32;      // exact, Size of a shader group handle
    constexpr uint32_t shaderGroupBaseAlignment = 64;   // max, Alignment for SBT base addresses
    constexpr uint32_t shaderGroupHandleAlignment = 32; // max, Alignment for SBT record addresses
    constexpr uint32_t maxShaderGroupStride = 4096;     // min, Maximum SBT record size
#endif
}

#ifdef EVA_ENABLE_RAYTRACING
    struct AsBuildSizesInfo;
    struct AsCreateInfo;
    struct AsBuildInfo;
    struct ShaderBindingTable;
    struct ShaderGroupHandle;
#endif



struct DeviceSettings {
    bool enableGraphicsQueues;
    bool enableComputeQueues;
    bool enableTransferQueues;
#ifdef EVA_ENABLE_WINDOW
    bool enableWindow;
#endif
#ifdef EVA_ENABLE_RAYTRACING
    bool enableRaytracing;
#endif
    // bool operator==(const DeviceSettings&) const = default;
    bool operator<=(const DeviceSettings& other) const {
        return (!enableGraphicsQueues || other.enableGraphicsQueues) &&
               (!enableComputeQueues  || other.enableComputeQueues)  &&
               (!enableTransferQueues || other.enableTransferQueues)
#ifdef EVA_ENABLE_WINDOW
               && (!enableWindow      || other.enableWindow)
#endif
#ifdef EVA_ENABLE_RAYTRACING
               && (!enableRaytracing  || other.enableRaytracing)
#endif
               ;
    }
};


enum QueueType {
    queue_graphics, 
    queue_compute, 
    queue_transfer, 
    queue_max,
};


enum class OwnershipTransferOpType {
    none,
    release,
    acquire,
};


class Runtime {
    VULKAN_CLASS_COMMON
    ~Runtime();
    Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Device createDevice(const DeviceSettings& settings);
    
public:
    static Runtime& get();    // singleton pattern
    uint32_t deviceCount() const;
    Device device(int gpuIndex=-1); 
    Device device(DeviceSettings settings);

#ifdef EVA_ENABLE_WINDOW
    Window createWindow(WindowCreateInfo info);
    void destroyWindow(Window window);
#endif
};


class Device {
    VULKAN_CLASS_COMMON2(Device)

public:
    void reportGPUQueueFamilies() const;
    void reportAssignedQueues() const;

    uint32_t queueCount(QueueType type) const;
    bool supportPresent(QueueType type) const;
    Queue queue(QueueType type, uint32_t index=0) const;
    QueueSelector queue(uint32_t index=0) const;

    CommandPool createCommandPool(QueueType type, COMMAND_POOL_CREATE flags=COMMAND_POOL_CREATE::NONE);
    CommandPool setDefalutCommandPool(QueueType type, CommandPool cmdPool);
    CommandBuffer newCommandBuffer(QueueType type, COMMAND_POOL_CREATE poolFlags=COMMAND_POOL_CREATE::NONE);
    std::vector<CommandBuffer> newCommandBuffers(uint32_t count, QueueType type, COMMAND_POOL_CREATE poolFlags=COMMAND_POOL_CREATE::NONE);
    
    Fence createFence(bool signaled=false);
    Result waitFences(std::vector<Fence> fences, bool waitAll, uint64_t timeout=uint64_t(-1));
    void resetFences(std::vector<Fence> fences);
    Semaphore createSemaphore();
    template <int N> auto createSemaphores()
    {
        return [this]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(((void)I, createSemaphore())...);
        }(std::make_index_sequence<N>{});
    }

    ShaderModule createShaderModule(const ShaderModuleCreateInfo& info);
    ComputePipeline createComputePipeline(const ComputePipelineCreateInfo& info);

    Buffer createBuffer(const BufferCreateInfo& info) ;
    Image createImage(const ImageCreateInfo& info);
    Sampler createSampler(const SamplerCreateInfo& info);
    DescriptorSetLayout createDescriptorSetLayout(DescriptorSetLayoutDesc desc); // call-by-value is ok because at least one copy is necessary for lvalue
    PipelineLayout createPipelineLayout(PipelineLayoutDesc desc);
    DescriptorPool createDescriptorPool(const DescriptorPoolCreateInfo& info);

#ifdef EVA_ENABLE_RAYTRACING
    RaytracingPipeline createRaytracingPipeline(const RaytracingPipelineCreateInfo& info);

    uint32_t shaderGroupHandleSize() const;
    uint32_t shaderGroupHandleAlignment() const;
    uint32_t shaderGroupBaseAlignment() const;
    uint32_t asBufferOffsetAlignment() const;
    uint32_t minAccelerationStructureScratchOffsetAlignment() const;
    AsBuildSizesInfo getBuildSizesInfo(const AsBuildInfo& info) const;
    AccelerationStructure createAccelerationStructure(const AsCreateInfo& info) ;
#endif

};


class Queue {
    VULKAN_CLASS_COMMON2(Queue)
    QueueType _type = queue_max;
public:

    QueueType type() const;

    uint32_t queueFamilyIndex() const;

    uint32_t index() const;

    float priority() const;

    Queue submit(
        CommandBuffer cmdBuffer
    );

    Queue submit(
        std::vector<CommandBuffer> cmdBuffers
    );

    Queue submit(
        std::vector<SubmissionBatchInfo>&& batches
    );

    Queue submit(
        std::vector<SubmissionBatchInfo>&& batches,
        std::optional<Fence> fence
    );

    Queue waitIdle();
};


class CommandPool {
    VULKAN_CLASS_COMMON2(CommandPool)
public:

    QueueType type() const;

    std::vector<CommandBuffer> newCommandBuffers(
        uint32_t count
    );

    CommandBuffer newCommandBuffer();
};


class CommandBuffer {
    VULKAN_CLASS_COMMON2(CommandBuffer)
public:

    QueueType type() const;

    uint32_t queueFamilyIndex() const;

    CommandBuffer submit(uint32_t index=0) const;

    Queue lastSubmittedQueue() const;
    
    void wait() const {
        lastSubmittedQueue().waitIdle();
    }

    CommandBuffer reset(bool keepCapacity=false);

    CommandBuffer begin(
        COMMAND_BUFFER_USAGE flags=COMMAND_BUFFER_USAGE::NONE
    );

    CommandBuffer end();

    CommandBuffer bindPipeline(
        Pipeline pipeline
    );

    CommandBuffer bindDescSets(
        PipelineLayout layout, 
        PIPELINE_BIND_POINT bindPoint,
        std::vector<DescriptorSet> descSets, 
        uint32_t firstSet=0
    );

    CommandBuffer bindDescSets(
        std::vector<DescriptorSet> descSets, 
        uint32_t firstSet=0
    );

    CommandBuffer setPushConstants(
        PipelineLayout layout, 
        SHADER_STAGE stageFlags, 
        uint32_t offset, 
        uint32_t size,
        const void* values
    );

    CommandBuffer setPushConstants(
        uint32_t offset, 
        uint32_t size, 
        const void* data
    );

    CommandBuffer barrier(
        std::vector<BarrierInfo> barrierInfos
    );

    CommandBuffer barrier(
        BarrierInfo barrierInfo
    );

	CommandBuffer copyBuffer(
        Buffer src, 
        Buffer dst, 
        uint64_t srcOffset = 0, 
        uint64_t dstOffset = 0, 
        uint64_t size = EVA_WHOLE_SIZE
    );

    CommandBuffer copyBuffer(
        BufferRange src,
        BufferRange dst 
    );

    CommandBuffer copyImage(
        Image src,
        Image dst,
        std::vector<CopyRegion> regions = {}
    );

    CommandBuffer copyBufferToImage(
        BufferRange src, 
        Image dst, 
        std::vector<CopyRegion> regions = {}
    );
    
    CommandBuffer copyImageToBuffer(
        Image src, 
        BufferRange dst, 
        std::vector<CopyRegion> regions = {}
    );

    CommandBuffer dispatch(
        uint32_t groupCountX, 
        uint32_t groupCountY=1, 
        uint32_t groupCountZ=1
    );

    CommandBuffer dispatch2(
        uint32_t numThreadsInX,
        uint32_t numThreadsInY=1,
        uint32_t numThreadsInZ=1
    );

#ifdef EVA_ENABLE_RAYTRACING
    CommandBuffer traceRays(
        ShaderBindingTable hitGroupSbt,
        uint32_t width,
        uint32_t height = 1,
        uint32_t depth = 1
    );

    CommandBuffer traceRays(
        uint32_t width,
        uint32_t height = 1,
        uint32_t depth = 1
    );

    CommandBuffer buildAccelerationStructures(
        const AsBuildInfo& info
    );

    CommandBuffer buildAccelerationStructures(
        const std::vector<AsBuildInfo>& infos
    );
#endif
};


class Fence {
    VULKAN_CLASS_COMMON2(Fence)
public:

    Result wait(bool autoReset = false, uint64_t timeout=uint64_t(-1)) const;
    void reset() const;
    bool isSignaled() const;
};


class Semaphore {
    VULKAN_CLASS_COMMON2(Semaphore)
public:

    SemaphoreStage operator()(PIPELINE_STAGE stage) const;

};


class ShaderModule {
    VULKAN_CLASS_COMMON2(ShaderModule)
public:

    bool hasReflect() const;
    void discardReflect() ;
    PipelineLayoutDesc extractPipelineLayoutDesc() const;

    operator uint64_t() const;
};


class GraphicsPipeline {};


class ComputePipeline {
    VULKAN_CLASS_COMMON2(ComputePipeline)
public:

    PipelineLayout layout() const;
    DescriptorSetLayout descSetLayout(uint32_t setId=0) const;
};





class Buffer {
    VULKAN_CLASS_COMMON2(Buffer)
public:
    
    uint8_t* map(
        uint64_t offset=0, 
        uint64_t size=EVA_WHOLE_SIZE
    );
 
    void flush(
        uint64_t offset=0, 
        uint64_t size=EVA_WHOLE_SIZE
    ) const;

    void invalidate(
        uint64_t offset=0, 
        uint64_t size=EVA_WHOLE_SIZE
    ) const;
    
    void unmap();
    uint64_t size() const;
    BUFFER_USAGE usage() const;
    MEMORY_PROPERTY memoryProperties() const;

    DeviceAddress deviceAddress() const;

    BufferRange operator()(
        uint64_t offset=0, 
        uint64_t size=EVA_WHOLE_SIZE
    );

};


class Image {
    VULKAN_CLASS_COMMON2(Image)
public:

    ImageView view() const;
    ImageView view(ImageViewDesc&& desc) const;

    // operator ImageMemoryBarrier() const;
    // ImageMemoryBarrier operator/(IMAGE_LAYOUT newLayout) const;
    ImageMemoryBarrier operator()(IMAGE_LAYOUT oldLayout, IMAGE_LAYOUT newLayout) const;
};


class ImageView {
    VULKAN_CLASS_COMMON2(ImageView)
public:
};


class Sampler{
    VULKAN_CLASS_COMMON2(Sampler)
public:
};


class DescriptorSetLayout {
    VULKAN_CLASS_COMMON2(DescriptorSetLayout)
public:

    // const VkDescriptorSetLayoutBinding& bindingInfo(
    //     uint32_t bindingId, 
    //     bool exact=true
    // ) const;
        
    // const std::map<uint32_t, VkDescriptorSetLayoutBinding>& bindingInfos() const;
};


class PipelineLayout {
    VULKAN_CLASS_COMMON2(PipelineLayout)
public:

    DescriptorSetLayout descSetLayout(uint32_t setId) const;
};


class DescriptorPool {
    VULKAN_CLASS_COMMON2(DescriptorPool)
public:

    std::vector<DescriptorSet> operator()(
        std::vector<DescriptorSetLayout> layouts
    );

    DescriptorSet operator()(DescriptorSetLayout layout);

    std::vector<DescriptorSet> operator()(
        DescriptorSetLayout layout,
        uint32_t count
    );

    template<typename... Layouts>
    auto operator()(Layouts... layouts) requires (std::is_same_v<Layouts, DescriptorSetLayout> && ...)
    {
        auto sets = (*this)(std::vector<DescriptorSetLayout>{ layouts... });

        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(sets[I]...);
        }(std::index_sequence_for<Layouts...>{});
    }
};


class DescriptorSet {
    VULKAN_CLASS_COMMON2(DescriptorSet)
public:

    DescriptorSet write(
        std::vector<Descriptor> descriptors, 
        uint32_t startBindingId=0, 
        uint32_t startArrayOffset=0
    );

    DescriptorSet operator=(
        std::vector<DescriptorSet>&& data
    );
};

inline DescriptorSet DescriptorPool::operator()(DescriptorSetLayout layout) 
{
    return (*this)(std::vector<DescriptorSetLayout>{layout})[0];
}

inline std::vector<DescriptorSet> DescriptorPool::operator()(DescriptorSetLayout layout, uint32_t count) 
{
    return (*this)(std::vector<DescriptorSetLayout>(count, layout));
}





///////////////////////////////////////////////////////////////////////////////
struct BindingInfo {
    uint32_t binding;
    DESCRIPTOR_TYPE descriptorType;
    uint32_t descriptorCount;
    SHADER_STAGE stageFlags;

    BindingInfo& operator|=(BindingInfo&& other) {
        if (binding == other.binding 
            && descriptorType == other.descriptorType 
            && descriptorCount == other.descriptorCount)
            stageFlags |= other.stageFlags;
        else
            throw;
        return *this;
    }
};


struct DescriptorSetLayoutDesc {
    std::map<uint32_t, BindingInfo> bindings;

    DescriptorSetLayoutDesc() = default;

    DescriptorSetLayoutDesc(const DescriptorSetLayoutDesc&) = default;
    
    DescriptorSetLayoutDesc(DescriptorSetLayoutDesc&& other) = default;

    DescriptorSetLayoutDesc& operator|=(DescriptorSetLayoutDesc&& other) 
    {
        for (auto& [rId, rBinding] : other.bindings) 
        {
            auto it = bindings.find(rId);
            if (it != bindings.end()) 
                it->second |= std::move(rBinding);
            else 
                bindings.emplace_hint(it, rId, std::move(rBinding));

        }
        return *this;
    }
};


struct PushConstantRange {
    SHADER_STAGE stageFlags;
    uint32_t offset;
    uint32_t size;

    PushConstantRange() : offset(0), size(0), stageFlags(SHADER_STAGE::NONE) {};

    PushConstantRange(uint32_t offset, uint32_t size, SHADER_STAGE stageFlags=SHADER_STAGE::NONE)
    : offset(offset), size(size), stageFlags(stageFlags) {}

    PushConstantRange(uint32_t size, SHADER_STAGE stageFlags=SHADER_STAGE::NONE)
    : offset(0), size(size), stageFlags(stageFlags) {}

    // PushConstantRange&& operator|=(SHADER_STAGE stageFlags) &&
    // {
    //     this->stageFlags |= stageFlags;
    //     return std::move(*this);
    // }

    PushConstantRange& operator|=(PushConstantRange&& other)
    {
        offset = std::min(offset, other.offset);
        size = std::max(offset + size, other.offset + other.size) - offset;
        stageFlags |= other.stageFlags;
        return *this;
    }
};




struct PipelineLayoutDesc {
    std::map<uint32_t, DescriptorSetLayoutDesc> setLayouts;
    /*
    * [VUID-VkPipelineLayoutCreateInfo-pPushConstantRanges-00292]
    * : Any two elements of pPushConstantRanges must not include the same stage in stageFlags.
    * 
    * std::vector<PushConstantRange> pushConstants;  
    *  - this cannot guarantee the above rule
    * 
    * std::map<SHADER_STAGE, PushConstantRange> pushConstants;
    *  - Each SHADER_STAGE key must contain only one bit, which is difficult to use.
    * 
    * PushConstantRange pushConstant;
    *  - It contains one push constant range across all shader stages in the pipeline.
    *  - It may be faster than the stage-granular version when calling vkCmdPushConstants.
    */
    std::unique_ptr<PushConstantRange> pushConstant; // only one push constant range is allowed

    PipelineLayoutDesc() = default;

    PipelineLayoutDesc(const PipelineLayoutDesc& other) 
    : setLayouts(other.setLayouts)
    , pushConstant(other.pushConstant ? std::make_unique<PushConstantRange>(*other.pushConstant) : nullptr)
    {}
    
    PipelineLayoutDesc(PipelineLayoutDesc&& other) = default;

    PipelineLayoutDesc& operator|=(PipelineLayoutDesc&& other)
    {
        for (auto& [rId, rSetLayout] : other.setLayouts) 
        {
            auto it = setLayouts.find(rId);
            if (it != setLayouts.end()) 
                it->second |= std::move(rSetLayout);
            else 
                setLayouts.emplace_hint(it, rId, std::move(rSetLayout));
        }

        if (other.pushConstant) 
        {
            if (pushConstant)
                *pushConstant |= std::move(*other.pushConstant);
            else
                pushConstant = std::move(other.pushConstant);
        }

        return *this;
    }

};


struct BufferDescriptor {
    Buffer buffer;
    uint64_t offset;
    uint64_t size;

    BufferDescriptor(Buffer buffer) 
    : buffer(buffer), offset(0), size(EVA_WHOLE_SIZE) {}

    BufferDescriptor(BufferRange range);
};


struct ImageDescriptor {
    std::optional<ImageView> imageView;
    std::optional<Sampler> sampler; 
    IMAGE_LAYOUT imageLayout;

    ImageDescriptor(Image image)
    : imageView(image.view())
    , imageLayout(IMAGE_LAYOUT::MAX_ENUM) {}

    ImageDescriptor(ImageView imageView)
    : imageView(imageView)
    , imageLayout(IMAGE_LAYOUT::MAX_ENUM) {}
};


inline ImageDescriptor&& operator/(ImageDescriptor&& image, Sampler sampler) 
{
    // EVA_ASSERT(!image.sampler);
    image.sampler = sampler;
    return std::move(image);
}

inline ImageDescriptor&& operator/(ImageDescriptor&& image, IMAGE_LAYOUT layout) 
{
    // EVA_ASSERT(image.imageLayout == IMAGE_LAYOUT::MAX_ENUM);
    image.imageLayout = layout;
    return std::move(image);
}


inline std::vector<DescriptorSet> operator,(DescriptorSet lhs, DescriptorSet rhs)
{
    return {lhs, rhs};
}



struct CopyRegion {
    uint64_t bufferOffset=0;
    uint32_t bufferRowLength=0;
    uint32_t bufferImageHeight=0;
    uint32_t offsetX=0;
    uint32_t offsetY=0;
    uint32_t offsetZ=0;
    uint32_t baseLayer=0;
    uint32_t width=0; 
    uint32_t height=0; 
    uint32_t depth=0;
    uint32_t layerCount=0;
};


struct SpvBlob {
    std::shared_ptr<uint32_t[]> data;
    size_t sizeInBytes; // in bytes

    static SpvBlob readFrom(const char* filepath);
};


struct ShaderModuleCreateInfo {
    SHADER_STAGE stage;
    const SpvBlob& spv;
    bool withSpirvReflect = true;
};

using ShaderInput = std::variant<SpvBlob, ShaderModule>;


template <uint32_t ID, typename T>
struct ConstantID {
    T value;
    ConstantID(T v) : value(v) {}
};

// template <uint32_t ID>
// struct ConstantID<ID, bool> {
//     uint32_t value;
//     ConstantID(bool v) : value(v ? 1u : 0u) {}
// };


template<int ID, class T>
inline auto constant_id(T v)
{
    return ConstantID<ID, T>{v};
}

/*
* VUID-VkSpecializationMapEntry-constantID-00776: 
* If the specialization constant is of type boolean, size must be the byte size of VkBool32.
* And in Vulkan, VkBool32 is defined as uint32_t.
*/
template<int ID>
inline auto constant_id(bool v)
{
    return ConstantID<ID, uint32_t>{ v ? 1u : 0u };
}


struct SpecializationMapEntry {
    uint32_t    constantID;
    uint32_t    offset;
    size_t      size;
};


struct SpecializationInfo {
    uint32_t                        mapEntryCount;
    const SpecializationMapEntry*   pMapEntries;
    size_t                          dataSize;
    const void*                     pData;
};


class SpecializationConstant {
    std::map<uint32_t, std::vector<uint8_t>> orderedConstants; // key: constantID, value: bytes of the constant value

    mutable std::optional<std::vector<SpecializationMapEntry>> cachedMapEntries;
    mutable std::optional<std::vector<uint8_t>> cachedData;
    mutable std::optional<SpecializationInfo> cachedSpecInfo;
    
    void buildCache() const;
public:

    template<uint32_t ID, typename T>
    void addConstant(ConstantID<ID, T> constant) 
    {
        auto it = orderedConstants.find(ID);
        if (it != orderedConstants.end()) throw;

        std::vector<uint8_t> newConstant(sizeof(constant.value));
        std::memcpy(newConstant.data(), &constant.value, sizeof(constant.value));
        // *((T*) newConstant.data()) = constant.value; 
        orderedConstants.emplace_hint(it, ID, std::move(newConstant));

        cachedMapEntries.reset();
        cachedData.reset();
        cachedSpecInfo.reset();
    }

    template<typename... ConstantIDs>
    SpecializationConstant(ConstantIDs... constantIDs) 
    {
        (addConstant(constantIDs), ...);
    }
    
    SpecializationConstant() = default;

    SpecializationConstant(const SpecializationConstant& other) 
    : orderedConstants(other.orderedConstants) {}

    SpecializationConstant(SpecializationConstant&& other) 
    : orderedConstants(std::move(other.orderedConstants)) {}
    
    bool empty() const { return orderedConstants.empty(); }

    const SpecializationInfo* getInfo() const
    {
        if (empty())  return nullptr;
        buildCache();
        return &cachedSpecInfo.value();
    }

    bool operator==(const SpecializationConstant& other) const noexcept 
    {
        return orderedConstants == other.orderedConstants;
    }

    uint64_t hash() const noexcept;
};


struct ShaderStage {
    std::optional<ShaderInput> shader;
    SpecializationConstant specialization;

    ShaderStage() : shader(std::nullopt), specialization() {}
    ShaderStage(ShaderInput shader, SpecializationConstant&& spec ={})
    : shader(shader), specialization(std::move(spec)) {}

    template<typename ShaderType>
    ShaderStage(ShaderType shader, SpecializationConstant&& spec ={})
    : shader(shader), specialization(std::move(spec)) {}

    template<int ID, typename T>
    ShaderStage operator+(ConstantID<ID, T> constant) && 
    {
        specialization.addConstant(constant);
        return std::move(*this);
    }

    bool operator==(const ShaderStage& other) const noexcept;
};


template<int ID, typename T>
inline ShaderStage operator+(ShaderInput shader, ConstantID<ID, T> constant) 
{
    return ShaderStage(shader) + constant;
}


struct ComputePipelineCreateInfo {
    ShaderStage csStage;
    std::optional<PipelineLayout> layout;
    bool autoLayoutAllowAllStages = false;
};


struct BufferCreateInfo {
    uint64_t size;
    BUFFER_USAGE usage;
    MEMORY_PROPERTY reqMemProps;
};


struct ImageCreateInfo {
    IMAGE_CREATE flags = IMAGE_CREATE::NONE;
    FORMAT format;
    struct Extent {
        uint32_t width;
        uint32_t height = 1;
        uint32_t depth = 1;
    } extent;
    uint32_t arrayLayers = 1;
    IMAGE_USAGE usage;
    bool preInitialized = false; // if true, initialLayout is VK_IMAGE_LAYOUT_PREINITIALIZED, else VK_IMAGE_LAYOUT_UNDEFINED
    MEMORY_PROPERTY reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL;
};


struct ComponentMapping {
    COMPONENT_SWIZZLE r = COMPONENT_SWIZZLE::IDENTITY;
    COMPONENT_SWIZZLE g = COMPONENT_SWIZZLE::IDENTITY;
    COMPONENT_SWIZZLE b = COMPONENT_SWIZZLE::IDENTITY;
    COMPONENT_SWIZZLE a = COMPONENT_SWIZZLE::IDENTITY;
};


struct ImageViewDesc {
    IMAGE_VIEW_TYPE viewType = IMAGE_VIEW_TYPE::MAX_ENUM;
    FORMAT format = FORMAT::MAX_ENUM;
    ComponentMapping components = {};
    // VkImageSubresourceRange subresourceRange;

    bool operator==(const ImageViewDesc& other) const {
        return viewType == other.viewType &&
               format   == other.format   &&
               components.r == other.components.r &&
               components.g == other.components.g &&
               components.b == other.components.b &&
               components.a == other.components.a;
    }
};


struct SamplerCreateInfo {
    FILTER magFilter = FILTER::LINEAR;
    FILTER minFilter = FILTER::LINEAR;
    SAMPLER_MIPMAP_MODE mipmapMode = SAMPLER_MIPMAP_MODE::LINEAR;
    SAMPLER_ADDRESS_MODE addressModeU = SAMPLER_ADDRESS_MODE::REPEAT;
    SAMPLER_ADDRESS_MODE addressModeV = SAMPLER_ADDRESS_MODE::REPEAT;
    SAMPLER_ADDRESS_MODE addressModeW = SAMPLER_ADDRESS_MODE::REPEAT;
    float mipLodBias = 0.0f;
    Bool32 anisotropyEnable = EVA_FALSE;
    float maxAnisotropy = 1.0f;
    Bool32 compareEnable = EVA_FALSE;
    COMPARE_OP compareOp = COMPARE_OP::ALWAYS;
    float minLod = 0.0f;
    float maxLod = EVA_LOD_CLAMP_NONE;
    BORDER_COLOR borderColor = BORDER_COLOR::INT_OPAQUE_BLACK;
    Bool32 unnormalizedCoordinates = EVA_FALSE;
};


inline ImageView Image::view() const
{
    return view(ImageViewDesc{});
}


struct DescriptorPoolSize {
    DESCRIPTOR_TYPE type;
    uint32_t descriptorCount;
};


constexpr inline DescriptorPoolSize operator<=(
    DESCRIPTOR_TYPE type, int count)
{
    return {
        .type = type,
        .descriptorCount = (uint32_t)count,
    };
}


struct DescriptorPoolCreateInfo {
    std::vector<DescriptorPoolSize> maxTypes;
    uint32_t maxSets;
};



struct QueueSelector {
    const Device device;
    const uint32_t index;

    QueueSelector(Device device, uint32_t index) 
    : device(device), index(index) {} 

    Queue operator()(CommandBuffer cmdBuffer) const
    {
        return device.queue(cmdBuffer.type(), index);
    }

    Queue submit(CommandBuffer cmdBuffer) const
    {
        return (*this)(cmdBuffer).submit(cmdBuffer);
    }

    Queue submit(std::vector<CommandBuffer> cmdBuffers) const
    {
        return (*this)(cmdBuffers[0]).submit(std::move(cmdBuffers));
    }

    Queue submit(std::vector<SubmissionBatchInfo>&& batches, std::optional<Fence> fence = std::nullopt) const
    {
        return (*this)(std::get<1>(batches[0])[0]).submit(std::move(batches), fence);
    }
};




/*
버퍼 range class가 꼭 필요한가?
버퍼 range 필요 시점:
- vkMapMemory
- vkFlushMappedMemoryRanges, vkInvalidateMappedMemoryRanges
- vkCmdCopyBuffer, vkCmdUpdateBuffer, vkCmdFillBuffer 
- VkDescriptorBufferInfo (vkUpdateDescriptorSets의 인자)
- VkBufferMemoryBarrier 
- VkBufferViewCreateInfo 
*/
struct BufferRange {
    Buffer buffer;
    const uint64_t offset;
    const uint64_t size;

    BufferRange() 
    : buffer({})
    , offset(0)
    , size(0) {};
    
    BufferRange(Buffer buffer, 
        uint64_t offset=0, 
        uint64_t size=EVA_WHOLE_SIZE)
    : buffer(buffer)
    , offset(offset)
    , size(size==EVA_WHOLE_SIZE ? buffer.size() - offset : size) {}

    BufferRange(const BufferRange&) = default;
    BufferRange(BufferRange&& other) = default;
    
    BufferRange& operator=(const BufferRange& other)
    {
        new (this) BufferRange(other);
        return *this;
    }

    BufferRange& operator=(BufferRange&& other)
    {
        new (this) BufferRange(std::move(other));
        return *this;
    }

    operator bool() const
    {
        return size != 0;
    }

    void flush() const
    {
        buffer.flush(offset, size);
    }

    void invalidate() const
    {
        buffer.invalidate(offset, size);
    }

    BUFFER_USAGE usage() const
    {
        return buffer.usage();
    }

    MEMORY_PROPERTY memoryProperties() const
    {
        return buffer.memoryProperties();
    }

    DeviceAddress deviceAddress() const
    {
        return buffer.deviceAddress() + offset;
    }
};


inline BufferRange Buffer::operator()(uint64_t offset, uint64_t size)
{    
    if (size == EVA_WHOLE_SIZE) 
    {
        // EVA_ASSERT(offset < this->size());
        size = this->size() - offset;
    }
    // else EVA_ASSERT(offset + size <= this->size());

    return {*this, offset, size};
}


inline BufferDescriptor::BufferDescriptor(BufferRange range)
: buffer(range.buffer), offset(range.offset), size(range.size) {}



/*
The old layout must either be VK_IMAGE_LAYOUT_UNDEFINED, or match the
current layout of the image subresource range. If the old layout matches the current layout of the
image subresource range, the transition preserves the contents of that range. If the old layout is
VK_IMAGE_LAYOUT_UNDEFINED, the contents of that range may be discarded.
*/
/*
When transitioning the image to VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR or
VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, there is no need to delay subsequent processing,
or perform any visibility operations (as vkQueuePresentKHR performs automatic
visibility operations). To achieve this, the dstAccessMask member of the
VkImageMemoryBarrier should be 0, and the dstStageMask parameter should be
VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
*/
/*
• VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is equivalent to VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT with
  VkAccessFlags2 set to 0 when specified in the second synchronization scope, but equivalent to
  VK_PIPELINE_STAGE_2_NONE in the first scope.
• VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT is equivalent to VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
  with VkAccessFlags2 set to 0 when specified in the first synchronization scope, but equivalent to
  VK_PIPELINE_STAGE_2_NONE in the second scope.
*/
/*
Accesses to the acceleration structure scratch buffers as identified by the
VkAccelerationStructureBuildGeometryInfoKHR::scratchData buffer device addresses must be
synchronized with the VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR pipeline stage and
an access type of (VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR). Accesses to each
VkAccelerationStructureBuildGeometryInfoKHR::srcAccelerationStructure and
VkAccelerationStructureBuildGeometryInfoKHR::dstAccelerationStructure must be synchronized
with the VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR pipeline stage and an access type
of VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR or
VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, as appropriate.
Accesses to other input buffers as identified by any used values of
VkAccelerationStructureGeometryMotionTrianglesDataNV::vertexData,
VkAccelerationStructureGeometryTrianglesDataKHR::vertexData,
VkAccelerationStructureGeometryTrianglesDataKHR::indexData,
VkAccelerationStructureGeometryTrianglesDataKHR::transformData,
VkAccelerationStructureGeometryAabbsDataKHR::data, and
VkAccelerationStructureGeometryInstancesDataKHR::data must be synchronized with the
VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR pipeline stage and an access type of
VK_ACCESS_SHADER_READ_BIT.
*/
struct SYNC_SCOPE {
    struct T {
        PIPELINE_STAGE stage;
        ACCESS access;
        
        bool operator<(const T& other) const {
            if (stage != other.stage) return stage < other.stage;
            return access < other.access;
        }

        // bool operator==(const T& other) const {
        //     return stage == other.stage && access == other.access;
        // }
    } scope;

    SYNC_SCOPE(T scope) : scope(scope) {}
    SYNC_SCOPE(PIPELINE_STAGE stage) : scope({stage, ACCESS::NONE}) {}

    inline static T NONE                = {PIPELINE_STAGE::NONE, ACCESS::NONE};
    inline static T ALL                 = {PIPELINE_STAGE::ALL_COMMANDS, ACCESS::MEMORY_READ | ACCESS::MEMORY_WRITE};
    inline static T ALL_READ            = {PIPELINE_STAGE::ALL_COMMANDS, ACCESS::MEMORY_READ};
    inline static T ALL_WRITE           = {PIPELINE_STAGE::ALL_COMMANDS, ACCESS::MEMORY_WRITE};
    inline static T COMPUTE_READ        = {PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ};
    inline static T COMPUTE_WRITE       = {PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE};
    inline static T RAYTRACING_READ     = {PIPELINE_STAGE::RAY_TRACING_SHADER, ACCESS::SHADER_READ};
    inline static T RAYTRACING_READ_AS  = {PIPELINE_STAGE::RAY_TRACING_SHADER, ACCESS::ACCELERATION_STRUCTURE_READ};
    inline static T RAYTRACING_WRITE    = {PIPELINE_STAGE::RAY_TRACING_SHADER, ACCESS::SHADER_WRITE};
    inline static T TRANSFER_SRC        = {PIPELINE_STAGE::TRANSFER, ACCESS::TRANSFER_READ};
    inline static T TRANSFER_DST        = {PIPELINE_STAGE::TRANSFER, ACCESS::TRANSFER_WRITE};
    inline static T ASBUILD_READ        = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::SHADER_READ};
    inline static T ASBUILD_READ_AS     = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::ACCELERATION_STRUCTURE_READ};
    inline static T ASBUILD_WRITE_AS    = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::ACCELERATION_STRUCTURE_WRITE};
    inline static T ASBUILD_READ_WRITE_AS  = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::ACCELERATION_STRUCTURE_READ | ACCESS::ACCELERATION_STRUCTURE_WRITE};
    inline static T PRESENT_SRC         = {(PIPELINE_STAGE)(uint64_t)-1, (ACCESS)(uint64_t)-1};
};

inline SYNC_SCOPE operator,(PIPELINE_STAGE stage, ACCESS access)
{
    return SYNC_SCOPE::T{stage, access};
}


struct MemoryBarrier {
    SYNC_SCOPE srcMask = SYNC_SCOPE::NONE;
    SYNC_SCOPE dstMask = SYNC_SCOPE::NONE;
};

inline MemoryBarrier operator/(SYNC_SCOPE mask1, SYNC_SCOPE mask2)
{
    return {mask1, mask2};
}


struct BufferMemoryBarrier {
    SYNC_SCOPE srcMask = SYNC_SCOPE::NONE;
    SYNC_SCOPE dstMask = SYNC_SCOPE::NONE;
    OwnershipTransferOpType opType = OwnershipTransferOpType::none;
    QueueType pairedQueue = queue_max;
    // const Buffer& buffer;
    // uint64_t offset = 0;
    // uint64_t size = EVA_WHOLE_SIZE;
    BufferRange buffer;

    BufferMemoryBarrier(Buffer buffer) : buffer(buffer) {}
    BufferMemoryBarrier(BufferRange buffer) : buffer(buffer) {}
};

inline BufferMemoryBarrier&& operator/(SYNC_SCOPE mask, BufferMemoryBarrier&& barrier)
{
    barrier.srcMask = mask;
    return std::move(barrier);
}

inline BufferMemoryBarrier&& operator/(BufferMemoryBarrier&& barrier, SYNC_SCOPE mask)
{
    barrier.dstMask = mask;
    return std::move(barrier);
}

// inline BufferMemoryBarrier&& operator-(QueueType queueType, BufferMemoryBarrier&& barrier)
// {
//     barrier.opType = OwnershipTransferOpType::acquire;
//     barrier.pairedQueue = queueType;
//     return std::move(barrier);
// }

// inline BufferMemoryBarrier&& operator-(BufferMemoryBarrier&& barrier, QueueType queueType)
// {
//     barrier.opType = OwnershipTransferOpType::release;
//     barrier.pairedQueue = queueType;
//     return std::move(barrier);
// }


struct ImageMemoryBarrier {
    SYNC_SCOPE srcMask = SYNC_SCOPE::NONE;
    SYNC_SCOPE dstMask = SYNC_SCOPE::NONE;
    IMAGE_LAYOUT oldLayout = IMAGE_LAYOUT::UNDEFINED;
    IMAGE_LAYOUT newLayout = IMAGE_LAYOUT::UNDEFINED;
    OwnershipTransferOpType opType = OwnershipTransferOpType::none;
    QueueType pairedQueue = queue_max;
    Image image;
    // VkImageSubresourceRange subresourceRange = {};

    ImageMemoryBarrier(Image image) : image(image) {}
};


// inline Image::operator ImageMemoryBarrier() const 
// { 
//     return {
//         .image = *this,
//     };
// }

// inline ImageMemoryBarrier Image::operator/(IMAGE_LAYOUT newLayout) const 
// { 
//     return {
//         .newLayout = newLayout,
//         .image = *this,
//     };
// }

inline ImageMemoryBarrier Image::operator()(IMAGE_LAYOUT oldLayout, IMAGE_LAYOUT newLayout) const
{ 
    ImageMemoryBarrier barrier(*this);
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    return barrier;
}

inline ImageMemoryBarrier&& operator/(SYNC_SCOPE mask, ImageMemoryBarrier&& barrier)
{
    barrier.srcMask = mask;
    return std::move(barrier);
}

inline ImageMemoryBarrier&& operator/(ImageMemoryBarrier&& barrier, SYNC_SCOPE mask)
{
    barrier.dstMask = mask;
    return std::move(barrier);
}



struct SemaphoreStage {
    const Semaphore sem;
    const PIPELINE_STAGE stage;

    SemaphoreStage(
        Semaphore sem, 
        PIPELINE_STAGE stage=PIPELINE_STAGE::ALL_COMMANDS) 
    : sem(sem), stage(stage) {}
};

inline SemaphoreStage Semaphore::operator()(PIPELINE_STAGE stage) const
{
    return {*this, stage};
}


inline std::vector<SemaphoreStage> operator,(SemaphoreStage sem1, SemaphoreStage sem2)
{
    return {sem1, sem2};
}   

inline std::vector<SemaphoreStage>&& operator,(std::vector<SemaphoreStage>&& sems, SemaphoreStage sem)
{
    sems.push_back(sem);
    return std::move(sems);
}

inline std::vector<CommandBuffer> operator,(CommandBuffer cmdBuffer1, CommandBuffer cmdBuffer2)
{
    return {cmdBuffer1, cmdBuffer2};
}

inline std::vector<CommandBuffer>&& operator,(std::vector<CommandBuffer>&& cmdBuffers, CommandBuffer cmdBuffer)
{
    cmdBuffers.push_back(cmdBuffer);
    return std::move(cmdBuffers);
}

inline SubmissionBatchInfo operator/(SemaphoreStage sem, CommandBuffer cmdBuffer)
{
    return {{sem}, {cmdBuffer}, {}};
}

inline SubmissionBatchInfo operator/(std::vector<SemaphoreStage>&& sems, CommandBuffer cmdBuffer)
{
    return {std::move(sems), {cmdBuffer}, {}};
}

inline SubmissionBatchInfo operator/(CommandBuffer cmdBuffer, SemaphoreStage sem)
{
    return {{}, {cmdBuffer}, {sem}};
}

inline SubmissionBatchInfo operator/(CommandBuffer cmdBuffer, std::vector<SemaphoreStage>&& sems)
{
    return {{}, {cmdBuffer}, std::move(sems)};
}

inline SubmissionBatchInfo operator/(SemaphoreStage sem, std::vector<CommandBuffer>&& cmdBuffers)
{
    return {{sem}, std::move(cmdBuffers), {}};
}

inline SubmissionBatchInfo operator/(std::vector<SemaphoreStage>&& sems, std::vector<CommandBuffer>&& cmdBuffers)
{
    return {std::move(sems), std::move(cmdBuffers), {}};
}

inline SubmissionBatchInfo operator/(std::vector<CommandBuffer>&& cmdBuffers, SemaphoreStage sem)
{
    return {{}, std::move(cmdBuffers), {sem}};
}

inline SubmissionBatchInfo operator/(std::vector<CommandBuffer>&& cmdBuffers, std::vector<SemaphoreStage>&& sems)
{
    return {{}, std::move(cmdBuffers), std::move(sems)};
}

inline SubmissionBatchInfo&& operator/(SubmissionBatchInfo&& batch, SemaphoreStage sem)
{
    std::get<2>(batch).push_back(sem);
    return std::move(batch);
}

inline SubmissionBatchInfo&& operator/(SubmissionBatchInfo&& batch, std::vector<SemaphoreStage>&& sems)
{
    std::get<2>(batch) = std::move(sems);
    return std::move(batch);
}

struct Waiting {};
inline void waiting(Waiting w){}

struct Submitting {
private:
    friend Submitting operator<<(Queue queue, SubmissionBatchInfo&& batch);
    friend Submitting operator<<(Queue queue, CommandBuffer cmdBuffer);
    friend Submitting operator<<(Queue queue, std::vector<CommandBuffer>&& cmdBuffers);
    friend Submitting&& operator<<(Submitting&& submitting, SubmissionBatchInfo&& batch);
    friend Submitting&& operator<<(Submitting&& submitting, CommandBuffer cmdBuffer);
    friend Submitting&& operator<<(Submitting&& submitting, std::vector<CommandBuffer>&& cmdBuffers);
    friend void operator<<(Submitting&& submitting, Fence fence);
    friend void operator<<(Submitting&& submitting, void(Waiting));

    Submitting() = delete;
    Submitting(const Submitting&) = delete;
    Submitting(Submitting&&) = delete;

    Submitting(Queue queue, SubmissionBatchInfo&& batch) : queue(queue)
    {
        batches.emplace_back(std::move(batch));
    }
    Queue queue;
    std::vector<SubmissionBatchInfo> batches;
    bool isWaiting = false;
    std::optional<Fence> fence;
    
public:
    ~Submitting() { 
        queue.submit(std::move(batches), fence); 
       
        if (isWaiting) {
            queue.waitIdle();
        }
    }
};


inline Submitting operator<<(Queue queue, CommandBuffer cmdBuffer)
{
    return Submitting(queue, {{}, {cmdBuffer}, {}});
}

inline Submitting operator<<(Queue queue, std::vector<CommandBuffer>&& cmdBuffers)
{
	
    return Submitting(queue, SubmissionBatchInfo{
		    std::vector<SemaphoreStage>{}
		    , std::move(cmdBuffers)
		    , std::vector<SemaphoreStage>{}}
		    );
}

inline Submitting operator<<(Queue queue, SubmissionBatchInfo&& batch)
{
    return Submitting(queue, std::move(batch));
}

inline Submitting operator<<(QueueSelector queueSelector, CommandBuffer cmdBuffer)
{
    return operator<<(queueSelector(cmdBuffer), cmdBuffer);
}

inline Submitting operator<<(QueueSelector queueSelector, std::vector<CommandBuffer>&& cmdBuffers)
{
    return operator<<(queueSelector(cmdBuffers[0]), std::move(cmdBuffers));
}

inline Submitting operator<<(QueueSelector queueSelector, SubmissionBatchInfo&& batch)
{
    return operator<<(queueSelector(std::get<1>(batch)[0]), std::move(batch));
}

inline Submitting&& operator<<(Submitting&& submitting, CommandBuffer cmdBuffer)
{
    submitting.batches.emplace_back(
        std::vector<SemaphoreStage>{}, 
        std::vector<CommandBuffer>{cmdBuffer}, 
        std::vector<SemaphoreStage>{});
    return std::move(submitting);
}

inline Submitting&& operator<<(Submitting&& submitting, std::vector<CommandBuffer>&& cmdBuffers)
{
    submitting.batches.emplace_back(
        std::vector<SemaphoreStage>{}, 
        std::move(cmdBuffers), 
        std::vector<SemaphoreStage>{});
    return std::move(submitting);
}

inline Submitting&& operator<<(Submitting&& submitting, SubmissionBatchInfo&& batch)
{   
    submitting.batches.emplace_back(std::move(batch));
    return std::move(submitting);
}

inline void operator<<(Submitting&& submitting, Fence fence)
{
    submitting.fence = fence;
}

inline void operator<<(Submitting&& submitting, void(Waiting))
{
    submitting.isWaiting = true;
}





#ifdef EVA_ENABLE_WINDOW
struct WindowCreateInfo {
    const char* title;
    uint32_t width;
    uint32_t height;
    bool hidden = false;

    Device device;
    IMAGE_USAGE swapChainImageUsage;
    uint32_t minSwapChainImages = 0;
    FORMAT swapChainImageFormat = FORMAT::B8G8R8A8_SRGB;
    COLOR_SPACE swapChainImageColorSpace = COLOR_SPACE::SRGB_NONLINEAR;
    PRESENT_MODE preferredPresentMode = PRESENT_MODE::FIFO;

    /*
    * prePresentCommandBuffer needs to be allocated:
    * - prePresentCommandPool을 지정하였다면, 우선적으로 그것을 사용하여 생성
    * - prePresentCommandPool이 지정되지 않았다면, device의 prePresentCommandPoolType 타입의 (디바이스에 내제된)기본 커맨드 풀을 사용하여 생성
    */
    CommandPool prePresentCommandPool;
    QueueType prePresentCommandPoolType = queue_graphics; // must be compatible with the present queue family
    COMMAND_POOL_CREATE prePresentCommandPoolFlags = COMMAND_POOL_CREATE::NONE;
};


class Window {
    VULKAN_CLASS_COMMON2(Window)

public:
    const std::vector<Image>& swapChainImages() const;

    void recordPrePresentCommands(std::function<void(CommandBuffer, Image)> recordFunc);

    uint32_t acquireNextImageIndex(Semaphore onNextScImageWritable) const;
    void present(Queue queue, std::vector<Semaphore> waitSemaphore, uint32_t imageIndex) const;

    std::pair<CommandBuffer, Semaphore> getNextPresentingContext(Semaphore onNextScImageWritable) const;
    void present(Queue queue) const;

    bool shouldClose() const;
    void pollEvents() const;

    // Bring window to front and request user attention
    void focus() const;

    void setTitle(const char* title) const;

    // Input callback setters
    void setMouseButtonCallback(void (*callback)(int button, int action, double xpos, double ypos));
    void setKeyCallback(void (*callback)(int key, int action, int mods));
    void setCursorPosCallback(void (*callback)(double xpos, double ypos));
    void setScrollCallback(void (*callback)(double xoffset, double yoffset));
};
#endif // EVA_ENABLE_WINDOW


#ifdef EVA_ENABLE_RAYTRACING
class RaytracingPipeline {
    VULKAN_CLASS_COMMON2(RaytracingPipeline)
public:

    PipelineLayout layout() const;
    DescriptorSetLayout descSetLayout(uint32_t setId0=0) const;
    ShaderGroupHandle getHitGroupHandle(uint32_t groupIndex) const;
    void setHitGroupSbt(ShaderBindingTable sbt);
};


class AccelerationStructure {
    VULKAN_CLASS_COMMON2(AccelerationStructure)
public:
    DeviceAddress deviceAddress() const;
};


struct ShaderGroupHandle {
    uint8_t data[portable::shaderGroupHandleSize];
};


struct HitGroup {
    ShaderStage chitStage;
    ShaderStage ahitStage;
    ShaderStage isecStage;
};


struct RaytracingPipelineCreateInfo {
    ShaderStage rgenStage;
    std::vector<ShaderStage> missStages;
    std::vector<HitGroup> hitGroups;
    uint32_t maxRecursionDepth = 1;
    std::optional<PipelineLayout> layout;
    bool autoLayoutAllowAllStages = false;
};


// directly match VkTransformMatrixKHR
struct TransformMatrix {
    float    matrix[3][4];
};


// directly match VkAccelerationStructureInstanceKHR
struct AccelerationStructureInstance {
    TransformMatrix     transform;
    uint32_t            instanceCustomIndex:24;
    uint32_t            mask:8;
    uint32_t            instanceShaderBindingTableRecordOffset:24;
    GEOMETRY_INSTANCE   flags:8;
    uint64_t            accelerationStructureReference;
};


struct ShaderBindingTable {
    Buffer buffer;
    uint32_t recordSize;
    uint32_t numRecords;
};


struct AABB
{
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};


struct StridedBuffer {
    BufferRange buffer;
    uint32_t stride;
};


struct AsBuildSizesInfo {
    DeviceSize accelerationStructureSize;
    DeviceSize updateScratchSize;
    DeviceSize buildScratchSize;
};


struct AsCreateInfo {
    ACCELERATION_STRUCTURE_TYPE asType;
    BufferRange internalBuffer;     // offset and size required for AS data storage
    uint64_t size;                  // Although it could be fed via internalBuffer.size, it is explicitly specified to prevent user mistakes
};


struct AsBuildInfo {
    BUILD_ACCELERATION_STRUCTURE buildFlags;
    GEOMETRY_TYPE geometryType;
    std::vector<uint32_t> primitiveCounts; // primitive count for each geometry
    AccelerationStructure srcAs;
    AccelerationStructure dstAs;
    BufferRange scratchBuffer;

    struct Triangles {
        StridedBuffer vertexInput;
        StridedBuffer indexInput;
        std::vector<uint32_t> vertexCounts;

        struct Geometry {
            GEOMETRY flags;
            StridedBuffer vertexInput;
            StridedBuffer indexInput;  // stride must be 0, 2, or 4
            BufferRange transformBuffer;
        };
        std::vector<Geometry> eachGeometry;
    };

    struct Aabbs{
        StridedBuffer aabbInput;

        struct Geometry {
            GEOMETRY flags;
            StridedBuffer aabbInput;
        };
        std::vector<Geometry> eachGeometry;
    };

    struct Instances{
        BufferRange instanceInput;
    };

    using Inputs = std::variant<Triangles, Aabbs, Instances>;
    Inputs inputs;
};
#else
class RaytracingPipeline {};
class AccelerationStructure {};
#endif // EVA_ENABLE_RAYTRACING


} // namespace eva



namespace std {
    template<>
    struct hash<eva::ShaderStage> {
        size_t operator()(const eva::ShaderStage& stage) const noexcept;
    };
} 

inline auto alignTo = [](auto value, auto alignment) -> decltype(value) {
    return (value + (decltype(value))alignment - 1) & ~((decltype(value))alignment - 1);
};


#endif // EVA_RUNTIME_H
