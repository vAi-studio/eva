#include "eva-runtime.h"
#include "input.h"
#include "camera.h"
#include <thread>
#include <chrono>
#include <cstdio>

#include <eva-error.h>

using namespace eva;


static Device device = Runtime::get().device({
    .enableGraphicsQueues = true,
    .enablePresent = true,
    .enableRaytracing = true
});

struct Sphere {
    float center[3];
    float radius;

    inline static const char* isecSrc = SHADER_OUTPUT_DIR"/sphere.rint.spv";

    Sphere(float x, float y, float z, float r) 
    : center{x, y, z}, radius(r) {}
    AABB getAABB() const {
        return {
            center[0]-radius, center[1]-radius, center[2]-radius,
            center[0]+radius, center[1]+radius, center[2]+radius
        };
    }
};

struct Cylinder {
    float center[3];
    float radius;
    float height;

    inline static const char* isecSrc = SHADER_OUTPUT_DIR"/cylinder.rint.spv";
    
    Cylinder(float x, float y, float z, float r, float h) 
    : center{x, y, z}, radius(r), height(h) {}
    AABB getAABB() const {
        return {
            center[0] - radius, center[1] - height*0.5f, center[2] - radius,
            center[0] + radius, center[1] + height*0.5f, center[2] + radius
        };
    }
};


static DescriptorPool pool = device.createDescriptorPool({
    .maxTypes = { 
        DESCRIPTOR_TYPE::ACCELERATION_STRUCTURE <= 32,
        DESCRIPTOR_TYPE::STORAGE_IMAGE <= 32,
        DESCRIPTOR_TYPE::UNIFORM_BUFFER <= 32,
    },
    .maxSets = 32
});

static const uint32_t asAlign = device.asBufferOffsetAlignment();
static const uint32_t scratchAlign = device.minAccelerationStructureScratchOffsetAlignment();
static const uint32_t sbtAlign = portable::shaderGroupBaseAlignment;
static const uint32_t sbtRecordAlign = portable::shaderGroupHandleAlignment;
static const uint32_t handleSize = portable::shaderGroupHandleSize;

// Global input engine and camera (need to be accessible from callbacks)
static InputEngine inputEngine;
static FpsCamera camera;


int main()
{
    Window window = Runtime::get().createWindow({
        .title = "Vulkan Window",
        .width = 1200,
        .height = 800,
        .device = device,
        .swapChainImageUsage = IMAGE_USAGE::TRANSFER_DST,
        .swapChainImageFormat = FORMAT::R8G8B8A8_UNORM
    });

    // Initialize camera
    camera.initFps(float3(0.0f, 0.0f, 5.0f), 0.0f, 0.0f);
    camera.setScreenSize(1200.0f, 800.0f);
    camera.setFovY(45.0f);

    // Register input callbacks
    window.setMouseButtonCallback([](int button, int action, double xpos, double ypos) {
        inputEngine.onMouseButton(button, action, xpos, ypos);
    });

    window.setKeyCallback([](int key, int action, int mods) {
        inputEngine.onKey(key, action, mods);
    });

    window.setCursorPosCallback([](double xpos, double ypos) {
        inputEngine.onCursorPos(xpos, ypos);
    });

    window.setScrollCallback([](double xoffset, double yoffset) {
        inputEngine.onScroll(xoffset, yoffset);
    });

    std::vector<Sphere> spheres = {
        Sphere(-0.5f, 0.f, 0.f, 0.5f),
        Sphere(0.5f, 0.f, 0.f, 0.5f),
    };

    std::vector<Cylinder> cylinders = {
        Cylinder(2.0f, -2.f, -1.f, 0.7f, 0.1f),
        Cylinder(2.0f, -0.f, -1.f, 0.1f, 4.0f),
    };

    Sphere largeSphere(0.0f, 0.0f, -30.0f, 16.0f);

    // Scene geometry info
    std::vector<AABB> aabbs = {
        spheres[0].getAABB(),       // blas 0 geom 0
        spheres[1].getAABB(),
        cylinders[0].getAABB(),     // blas 0 geom 1
        cylinders[1].getAABB(),

        largeSphere.getAABB(),      // blas 1 geom 0
    };

    uint32_t geometryCountInBlas0 = 2;
    uint32_t primitiveCountInBlas0 = spheres.size() + cylinders.size();

    uint32_t geometryCountInBlas1 = 1;
    uint32_t primitiveCountInBlas1 = 1;

    /*
    - instance0 -> blas0 (2 spheres + 2 cylinders)
    - instance1 -> blas1 (1 large sphere)
    */
    std::vector<AccelerationStructureInstance> instances(2);
    

    //////////////////////////////////////////////////////////////////////////////
    // Preparing input data for BLAS & TLAS build  
    //////////////////////////////////////////////////////////////////////////////
    Buffer aabbBuffer = device.createBuffer({
        .size = sizeof(AABB) * aabbs.size(),
        .usage = BUFFER_USAGE::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY 
                | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
    });
    memcpy(aabbBuffer.map(), aabbs.data(), sizeof(AABB) * aabbs.size());
    aabbBuffer.unmap();

    Buffer stagingBuffer = device.createBuffer({
        .size = sizeof(AccelerationStructureInstance) * instances.size(),
        .usage = BUFFER_USAGE::TRANSFER_SRC,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
    });
    
    // instances buffer must be device local as stated in the spec
    Buffer instanceBuffer = device.createBuffer({
        .size = sizeof(AccelerationStructureInstance) * instances.size(),
        .usage = BUFFER_USAGE::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY 
                | BUFFER_USAGE::SHADER_DEVICE_ADDRESS
                | BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL,
    });


    //////////////////////////////////////////////////////////////////////////////
    // Preparing acceleration structures (BLAS & TLAS) and scratch buffers  
    //////////////////////////////////////////////////////////////////////////////
    /* Create BLAS & TLAS build info and buffers for intermediate(scratch) and final as data */
    std::vector<AsBuildInfo> blasInfos(instances.size());
    auto& blasInfo0 = blasInfos[0] = {
        .buildFlags = BUILD_ACCELERATION_STRUCTURE::PREFER_FAST_TRACE,
        .geometryType = GEOMETRY_TYPE::AABBS,
        .primitiveCounts = { (uint32_t)spheres.size(), (uint32_t)cylinders.size() },
    };
    auto& blasInfo1 = blasInfos[1] = {
        .buildFlags = BUILD_ACCELERATION_STRUCTURE::PREFER_FAST_TRACE,
        .geometryType = GEOMETRY_TYPE::AABBS,
        .primitiveCounts = { 1 },
    };
    ASSERT_(blasInfo0.primitiveCounts.size() == geometryCountInBlas0);
    ASSERT_(blasInfo1.primitiveCounts.size() == geometryCountInBlas1);

    AsBuildInfo tlasInfo = {
        .buildFlags = BUILD_ACCELERATION_STRUCTURE::PREFER_FAST_TRACE,
        .geometryType = GEOMETRY_TYPE::INSTANCES,
        .primitiveCounts = { (uint32_t) blasInfos.size() },
    };

    auto blas0Size = device.getBuildSizesInfo(blasInfo0);
    auto blas1Size = device.getBuildSizesInfo(blasInfo1);
    auto tlasSize = device.getBuildSizesInfo(tlasInfo);
    auto blas1Offset = alignTo(blas0Size.accelerationStructureSize, asAlign);
    auto tlasOffset = alignTo(blas1Offset + blas1Size.accelerationStructureSize, asAlign);
    auto scratch1Offset = alignTo(blas0Size.buildScratchSize, scratchAlign);

    auto totalAsSize = tlasOffset + tlasSize.accelerationStructureSize;
    auto scratchSize = std::max({scratch1Offset + blas1Size.buildScratchSize, tlasSize.buildScratchSize});

    Buffer asBuffer = device.createBuffer({
        .size = (size_t) totalAsSize,
        .usage = BUFFER_USAGE::ACCELERATION_STRUCTURE_STORAGE 
                | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL,
    });

    Buffer scratchBuffer = device.createBuffer({
        .size = scratchSize,
        .usage = BUFFER_USAGE::STORAGE_BUFFER 
                | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL,
    });

    AccelerationStructure blas0 = device.createAccelerationStructure({
        .asType = ACCELERATION_STRUCTURE_TYPE::BOTTOM_LEVEL,
        .internalBuffer = asBuffer(0),
        .size = blas0Size.accelerationStructureSize,
    });
    AccelerationStructure blas1 = device.createAccelerationStructure({
        .asType = ACCELERATION_STRUCTURE_TYPE::BOTTOM_LEVEL,
        .internalBuffer = asBuffer(blas1Offset),
        .size = blas1Size.accelerationStructureSize,
    });
    AccelerationStructure tlas = device.createAccelerationStructure({
        .asType = ACCELERATION_STRUCTURE_TYPE::TOP_LEVEL,
        .internalBuffer = asBuffer(tlasOffset),
        .size = tlasSize.accelerationStructureSize,
    });


    //////////////////////////////////////////////////////////////////////////////
    // Recording commands for building BLAS & TLAS  
    //////////////////////////////////////////////////////////////////////////////
    instances[0] = {
        .transform = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
        },
        .mask = 0xFF,
        .instanceShaderBindingTableRecordOffset = 0,
        .accelerationStructureReference = blas0.deviceAddress()
    };
    instances[1] = {
        .transform = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
        },
        .mask = 0xFF,
        .instanceShaderBindingTableRecordOffset = (uint32_t) geometryCountInBlas0,
        .accelerationStructureReference = blas1.deviceAddress()
    }; 
    memcpy(stagingBuffer.map(), instances.data(), sizeof(AccelerationStructureInstance) * instances.size());
    stagingBuffer.unmap();

    blasInfo0.dstAs = blas0;
    blasInfo0.scratchBuffer = scratchBuffer;
    blasInfo0.inputs = AsBuildInfo::Aabbs{
        .aabbInput = { aabbBuffer, sizeof(AABB) },
    };

    blasInfo1.dstAs = blas1;
    blasInfo1.scratchBuffer = scratchBuffer(scratch1Offset);
    blasInfo1.inputs = AsBuildInfo::Aabbs{
        .aabbInput = { aabbBuffer(sizeof(AABB) * primitiveCountInBlas0), sizeof(AABB) }, 
    };

    tlasInfo.dstAs = tlas;
    tlasInfo.scratchBuffer = scratchBuffer;
    tlasInfo.inputs = AsBuildInfo::Instances{
        .instanceInput = instanceBuffer,
    };

    auto buildCommands = device.newCommandBuffer(queue_graphics)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .buildAccelerationStructures(blasInfos)
        .copyBuffer(stagingBuffer, instanceBuffer)
        .barrier({
            SYNC_SCOPE::ASBUILD_WRITE_AS / scratchBuffer / SYNC_SCOPE::ASBUILD_READ_WRITE_AS,
            SYNC_SCOPE::ASBUILD_WRITE_AS / asBuffer(0, tlasOffset) / SYNC_SCOPE::ASBUILD_READ_AS,
            SYNC_SCOPE::TRANSFER_DST / instanceBuffer / SYNC_SCOPE::ASBUILD_READ,
        })
        .buildAccelerationStructures(tlasInfo)
        .barrier(SYNC_SCOPE::ASBUILD_WRITE_AS / asBuffer(tlasOffset) / SYNC_SCOPE::RAYTRACING_READ_AS)
    .end()
    .submit();


    //////////////////////////////////////////////////////////////////////////////
    // Raytracing pipeline creation
    //////////////////////////////////////////////////////////////////////////////
    SpvBlob blob_chit = SpvBlob::readFrom(SHADER_OUTPUT_DIR"/chit.rchit.spv");
    
    auto rtPipeline = device.createRaytracingPipeline({
        .rgenStage = SpvBlob::readFrom(SHADER_OUTPUT_DIR"/raygen.rgen.spv"),
        .missStages = { SpvBlob::readFrom(SHADER_OUTPUT_DIR"/miss.rmiss.spv") },
        .hitGroups = {
            {
                .chitStage = blob_chit,
                .isecStage = SpvBlob::readFrom(Sphere::isecSrc),
            },
            {
                .chitStage = blob_chit,
                .isecStage = SpvBlob::readFrom(Cylinder::isecSrc)
            }
        },
        .maxRecursionDepth = 1,
    });


    //////////////////////////////////////////////////////////////////////////////
    // Shader binding table for hit group
    //////////////////////////////////////////////////////////////////////////////
    ShaderGroupHandle handle_sphere = rtPipeline.getHitGroupHandle(0);
    ShaderGroupHandle handle_cylinder = rtPipeline.getHitGroupHandle(1);
    
    uint32_t recordSize = std::max(
        alignTo(handleSize + sizeof(Sphere)*spheres.size(), sbtRecordAlign),
        alignTo(handleSize + sizeof(Cylinder)*cylinders.size(), sbtRecordAlign)
    );

    uint32_t numRecords = 0;        // it must match the number of all geometries across BLASs
    for (int i=0; i<blasInfos.size(); ++i)
        numRecords += blasInfos[i].primitiveCounts.size();

    Buffer hitGpSbtBuffer = device.createBuffer({
        .size = recordSize * numRecords,
        .usage = BUFFER_USAGE::SHADER_BINDING_TABLE | BUFFER_USAGE::SHADER_DEVICE_ADDRESS,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
    });
    {
        uint8_t* pSbtRecord = (uint8_t*) hitGpSbtBuffer.map();
        uint8_t* pSbtRecordOffset = pSbtRecord;

        std::memcpy(pSbtRecordOffset, &handle_sphere, handleSize);
        pSbtRecordOffset += handleSize;
        for (uint32_t i = 0; i < spheres.size(); i++) 
        {
            std::memcpy(pSbtRecordOffset, &spheres[i], sizeof(Sphere));
            pSbtRecordOffset += sizeof(Sphere);
        }

        pSbtRecord += recordSize;
        pSbtRecordOffset = pSbtRecord;
        
        std::memcpy(pSbtRecordOffset, &handle_cylinder, handleSize);
        pSbtRecordOffset += handleSize;
        for (uint32_t i = 0; i < cylinders.size(); i++) 
        {
            std::memcpy(pSbtRecordOffset, &cylinders[i], sizeof(Cylinder));
            pSbtRecordOffset += sizeof(Cylinder);
        }

        pSbtRecord += recordSize;
        pSbtRecordOffset = pSbtRecord;

        std::memcpy(pSbtRecordOffset, &handle_sphere, handleSize);
        pSbtRecordOffset += handleSize;
        std::memcpy(pSbtRecordOffset, &largeSphere, sizeof(Sphere));

        hitGpSbtBuffer.unmap();
    }
    rtPipeline.setHitGroupSbt({hitGpSbtBuffer, recordSize, numRecords});


    //////////////////////////////////////////////////////////////////////////////
    // Rendering
    //////////////////////////////////////////////////////////////////////////////
    const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    auto queue = device.queue(queue_graphics);

    Image renderImage = device.createImage({
        .format = FORMAT::R8G8B8A8_UNORM,
        .extent = {
            .width = 1200,
            .height = 800,
        },
        .usage = IMAGE_USAGE::STORAGE | IMAGE_USAGE::TRANSFER_SRC,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL,
        });

    Buffer uniformBuffer[MAX_FRAMES_IN_FLIGHT];
    float* pMap[MAX_FRAMES_IN_FLIGHT];
    DescriptorSet descSet[MAX_FRAMES_IN_FLIGHT];

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        uniformBuffer[i] = device.createBuffer({
            .size = sizeof(float) * 16,  // vec3 pos + float fov + vec3 X + pad + vec3 Y + pad + vec3 Z + pad
            .usage = BUFFER_USAGE::UNIFORM_BUFFER,
            .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT,
            });
        pMap[i] = (float*)uniformBuffer[i].map();

        descSet[i] = pool(rtPipeline.descSetLayout(0));
        descSet[i].write({ tlas, renderImage, uniformBuffer[i] });
    }

    auto& swapChainImages = window.swapChainImages();
    size_t numSwapChainImages = (uint32_t)swapChainImages.size();

    auto initCommandBuffer = device.newCommandBuffer(queue_graphics)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .barrier(SYNC_SCOPE::NONE / renderImage / SYNC_SCOPE::RAYTRACING_WRITE)
    .end()
    .submit();

    std::vector<std::vector<CommandBuffer>> renderCommandBuffers(MAX_FRAMES_IN_FLIGHT);
    
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        renderCommandBuffers[i].resize(numSwapChainImages);

        for (uint32_t j = 0; j < numSwapChainImages; ++j)
        {
            renderCommandBuffers[i][j] = device.newCommandBuffer(queue_graphics)
            .begin()
                .bindPipeline(rtPipeline)
                .bindDescSets({ descSet[i] })

                .traceRays(1200, 800)
                
                .barrier({
                    SYNC_SCOPE::RAYTRACING_WRITE / renderImage / SYNC_SCOPE::TRANSFER_SRC,
                    SYNC_SCOPE::NONE / swapChainImages[j] / SYNC_SCOPE::TRANSFER_DST
                })
                
                .copyImage(renderImage, swapChainImages[j])
                
                .barrier({
                    SYNC_SCOPE::TRANSFER_SRC / renderImage / SYNC_SCOPE::RAYTRACING_WRITE,
                    SYNC_SCOPE::TRANSFER_DST / swapChainImages[j] / SYNC_SCOPE::PRESENT_SRC
                })
            .end();
        }
    }

    std::vector<Semaphore> imageAvailable(MAX_FRAMES_IN_FLIGHT);
    std::vector<Fence> fence(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        imageAvailable[i] = device.createSemaphore();
        fence[i] = device.createFence(true);
    }

    std::vector<Semaphore> presentable(numSwapChainImages);
    for (uint32_t i = 0; i < numSwapChainImages; ++i)
    {
        presentable[i] = device.createSemaphore();
    }

    uint32_t currentFrame = 0;
    int count = 0;

    while (!window.shouldClose())
    {
        window.pollEvents();

        // Update camera with input
        camera.update(inputEngine);

        // Update input state for next frame
        inputEngine.update();

        fence[currentFrame].wait(true);

        uint32_t imgIndex = window.acquireNextImageIndex(imageAvailable[currentFrame]);

        // Update uniform data with camera position, FOV, and orientation
        float* uniformData = pMap[currentFrame];
        {
            const auto& camPos = camera.getCameraPos();
            const auto& camX = camera.getCameraX();
            const auto& camY = camera.getCameraY();
            const auto& camZ = camera.getCameraZ();

            // vec3 cameraPos + float yFov
            uniformData[0] = camPos.x;
            uniformData[1] = camPos.y;
            uniformData[2] = camPos.z;
            uniformData[3] = camera.getFovY();

            // vec3 cameraX + padding
            uniformData[4] = camX.x;
            uniformData[5] = camX.y;
            uniformData[6] = camX.z;
            uniformData[7] = 0.0f; // padding

            // vec3 cameraY + padding
            uniformData[8] = camY.x;
            uniformData[9] = camY.y;
            uniformData[10] = camY.z;
            uniformData[11] = 0.0f; // padding

            // vec3 cameraZ + padding
            uniformData[12] = camZ.x;
            uniformData[13] = camZ.y;
            uniformData[14] = camZ.z;
            uniformData[15] = 0.0f; // padding
        }

        queue << imageAvailable[currentFrame] / renderCommandBuffers[currentFrame][imgIndex] / presentable[imgIndex]
            << fence[currentFrame];

        window.present(queue, { presentable[imgIndex] }, imgIndex);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        count++;
    }

    queue.waitIdle();

    return 0;
}
