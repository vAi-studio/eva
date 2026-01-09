#include "timeChecker.hpp"
#include "eva-runtime.h"
#define SHADER_SPV(name) SHADER_OUTPUT_DIR "/" name ".comp.spv"

using namespace eva;


static Device device = Runtime::get().device({.enableComputeQueues = true});
static Queue queue = device.queue(QueueType::queue_compute);
static DescriptorPool destSetPool = device.createDescriptorPool({
    .maxTypes = {
        {DESCRIPTOR_TYPE::STORAGE_BUFFER, 100},
    },
    .maxSets = 100,
});



ComputePipeline getPipeline(const char* spvPath)
{
    SpvBlob spv = SpvBlob::readFrom(spvPath);
    return  device.createComputePipeline({.csStage = spv});
}


void evalMmNaive(Buffer A, Buffer B, Buffer C, uint32_t M, uint32_t N, uint32_t K, uint32_t iter)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("mm-naive"));

    DescriptorSet descSet = destSetPool(pipeline.layout().descSetLayout(0));
    descSet.write({C, A, B});

    struct Params {
        uint32_t M;
        uint32_t N;
        uint32_t K;
    } params { M, N, K };

    CommandBuffer cmdBuff = device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::SIMULTANEOUS_USE)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(M, N)
    .end();

    for (int i = 0; i < iter; ++i) {
        queue << cmdBuff;
    }

    queue.waitIdle();
}

void evalMmNaiveT(Buffer A, Buffer B, Buffer C, uint32_t M, uint32_t N, uint32_t K, uint32_t iter)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("mm-naive-t"));

    DescriptorSet descSet = destSetPool(pipeline.layout().descSetLayout(0));
    descSet.write({C, A, B});

    struct Params {
        uint32_t M;
        uint32_t N;
        uint32_t K;
    } params { M, N, K };

    CommandBuffer cmdBuff = device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::SIMULTANEOUS_USE)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(M, N)
    .end();

    for (int i = 0; i < iter; ++i) {
        queue << cmdBuff;
    }

    queue.waitIdle();
}


void evalMatMul(const char* spvPath, Buffer A, Buffer B, Buffer C, uint32_t M, uint32_t N, uint32_t K, uint32_t iter)
{
    ComputePipeline pipeline = getPipeline(spvPath);

    DescriptorSet descSet = destSetPool(pipeline.layout().descSetLayout(0));
    descSet.write({C, A, B});

    struct Params {
        uint32_t M;
        uint32_t N;
        uint32_t K;
    } params { M, N, K };

    CommandBuffer cmdBuff = device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::SIMULTANEOUS_USE)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(M, N)
        .barrier(SYNC_SCOPE::COMPUTE_WRITE / C / SYNC_SCOPE::COMPUTE_WRITE)
    .end();

    for (int i = 0; i < iter; ++i) {
        queue << cmdBuff;
    }

    queue.waitIdle();
}

int main()
{
    uint32_t M = 4096;
    uint32_t K = 4096;
    uint32_t N = 4096;

    Buffer A = device.createBuffer({
        .size = M * K * sizeof(float),
        .usage = BUFFER_USAGE::STORAGE_BUFFER | BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL
    });
    Buffer B = device.createBuffer({
        .size = K * N * sizeof(float),
        .usage = BUFFER_USAGE::STORAGE_BUFFER | BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL
    });
    Buffer C = device.createBuffer({
        .size = M * N * sizeof(float),
        .usage = BUFFER_USAGE::STORAGE_BUFFER | BUFFER_USAGE::TRANSFER_SRC,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL
    });

    Buffer staging = device.createBuffer({
        .size = std::max({A.size(), B.size(), C.size()}),
        .usage = BUFFER_USAGE::TRANSFER_SRC | BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT
    });
    uint8_t* ptr = staging.map();

    std::vector<float> aData(M * K);
    std::vector<float> bData(K * N);
    for (uint32_t i = 0; i < M * K; ++i)
        aData[i] = static_cast<float>(i % 100) / 100.0f;
    for (uint32_t i = 0; i < K * N; ++i)
        bData[i] = static_cast<float>((i * 7) % 100) / 100.0f;
    
    std::memcpy(ptr, aData.data(), A.size());
    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .copyBuffer(staging(0, A.size()), A)
    .end().submit().wait();

    std::memcpy(ptr, bData.data(), B.size());
    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .copyBuffer(staging(0, B.size()), B)
    .end().submit().wait();
    
    int iter = 10;
    {
        TimeChecker timer("matmul : {} iterations", iter);
        evalMatMul(SHADER_SPV("mm-naive"), A, B, C, M, N, K, iter);
    }

    {
        TimeChecker timer("transposed matmul : {} iterations", iter);
        evalMatMul(SHADER_SPV("mm-naive-t"), A, B, C, M, N, K, iter);
    }

    // device.newCommandBuffer(QueueType::queue_compute)
    // .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
    //     .copyBuffer(C, staging(0, C.size()))
    // .end().submit().wait();

    // std::vector<float> cData(M * N);
    // std::memcpy(cData.data(), ptr, C.size());
    // staging.unmap();

    // for (uint32_t m = 0; m < 5; ++m) 
    // {
    //     for (uint32_t n = 0; n < 5; ++n) 
    //     {
    //         printf("%0.4f ", cData[m * N + n]);
    //     }
    //     printf("\n");
    // }

    return 0;
}