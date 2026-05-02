// This sample requires EVA_ENABLE_PERFORMANCE_QUERY.
// Configure with: cmake -DEVA_ENABLE_PERFORMANCE_QUERY=ON ..
#include "eva-runtime.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <map>

#define SHADER_SPV(name) SHADER_OUTPUT_DIR "/" name ".comp.spv"

using namespace eva;


static Device device = Runtime::get().device({.enableComputeQueues = true});
static Queue queue = device.queue(QueueType::queue_compute);
static DescriptorPool descPool = device.createDescriptorPool({
    .maxTypes = {
        {DESCRIPTOR_TYPE::STORAGE_BUFFER, 10},
    },
    .maxSets = 10,
});


int main()
{
    if (!device.supportsPerformanceQueries())
    {
        printf("Performance queries not supported. Aborting.\n");
        return 1;
    }

    // --- Enumerate counters and write to file ---
    auto counters = device.enumeratePerformanceCounters(QueueType::queue_compute);

    // Group by category, preserving order of first appearance
    std::vector<std::string> categoryOrder;
    std::map<std::string, std::vector<uint32_t>> byCategory;
    for (uint32_t i = 0; i < counters.size(); ++i) {
        auto& cat = counters[i].category;
        if (byCategory.find(cat) == byCategory.end())
            categoryOrder.push_back(cat);
        byCategory[cat].push_back(i);
    }

    // Find max name width for alignment
    size_t maxNameLen = 0;
    for (auto& c : counters)
        if (c.name.size() > maxNameLen) maxNameLen = c.name.size();

    FILE* f = fopen(PROJECT_CURRENT_DIR "/counters.txt", "w");
    fprintf(f, "Performance Counters (%zu total)\n\n", counters.size());
    for (auto& cat : categoryOrder) {
        auto& indices = byCategory[cat];
        fprintf(f, "  [%s] (%zu counters)\n", cat.c_str(), indices.size());
        for (uint32_t idx : indices)
            fprintf(f, "    %4u  %-*s  %s\n", idx, (int)maxNameLen, counters[idx].name.c_str(), counters[idx].description.c_str());
        fprintf(f, "\n");
    }
    fclose(f);
    printf("Counter list written to: %s/counters.txt\n", PROJECT_CURRENT_DIR);

    // Select first few counters (up to 4)
    std::vector<uint32_t> counterIndices;
    for (uint32_t i = 0; i < counters.size() && i < 4; ++i)
        counterIndices.push_back(i);

    uint32_t numPasses = device.getPerformanceQueryPasses(QueueType::queue_compute, counterIndices);
    printf("Selected %zu counters, required passes: %u\n", counterIndices.size(), numPasses);
    printf("-------------------------------------------\n");

    // --- Setup buffers ---
    const uint32_t N = 1024 * 1024;
    const uint64_t bufferSize = N * sizeof(float);

    Buffer A = device.createBuffer({
        .size = bufferSize,
        .usage = BUFFER_USAGE::STORAGE_BUFFER | BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL
    });
    Buffer B = device.createBuffer({
        .size = bufferSize,
        .usage = BUFFER_USAGE::STORAGE_BUFFER | BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL
    });
    Buffer C = device.createBuffer({
        .size = bufferSize,
        .usage = BUFFER_USAGE::STORAGE_BUFFER | BUFFER_USAGE::TRANSFER_SRC,
        .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL
    });

    Buffer staging = device.createBuffer({
        .size = bufferSize,
        .usage = BUFFER_USAGE::TRANSFER_SRC | BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT
    });

    uint8_t* ptr = staging.map();

    std::vector<float> aData(N), bData(N);
    for (uint32_t i = 0; i < N; ++i) {
        aData[i] = static_cast<float>(i) / N;
        bData[i] = static_cast<float>(N - i) / N;
    }

    std::memcpy(ptr, aData.data(), bufferSize);
    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .copyBuffer(staging(0, bufferSize), A)
    .end().submit().wait();

    std::memcpy(ptr, bData.data(), bufferSize);
    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .copyBuffer(staging(0, bufferSize), B)
    .end().submit().wait();

    // --- Pipeline setup ---
    SpvBlob spv = SpvBlob::readFrom(SHADER_SPV("vector-add"));
    ComputePipeline pipeline = device.createComputePipeline({.csStage = spv});

    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({C, A, B});

    struct { uint32_t N; } params { N };

    // --- Performance profiling ---
    QueryPool perfPool = device.createPerformanceQueryPool(
        QueueType::queue_compute, counterIndices);

    device.acquireProfilingLock();

    for (uint32_t pass = 0; pass < numPasses; ++pass)
    {
        perfPool.reset();

        device.newCommandBuffer(QueueType::queue_compute)
        .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
            .beginQuery(perfPool, 0)
            .bindPipeline(pipeline)
            .setPushConstants(0, sizeof(params), &params)
            .bindDescSets({descSet})
            .dispatch2(N)
            .endQuery(perfPool, 0)
        .end().submit().wait();
    }

    device.releaseProfilingLock();

    // --- Read results ---
    auto results = perfPool.getResults(0);

    printf("[Results] C = A + B  (%u elements)\n", N);
    for (uint32_t i = 0; i < counterIndices.size(); ++i)
    {
        auto& c = counters[counterIndices[i]];
        auto& r = reinterpret_cast<PerformanceCounterResult*>(results.data())[i];

        printf("  %-40s ", c.name.c_str());
        switch (c.storage) {
        case PERFORMANCE_COUNTER_STORAGE::INT32:   printf("%d",   r.i32); break;
        case PERFORMANCE_COUNTER_STORAGE::INT64:   printf("%lld", r.i64); break;
        case PERFORMANCE_COUNTER_STORAGE::UINT32:  printf("%u",   r.u32); break;
        case PERFORMANCE_COUNTER_STORAGE::UINT64:  printf("%llu", r.u64); break;
        case PERFORMANCE_COUNTER_STORAGE::FLOAT32: printf("%.4f", r.f32); break;
        case PERFORMANCE_COUNTER_STORAGE::FLOAT64: printf("%.4f", r.f64); break;
        default: printf("?"); break;
        }
        printf("\n");
    }

    // --- Verify ---
    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .copyBuffer(C, staging(0, bufferSize))
    .end().submit().wait();

    std::vector<float> cData(N);
    std::memcpy(cData.data(), ptr, bufferSize);
    staging.unmap();

    printf("-------------------------------------------\n");
    printf("[Verification] C = A + B\n");
    for (uint32_t i = 0; i < 5; ++i) {
        float expected = aData[i] + bData[i];
        printf("  C[%d] = %.6f  (expected %.6f)\n", i, cData[i], expected);
    }

    return 0;
}
