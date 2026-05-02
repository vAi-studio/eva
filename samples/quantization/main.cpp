#include "timeChecker.hpp"
#include "eva-runtime.h"
#include <random>
#include <cmath>
#include <algorithm>
#define SHADER_SPV(name) SHADER_OUTPUT_DIR "/" name ".comp.spv"

using namespace eva;


static Device device = Runtime::get().device({.enableComputeQueues = true});
static Queue queue = device.queue(QueueType::queue_compute);
static DescriptorPool descPool = device.createDescriptorPool({
    .maxTypes = {
        {DESCRIPTOR_TYPE::STORAGE_BUFFER, 100},
    },
    .maxSets = 100,
});


// ---------------------------------------------------------------------------
// GPU operations
// ---------------------------------------------------------------------------

ComputePipeline getPipeline(const char* spvPath)
{
    SpvBlob spv = SpvBlob::readFrom(spvPath);
    return device.createComputePipeline({.csStage = spv});
}

void uploadBuffer(Buffer staging, Buffer dst, const void* data, uint64_t size)
{
    uint8_t* ptr = staging.map();
    std::memcpy(ptr, data, size);
    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .copyBuffer(staging(0, size), dst)
    .end().submit().wait();
}

void downloadBuffer(Buffer staging, Buffer src, void* data, uint64_t size)
{
    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .copyBuffer(src, staging(0, size))
    .end().submit().wait();
    uint8_t* ptr = staging.map();
    std::memcpy(data, ptr, size);
}

float findScale(Buffer fpBuf, Buffer intBuf, Buffer maxBuf, Buffer staging,
                uint32_t numElements)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("quantize"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({fpBuf, intBuf, maxBuf});

    uint32_t zero = 0;
    uploadBuffer(staging, maxBuf, &zero, sizeof(uint32_t));

    struct Params { int numElements; int mode; float scale; };
    Params params { (int)numElements, 0, 0.0f };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(numElements, 1)
    .end().submit().wait();

    uint32_t maxAbsBits;
    downloadBuffer(staging, maxBuf, &maxAbsBits, sizeof(uint32_t));
    float maxAbs;
    std::memcpy(&maxAbs, &maxAbsBits, sizeof(float));

    float scale = maxAbs / 127.0f;
    if (scale == 0.0f) scale = 1.0f;
    return scale;
}

float quantizeBuffer(Buffer fpBuf, Buffer intBuf, Buffer maxBuf, Buffer staging,
                     uint32_t numElements)
{
    float scale = findScale(fpBuf, intBuf, maxBuf, staging, numElements);

    ComputePipeline pipeline = getPipeline(SHADER_SPV("quantize"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({fpBuf, intBuf, maxBuf});

    struct Params { int numElements; int mode; float scale; };
    Params params { (int)numElements, 1, scale };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(numElements, 1)
    .end().submit().wait();

    return scale;
}

// Fused quantize + pack (optionally with transpose for B)
// For A:   rows=M, K4=K/4, baseStride=1, rowStride=K
// For B^T: rows=N, K4=K/4, baseStride=N, rowStride=1
void quantizePackBuffer(Buffer fpBuf, Buffer packedBuf,
                        uint32_t rows, uint32_t K4,
                        int baseStride, int rowStride, float scale)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("quantize-pack"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({fpBuf, packedBuf});

    struct Params { int rows; int K4; int baseStride; int rowStride; float scale; };
    Params params { (int)rows, (int)K4, baseStride, rowStride, scale };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(K4, rows)
    .end().submit().wait();
}

void transposeBuffer(Buffer src, Buffer dst, uint32_t rows, uint32_t cols)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("transpose"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({src, dst});

    struct Params { int rows; int cols; } params { (int)rows, (int)cols };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(cols, rows)
    .end().submit().wait();
}

void packBuffer(Buffer intBuf, Buffer packedBuf, uint32_t numElements)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("pack"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({intBuf, packedBuf});

    int numPacked = (int)(numElements / 4);

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(numPacked), &numPacked)
        .bindDescSets({descSet})
        .dispatch2(numPacked, 1)
    .end().submit().wait();
}

void matmulFP32(Buffer A, Buffer B, Buffer C,
                uint32_t M, uint32_t N, uint32_t K)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("matmul-fp32"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({C, A, B});

    struct Params { uint32_t M, N, K; } params { M, N, K };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(M, N)
    .end().submit().wait();
}

void matmulINT8(Buffer A, Buffer B, Buffer C,
                uint32_t M, uint32_t N, uint32_t K)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("matmul-int8"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({C, A, B});

    struct Params { uint32_t M, N, K; } params { M, N, K };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(M, N)
    .end().submit().wait();
}

void matmulINT8Packed(Buffer A_packed, Buffer BT_packed, Buffer C,
                      uint32_t M, uint32_t N, uint32_t K4)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("matmul-int8-packed"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({C, A_packed, BT_packed});

    struct Params { uint32_t M, N, K4; } params { M, N, K4 };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(M, N)
    .end().submit().wait();
}

void convertToFP16Packed(Buffer fpBuf, Buffer packedBuf, uint32_t numElements)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("convert-fp16-pack"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({fpBuf, packedBuf});

    int numPairs = (int)(numElements / 2);

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(numPairs), &numPairs)
        .bindDescSets({descSet})
        .dispatch2(numPairs, 1)
    .end().submit().wait();
}

void matmulFP16Packed(Buffer A_packed, Buffer B_packed, Buffer C,
                      uint32_t M, uint32_t N, uint32_t K)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("matmul-fp16-packed"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({C, A_packed, B_packed});

    struct Params { int M, N, K; } params { (int)M, (int)N, (int)K };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(N, M)
    .end().submit().wait();
}

void convertToBF16Packed(Buffer fpBuf, Buffer packedBuf, uint32_t numElements)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("convert-bf16-pack"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({fpBuf, packedBuf});

    int numPairs = (int)(numElements / 2);

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(numPairs), &numPairs)
        .bindDescSets({descSet})
        .dispatch2(numPairs, 1)
    .end().submit().wait();
}

void matmulBF16Packed(Buffer A_packed, Buffer B_packed, Buffer C,
                      uint32_t M, uint32_t N, uint32_t K)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("matmul-bf16-packed"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({C, A_packed, B_packed});

    struct Params { int M, N, K; } params { (int)M, (int)N, (int)K };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(N, M)
    .end().submit().wait();
}

void convertToFP16Native(Buffer fpBuf, Buffer outBuf, uint32_t numElements)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("convert-fp16-native"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({fpBuf, outBuf});

    int n = (int)numElements;

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(n), &n)
        .bindDescSets({descSet})
        .dispatch2(numElements, 1)
    .end().submit().wait();
}

void matmulFP16Native(Buffer A, Buffer B, Buffer C,
                      uint32_t M, uint32_t N, uint32_t K)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("matmul-fp16-native"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({C, A, B});

    struct Params { int M, N, K; } params { (int)M, (int)N, (int)K };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(N, M)
    .end().submit().wait();
}

void dequantize(Buffer intBuf, Buffer fpBuf, uint32_t numElements, float scale)
{
    ComputePipeline pipeline = getPipeline(SHADER_SPV("dequantize"));
    DescriptorSet descSet = descPool(pipeline.layout().descSetLayout(0));
    descSet.write({intBuf, fpBuf});

    struct Params { int numElements; float scale; } params { (int)numElements, scale };

    device.newCommandBuffer(QueueType::queue_compute)
    .begin(COMMAND_BUFFER_USAGE::ONE_TIME_SUBMIT)
        .bindPipeline(pipeline)
        .setPushConstants(0, sizeof(params), &params)
        .bindDescSets({descSet})
        .dispatch2(numElements, 1)
    .end().submit().wait();
}


// ---------------------------------------------------------------------------
// Error analysis
// ---------------------------------------------------------------------------

void printErrorMetrics(const char* label,
                       const std::vector<float>& ref,
                       const std::vector<float>& test, uint32_t count)
{
    float  maxErr = 0.0f;
    double sumErr = 0.0, sumSqErr = 0.0, sumRefSq = 0.0;

    for (uint32_t i = 0; i < count; ++i) {
        float e = std::abs(ref[i] - test[i]);
        maxErr = std::max(maxErr, e);
        sumErr += e;
        sumSqErr += (double)e * e;
        sumRefSq += (double)ref[i] * ref[i];
    }

    printf("  %-22s | max %.4f | mean %.4f | relRMSE %.2f%%\n",
           label, maxErr, (float)(sumErr / count),
           std::sqrt(sumSqErr / sumRefSq) * 100.0);
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    const uint32_t M = 1024, K = 1024, N = 1024;
    const uint32_t K4 = K / 4;
    const uint32_t K2 = K / 2;
    const uint32_t N2 = N / 2;

    // Random input
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> aData(M * K), bData(K * N);
    for (auto& v : aData) v = dist(rng);
    for (auto& v : bData) v = dist(rng);

    // GPU buffers
    auto storageFlags = BUFFER_USAGE::STORAGE_BUFFER | BUFFER_USAGE::TRANSFER_DST | BUFFER_USAGE::TRANSFER_SRC;
    auto makeBuf = [&](uint64_t size) {
        return device.createBuffer({ .size = size, .usage = storageFlags, .reqMemProps = MEMORY_PROPERTY::DEVICE_LOCAL });
    };

    Buffer A_fp32    = makeBuf(M * K * sizeof(float));
    Buffer B_fp32    = makeBuf(K * N * sizeof(float));
    Buffer C_fp32    = makeBuf(M * N * sizeof(float));
    Buffer A_int     = makeBuf(M * K * sizeof(int32_t));
    Buffer B_int     = makeBuf(K * N * sizeof(int32_t));
    Buffer C_int     = makeBuf(M * N * sizeof(int32_t));
    Buffer C_deq     = makeBuf(M * N * sizeof(float));
    Buffer maxBuf    = makeBuf(sizeof(uint32_t));
    Buffer A_packed      = makeBuf(M * K4 * sizeof(uint32_t));
    Buffer BT_packed     = makeBuf(N * K4 * sizeof(uint32_t));
    Buffer A_fp16_packed = makeBuf(M * K2 * sizeof(uint32_t));
    Buffer B_fp16_packed = makeBuf(K * N2 * sizeof(uint32_t));
    Buffer A_bf16_packed = makeBuf(M * K2 * sizeof(uint32_t));
    Buffer B_bf16_packed = makeBuf(K * N2 * sizeof(uint32_t));
    Buffer C_fp16        = makeBuf(M * N * sizeof(float));
    Buffer A_fp16_native = makeBuf(M * K * sizeof(uint16_t));
    Buffer B_fp16_native = makeBuf(K * N * sizeof(uint16_t));

    Buffer staging = device.createBuffer({
        .size = M * K * sizeof(float),
        .usage = BUFFER_USAGE::TRANSFER_SRC | BUFFER_USAGE::TRANSFER_DST,
        .reqMemProps = MEMORY_PROPERTY::HOST_VISIBLE | MEMORY_PROPERTY::HOST_COHERENT
    });

    uploadBuffer(staging, A_fp32, aData.data(), M * K * sizeof(float));
    uploadBuffer(staging, B_fp32, bData.data(), K * N * sizeof(float));

    printf("=== Quantized MatMul Demo ===\n");
    printf("Matrix size: %u x %u x %u\n\n", M, K, N);

    std::vector<float> refResult(M * N), unpackedResult(M * N), packedResult(M * N);
    std::vector<float> fp16PackedResult(M * N), fp16NativeResult(M * N), bf16PackedResult(M * N);

    // ===================================================================
    // Group 1: FP32 (baseline)
    // ===================================================================
    printf("--- [1] FP32 ---\n");
    {
        TimeChecker total("  total");
        {
            TimeChecker timer("  matmul");
            matmulFP32(A_fp32, B_fp32, C_fp32, M, N, K);
        }
    }
    downloadBuffer(staging, C_fp32, refResult.data(), M * N * sizeof(float));

    // ===================================================================
    // Group 2: INT8 (int32 storage per element)
    // ===================================================================
    printf("\n--- [2] INT8 ---\n");
    float scaleA, scaleB;
    float combinedScale;
    {
        TimeChecker total("  total");
        {
            TimeChecker timer("  quantize A + B");
            scaleA = quantizeBuffer(A_fp32, A_int, maxBuf, staging, M * K);
            scaleB = quantizeBuffer(B_fp32, B_int, maxBuf, staging, K * N);
        }
        combinedScale = scaleA * scaleB;
        {
            TimeChecker timer("  matmul");
            matmulINT8(A_int, B_int, C_int, M, N, K);
        }
        {
            TimeChecker timer("  dequantize");
            dequantize(C_int, C_deq, M * N, combinedScale);
        }
    }
    downloadBuffer(staging, C_deq, unpackedResult.data(), M * N * sizeof(float));

    // ===================================================================
    // Group 3: INT8 Packed (4x int8 in uint32, DP4A-style)
    // ===================================================================
    printf("\n--- [3] INT8 Packed ---\n");
    {
        TimeChecker total("  total");
        {
            TimeChecker timer("  findScale A + B");
            scaleA = findScale(A_fp32, A_int, maxBuf, staging, M * K);
            scaleB = findScale(B_fp32, B_int, maxBuf, staging, K * N);
        }
        combinedScale = scaleA * scaleB;
        {
            TimeChecker timer("  quantize+pack A + quantize+transpose+pack B");
            quantizePackBuffer(A_fp32,  A_packed,  M, K4, 1, K, scaleA);  // A: consecutive
            quantizePackBuffer(B_fp32, BT_packed,  N, K4, N, 1, scaleB);  // B^T: transposed
        }
        {
            TimeChecker timer("  matmul");
            matmulINT8Packed(A_packed, BT_packed, C_int, M, N, K4);
        }
        {
            TimeChecker timer("  dequantize");
            dequantize(C_int, C_deq, M * N, combinedScale);
        }
    }
    downloadBuffer(staging, C_deq, packedResult.data(), M * N * sizeof(float));

    // ===================================================================
    // Group 4: FP16 Packed (2x fp16 in uint32)
    // ===================================================================
    printf("\n--- [4] FP16 Packed ---\n");
    {
        TimeChecker total("  total");
        {
            TimeChecker timer("  convert+pack A + B");
            convertToFP16Packed(A_fp32, A_fp16_packed, M * K);
            convertToFP16Packed(B_fp32, B_fp16_packed, K * N);
        }
        {
            TimeChecker timer("  matmul");
            matmulFP16Packed(A_fp16_packed, B_fp16_packed, C_fp16, M, N, K);
        }
    }
    downloadBuffer(staging, C_fp16, fp16PackedResult.data(), M * N * sizeof(float));

    // ===================================================================
    // Group 5: FP16 Native (float16_t, true 16-bit storage)
    // ===================================================================
    printf("\n--- [5] FP16 Native ---\n");
    {
        TimeChecker total("  total");
        {
            TimeChecker timer("  convert A + B");
            convertToFP16Native(A_fp32, A_fp16_native, M * K);
            convertToFP16Native(B_fp32, B_fp16_native, K * N);
        }
        {
            TimeChecker timer("  matmul");
            matmulFP16Native(A_fp16_native, B_fp16_native, C_fp16, M, N, K);
        }
    }
    downloadBuffer(staging, C_fp16, fp16NativeResult.data(), M * N * sizeof(float));

    // ===================================================================
    // Group 6: BF16 Packed (2x bf16 in uint32)
    // ===================================================================
    printf("\n--- [6] BF16 Packed ---\n");
    {
        TimeChecker total("  total");
        {
            TimeChecker timer("  convert+pack A + B");
            convertToBF16Packed(A_fp32, A_bf16_packed, M * K);
            convertToBF16Packed(B_fp32, B_bf16_packed, K * N);
        }
        {
            TimeChecker timer("  matmul");
            matmulBF16Packed(A_bf16_packed, B_bf16_packed, C_fp16, M, N, K);
        }
    }
    downloadBuffer(staging, C_fp16, bf16PackedResult.data(), M * N * sizeof(float));

    // ===================================================================
    // Results
    // ===================================================================
    printf("\n--- Error vs FP32 Reference ---\n");
    printErrorMetrics("INT8",         refResult, unpackedResult,   M * N);
    printErrorMetrics("INT8 Packed",  refResult, packedResult,     M * N);
    printErrorMetrics("FP16 Packed",  refResult, fp16PackedResult, M * N);
    printErrorMetrics("FP16 Native",  refResult, fp16NativeResult, M * N);
    printErrorMetrics("BF16 Packed",  refResult, bf16PackedResult, M * N);

    return 0;
}
