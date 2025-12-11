#ifndef EVA_BUFFER_POOL_H
#define EVA_BUFFER_POOL_H

#include "eva-runtime.h"
#include <map>
#include <unordered_map>
#include <cstdio>
#include <memory>

namespace eva {

struct BufferPoolConfig {
    Device device;
    int logLevel = 0;  // 0: off, 1: basic, 2: verbose
};


class BufferPool;

struct PooledBuffer
{
    Buffer buffer;
    std::shared_ptr<BufferPool> owner;

    PooledBuffer() = default;
    PooledBuffer(Buffer buf, std::shared_ptr<BufferPool> pool) : buffer(std::move(buf)), owner(std::move(pool)) {}

    PooledBuffer(const PooledBuffer&) = default;
    PooledBuffer& operator=(const PooledBuffer&) = default;
    PooledBuffer(PooledBuffer&&) = default;
    PooledBuffer& operator=(PooledBuffer&&) = default;

    operator bool() const { return (bool)buffer; }
    void release();
};


class BufferPool : public std::enable_shared_from_this<BufferPool>
{
    friend struct PooledBuffer;

    Device device;

    std::unordered_map<
        BUFFER_USAGE,
        std::multimap<size_t, std::pair<Buffer, MEMORY_PROPERTY>>> bufferPool;

    size_t totalAllocated = 0;

public:
    int logLevel;

    BufferPool(BufferPoolConfig config) 
    : device(config.device), logLevel(config.logLevel) {}

    PooledBuffer requestBuffer(
        BUFFER_USAGE usageFlags,
        MEMORY_PROPERTY reqMemProps,
        size_t minSize,
        size_t maxSize = size_t(-1)
    )
    {
        auto& subPool = bufferPool[usageFlags];

        if (logLevel >= 1)
            std::printf("[BufferPool] ");
        if (logLevel >= 2)
        {
            if (subPool.empty())
                std::printf("(empty)");
            else {
                std::printf("{ ");
                for (auto& [sz, _] : subPool)
                   std::printf("%zu, ", sz);
                std::printf("}");
            }
            std::printf("\n");

            std::printf("request: min=%zu, max=%zu\n", minSize, maxSize);
        }

        for (auto it = subPool.lower_bound(minSize);
            it != subPool.end() && it->first <= maxSize;
            ++it)
        {
            const auto& [buffer, memProps] = it->second;
            if (hasFlag(memProps, reqMemProps))
            {
                Buffer result = std::move(it->second.first);
                subPool.erase(it);
                if (logLevel >= 1)
                    std::printf("=> reuse %zu bytes\n", result.size());
                return PooledBuffer{std::move(result), shared_from_this()};
            }
        }

        // Create new buffer
        totalAllocated += minSize;
        if (logLevel >= 1)
            std::printf("=> alloc %zu bytes (total: %zu)\n", minSize, totalAllocated);
        return PooledBuffer{
            device.createBuffer({
                .size = minSize,
                .usage = usageFlags,
                .reqMemProps = reqMemProps
            }),
            shared_from_this()
        };
    }
};


inline void PooledBuffer::release()
{
    if (buffer && owner)
    {
        auto memProps = buffer.memoryProperties();
        auto size = buffer.size();
        auto usage = buffer.usage();

        owner->bufferPool[usage].emplace(
            size,
            std::make_pair(std::move(buffer), memProps)
        );
    }
    buffer = Buffer{};
    owner.reset();
}

} // namespace eva

#endif // EVA_BUFFER_POOL_H
