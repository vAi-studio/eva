#pragma once

#include "eva-runtime.h"

namespace eva {

void dumpCapturedPipelineBinaryIfRequested(
    const ComputePipeline& pipeline,
    const ComputePipelineCreateInfo& info,
    uint32_t workgroupSizeX,
    uint32_t workgroupSizeY,
    uint32_t workgroupSizeZ);

} // namespace eva
