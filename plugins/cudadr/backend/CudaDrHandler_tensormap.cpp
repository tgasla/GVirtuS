/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDrHandler.h"

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

// TODO: test
CUDA_DRIVER_HANDLER(TensorMapEncodeTiled) {
    CUtensorMap tensorMap;
    CUtensorMapDataType tensorDataType = input_buffer->Get<CUtensorMapDataType>();
    cuuint32_t tensorRank = input_buffer->Get<cuuint32_t>();
    void* globalAddress = input_buffer->GetFromMarshal<void*>();
    const cuuint64_t* globalDim = input_buffer->Assign<cuuint64_t>(tensorRank * sizeof(cuuint64_t));
    const cuuint64_t* globalStrides =
        input_buffer->Assign<cuuint64_t>(tensorRank * sizeof(cuuint64_t));
    const cuuint32_t* boxDim = input_buffer->Assign<cuuint32_t>(tensorRank * sizeof(cuuint32_t));
    const cuuint32_t* elementStrides =
        input_buffer->Assign<cuuint32_t>(tensorRank * sizeof(cuuint32_t));
    CUtensorMapInterleave interleave = input_buffer->Get<CUtensorMapInterleave>();
    CUtensorMapSwizzle swizzle = input_buffer->Get<CUtensorMapSwizzle>();
    CUtensorMapL2promotion l2Promotion = input_buffer->Get<CUtensorMapL2promotion>();
    CUtensorMapFloatOOBfill oobFill = input_buffer->Get<CUtensorMapFloatOOBfill>();

    CUresult exit_code = cuTensorMapEncodeTiled(
        &tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim,
        elementStrides, interleave, swizzle, l2Promotion, oobFill);

    if (exit_code == CUDA_SUCCESS) {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add(&tensorMap, sizeof(CUtensorMap));
        return std::make_shared<Result>((cudaError_t)exit_code, out);
    }

    return std::make_shared<Result>((cudaError_t)exit_code);
}