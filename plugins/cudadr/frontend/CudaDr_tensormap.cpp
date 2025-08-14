/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDr.h"

using namespace std;

// TODO: test
extern "C" CUresult cuTensorMapEncodeTiled(
    CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
    void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(tensorDataType);
    CudaDrFrontend::AddVariableForArguments(tensorRank);
    CudaDrFrontend::AddDevicePointerForArguments(globalAddress);
    CudaDrFrontend::AddHostPointerForArguments(globalDim, tensorRank * sizeof(cuuint64_t));
    CudaDrFrontend::AddHostPointerForArguments(globalStrides, tensorRank * sizeof(cuuint64_t));
    CudaDrFrontend::AddHostPointerForArguments(boxDim, tensorRank * sizeof(cuuint32_t));
    CudaDrFrontend::AddHostPointerForArguments(elementStrides, tensorRank * sizeof(cuuint32_t));
    CudaDrFrontend::AddVariableForArguments(interleave);
    CudaDrFrontend::AddVariableForArguments(swizzle);
    CudaDrFrontend::AddVariableForArguments(l2Promotion);
    CudaDrFrontend::AddVariableForArguments(oobFill);

    CudaDrFrontend::Execute("cuTensorMapEncodeTiled");
    if (CudaDrFrontend::Success()) {
        memcpy(tensorMap, CudaDrFrontend::GetOutputHostPointer<CUtensorMap>(), sizeof(CUtensorMap));
    }

    return CudaDrFrontend::GetExitCode();
}