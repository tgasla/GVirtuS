/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDr.h"

using namespace std;

// This is in the CUDA headers and causes problems
// #define cuCtxPushCurrent cuCtxPushCurrent_v2
// So: cancel it
#ifdef cuStreamWriteValue32
#undef cuStreamWriteValue32
#endif

// Add backward compatibility
extern "C" CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value,
                                         unsigned int flags)
    __attribute__((alias("cuStreamWriteValue32_v2")));

extern "C" CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value,
                                            unsigned int flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(stream);
    CudaDrFrontend::AddVariableForArguments(addr);
    CudaDrFrontend::AddVariableForArguments(value);
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::Execute("cuStreamWriteValue32");
    return CudaDrFrontend::GetExitCode();
}