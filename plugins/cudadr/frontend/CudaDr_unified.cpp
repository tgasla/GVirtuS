/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDr.h"

using namespace std;

extern "C" CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
                                          CUdeviceptr ptr) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(attribute);
    CudaDrFrontend::AddVariableForArguments(ptr);

    CudaDrFrontend::Execute("cuPointerGetAttribute");

    if (CudaDrFrontend::Success()) {
        memcpy(data, CudaDrFrontend::GetOutputHostPointer<void>(), sizeof(void *));
    }

    return CudaDrFrontend::GetExitCode();
}