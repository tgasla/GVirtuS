/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaRt.h"

// Helper function to check if the attribute is related to memory pool reuse
// In that case, the value is interpreted as int, otherwise as cuuint64_t
bool isMemPoolReuseAttr(cudaMemPoolAttr attr) {
    return (attr == cudaMemPoolReuseFollowEventDependencies ||
            attr == cudaMemPoolReuseAllowOpportunistic ||
            attr == cudaMemPoolReuseAllowInternalDependencies);
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t* pool,
                                                            const cudaMemPoolProps* poolProps) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments<const cudaMemPoolProps>(poolProps);
    CudaRtFrontend::Execute("cudaMemPoolCreate");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t pool,
                                                                  cudaMemPoolAttr attr,
                                                                  void* value) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(pool);
    CudaRtFrontend::AddVariableForArguments(attr);
    CudaRtFrontend::Execute("cudaMemPoolGetAttribute");
    isMemPoolReuseAttr(attr)
        ? *(static_cast<int*>(value)) = CudaRtFrontend::GetOutputVariable<int>()
        : *(static_cast<cuuint64_t*>(value)) = CudaRtFrontend::GetOutputVariable<cuuint64_t>();
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t pool,
                                                                  cudaMemPoolAttr attr,
                                                                  void* value) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(pool);
    CudaRtFrontend::AddVariableForArguments(attr);
    isMemPoolReuseAttr(attr)
        ? CudaRtFrontend::AddVariableForArguments<int>(*(static_cast<int*>(value)))
        : CudaRtFrontend::AddVariableForArguments<cuuint64_t>(*(static_cast<cuuint64_t*>(value)));
    CudaRtFrontend::Execute("cudaMemPoolSetAttribute");
    return CudaRtFrontend::GetExitCode();
}

// TODO: implement
extern "C" __host__ cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool,
                                                     const cudaMemAccessDesc* descList,
                                                     size_t count) {
    cout << "cudaMemPoolSetAccess not yet implemented" << endl;
    return cudaErrorNotYetImplemented;
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaFreeAsync(void* devPtr, cudaStream_t hStream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddDevicePointerForArguments(hStream);
    CudaRtFrontend::Execute("cudaFreeAsync");
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void* devPtr) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::Execute("cudaIpcCloseMemHandle");
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t pool) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(pool);
    CudaRtFrontend::Execute("cudaMemPoolDestroy");
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaMallocAsync(void** devPtr, size_t size,
                                                          cudaStream_t hStream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::AddDevicePointerForArguments(hStream);
    CudaRtFrontend::Execute("cudaMallocAsync");
    // if (CudaRtFrontend::Success())
    //     *devPtr = CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t memPool,
                                                            size_t minBytesToKeep) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(memPool);
    CudaRtFrontend::AddVariableForArguments(minBytesToKeep);
    CudaRtFrontend::Execute("cudaMemPoolTrimTo");
    return CudaRtFrontend::GetExitCode();
}