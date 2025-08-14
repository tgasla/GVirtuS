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
 * Written By: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            School of Computer Science, University College Dublin
 */

#include <CudaRt_internal.h>

#include "CudaRt.h"

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                                            size_t sharedMem, cudaStream_t stream) {
    CudaRtFrontend::Prepare();

    CudaRtFrontend::AddVariableForArguments(gridDim);
    CudaRtFrontend::AddVariableForArguments(blockDim);
    CudaRtFrontend::AddVariableForArguments(sharedMem);
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaConfigureCall");
    cudaError_t cudaError = CudaRtFrontend::GetExitCode();
    return cudaError;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                                                const void *func) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(attr);
    CudaRtFrontend::AddVariableForArguments((gvirtus::common::pointer_t)func);
    CudaRtFrontend::Execute("cudaFuncGetAttributes");
    if (CudaRtFrontend::Success())
        memmove(attr, CudaRtFrontend::GetOutputHostPointer<cudaFuncAttributes>(),
                sizeof(cudaFuncAttributes));
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func,
                                                                 cudaFuncCache cacheConfig) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments((gvirtus::common::pointer_t)func);
    CudaRtFrontend::AddVariableForArguments(cacheConfig);
    CudaRtFrontend::Execute("cudaFuncSetCacheConfig");

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaLaunch(const void *entry) {
    Buffer *launch = CudaRtFrontend::GetLaunchBuffer();
    launch->Add<int>(0x4c41554e);
    launch->Add<gvirtus::common::pointer_t>((gvirtus::common::pointer_t)entry);
    CudaRtFrontend::Execute("cudaLaunch", launch);
    cudaError_t error = CudaRtFrontend::GetExitCode();
    return error;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaSetDoubleForDevice(double *d) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(d);
    CudaRtFrontend::Execute("cudaSetDoubleForDevice");
    if (CudaRtFrontend::Success()) *d = *(CudaRtFrontend::GetOutputHostPointer<double>());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaSetDoubleForHost(double *d) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(d);
    CudaRtFrontend::Execute("cudaSetDoubleForHost");
    if (CudaRtFrontend::Success()) *d = *(CudaRtFrontend::GetOutputHostPointer<double>());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size,
                                                            size_t offset) {
    const void *pointer = *(const void **)arg;

    Buffer *launch = CudaRtFrontend::GetLaunchBuffer();

    if (CudaRtFrontend::isMappedMemory(*(void **)arg)) {
        gvirtus::common::mappedPointer p = CudaRtFrontend::getMappedPointer(*(void **)arg);
        pointer = (const void *)p.pointer;
        CudaRtFrontend::addtoManage(*(void **)arg);
        cerr << "Async device: " << p.pointer << " host: " << *(void **)arg << endl;
        cudaMemcpyAsync(p.pointer, *(void **)arg, p.size, cudaMemcpyHostToDevice, NULL);
    }

    launch->Add<int>(0x53544147);
    launch->Add<char>(static_cast<char *>(const_cast<void *>((void *)&pointer)), size);
    launch->Add(size);
    launch->Add(offset);
    CudaRtFrontend::Execute("cudaSetupArgument");
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                                    const void *func, void **args) {
    CudaRtFrontend::Prepare();

    // A vector with the mapped pointers to be marshalled and unmarshalled
    vector<gvirtus::common::mappedPointer> mappedPointers;

    CudaRtFrontend::AddDevicePointerForArguments(func);
    CudaRtFrontend::AddVariableForArguments(config->gridDim);
    CudaRtFrontend::AddVariableForArguments(config->blockDim);
    CudaRtFrontend::AddVariableForArguments(config->numAttrs);
    CudaRtFrontend::AddDevicePointerForArguments(config->attrs);
    CudaRtFrontend::AddVariableForArguments(config->dynamicSmemBytes);
    CudaRtFrontend::AddDevicePointerForArguments(config->stream);

    std::string deviceFunc = CudaRtFrontend::getDeviceFunc(const_cast<void *>(func));
    NvInfoFunction infoFunction = CudaRtFrontend::getInfoFunc(deviceFunc);

    size_t argsPayloadSize = 0;
    for (NvInfoKParam infoKParam : infoFunction.params) {
        argsPayloadSize += infoKParam.size_bytes();
        // argsPayloadSize += (infoKParam.size & 0xf8) >> 2; // masking to get the size in bytes
    }

    byte *pArgsPayload = static_cast<byte *>(malloc(argsPayloadSize));
    memset(pArgsPayload, 0x00, argsPayloadSize);

    for (NvInfoKParam infoKParam : infoFunction.params) {
        byte *p = pArgsPayload + infoKParam.offset;
        memcpy(p, args[infoKParam.ordinal], infoKParam.size_bytes());
        // memcpy(p, args[infoKParam.ordinal], (infoKParam.size & 0xf8) >> 2); // masking to get the
        // size in bytes
    }

    CudaRtFrontend::AddHostPointerForArguments<byte>(pArgsPayload, argsPayloadSize);

    CudaRtFrontend::Execute("cudaLaunchKernelExC");
    cudaError_t cudaError = CudaRtFrontend::GetExitCode();
    free(pArgsPayload);

    return cudaError;
}

// TODO: needs testing
extern "C" __host__ cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn,
                                                   void *userData) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::AddVariableForArguments((gvirtus::common::pointer_t)fn);
    CudaRtFrontend::AddHostPointerForArguments(userData);
    CudaRtFrontend::Execute("cudaLaunchHostFunc");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                                 void **args, size_t sharedMem,
                                                 cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(func);
    CudaRtFrontend::AddVariableForArguments(gridDim);
    CudaRtFrontend::AddVariableForArguments(blockDim);
    CudaRtFrontend::AddVariableForArguments(sharedMem);
    CudaRtFrontend::AddDevicePointerForArguments(stream);

    std::string deviceFunc = CudaRtFrontend::getDeviceFunc(func);
    NvInfoFunction infoFunction = CudaRtFrontend::getInfoFunc(deviceFunc);

    // to pass the args to the backend we need their total size
    // we have this info in the infoFunction, so we can calculate the total size
    size_t argsPayloadSize = 0;
    for (const NvInfoKParam &p : infoFunction.params) {
        size_t end = p.offset + p.size_bytes();
        if (end > argsPayloadSize) {
            argsPayloadSize = end;
        }
    }

    // cout << "argsPayloadSize: " << argsPayloadSize << endl;

    byte *pArgsPayload = (byte *)calloc(argsPayloadSize, 1);
    for (NvInfoKParam infoKParam : infoFunction.params) {
        // cout << "Setting argument for ordinal: " << infoKParam.ordinal
        //      << ", offset: " << infoKParam.offset
        //      << ", size: " << infoKParam.size_bytes() << endl;
        memcpy(pArgsPayload + infoKParam.offset, args[infoKParam.ordinal], infoKParam.size_bytes());
    }

    CudaRtFrontend::AddHostPointerForArguments<byte>(pArgsPayload, argsPayloadSize);
    // cout << "GridDim: " << gridDim.x << "," << gridDim.y << "," << gridDim.z << endl;
    // cout << "BlockDim: " << blockDim.x << "," << blockDim.y << "," << blockDim.z << endl;
    // cout << "SharedMem: " << sharedMem << endl;
    // cout << "Stream: " << stream << endl;

    CudaRtFrontend::Execute("cudaLaunchKernel");
    free(pArgsPayload);
    return CudaRtFrontend::GetExitCode();
}