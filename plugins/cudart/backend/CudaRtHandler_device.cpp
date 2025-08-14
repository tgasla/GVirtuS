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
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include "CudaRtHandler.h"

using namespace log4cplus;

CUDA_ROUTINE_HANDLER(DeviceSetCacheConfig) {
    try {
        cudaFuncCache cacheConfig = input_buffer->Get<cudaFuncCache>();
        cudaError_t exit_code = cudaDeviceSetCacheConfig(cacheConfig);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);  //???
    }
}

CUDA_ROUTINE_HANDLER(DeviceSetLimit) {
    try {
        cudaLimit limit = input_buffer->Get<cudaLimit>();
        size_t value = input_buffer->Get<size_t>();
        cudaError_t exit_code = cudaDeviceSetLimit(limit, value);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);  //???
    }
}

CUDA_ROUTINE_HANDLER(IpcOpenMemHandle) {
    void* devPtr = NULL;
    try {
        cudaIpcMemHandle_t handle = input_buffer->Get<cudaIpcMemHandle_t>();
        unsigned int flags = input_buffer->Get<unsigned int>();
        cudaError_t exit_code = cudaIpcOpenMemHandle(&devPtr, handle, flags);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->AddMarshal(devPtr);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(DeviceEnablePeerAccess) {
    int peerDevice = input_buffer->Get<int>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    cudaError_t exit_code = cudaDeviceEnablePeerAccess(peerDevice, flags);
    return std::make_shared<Result>(exit_code);
}

CUDA_ROUTINE_HANDLER(DeviceDisablePeerAccess) {
    int peerDevice = input_buffer->Get<int>();
    cudaError_t exit_code = cudaDeviceDisablePeerAccess(peerDevice);
    return std::make_shared<Result>(exit_code);
}

CUDA_ROUTINE_HANDLER(DeviceCanAccessPeer) {
    int* canAccessPeer = input_buffer->Assign<int>();
    int device = input_buffer->Get<int>();
    int peerDevice = input_buffer->Get<int>();

    cudaError_t exit_code = cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(canAccessPeer);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(DeviceGetStreamPriorityRange) {
    int leastPriority;
    int greatestPriority;

    cudaError_t exit_code = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(leastPriority);
        out->Add(greatestPriority);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception:") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(DeviceGetAttribute) {
    int* value = input_buffer->Assign<int>();
    cudaDeviceAttr attr = input_buffer->Get<cudaDeviceAttr>();
    int device = input_buffer->Get<int>();
    cudaError_t exit_code = cudaDeviceGetAttribute(value, attr, device);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(value);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(IpcGetMemHandle) {
    cudaIpcMemHandle_t* handle = input_buffer->Assign<cudaIpcMemHandle_t>();
    void* devPtr = input_buffer->GetFromMarshal<void*>();

    cudaError_t exit_code = cudaIpcGetMemHandle(handle, devPtr);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(handle);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(IpcGetEventHandle) {
    cudaIpcEventHandle_t* handle = input_buffer->Assign<cudaIpcEventHandle_t>();
    cudaEvent_t event = input_buffer->Get<cudaEvent_t>();

    cudaError_t exit_code = cudaIpcGetEventHandle(handle, event);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(handle);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(ChooseDevice) {
    int* device = input_buffer->Assign<int>();
    const cudaDeviceProp* prop = input_buffer->Assign<cudaDeviceProp>();
    cudaError_t exit_code = cudaChooseDevice(device, prop);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(device);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetDevice) {
    try {
        int device;
        cudaError_t exit_code = cudaGetDevice(&device);
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "GetDevice executed. Device: " << device);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<int>(device);
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "added device to out buffer");
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(DeviceReset) {
    cudaError_t exit_code = cudaDeviceReset();
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    return std::make_shared<Result>(exit_code, out);
}

CUDA_ROUTINE_HANDLER(DeviceSynchronize) {
    cudaError_t exit_code = cudaDeviceSynchronize();

    return std::make_shared<Result>(exit_code);
}

CUDA_ROUTINE_HANDLER(GetDeviceCount) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        int* count = out->Delegate<int>();
        cudaError_t exit_code = cudaGetDeviceCount(count);
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "GetDeviceCount executed. Count: " << *count);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(GetDeviceProperties) {
    try {
        struct cudaDeviceProp* prop = input_buffer->Assign<struct cudaDeviceProp>();
        int device = input_buffer->Get<int>();
        cudaError_t exit_code = cudaGetDeviceProperties(prop, device);
        prop->canMapHostMemory = 0;
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->Add(prop, 1);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(SetDevice) {
    try {
        int device = input_buffer->Get<int>();
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "SetDevice: " << device);
        cudaError_t exit_code = cudaSetDevice(device);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(SetDeviceFlags) {
    try {
        int flags = input_buffer->Get<int>();
        cudaError_t exit_code = cudaSetDeviceFlags(flags);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(IpcOpenEventHandle) {
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        cudaEvent_t* event = input_buffer->Assign<cudaEvent_t>();
        cudaIpcEventHandle_t handle = input_buffer->Get<cudaIpcEventHandle_t>();
        cudaError_t exit_code = cudaIpcOpenEventHandle(event, handle);
        out->Add(event);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
    }
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
}

CUDA_ROUTINE_HANDLER(SetValidDevices) {
    try {
        int len = input_buffer->BackGet<int>();
        int* device_arr = input_buffer->Assign<int>(len);
        cudaError_t exit_code = cudaSetValidDevices(device_arr, len);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->Add(device_arr, len);
        return std::make_shared<Result>(exit_code, out);

    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(DeviceGetDefaultMemPool) {
    cudaMemPool_t memPool;
    int device = input_buffer->Get<int>();

    cudaError_t exit_code = cudaDeviceGetDefaultMemPool(&memPool, device);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("cudaDeviceGetDefaultMemPool: ")
                                            << exit_code);
    if (exit_code == cudaSuccess) {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->AddMarshal(memPool);
        return std::make_shared<Result>(exit_code, out);
    }
    return std::make_shared<Result>(exit_code);
}