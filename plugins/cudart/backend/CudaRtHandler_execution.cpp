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
#include "cuda_runtime_compat.h"

CUDA_ROUTINE_HANDLER(ConfigureCall) {
    try {
        dim3 gridDim = input_buffer->Get<dim3>();
        dim3 blockDim = input_buffer->Get<dim3>();
        size_t sharedMem = input_buffer->Get<size_t>();
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaError_t exit_code = cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(FuncGetAttributes) {
    try {
        cudaFuncAttributes *guestAttr = input_buffer->Assign<cudaFuncAttributes>();
        const char *handler = (const char *)(input_buffer->Get<pointer_t>());
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        cudaFuncAttributes *attr = out->Delegate<cudaFuncAttributes>();
        memmove(attr, guestAttr, sizeof(cudaFuncAttributes));
        cudaError_t exit_code = cudaFuncGetAttributes(attr, handler);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(FuncSetCacheConfig) {
    try {
        const char *handler = (const char *)(input_buffer->Get<pointer_t>());
        cudaFuncCache cacheConfig = input_buffer->Get<cudaFuncCache>();
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        cudaError_t exit_code = cudaFuncSetCacheConfig(handler, cacheConfig);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(LaunchKernel) {
    void *func = input_buffer->GetFromMarshal<void *>();
    dim3 gridDim = input_buffer->Get<dim3>();
    dim3 blockDim = input_buffer->Get<dim3>();
    size_t sharedMem = input_buffer->Get<size_t>();
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();

    std::string deviceFunc = pThis->getDeviceFunc(func);
    NvInfoFunction infoFunction = pThis->getInfoFunc(deviceFunc);

    size_t argsSize = 0;
    for (const NvInfoKParam &p : infoFunction.params) {
        size_t end = p.offset + p.size_bytes();
        if (end > argsSize) {
            argsSize = end;
        }
    }

    // cout << "argsSize: " << argsSize << endl;

    byte *pArgs = input_buffer->Assign<byte>(argsSize);

    // cudaLaunchKernel needs an array of pointers to the arguments,
    // so in the buffer we have the arguments packed in a byte array
    // and we need to create an array of pointers to the arguments
    // based on the offsets defined in the infoFunction.params
    void *args[infoFunction.params.size()];

    for (NvInfoKParam infoKParam : infoFunction.params) {
        args[infoKParam.ordinal] = (void *)(pArgs + infoKParam.offset);
        // cout << "Setting argument for ordinal: " << infoKParam.ordinal
        //     << "param: " << infoKParam.ordinal
        //     << ", offset: " << infoKParam.offset
        //     << ", size: " << infoKParam.size_bytes() << endl;
    }

    // cout << "function: " << deviceFunc << endl;
    // cout << "GridDim: " << gridDim.x << "," << gridDim.y << "," << gridDim.z
    // << endl; cout << "BlockDim: " << blockDim.x << "," << blockDim.y << ","
    // << blockDim.z << endl; cout << "SharedMem: " << sharedMem << endl; cout
    // << "Stream: " << stream << endl;

    cudaError_t exit_code = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "LaunchKernel exit_code: " << exit_code);
    return std::make_shared<Result>(exit_code);
}

CUDA_ROUTINE_HANDLER(Launch) {
    int ctrl;
    void *pointer;
    ctrl = input_buffer->Get<int>();
    if (ctrl != 0x434e34c) throw runtime_error("Expecting cudaConfigureCall");

    dim3 gridDim = input_buffer->Get<dim3>();
    dim3 blockDim = input_buffer->Get<dim3>();
    size_t sharedMem = input_buffer->Get<size_t>();
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();

    cudaError_t exit_code = cudaConfigureCall(gridDim, blockDim, sharedMem, stream);

    if (exit_code != cudaSuccess) return std::make_shared<Result>(exit_code);

    while ((ctrl = input_buffer->Get<int>()) == 0x53544147) {
        void *arg = input_buffer->AssignAll<char>();
        size_t size = input_buffer->Get<size_t>();
        size_t offset = input_buffer->Get<size_t>();
        exit_code = cudaSetupArgument(arg, size, offset);
        if (exit_code != cudaSuccess) return std::make_shared<Result>(exit_code);
    }

    if (ctrl != 0x4c41554e) throw runtime_error("Expecting cudaLaunch");
    const char *entry = (const char *)(input_buffer->Get<pointer_t>());

    char *__f = ((char *)pointer);

    exit_code = cudaLaunch(entry);
    return std::make_shared<Result>(exit_code);
}

CUDA_ROUTINE_HANDLER(SetDoubleForDevice) {
    try {
        double *guestD = input_buffer->Assign<double>();
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        double *d = out->Delegate<double>();
        memmove(d, guestD, sizeof(double));
        cudaError_t exit_code = cudaSetDoubleForDevice(d);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(SetDoubleForHost) {
    try {
        double *guestD = input_buffer->Assign<double>();
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        double *d = out->Delegate<double>();
        memmove(d, guestD, sizeof(double));
        cudaError_t exit_code = cudaSetDoubleForHost(d);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(SetupArgument) {
    try {
        size_t offset = input_buffer->BackGet<size_t>();
        size_t size = input_buffer->BackGet<size_t>();
        void *arg = input_buffer->Assign<char>(size);
        cudaError_t exit_code = cudaSetupArgument(arg, size, offset);
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}
