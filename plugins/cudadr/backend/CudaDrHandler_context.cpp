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
 * Written by: Flora Giannone <flora.giannone@studenti.uniparthenope.it>,
 *             Department of Applied Science
 */

#include "CudaDrHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

/*Create a CUDA context*/
CUDA_DRIVER_HANDLER(CtxCreate) {
    CUcontext pctx;
    unsigned int flags = input_buffer->Get<unsigned int>();
    CUdevice dev = input_buffer->Get<CUdevice>();
    CUresult exit_code = cuCtxCreate(&pctx, flags, dev);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(pctx);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Increment a context's usage-count*/
CUDA_DRIVER_HANDLER(CtxAttach) {
    unsigned int flags = input_buffer->Get<unsigned int>();
    CUcontext *pctx = input_buffer->Assign<CUcontext>();
    CUresult exit_code = cuCtxAttach(pctx, flags);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(pctx);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Destroy the current context or a floating CUDA context*/
CUDA_DRIVER_HANDLER(CtxDestroy) {
    CUcontext ctx = input_buffer->Get<CUcontext>();
    CUresult exit_code = cuCtxDestroy(ctx);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Decrement a context's usage-count.*/
CUDA_DRIVER_HANDLER(CtxDetach) {
    CUcontext tmp = input_buffer->Get<CUcontext>();
    CUcontext ctx = tmp;
    CUresult exit_code = cuCtxDetach(ctx);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Returns the device ID for the current context.*/
CUDA_DRIVER_HANDLER(CtxGetDevice) {
    CUdevice *device = input_buffer->Assign<CUdevice>();
    CUresult exit_code = cuCtxGetDevice(device);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(device);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Pops the current CUDA context from the current CPU thread.*/
CUDA_DRIVER_HANDLER(CtxPopCurrent) {
    CUcontext pctx;
    CUresult exit_code = cuCtxPopCurrent(&pctx);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(pctx);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Pushes a floating context on the current CPU thread.*/
CUDA_DRIVER_HANDLER(CtxPushCurrent) {
    CUcontext ctx = input_buffer->Get<CUcontext>();
    CUresult exit_code = cuCtxPushCurrent(ctx);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Block for a context's tasks to complete.*/
CUDA_DRIVER_HANDLER(CtxSynchronize) {
    return std::make_shared<Result>((cudaError_t) cuCtxSynchronize());
}

/* Disable peer access */
CUDA_DRIVER_HANDLER(CtxEnablePeerAccess) {
    CUcontext peerContext = input_buffer->Get<CUcontext>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    CUresult exit_code = cuCtxEnablePeerAccess(peerContext, flags);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/* Enable peer access */
CUDA_DRIVER_HANDLER(CtxDisablePeerAccess) {
    CUcontext peerContext = input_buffer->Get<CUcontext>();
    CUresult exit_code = cuCtxDisablePeerAccess(peerContext);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/* Check if two devices could be connected using peer to peer */
CUDA_DRIVER_HANDLER(DeviceCanAccessPeer) {
    int *canAccessPeer = input_buffer->Assign<int>();
    CUdevice dev = input_buffer->Get<CUdevice>();
    CUdevice devPeer = input_buffer->Get<CUdevice>();
    CUresult exit_code = cuDeviceCanAccessPeer(canAccessPeer, dev, devPeer);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(canAccessPeer);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

CUDA_DRIVER_HANDLER(DevicePrimaryCtxGetState) {
    CUdevice dev = input_buffer->Get<CUdevice>();
    unsigned int flags;
    int active;
    CUresult exit_code = cuDevicePrimaryCtxGetState(dev, &flags, &active);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(flags);
    out->Add(active);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

CUDA_DRIVER_HANDLER(CtxGetCurrent) {
    CUcontext pctx;
    CUresult exit_code = cuCtxGetCurrent(&pctx);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(pctx);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

CUDA_DRIVER_HANDLER(CtxSetCurrent) {
    CUcontext ctx = input_buffer->Get<CUcontext>();
    CUresult exit_code = cuCtxSetCurrent(ctx);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

CUDA_DRIVER_HANDLER(CtxGetLimit) {
    CUlimit limit = input_buffer->Get<CUlimit>();
    size_t value;
    CUresult exit_code = cuCtxGetLimit(&value, limit);
    if (exit_code == CUDA_SUCCESS) {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add(value);
        return std::make_shared<Result>((cudaError_t) exit_code, out);
    }
    return std::make_shared<Result>((cudaError_t) exit_code);
}

CUDA_DRIVER_HANDLER(CtxSetLimit) {
    CUlimit limit = input_buffer->Get<CUlimit>();
    size_t value = input_buffer->Get<size_t>();
    CUresult exit_code = cuCtxSetLimit(limit, value);
    return std::make_shared<Result>((cudaError_t) exit_code);
}