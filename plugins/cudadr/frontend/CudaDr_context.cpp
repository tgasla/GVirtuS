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
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDr.h"

using namespace std;

// This is in the CUDA headers and causes problems
// #define cuCtxPushCurrent cuCtxPushCurrent_v2
// So: cancel it
#ifdef cuCtxPushCurrent
#undef cuCtxPushCurrent
#endif

#ifdef cuCtxPopCurrent
#undef cuCtxPopCurrent
#endif

#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif

#ifdef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxReset
#endif

// Add backward compatibility
extern "C" CUresult cuCtxPushCurrent(CUcontext ctx) __attribute__((alias("cuCtxPushCurrent_v2")));
extern "C" CUresult cuCtxPopCurrent(CUcontext *pctx) __attribute__((alias("cuCtxPopCurrent_v2")));
extern "C" CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
    __attribute__((alias("cuDevicePrimaryCtxRelease_v2")));
extern "C" CUresult cuDevicePrimaryCtxReset(CUdevice dev)
    __attribute__((alias("cuDevicePrimaryCtxReset_v2")));

/*Create a CUDA context*/
extern "C" CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::Execute("cuCtxCreate");
    if (CudaDrFrontend::Success()) {
        *pctx = (CUcontext)(CudaDrFrontend::GetOutputDevicePointer());
    }
    return CudaDrFrontend::GetExitCode();
}

/*Increment a context's usage-count*/
extern "C" CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::AddHostPointerForArguments(pctx);
    CudaDrFrontend::Execute("cuCtxAttach");
    if (CudaDrFrontend::Success()) *pctx = (CUcontext)CudaDrFrontend::GetOutputDevicePointer();
    return CudaDrFrontend::GetExitCode();
}

/*Destroy the current context or a floating CUDA context*/
extern "C" CUresult cuCtxDestroy(CUcontext ctx) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(ctx);
    CudaDrFrontend::Execute("cuCtxDestroy");
    return CudaDrFrontend::GetExitCode();
}

/*Decrement a context's usage-count. */
extern "C" CUresult cuCtxDetach(CUcontext ctx) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(ctx);
    CudaDrFrontend::Execute("cuCtxDetach");
    return CudaDrFrontend::GetExitCode();
}

/*Returns the device ID for the current context.*/
extern "C" CUresult cuCtxGetDevice(CUdevice *device) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(device);
    CudaDrFrontend::Execute("cuCtxGetDevice");
    if (CudaDrFrontend::Success()) {
        *device = *(CudaDrFrontend::GetOutputHostPointer<CUdevice>());
    }
    return CudaDrFrontend::GetExitCode();
}

/*Pops the current CUDA context from the current CPU thread.*/
extern "C" CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
    CudaDrFrontend::Prepare();
    CUcontext ctx;
    pctx = &ctx;
    CudaDrFrontend::Execute("cuCtxPopCurrent");
    if (CudaDrFrontend::Success()) *pctx = (CUcontext)(CudaDrFrontend::GetOutputDevicePointer());
    return CudaDrFrontend::GetExitCode();
}

/*Pushes a floating context on the current CPU thread. */
extern "C" CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(ctx);
    CudaDrFrontend::Execute("cuCtxPushCurrent");
    return CudaDrFrontend::GetExitCode();
}

/*Block for a context's tasks to complete.*/
extern "C" CUresult cuCtxSynchronize(void) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::Execute("cuCtxSynchronize");
    return CudaDrFrontend::GetExitCode();
}

/* Disable peer access */
extern "C" CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(peerContext);
    CudaDrFrontend::Execute("cuCtxDisablePeerAccess");
    return CudaDrFrontend::GetExitCode();
}

/* Enable peer access */
extern "C" CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(peerContext);
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::Execute("cuCtxEnablePeerAccess");
    return CudaDrFrontend::GetExitCode();
}

/* Check if two devices could be connected using peer to peer */
extern "C" CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(canAccessPeer);
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::AddVariableForArguments(peerDev);
    CudaDrFrontend::Execute("cuDeviceCanAccessPeer");
    if (CudaDrFrontend::Success()) *canAccessPeer = *(CudaDrFrontend::GetOutputHostPointer<int>());
    return CudaDrFrontend::GetExitCode();
}

// TODO: test
extern "C" CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(pctx);
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::Execute("cuDevicePrimaryCtxRetain");
    if (CudaDrFrontend::Success()) {
        *pctx = (CUcontext)(CudaDrFrontend::GetOutputDevicePointer());
    }
    return CudaDrFrontend::GetExitCode();
}

// TODO: test
extern "C" CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::Execute("cuDevicePrimaryCtxRelease");
    return CudaDrFrontend::GetExitCode();
}

// TODO: test
extern "C" CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::Execute("cuDevicePrimaryCtxReset");
    return CudaDrFrontend::GetExitCode();
}

// TODO: test
extern "C" CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::Execute("cuDevicePrimaryCtxGetState");
    if (CudaDrFrontend::Success()) {
        *flags = CudaDrFrontend::GetOutputVariable<unsigned int>();
        *active = CudaDrFrontend::GetOutputVariable<int>();
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuCtxGetCurrent(CUcontext *pctx) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::Execute("cuCtxGetCurrent");
    if (CudaDrFrontend::Success()) {
        *pctx = (CUcontext)(CudaDrFrontend::GetOutputDevicePointer());
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuCtxSetCurrent(CUcontext ctx) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(ctx);
    CudaDrFrontend::Execute("cuCtxSetCurrent");
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuCtxGetLimit(size_t *value, CUlimit limit) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(limit);
    CudaDrFrontend::AddVariableForArguments(value);
    CudaDrFrontend::Execute("cuCtxGetLimit");
    if (CudaDrFrontend::Success()) {
        *value = CudaDrFrontend::GetOutputVariable<size_t>();
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(limit);
    CudaDrFrontend::AddVariableForArguments(value);
    CudaDrFrontend::Execute("cuCtxSetLimit");
    return CudaDrFrontend::GetExitCode();
}