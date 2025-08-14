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
 *             Department of Computer Science, University College Dublin
 */

#include "CudaRt.h"

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t* pStream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::Execute("cudaStreamCreate");
    if (CudaRtFrontend::Success())
        *pStream = (cudaStream_t)CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* pStream,
                                                                    unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaStreamCreateWithFlags");
    if (CudaRtFrontend::Success())
        *pStream = (cudaStream_t)CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaStreamDestroy");
    return CudaRtFrontend::GetExitCode();
}

/*cudaError_t cudaStreamWaitEvent	(	cudaStream_t 	stream,
cudaEvent_t 	event,
unsigned int 	flags
)	*/
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream,
                                                              cudaEvent_t event,
                                                              unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaStreamWaitEvent");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaStreamQuery");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t* pStream,
                                                                       unsigned int flags,
                                                                       int priority) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::AddVariableForArguments(priority);
    CudaRtFrontend::Execute("cudaStreamCreateWithPriority");
    if (CudaRtFrontend::Success())
        *pStream = (cudaStream_t)CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaStreamSynchronize");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments<cudaStreamCaptureMode>(*mode);
    CudaRtFrontend::Execute("cudaThreadExchangeStreamCaptureMode");
    if (CudaRtFrontend::Success())
        *mode = CudaRtFrontend::GetOutputVariable<cudaStreamCaptureMode>();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream,
                                                                 cudaStreamCaptureMode mode) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::AddVariableForArguments(mode);
    CudaRtFrontend::Execute("cudaStreamBeginCapture");
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI
cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaStreamIsCapturing");
    if (CudaRtFrontend::Success())
        *pCaptureStatus = CudaRtFrontend::GetOutputVariable<cudaStreamCaptureStatus>();
    return CudaRtFrontend::GetExitCode();
}

// TODO: implement
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(
    cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out,
    cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) {
    return cudaErrorNotYetImplemented;
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream,
                                                               cudaGraph_t* pGraph) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::AddHostPointerForArguments(pGraph);
    CudaRtFrontend::Execute("cudaStreamEndCapture");
    if (CudaRtFrontend::Success()) *pGraph = CudaRtFrontend::GetOutputVariable<cudaGraph_t>();
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs fixing
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream,
                                                                cudaStreamCallback_t callback,
                                                                void* userData,
                                                                unsigned int flags) {
    // IMPORTANT: cudaStreamCallback_t is a function pointer type
    // not sure how to handle it in the frontend
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    // CudaRtFrontend::AddDevicePointerForArguments(callback);
    CudaRtFrontend::AddDevicePointerForArguments(userData);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaStreamAddCallback");
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream,
                                                                int* priority) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(hStream);
    CudaRtFrontend::Execute("cudaStreamGetPriority");
    if (CudaRtFrontend::Success()) *priority = CudaRtFrontend::GetOutputVariable<int>();
    return CudaRtFrontend::GetExitCode();
}