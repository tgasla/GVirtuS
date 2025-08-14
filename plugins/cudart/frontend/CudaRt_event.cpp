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
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

#include "CudaRt.h"

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::Execute("cudaEventCreate");
    if (CudaRtFrontend::Success()) *event = (cudaEvent_t)CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event,
                                                                   unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaEventCreateWithFlags");
    if (CudaRtFrontend::Success()) *event = (cudaEvent_t)CudaRtFrontend::GetOutputDevicePointer();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::Execute("cudaEventDestroy");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                                               cudaEvent_t end) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(ms);
    CudaRtFrontend::AddDevicePointerForArguments(start);
    CudaRtFrontend::AddDevicePointerForArguments(end);
    CudaRtFrontend::Execute("cudaEventElapsedTime");
    if (CudaRtFrontend::Success()) *ms = *(CudaRtFrontend::GetOutputHostPointer<float>());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::Execute("cudaEventQuery");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaEventRecord");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(event);
    CudaRtFrontend::Execute("cudaEventSynchronize");
    return CudaRtFrontend::GetExitCode();
}
