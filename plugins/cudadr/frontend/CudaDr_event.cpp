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

/*Creates an event.*/
extern "C" CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(Flags);
    CudaDrFrontend::Execute("cuEventCreate");
    if (CudaDrFrontend::Success()) *phEvent = (CUevent)(CudaDrFrontend::GetOutputDevicePointer());
    return CudaDrFrontend::GetExitCode();
}

/*Destroys an event.*/
extern "C" CUresult cuEventDestroy(CUevent hEvent) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hEvent);
    CudaDrFrontend::Execute("cuEventDestroy");
    return CudaDrFrontend::GetExitCode();
}

/*Computes the elapsed time between two events.*/
extern "C" CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(pMilliseconds);
    CudaDrFrontend::AddDevicePointerForArguments((void *)hStart);
    CudaDrFrontend::AddDevicePointerForArguments((void *)hEnd);
    CudaDrFrontend::Execute("cuEventElapsedTime");
    if (CudaDrFrontend::Success())
        *pMilliseconds = *(CudaDrFrontend::GetOutputHostPointer<float>());
    return CudaDrFrontend::GetExitCode();
}

/*Queries an event's status.*/
extern "C" CUresult cuEventQuery(CUevent hEvent) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hEvent);
    CudaDrFrontend::Execute("cuEventQuery");
    return CudaDrFrontend::GetExitCode();
}

/*Records an event.*/
extern "C" CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hEvent);
    CudaDrFrontend::AddDevicePointerForArguments((void *)hStream);
    CudaDrFrontend::Execute("cuEventRecord");
    return CudaDrFrontend::GetExitCode();
}

/*Waits for an event to complete.*/
extern "C" CUresult cuEventSynchronize(CUevent hEvent) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void *)hEvent);
    CudaDrFrontend::Execute("cuEventSynchronize");
    return CudaDrFrontend::GetExitCode();
}
