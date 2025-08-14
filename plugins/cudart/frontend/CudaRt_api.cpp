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
 * Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaRt.h"

extern "C" __host__ cudaError_t CUDARTAPI cudaFuncSetAttribute(const void* func,
                                                               cudaFuncAttribute attr, int value) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(func);
    CudaRtFrontend::AddVariableForArguments(attr);
    CudaRtFrontend::AddVariableForArguments(value);
    CudaRtFrontend::Execute("cudaFuncSetAttribute");
    return CudaRtFrontend::GetExitCode();
}