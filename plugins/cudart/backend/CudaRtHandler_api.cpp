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
 *             Department of Computer Science,
 *             University College Dublin, Ireland
 */

#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(FuncSetAttribute) {
    const void* func = input_buffer->Get<const void*>();
    cudaFuncAttribute attr = input_buffer->Get<cudaFuncAttribute>();
    int value = input_buffer->Get<int>();

    cudaError_t err = cudaFuncSetAttribute(func, attr, value);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cudaFuncSetAttribute executed with status: " << err);

    return std::make_shared<Result>(err);
}