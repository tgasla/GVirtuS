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

#include <cuda.h>

#include "CudaRtHandler.h"

using namespace log4cplus;

bool isMemPoolReuseAttr(cudaMemPoolAttr attr) {
    return (attr == cudaMemPoolReuseFollowEventDependencies ||
            attr == cudaMemPoolReuseAllowOpportunistic ||
            attr == cudaMemPoolReuseAllowInternalDependencies);
}

CUDA_ROUTINE_HANDLER(MemPoolCreate) {
    cudaMemPool_t memPool;
    const cudaMemPoolProps poolProps = input_buffer->Get<cudaMemPoolProps>();
    cudaError_t exit_code = cudaMemPoolCreate(&memPool, &poolProps);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("cudaMemPoolCreate: ")
                                            << exit_code);
    return std::make_shared<Result>(exit_code);
}

CUDA_ROUTINE_HANDLER(MemPoolGetAttribute) {
    cudaMemPool_t memPool = input_buffer->Get<cudaMemPool_t>();
    cudaMemPoolAttr attr = input_buffer->Get<cudaMemPoolAttr>();

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    void* value = isMemPoolReuseAttr(attr)
                      ? static_cast<void*>(out->Delegate<int>())
                      : static_cast<void*>(out->Delegate<cuuint64_t>());

    cudaError_t exit_code = cudaMemPoolGetAttribute(memPool, attr, value);
    LOG4CPLUS_DEBUG(pThis->GetLogger(),
                    LOG4CPLUS_TEXT("cudaMemPoolGetAttribute: ") << exit_code);
    return std::make_shared<Result>(exit_code, out);
}

// TODO: needs testing
CUDA_ROUTINE_HANDLER(MemPoolSetAttribute) {
    cudaMemPool_t memPool = input_buffer->Get<cudaMemPool_t>();
    cudaMemPoolAttr attr = input_buffer->Get<cudaMemPoolAttr>();

    void* value =
        isMemPoolReuseAttr(attr)
            ? reinterpret_cast<void*>(input_buffer->Get<int>())
            : reinterpret_cast<void*>(input_buffer->Get<cuuint64_t>());

    cudaError_t exit_code = cudaMemPoolSetAttribute(memPool, attr, value);
    LOG4CPLUS_DEBUG(pThis->GetLogger(),
                    LOG4CPLUS_TEXT("cudaMemPoolSetAttribute: ") << exit_code);
    return std::make_shared<Result>(exit_code);
}

// TODO: needs testing
CUDA_ROUTINE_HANDLER(MemPoolDestroy) {
    cudaMemPool_t memPool = input_buffer->Get<cudaMemPool_t>();
    cudaError_t exit_code = cudaMemPoolDestroy(memPool);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("cudaMemPoolDestroy: ")
                                            << exit_code);
    return std::make_shared<Result>(exit_code);
}