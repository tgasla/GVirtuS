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
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(GetChannelDesc) {
    try {
        cudaChannelFormatDesc *guestDesc = input_buffer->Assign<cudaChannelFormatDesc>();
        cudaArray *array = (cudaArray *)input_buffer->GetFromMarshal<cudaArray *>();
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        cudaChannelFormatDesc *desc = out->Delegate<cudaChannelFormatDesc>();
        memmove(desc, guestDesc, sizeof(cudaChannelFormatDesc));
        cudaError_t exit_code = cudaGetChannelDesc(desc, array);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(CreateTextureObject) {
    cudaTextureObject_t tex = 0;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        cudaResourceDesc *pResDesc = input_buffer->Assign<cudaResourceDesc>();
        cudaTextureDesc *pTexDesc = CudaUtil::UnmarshalTextureDesc(input_buffer.get());
        cudaResourceViewDesc *pResViewDesc = input_buffer->Assign<cudaResourceViewDesc>();

        cudaError_t exit_code = cudaCreateTextureObject(&tex, pResDesc, pTexDesc, pResViewDesc);

        out->Add<cudaTextureObject_t>(tex);
        return std::make_shared<Result>(exit_code, out);

    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorUnknown);
    }
}