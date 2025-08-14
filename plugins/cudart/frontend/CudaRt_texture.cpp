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

extern "C" __host__ cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z,
                                                                          int w,
                                                                          cudaChannelFormatKind f) {
    cudaChannelFormatDesc desc;
    desc.x = x;
    desc.y = y;
    desc.z = z;
    desc.w = w;
    desc.f = f;
    return desc;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetChannelDesc(cudaChannelFormatDesc *desc,
                                                             const cudaArray *array) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(desc);
    CudaRtFrontend::AddDevicePointerForArguments(array);
    CudaRtFrontend::Execute("cudaGetChannelDesc");
    if (CudaRtFrontend::Success())
        memmove(desc, CudaRtFrontend::GetOutputHostPointer<cudaChannelFormatDesc>(),
                sizeof(cudaChannelFormatDesc));
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaCreateTextureObject(
    cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc,
    const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc) {
    Buffer *input_buffer = new Buffer();

    input_buffer->Add(pResDesc);
    CudaUtil::MarshalTextureDescForArguments(pTexDesc, input_buffer);
    input_buffer->Add(pResDesc);
    CudaRtFrontend::Prepare();
    CudaRtFrontend::Execute("cudaCreateTextureObject", input_buffer);
    if (CudaRtFrontend::Success())
        (*pTexObject) = CudaRtFrontend::GetOutputVariable<cudaTextureObject_t>();
    return CudaRtFrontend::GetExitCode();
}
