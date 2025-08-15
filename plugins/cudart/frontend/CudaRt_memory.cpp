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
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            School of Computer Science, University College Dublin
 */

#include "CudaRt.h"

using namespace std;
using gvirtus::common::mappedPointer;

cudaMemcpyKind inferMemcpyKind(void *dst, const void *src) {
    if (CudaRtFrontend::isDevicePointer(dst) && CudaRtFrontend::isDevicePointer(src)) {
        return cudaMemcpyDeviceToDevice;
    } else if (CudaRtFrontend::isDevicePointer(dst) && !CudaRtFrontend::isDevicePointer(src)) {
        return cudaMemcpyHostToDevice;
    } else if (!CudaRtFrontend::isDevicePointer(dst) && CudaRtFrontend::isDevicePointer(src)) {
        return cudaMemcpyDeviceToHost;
    } else {
        return cudaMemcpyHostToHost;
    }
}

cudaMemcpyKind inferMemcpyKindFromDevice(void *dst) {
    if (CudaRtFrontend::isDevicePointer(dst)) {
        return cudaMemcpyDeviceToDevice;
    }
    return cudaMemcpyDeviceToHost;
}

cudaMemcpyKind inferMemcpyKindToDevice(const void *src) {
    if (CudaRtFrontend::isDevicePointer(src)) {
        return cudaMemcpyDeviceToDevice;
    }
    return cudaMemcpyHostToDevice;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total) {
    // cout << "cudaMemGetInfo called" << endl;
    CudaRtFrontend::Prepare();
    CudaRtFrontend::Execute("cudaMemGetInfo");
    if (CudaRtFrontend::Success()) {
        *free = *CudaRtFrontend::GetOutputHostPointer<size_t>();
        *total = *CudaRtFrontend::GetOutputHostPointer<size_t>();
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
    if (CudaRtFrontend::isMappedMemory(devPtr)) {
        void *hostPointer = devPtr;

#ifdef DEBUG
        cerr << "Mapped pointer detected" << endl;
#endif

        mappedPointer remotePointer = CudaRtFrontend::getMappedPointer(devPtr);

        free(devPtr);
        devPtr = remotePointer.pointer;
    }

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::Execute("cudaFree");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray *array) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments((void *)array);
    CudaRtFrontend::Execute("cudaFreeArray");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    CudaRtFrontend::Prepare();
    // Achtung: skip adding devPtr
    CudaRtFrontend::AddSymbolForArguments((char *)symbol);
    CudaRtFrontend::Execute("cudaGetSymbolAddress");
    if (CudaRtFrontend::Success())
        *devPtr = CudaUtil::UnmarshalPointer(CudaRtFrontend::GetOutputString());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(size);
    CudaRtFrontend::AddSymbolForArguments((char *)symbol);
    CudaRtFrontend::Execute("cudaGetSymbolSize");
    if (CudaRtFrontend::Success()) *size = *(CudaRtFrontend::GetOutputHostPointer<size_t>());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **ptr, size_t size,
                                                        unsigned int flags) {
#ifdef DEBUG
    printf("Requesting cudaHostAlloc\n");
#endif
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here.
    if ((*ptr = malloc(size)) == NULL) return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                                                   unsigned int flags) {
#ifdef DEBUG
    printf("Requesting cudaHostGetDevicePointer\n");
#endif
    // Achtung: we can't use mapped memory
    return cudaErrorMemoryAllocation;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
#ifdef DEBUG
    printf("Requesting cudaHostGetFlags\n");
#endif
    // Achtung: falling back to the simplest method because we can't map memory
    *pFlags = cudaHostAllocDefault;
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(size);
    // cout << "cudaMalloc frontend size: " << size << endl;
    CudaRtFrontend::Execute("cudaMalloc");

    if (CudaRtFrontend::Success()) {
        *devPtr = CudaRtFrontend::GetOutputDevicePointer();
        // cout << "cudaMalloc frontend devPtr: " << *devPtr << endl;
        CudaRtFrontend::addDevicePointer(*devPtr);
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
                                                       cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMalloc3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t cudaMalloc3DArray(cudaArray_t *array,
                                                  const cudaChannelFormatDesc *desc,
                                                  cudaExtent extent, unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(desc);
    CudaRtFrontend::AddVariableForArguments(extent);
    CudaRtFrontend::AddVariableForArguments(flags);

    CudaRtFrontend::Execute("cudaMalloc3DArray");
    if (CudaRtFrontend::Success()) *array = *(CudaRtFrontend::GetOutputHostPointer<cudaArray_t>());
    return CudaRtFrontend::GetExitCode();
}
// FIXME: new mapping way

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocArray(cudaArray **arrayPtr,
                                                          const cudaChannelFormatDesc *desc,
                                                          size_t width, size_t height,
                                                          unsigned int flags) {
    CudaRtFrontend::Prepare();

    CudaRtFrontend::AddHostPointerForArguments(desc);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);

    CudaRtFrontend::Execute("cudaMallocArray");
    if (CudaRtFrontend::Success())
        *arrayPtr = (cudaArray *)CudaRtFrontend::GetOutputDevicePointer();

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here.
    if ((*ptr = malloc(size)) == NULL) return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch,
                                                          size_t width, size_t height) {
    CudaRtFrontend::Prepare();

    CudaRtFrontend::AddVariableForArguments(*pitch);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);
    CudaRtFrontend::Execute("cudaMallocPitch");

    if (CudaRtFrontend::Success()) {
        *devPtr = CudaRtFrontend::GetOutputDevicePointer();
        *pitch = CudaRtFrontend::GetOutputVariable<size_t>();
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice,
                                                              const void *src, int srcDevice,
                                                              size_t count, cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(dst);
    CudaRtFrontend::AddVariableForArguments(dstDevice);
    CudaRtFrontend::AddDevicePointerForArguments(src);
    CudaRtFrontend::AddVariableForArguments(srcDevice);
    CudaRtFrontend::AddVariableForArguments(count);
    CudaRtFrontend::AddDevicePointerForArguments(stream);

    CudaRtFrontend::Execute("cudaMemcpyPeerAsync");

    //    if (CudaRtFrontend::Success()) {
    //        // *dst = CudaRtFrontend::GetOutputDevicePointer();
    //        // *src = CudaRtFrontend::GetOutputDevicePointer();
    //    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ CUDARTAPI cudaError_t cudaMallocManaged(void **devPtr, size_t size,
                                                            unsigned flags) {
    *devPtr = malloc(size);

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaMallocManaged");

    if (CudaRtFrontend::Success()) {
        void *remotePointer = CudaRtFrontend::GetOutputDevicePointer();

        mappedPointer host;
        host.pointer = remotePointer;
        host.size = size;

#ifdef DEBUG
        cerr << "device: " << std::hex << hp << " host: " << *devPtr << endl;
#endif
        CudaRtFrontend::addMappedPointer(*devPtr, host);
    } else {
        free(*devPtr);
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy3D(const cudaMemcpy3DParms *p) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(p);

    unsigned int width = p->extent.width;
    unsigned int num_faces = p->extent.depth;
    unsigned int num_layers = 1;
    unsigned int cubemap_size = width * width * num_faces;
    unsigned int size = cubemap_size * num_layers * (p->srcPtr.pitch / p->extent.width);

    CudaRtFrontend::AddHostPointerForArguments<char>(
        static_cast<char *>(const_cast<void *>(p->srcPtr.ptr)), size);
    CudaRtFrontend::Execute("cudaMemcpy3D");
    if (CudaRtFrontend::Success()) {
        return CudaRtFrontend::GetExitCode();
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count,
                                                     cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            if (memmove(dst, src, count) == NULL) return cudaErrorInvalidValue;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddHostPointerForArguments<char>(
                static_cast<char *>(const_cast<void *>(src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            if (CudaRtFrontend::Success()) {
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            }
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                                                       size_t spitch, size_t width, size_t height,
                                                       cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    char *dst_bytes = static_cast<char *>(dst);
    const char *src_bytes = static_cast<const char *>(src);

    switch (kind) {
        case cudaMemcpyHostToHost:
            if (dpitch < width) return cudaErrorInvalidValue;

            if (dpitch == spitch) {
                if (memcpy(dst_bytes, src_bytes, spitch * height) == NULL)
                    return cudaErrorInvalidValue;
            } else {
                for (int i = 0; i < height; i++) {
                    if (memcpy(dst_bytes + (dpitch * i), src_bytes + (spitch * i), width) == NULL)
                        return cudaErrorInvalidValue;
                }
            }
            return cudaSuccess;
        case cudaMemcpyHostToDevice:
            // Use original pointers or cast again if needed for frontend calls
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddHostPointerForArguments<char>(const_cast<char *>(src_bytes),
                                                             spitch * height);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2D");
            break;
        case cudaMemcpyDeviceToHost:
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2D");
            if (CudaRtFrontend::Success())
                memmove(dst_bytes, CudaRtFrontend::GetOutputHostPointer<char>(dpitch * height),
                        dpitch * height);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2D");
            break;
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(
    cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
    size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DArrayToArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch,
                                                            const void *src, size_t spitch,
                                                            size_t width, size_t height,
                                                            cudaMemcpyKind kind,
                                                            cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch,
                                                                cudaArray_const_t src,
                                                                size_t wOffset, size_t hOffset,
                                                                size_t width, size_t height,
                                                                cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
            // FIXME: implement
            return cudaErrorInvalidMemcpyDirection;
            break;

        case cudaMemcpyDeviceToHost:
            // pass contenuto source
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2DFromArray");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(dpitch * height),
                        dpitch * height);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2DFromArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(
    void *dst, size_t dpitch, const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
    size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DFromArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray *dst, size_t wOffset,
                                                              size_t hOffset, const void *src,
                                                              size_t spitch, size_t width,
                                                              size_t height, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyDeviceToHost:
            // FIXME: implement
            return cudaErrorInvalidMemcpyDirection;

        case cudaMemcpyHostToDevice:
            // pass contenuto source
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddHostPointerForArguments<char>(
                static_cast<char *>(const_cast<void *>(src)), spitch * height);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            break;
    }
    CudaRtFrontend::Execute("cudaMemcpy2DToArray");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(
    cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width,
    size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const cudaMemcpy3DParms *p,
                                                            cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy3DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                                          cudaMemcpyKind kind,
                                                          cudaStream_t stream) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();
    // cout << "cudaMemcpyAsync frontend: "
    //      << "dst: " << dst << ", src: " << src << ", count: " << count
    //      << ", kind: " << kind << ", stream: " << stream << endl;

    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::AddDevicePointerForArguments(stream);
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            if (memmove(dst, src, count) == NULL) {
                return cudaErrorInvalidValue;
            }
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            // cout << "cudaMemcpyAsync HostToDevice" << endl;
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddHostPointerForArguments<char>(
                static_cast<char *>(const_cast<void *>(src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::AddDevicePointerForArguments(stream);
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            break;
        case cudaMemcpyDeviceToHost:
            // cout << "cudaMemcpyAsync DeviceToHost" << endl;
            /* NOTE: adding a fake host pointer */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::AddDevicePointerForArguments(stream);
            // cout << "cudaMemcpyAsync DeviceToHost: "
            //      << "dst: " << dst << ", src: " << src << ", count: " << count
            //      << ", kind: " << kind << ", stream: " << stream << endl;
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            if (CudaRtFrontend::Success()) {
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            }
            break;
        case cudaMemcpyDeviceToDevice:
            // cout << "cudaMemcpyAsync DeviceToDevice" << endl;
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::AddDevicePointerForArguments(stream);
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            break;
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, const cudaArray *src,
                                                              size_t wOffset, size_t hOffset,
                                                              size_t count, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToDevice:
        case cudaMemcpyHostToHost:
            // FIXME: implement
            return cudaErrorInvalidMemcpyDirection;
        case cudaMemcpyDeviceToHost:
            // pass contenuto source
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromArray");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaMemcpyArrayToArray(cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const cudaArray *src,
                       size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToDevice:
            // FIXME: implement
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(wOffsetDst);
            CudaRtFrontend::AddVariableForArguments(hOffsetDst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffsetSrc);
            CudaRtFrontend::AddVariableForArguments(hOffsetSrc);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyArrayToArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const cudaArray *src,
                                                                   size_t wOffset, size_t hOffset,
                                                                   size_t count,
                                                                   cudaMemcpyKind kind,
                                                                   cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                                               size_t count, size_t offset,
                                                               cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKindFromDevice(dst);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
        case cudaMemcpyHostToDevice:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
        case cudaMemcpyDeviceToHost:
            // Achtung: adding a fake host pointer
            CudaRtFrontend::AddDevicePointerForArguments((void *)0x666);
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments((char *)symbol);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromSymbol");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments((char *)symbol);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromSymbol");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                                                    size_t count, size_t offset,
                                                                    cudaMemcpyKind kind,
                                                                    cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromSymbolAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(cudaArray *dst, size_t wOffset,
                                                            size_t hOffset, const void *src,
                                                            size_t count, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKind(dst, src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToHost:
            // FIXME: implement
            return cudaErrorInvalidMemcpyDirection;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments((void *)dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddHostPointerForArguments<char>(
                static_cast<char *>(const_cast<void *>(src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToArray");
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments((void *)dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(cudaArray *dst, size_t wOffset,
                                                                 size_t hOffset, const void *src,
                                                                 size_t count, cudaMemcpyKind kind,
                                                                 cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src,
                                                             size_t count, size_t offset,
                                                             cudaMemcpyKind kind) {
    if (kind == cudaMemcpyDefault) {
        kind = inferMemcpyKindToDevice(src);
    }

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments((char *)symbol);
            CudaRtFrontend::AddHostPointerForArguments<char>(
                static_cast<char *>(const_cast<void *>(src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToSymbol");
            break;
        case cudaMemcpyDeviceToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments((char *)symbol);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToSymbol");
            break;
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol,
                                                                  const void *src, size_t count,
                                                                  size_t offset,
                                                                  cudaMemcpyKind kind,
                                                                  cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToSymbolAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int c, size_t count) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(c);
    CudaRtFrontend::AddVariableForArguments(count);
    CudaRtFrontend::Execute("cudaMemset");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value,
                                                            size_t width, size_t height,
                                                            cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(pitch);
    CudaRtFrontend::AddVariableForArguments(value);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);
    CudaRtFrontend::Execute("cudaMemset2DAsync");
    // TODO: backend side needs implementation
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int c, size_t count,
                                                          cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(c);
    CudaRtFrontend::AddVariableForArguments(count);
    CudaRtFrontend::Execute("cudaMemset");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value,
                                                       size_t width, size_t height) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(pitch);
    CudaRtFrontend::AddVariableForArguments(value);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);
    CudaRtFrontend::Execute("cudaMemset2D");
    // TO-DO si deve fare sul serio solo la parte di backend ovviamente
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset3D(cudaPitchedPtr pitchDevPtr, int value,
                                                       cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMemset3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size,
                                                           unsigned int flags) {
    if (ptr == NULL || size == 0) {
        return cudaErrorInvalidValue;
    } else if (CudaRtFrontend::isMappedMemory(ptr)) {
        // Memory is already registered
        return cudaErrorHostMemoryAlreadyRegistered;
    }

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(ptr);  // send the memory address
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaHostRegister");
    if (CudaRtFrontend::Success()) {
        // backend_ptr tracking is done here and also in the backend
        void *backend_ptr = CudaRtFrontend::GetOutputDevicePointer();
        mappedPointer host;
        host.pointer = backend_ptr;
        host.size = size;
        CudaRtFrontend::addMappedPointer(ptr, host);
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr) {
    if (!CudaRtFrontend::isMappedMemory(ptr)) {
        return cudaErrorHostMemoryNotRegistered;
    }

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(ptr);
    CudaRtFrontend::Execute("cudaHostUnregister");
    if (CudaRtFrontend::Success()) {
        CudaRtFrontend::removeMappedPointer(ptr);
    }
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI
cudaPointerGetAttributes(cudaPointerAttributes *attributes, const void *ptr) {
    cout << "cudaPointerGetAttributes frontend" << endl;
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(attributes);
    CudaRtFrontend::AddHostPointerForArguments(ptr);
    CudaRtFrontend::Execute("cudaPointerGetAttributes");
    if (CudaRtFrontend::Success()) {
        *attributes = *(CudaRtFrontend::GetOutputHostPointer<cudaPointerAttributes>());
    }
    return CudaRtFrontend::GetExitCode();
}