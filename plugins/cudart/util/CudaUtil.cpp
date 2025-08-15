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

#include <CudaUtil.h>
#include <cuda.h>

#include <cstdio>
#include <iostream>

using namespace std;

CudaUtil::CudaUtil() {}

CudaUtil::CudaUtil(const CudaUtil& orig) {}

CudaUtil::~CudaUtil() {}

char* CudaUtil::MarshalHostPointer(const void* ptr) {
    char* marshal = new char[CudaUtil::MarshaledHostPointerSize];
    MarshalHostPointer(ptr, marshal);
    return marshal;
}

void CudaUtil::MarshalHostPointer(const void* ptr, char* marshal) {
#ifdef _WIN32
    sprintf_s(marshal, 10, "%p", ptr);
#else
    sprintf(marshal, "%p", ptr);
#endif
}

char* CudaUtil::MarshalDevicePointer(const void* devPtr) {
    char* marshal = new char[CudaUtil::MarshaledDevicePointerSize];
    MarshalDevicePointer(devPtr, marshal);
    return marshal;
}

void CudaUtil::MarshalDevicePointer(const void* devPtr, char* marshal) {
#ifdef _WIN32
    sprintf_s(marshal, 10, "%p", devPtr);
#else
    sprintf(marshal, "%p", devPtr);
#endif
}

Buffer* CudaUtil::MarshalFatCudaBinary(__fatBinC_Wrapper_t* bin, Buffer* marshal) {
    if (marshal == NULL) marshal = new Buffer();

    marshal->Add(bin->magic);
    marshal->Add(bin->version);

    struct fatBinaryHeader* header = (fatBinaryHeader*)bin->data;
    size_t size = header->fatSize + (unsigned long long)header->headerSize;

    marshal->Add(size);
    marshal->Add((char*)(bin->data), size);

    return marshal;
}

__fatBinC_Wrapper_t* CudaUtil::UnmarshalFatCudaBinary(Buffer* marshal) {
    void* raw = std::aligned_alloc(8, sizeof(__fatBinC_Wrapper_t));
    __fatBinC_Wrapper_t* bin = new (raw) __fatBinC_Wrapper_t;
    size_t size;

    bin->magic = marshal->Get<int>();
    bin->version = marshal->Get<int>();
    size = marshal->Get<size_t>();
    bin->data = (const long long unsigned int*)marshal->Get<char>(size);
    bin->filename_or_fatbins = NULL;
    return bin;
}

Buffer* CudaUtil::MarshalTextureDescForArguments(const cudaTextureDesc* tex,
                                                 Buffer* marshal = NULL) {
    if (marshal == NULL) marshal = new Buffer();
    if (tex->addressMode != NULL) {
        marshal->AddConst<int>(1);
        marshal->AddConst<cudaTextureAddressMode>(tex->addressMode[0]);
        marshal->AddConst<cudaTextureAddressMode>(tex->addressMode[1]);
        marshal->AddConst<cudaTextureAddressMode>(tex->addressMode[2]);
        marshal->AddConst<cudaTextureFilterMode>(tex->filterMode);
        marshal->AddConst<cudaTextureReadMode>(tex->readMode);
        marshal->AddConst<int>(tex->sRGB);
        marshal->AddConst<int>(tex->normalizedCoords);
        marshal->AddConst<unsigned int>(tex->maxAnisotropy);
        marshal->AddConst<cudaTextureFilterMode>(tex->mipmapFilterMode);
        marshal->AddConst<float>(tex->mipmapLevelBias);
        marshal->AddConst<float>(tex->minMipmapLevelClamp);
        marshal->AddConst<float>(tex->maxMipmapLevelClamp);
    } else {
        marshal->AddConst<int>(0);
        marshal->AddConst<cudaTextureDesc>(tex);
    }

    return marshal;
}

cudaTextureDesc* CudaUtil::UnmarshalTextureDesc(Buffer* marshal) {
    int check = marshal->Get<int>();
    cudaTextureDesc* tex = new cudaTextureDesc;
    if (check == 1) {
        tex->addressMode[0] = marshal->Get<cudaTextureAddressMode>();
        tex->addressMode[1] = marshal->Get<cudaTextureAddressMode>();
        tex->addressMode[2] = marshal->Get<cudaTextureAddressMode>();
        tex->filterMode = marshal->Get<cudaTextureFilterMode>();
        tex->readMode = marshal->Get<cudaTextureReadMode>();
        tex->sRGB = marshal->Get<int>();
        tex->normalizedCoords = marshal->Get<int>();
        tex->maxAnisotropy = marshal->Get<unsigned int>();
        tex->mipmapFilterMode = marshal->Get<cudaTextureFilterMode>();
        tex->mipmapLevelBias = marshal->Get<float>();
        tex->minMipmapLevelClamp = marshal->Get<float>();
        tex->maxMipmapLevelClamp = marshal->Get<float>();
        return tex;
    } else
        return marshal->Assign<cudaTextureDesc>();
}
