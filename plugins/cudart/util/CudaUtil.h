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

/**
 * @file   CudaUtil.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 11 17:16:48 2009
 *
 * @brief
 *
 *
 */

#ifndef _CUDAUTIL_H
#define _CUDAUTIL_H

#include <cuda_runtime_api.h>
#include <fatbinary_section.h>
#include <gvirtus/communicators/Buffer.h>
#include <texture_types.h>

#include <cstdlib>
#include <iostream>

using gvirtus::communicators::Buffer;

struct __align__(8) fatBinaryHeader {
    unsigned int magic;
    unsigned short version;
    unsigned short headerSize;
    unsigned long long int fatSize;
};
/**
 * CudaUtil contains facility functions used by gVirtuS. These functions
 * includes the ones for marshalling and unmarshalling pointers and "CUDA fat
 * binaries".
 */
class CudaUtil {
   public:
    CudaUtil();
    CudaUtil(const CudaUtil &orig);
    virtual ~CudaUtil();
    static const size_t MarshaledDevicePointerSize = sizeof(void *) * 2 + 3;
    static const size_t MarshaledHostPointerSize = sizeof(void *) * 2 + 3;
    static char *MarshalHostPointer(const void *ptr);
    static void MarshalHostPointer(const void *ptr, char *marshal);
    static char *MarshalDevicePointer(const void *devPtr);
    static void MarshalDevicePointer(const void *devPtr, char *marshal);
    static inline void *UnmarshalPointer(const char *marshal) {
        return (void *)strtoul(marshal, NULL, 16);
    }
    template <class T>
    static inline gvirtus::common::pointer_t MarshalPointer(const T ptr) {
        return static_cast<gvirtus::common::pointer_t>(ptr);
    }
    static Buffer *MarshalFatCudaBinary(__fatBinC_Wrapper_t *bin, Buffer *marshal = NULL);
    static __fatBinC_Wrapper_t *UnmarshalFatCudaBinary(Buffer *marshal);
    static Buffer *MarshalTextureDescForArguments(const cudaTextureDesc *tex, Buffer *marshal);
    static cudaTextureDesc *UnmarshalTextureDesc(Buffer *marshal);

    /**
     * CudaVar is a data structure used for storing information about shared
     * variables.
     */
    struct CudaVar {
        char fatCubinHandle[MarshaledHostPointerSize];
        char hostVar[MarshaledDevicePointerSize];
        char deviceAddress[255];
        char deviceName[255];
        int ext;
        int size;
        int constant;
        int global;
    };

   private:
};

#endif /* _CUDAUTIL_H */
