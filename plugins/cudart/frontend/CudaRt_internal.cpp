
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

#include <cstdio>
#include <elf.h>
#include <CudaRt_internal.h>

#include "CudaRt.h"

/*
 Routines not found in the cuda's header files.
 KEEP THEM WITH CARE
 */

extern "C" __host__ void **__cudaRegisterFatBinary(void *fatCubin) {
  /* Fake host pointer */
  __fatBinC_Wrapper_t *bin = (__fatBinC_Wrapper_t *)fatCubin;
  char *data = (char *)bin->data;

    NvFatCubin *pFatCubin = (NvFatCubin *)data;
    // check so its really an elf file
    Elf64_Ehdr *eh = &(pFatCubin->elf);
    if(!strncmp((char*)eh->e_ident, "\177ELF", 4)) {

        /* Section header table :  */
        Elf64_Shdr *sh_table = static_cast<Elf64_Shdr *>(malloc(eh->e_shentsize * eh->e_shnum));

        byte *baseAddr = (byte *) eh;
        for (uint32_t i = 0; i < eh->e_shnum; i++) {
            Elf64_Shdr *shdrSrc = (Elf64_Shdr *) (baseAddr + (off_t) eh->e_shoff + i * eh->e_shentsize);
            memcpy(&sh_table[i], shdrSrc, eh->e_shentsize);
        }

        char *sh_str = static_cast<char *>(malloc(sh_table[eh->e_shstrndx].sh_size));
        if (sh_str) {
            memcpy(sh_str, baseAddr + sh_table[eh->e_shstrndx].sh_offset, sh_table[eh->e_shstrndx].sh_size);

            for (uint32_t i = 0; i < eh->e_shnum; i++) {

                char *szSectionName = (sh_str + sh_table[i].sh_name);
                if (strncmp(".nv.info.", szSectionName, strlen(".nv.info.")) == 0) {
                    char *szFuncName = szSectionName + strlen(".nv.info.");

                    byte *p = (byte *) eh + sh_table[i].sh_offset;

                    NvInfoFunction infoFunction;
                    size_t size;
                    NvInfoAttribute *pAttr = (NvInfoAttribute *) p;
                    while (pAttr < (NvInfoAttribute *) ((byte *) p + sh_table[i].sh_size)) {
                        size = 0;
                        switch (pAttr->fmt) {
                            case EIFMT_SVAL:
                                size = sizeof(NvInfoAttribute) + pAttr->value;
                                break;
                            case EIFMT_NVAL:
                                size = sizeof(NvInfoAttribute);
                                break;
                            case EIFMT_HVAL:
                                size = sizeof(NvInfoAttribute);
                                break;

                        }
                        if (pAttr->attr == EIATTR_KPARAM_INFO) {
                            NvInfoKParam *nvInfoKParam = (NvInfoKParam *) pAttr;

                            NvInfoKParam nvInfoKParam1;
                            nvInfoKParam1.index = nvInfoKParam->index;
                            nvInfoKParam1.ordinal = nvInfoKParam->ordinal;
                            nvInfoKParam1.offset = nvInfoKParam->offset;
                            nvInfoKParam1.a = nvInfoKParam->a;
                            nvInfoKParam1.size = nvInfoKParam->size;
                            nvInfoKParam1.b = nvInfoKParam->b;
                            infoFunction.params.push_back(nvInfoKParam1);
                        }
                        pAttr = (NvInfoAttribute *) ((byte *) pAttr + size);
                    }
                    CudaRtFrontend::addDeviceFunc2InfoFunc(szFuncName, infoFunction);
                }
            }
            free(sh_str);
        }
        free(sh_table);


        Buffer *input_buffer = new Buffer();
        input_buffer->AddString(CudaUtil::MarshalHostPointer((void **) bin));
        input_buffer = CudaUtil::MarshalFatCudaBinary(bin, input_buffer);

        CudaRtFrontend::Prepare();
        CudaRtFrontend::Execute("cudaRegisterFatBinary", input_buffer);
        if (CudaRtFrontend::Success()) return (void **) fatCubin;
    }
  return NULL;
}

extern "C" __host__ void **__cudaRegisterFatBinaryEnd(void *fatCubin) {
    /* Fake host pointer */
  __fatBinC_Wrapper_t *bin = (__fatBinC_Wrapper_t *)fatCubin;
  char *data = (char *)bin->data;

  Buffer *input_buffer = new Buffer();
  input_buffer->AddString(CudaUtil::MarshalHostPointer((void **)bin));
  input_buffer = CudaUtil::MarshalFatCudaBinary(bin, input_buffer);

  CudaRtFrontend::Prepare();
  CudaRtFrontend::Execute("cudaRegisterFatBinaryEnd", input_buffer);
  if (CudaRtFrontend::Success()) return (void **)fatCubin;
  return NULL;
}

extern "C" __host__ void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    CudaRtFrontend::Execute("cudaUnregisterFatBinary");
}

extern "C" __host__ void __cudaRegisterFunction(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize) {

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));

  CudaRtFrontend::AddVariableForArguments((gvirtus::common::pointer_t)hostFun);
  CudaRtFrontend::AddStringForArguments(deviceFun);
  CudaRtFrontend::AddStringForArguments(deviceName);
  CudaRtFrontend::AddVariableForArguments(thread_limit);
  CudaRtFrontend::AddHostPointerForArguments(tid);
  CudaRtFrontend::AddHostPointerForArguments(bid);
  CudaRtFrontend::AddHostPointerForArguments(bDim);
  CudaRtFrontend::AddHostPointerForArguments(gDim);
  CudaRtFrontend::AddHostPointerForArguments(wSize);

  CudaRtFrontend::Execute("cudaRegisterFunction");

  deviceFun = CudaRtFrontend::GetOutputString();
  tid = CudaRtFrontend::GetOutputHostPointer<uint3>();
  bid = CudaRtFrontend::GetOutputHostPointer<uint3>();
  bDim = CudaRtFrontend::GetOutputHostPointer<dim3>();
  gDim = CudaRtFrontend::GetOutputHostPointer<dim3>();
  wSize = CudaRtFrontend::GetOutputHostPointer<int>();

  CudaRtFrontend::addHost2DeviceFunc((void*)hostFun,deviceFun);
}

extern "C" __host__ void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                           char *deviceAddress,
                                           const char *deviceName, int ext,
                                           int size, int constant, int global) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(
      CudaUtil::MarshalHostPointer(fatCubinHandle));
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(hostVar));
  CudaRtFrontend::AddStringForArguments(deviceAddress);
  CudaRtFrontend::AddStringForArguments(deviceName);
  CudaRtFrontend::AddVariableForArguments(ext);
  CudaRtFrontend::AddVariableForArguments(size);
  CudaRtFrontend::AddVariableForArguments(constant);
  CudaRtFrontend::AddVariableForArguments(global);
  CudaRtFrontend::Execute("cudaRegisterVar");
}

extern "C" __host__ void __cudaRegisterShared(void **fatCubinHandle,
                                              void **devicePtr) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(
      CudaUtil::MarshalHostPointer(fatCubinHandle));
  CudaRtFrontend::AddStringForArguments((char *)devicePtr);
  CudaRtFrontend::Execute("cudaRegisterShared");
}

extern "C" __host__ void __cudaRegisterSharedVar(void **fatCubinHandle,
                                                 void **devicePtr, size_t size,
                                                 size_t alignment,
                                                 int storage) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(
      CudaUtil::MarshalHostPointer(fatCubinHandle));
  CudaRtFrontend::AddStringForArguments((char *)devicePtr);
  CudaRtFrontend::AddVariableForArguments(size);
  CudaRtFrontend::AddVariableForArguments(alignment);
  CudaRtFrontend::AddVariableForArguments(storage);
  CudaRtFrontend::Execute("cudaRegisterSharedVar");
}

extern "C" __host__ int __cudaSynchronizeThreads(void **x, void *y) {
  // FIXME: implement
  std::cerr << "*** Error: __cudaSynchronizeThreads() not yet implemented!"
            << std::endl;
  return 0;
}

#if CUDA_VERSION >= 9020
    extern "C" __host__ __device__ unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
        CudaRtFrontend::Prepare();
        CudaRtFrontend::AddVariableForArguments(gridDim);
        CudaRtFrontend::AddVariableForArguments(blockDim);
        CudaRtFrontend::AddVariableForArguments(sharedMem);
        CudaRtFrontend::AddDevicePointerForArguments(stream);

        CudaRtFrontend::Execute("cudaPushCallConfiguration");

        return CudaRtFrontend::GetExitCode();
    }


    extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim,
                                                                dim3 *blockDim,
                                                                size_t *sharedMem,
                                                                cudaStream_t *stream) {
        CudaRtFrontend::Prepare();

        CudaRtFrontend::Execute("cudaPopCallConfiguration");

        *gridDim = CudaRtFrontend::GetOutputVariable<dim3>();
        *blockDim = CudaRtFrontend::GetOutputVariable<dim3>();
        *sharedMem = CudaRtFrontend::GetOutputVariable<size_t>();
        cudaStream_t stream1 = CudaRtFrontend::GetOutputVariable<cudaStream_t>();

        memcpy(stream, &stream1, sizeof(cudaStream_t));
        return CudaRtFrontend::GetExitCode();
    }
#endif
