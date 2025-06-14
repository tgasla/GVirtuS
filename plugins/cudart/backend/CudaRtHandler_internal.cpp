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
 */

#include <CudaUtil.h>

#include "CudaRtHandler.h"
#include <cuda.h>

using namespace std;
using namespace log4cplus;

typedef struct __cudaFatCudaBinary2HeaderRec {
    unsigned int magic;
    unsigned int version;
    unsigned long long int length;
} __cudaFatCudaBinary2Header;

enum FatBin2EntryType { FATBIN_2_PTX = 0x1 };

typedef struct __cudaFatCudaBinary2EntryRec {
    unsigned int type;
    unsigned int binary;
    unsigned long long int binarySize;
    unsigned int unknown2;
    unsigned int kindOffset;
    unsigned int unknown3;
    unsigned int unknown4;
    unsigned int name;
    unsigned int nameSize;
    unsigned long long int flags;
    unsigned long long int unknown7;
    unsigned long long int uncompressedBinarySize;
} __cudaFatCudaBinary2Entry;

long long COMPRESSED_PTX = 0x0000000000001000LL;

typedef struct __cudaFatCudaBinaryRec2 {
    int magic;
    int version;
    const unsigned long long *fatbinData;
    char *f;
} __cudaFatCudaBinary2;

extern "C" {
    void **__cudaRegisterFatBinary(void *fatCubin);
    void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
    void __cudaUnregisterFatBinary(void **fatCubinHandle);
    void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                                        char *deviceFun, const char *deviceName,
                                        int thread_limit, uint3 *tid, uint3 *bid,
                                        dim3 *bDim, dim3 *gDim, int *wSize);
    void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                    char *deviceAddress, const char *deviceName,
                                    int ext, int size, int constant, int global);
    void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
                                        size_t size, size_t alignment, int storage);
    void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr);
    cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim,
                                                            dim3 *blockDim,
                                                            size_t *sharedMem,
                                                            cudaStream_t *stream);
    __host__ __device__ unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim,
                                                                            dim3 blockDim,
                                                                            size_t sharedMem,
                                                                            cudaStream_t stream);
}

static bool initialized = false;
static char **constStrings;
static size_t constStrings_size = 0;
static size_t constStrings_length = 0;

static void init() {
    initialized = true;
}

const char *get_const_string(const char *s) {
    if (!initialized) init();
    size_t i;
    for (i = 0; i < constStrings_length; i++)
        if (!strcmp(s, constStrings[i])) return constStrings[i];
    if (i >= constStrings_size) {
        constStrings_size += 2048;
        constStrings =
            (char **)realloc(constStrings, sizeof(char *) * constStrings_size);
    }
    constStrings[constStrings_length] = strdup(s);
    return constStrings[constStrings_length++];
}

CUDA_ROUTINE_HANDLER(RegisterFatBinary) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RegisterFatBinary"));
    LOG4CPLUS_DEBUG(logger, "Entering in RegisterFatBinary");

  try {
    char *handler = input_buffer->AssignString();
    __fatBinC_Wrapper_t *fatBin =
        CudaUtil::UnmarshalFatCudaBinaryV2(input_buffer.get());
    void **bin = __cudaRegisterFatBinary((void *)fatBin);
    pThis->RegisterFatBinary(handler, bin);

      char *data = (char *)fatBin->data;

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
                      pThis->addDeviceFunc2InfoFunc(szFuncName, infoFunction);
                  }
              }
              free(sh_str);
          }
          free(sh_table);
      }

#ifdef DEBUG
    cudaError_t error = cudaGetLastError();
    if (error != 0) {
      cerr << "error executing RegisterFatBinary: " << _cudaGetErrorEnum(error)
           << endl;
    }
#endif
    return std::make_shared<Result>(cudaSuccess);
  } catch (const std::exception& e) {
      cerr << e.what() << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(RegisterFatBinaryEnd) {
  try {
    char *handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    __cudaRegisterFatBinaryEnd(fatCubinHandle);
    cudaError_t error = cudaGetLastError();
    return std::make_shared<Result>(error);
  } catch (const std::exception& e) {
      cerr << e.what() << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }

}

CUDA_ROUTINE_HANDLER(UnregisterFatBinary) {
  try {
    char *handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    __cudaUnregisterFatBinary(fatCubinHandle);
    pThis->UnregisterFatBinary(handler);
    return std::make_shared<Result>(cudaSuccess);
  } catch (const std::exception& e) {
      cerr << e.what() << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(RegisterFunction) {
  try {
    char *handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    const char *hostfun = (const char *)(input_buffer->Get<pointer_t>());
    char *deviceFun = strdup(input_buffer->AssignString());
    const char *deviceName = strdup(input_buffer->AssignString());
    int thread_limit = input_buffer->Get<int>();
    uint3 *tid = input_buffer->Assign<uint3>();
    uint3 *bid = input_buffer->Assign<uint3>();
    dim3 *bDim = input_buffer->Assign<dim3>();
    dim3 *gDim = input_buffer->Assign<dim3>();
    int *wSize = input_buffer->Assign<int>();
    __cudaRegisterFunction(fatCubinHandle, hostfun, deviceFun, deviceName,
                           thread_limit, tid, bid, bDim, gDim, wSize);

#ifdef DEBUG
    cudaError_t error = cudaGetLastError();
    if (error != 0) {
      cerr << "error executing RegisterFunction: " << _cudaGetErrorEnum(error)
           << endl;
    }
#endif

    std::shared_ptr<Buffer> output_buffer = std::make_shared<Buffer>();

    output_buffer->AddString(deviceFun);
    output_buffer->Add(tid);
    output_buffer->Add(bid);
    output_buffer->Add(bDim);
    output_buffer->Add(gDim);
    output_buffer->Add(wSize);

    pThis->addHost2DeviceFunc((void *) hostfun, deviceFun);

    return std::make_shared<Result>(cudaSuccess, output_buffer);
  } catch (const std::exception& e) {
      cerr << e.what() << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(RegisterVar) {
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RegisterVar"));
  LOG4CPLUS_DEBUG(logger, "Entering in RegisterVar");
  try {
    char *handler = input_buffer->AssignString();
    LOG4CPLUS_DEBUG(logger, "Handler: " << handler);
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    LOG4CPLUS_DEBUG(logger, "FatCubinHandle: " << fatCubinHandle);
    char *hostVar = input_buffer->AssignString();
    LOG4CPLUS_DEBUG(logger, "HostVar: " << hostVar);
    char *deviceAddress = strdup(input_buffer->AssignString());
    LOG4CPLUS_DEBUG(logger, "DeviceAddress: " << deviceAddress);
    const char *deviceName = strdup(input_buffer->AssignString());
    LOG4CPLUS_DEBUG(logger, "DeviceName: " << deviceName);
    int ext = input_buffer->Get<int>();
    int size = input_buffer->Get<int>();
    int constant = input_buffer->Get<int>();
    int global = input_buffer->Get<int>();
    LOG4CPLUS_DEBUG(logger, "Calling RegisterVar: " << deviceName << " for "
                      << fatCubinHandle << " with hostVar: " << hostVar
                      << ", deviceAddress: " << deviceAddress
                      << ", deviceName: " << deviceName
                      << ", ext: " << ext << ", size: " << size
                      << ", constant: " << constant
                      << ", global: " << global);
    __cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext,
                      size, constant, global);
    cudaError_t error = cudaGetLastError();
    LOG4CPLUS_DEBUG(logger, "RegisterVar Executed");
    if (error != cudaSuccess) {
      LOG4CPLUS_DEBUG(logger, "error executing RegisterVar: " << _cudaGetErrorEnum(error));
    }
  } catch (const std::exception& e) {
      LOG4CPLUS_DEBUG(logger, "Exception" << e.what() << " in RegisterVar");
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }

  return std::make_shared<Result>(cudaSuccess);
}

CUDA_ROUTINE_HANDLER(RegisterSharedVar) {
  try {
    char *handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    void **devicePtr = (void **)input_buffer->AssignString();
    size_t size = input_buffer->Get<size_t>();
    size_t alignment = input_buffer->Get<size_t>();
    int storage = input_buffer->Get<int>();
    __cudaRegisterSharedVar(fatCubinHandle, devicePtr, size, alignment,
                            storage);
#ifdef DEBUG
    cout << "Registered SharedVar " << (char *)devicePtr << endl;
    cudaError_t error = cudaGetLastError();
    if (error != 0)
      cerr << "error executing RegisterSharedVar: " << _cudaGetErrorEnum(error) << endl;
#endif

  } catch (const std::exception& e) {
      cerr << e.what() << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }

  return std::make_shared<Result>(cudaSuccess);
}

CUDA_ROUTINE_HANDLER(RegisterShared) {
  try {
    char *handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    char *devPtr = strdup(input_buffer->AssignString());
    __cudaRegisterShared(fatCubinHandle, (void **)devPtr);

#ifdef DEBUG
    cout << "Registerd Shared " << (char *)devPtr << " for " << fatCubinHandle
         << endl;
    cudaError_t error = cudaGetLastError();
    if (error != 0) {
      cerr << "error executing RegisterSharedVar: " << _cudaGetErrorEnum(error)
           << endl;
    }
#endif

  } catch (const std::exception& e) {
      cerr << e.what() << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }

  return std::make_shared<Result>(cudaSuccess);
}

#if (CUDART_VERSION >= 9020)
    CUDA_ROUTINE_HANDLER(PushCallConfiguration) {
        try {
            dim3 gridDim = input_buffer->Get<dim3>();
            dim3 blockDim = input_buffer->Get<dim3>();
            size_t sharedMem = input_buffer->Get<size_t>();
            cudaStream_t stream = input_buffer->Get<cudaStream_t>();
            cudaError_t exit_code = static_cast<cudaError_t>(__cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream));
            return std::make_shared<Result>(exit_code);
        } catch (const std::exception& e) {
      cerr << e.what() << endl;
            return std::make_shared<Result>(cudaErrorMemoryAllocation);
        }

    }

    CUDA_ROUTINE_HANDLER(PopCallConfiguration) {
        try {
            dim3 gridDim,blockDim;
            size_t sharedMem;
            cudaStream_t stream;
            cudaError_t exit_code = static_cast<cudaError_t>(__cudaPopCallConfiguration(&gridDim, &blockDim, &sharedMem, &stream));
            std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

            out->Add(gridDim);
            out->Add(blockDim);
            out->AddMarshal(sharedMem);
            out->AddMarshal(stream);

            return std::make_shared<Result>(exit_code, out);
        } catch (const std::exception& e) {
      cerr << e.what() << endl;
            return std::make_shared<Result>(cudaErrorMemoryAllocation);
        }

    }

#endif
