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
 * Edited by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             School of ComputerUniversity College Dublin
 *
 */

#include <CudaRt_internal.h>
#include <CudaUtil.h>
#include <cuda.h>
#include <lz4.h>

#include "CudaRtHandler.h"

using namespace std;
using namespace log4cplus;

extern "C" {
void **__cudaRegisterFatBinary(void *fatCubin);
void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
void __cudaUnregisterFatBinary(void **fatCubinHandle);
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize);
void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                       const char *deviceName, int ext, int size, int constant, int global);
void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr, size_t size, size_t alignment,
                             int storage);
void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr);
cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem,
                                                 cudaStream_t *stream);
__host__ __device__ unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                                   size_t sharedMem,
                                                                   cudaStream_t stream);
}

// Helper: allocate and copy section headers table
Elf64_Shdr *copySectionHeaders(const Elf64_Ehdr *eh) {
    Elf64_Shdr *sh_table = (Elf64_Shdr *)malloc(eh->e_shentsize * eh->e_shnum);
    if (!sh_table) return nullptr;

    byte *baseAddr = (byte *)eh;
    for (uint32_t i = 0; i < eh->e_shnum; i++) {
        Elf64_Shdr *src = (Elf64_Shdr *)(baseAddr + eh->e_shoff + i * eh->e_shentsize);
        memcpy(&sh_table[i], src, eh->e_shentsize);
    }
    return sh_table;
}

// Helper: allocate and copy section header string table
char *copySectionHeaderStrTable(const Elf64_Ehdr *eh, Elf64_Shdr *sh_table) {
    // uint8_t *sh_str_table_bytes = (uint8_t *)&sh_table[eh->e_shstrndx];
    // for (int i = 0; i < sizeof(Elf64_Shdr); i++) {
    //     printf("%02x ", sh_str_table_bytes[i]);
    // }
    // printf("\n");

    size_t offset = sh_table[eh->e_shstrndx].sh_offset;
    size_t size = sh_table[eh->e_shstrndx].sh_size;
    // cout << "sh_table address: " << hex << (void *)sh_table << dec << endl;
    // cout << "offset address: " << hex << (void
    // *)(&sh_table[eh->e_shstrndx].sh_offset) << dec << endl; cout << "size
    // address: " << hex << (void *)(&sh_table[eh->e_shstrndx].sh_size) << dec
    // << endl; cout << "section header string table index: " << eh->e_shstrndx
    // << " has offset: " << offset << " and size: " << size << endl;
    char *sh_str = (char *)malloc(size);
    if (!sh_str) return nullptr;

    byte *baseAddr = (byte *)eh;
    memcpy(sh_str, baseAddr + offset, size);
    return sh_str;
}

// Helper: parse NvInfo sections and register functions
void parseNvInfoSections(const Elf64_Ehdr *eh, Elf64_Shdr *sh_table, char *sh_str,
                         CudaRtHandler *pThis) {
    byte *baseAddr = (byte *)eh;
    for (uint32_t i = 0; i < eh->e_shnum; i++) {
        char *sectionName = sh_str + sh_table[i].sh_name;
        if (strncmp(".nv.info.", sectionName, strlen(".nv.info.")) != 0) {
            continue;
        }

        char *funcName = sectionName + strlen(".nv.info.");
        byte *sectionData = baseAddr + sh_table[i].sh_offset;

        NvInfoFunction infoFunction;

        NvInfoAttribute *pAttr = (NvInfoAttribute *)sectionData;
        byte *sectionEnd = sectionData + sh_table[i].sh_size;

        while ((byte *)pAttr < sectionEnd) {
            size_t size = sizeof(NvInfoAttribute);
            if (pAttr->fmt == EIFMT_SVAL) {
                size += pAttr->value;
            }
            if (pAttr->attr == EIATTR_KPARAM_INFO) {
                NvInfoKParam *nvInfoKParam = (NvInfoKParam *)pAttr;
                infoFunction.params.push_back(*nvInfoKParam);
            }
            pAttr = (NvInfoAttribute *)((byte *)pAttr + size);
        }
        pThis->addDeviceFunc2InfoFunc(funcName, infoFunction);
    }
}

CUDA_ROUTINE_HANDLER(RegisterFatBinary) {
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "Entering in RegisterFatBinary");

    try {
        char *handler = input_buffer->AssignString();
        __fatBinC_Wrapper_t *fatBin = CudaUtil::UnmarshalFatCudaBinary(input_buffer.get());
        if (fatBin->magic != FATBINWRAPPER_MAGIC) {
            LOG4CPLUS_ERROR(pThis->GetLogger(),
                            "*** Error: Invalid fat binary wrapper magic number");
            return std::make_shared<Result>(cudaErrorInvalidValue);
        }
        void **bin = __cudaRegisterFatBinary((void *)fatBin);
        pThis->RegisterFatBinary(handler, bin);

        struct fatBinaryHeader *fatBinHdr = (struct fatBinaryHeader *)fatBin->data;
        if (fatBinHdr->magic != FATBIN_MAGIC) {
            LOG4CPLUS_ERROR(pThis->GetLogger(),
                            "*** Error: Invalid fat binary header magic number");
            return std::make_shared<Result>(cudaErrorInvalidValue);
        }

        // cout << "Fat binary header size: " << fatBinHdr->headerSize << endl;
        // cout << "Fat binary size: " << fatBinHdr->fatSize << endl;

        uint8_t *data_ptr = (uint8_t *)fatBin->data + fatBinHdr->headerSize;
        size_t remaining_size = fatBinHdr->fatSize;
        std::vector<char> cubin;
        while (remaining_size > 0) {
            fatBinData_t *fatBinData = (fatBinData_t *)data_ptr;
            data_ptr += fatBinData->headerSize;

            if (fatBinData->uncompressedPayload != 0) {
                uint8_t *compressed_data = data_ptr;
                int compressed_size = fatBinData->payloadSize;

                // cout << "Uncompressed payload: " <<
                // fatBinData->uncompressedPayload << endl; Prepare output
                // buffer with the expected decompressed size
                cubin.resize(fatBinData->uncompressedPayload);

                // Decompress - LZ4_decompress_safe returns decompressed size or
                // < 0 on error
                int decompressed_size =
                    LZ4_decompress_safe((const char *)compressed_data, cubin.data(),
                                        compressed_size, fatBinData->uncompressedPayload);
                if (decompressed_size < 0) {
                    LOG4CPLUS_ERROR(
                        pThis->GetLogger(),
                        "*** Error: LZ4 decompression failed with code " << decompressed_size);
                    return nullptr;  // Decompression failed
                }
                // Advance pointer for next usage (if needed)
                data_ptr += fatBinData->paddedPayloadSize;
            } else {
                cubin.resize(fatBinData->paddedPayloadSize);
                memcpy(cubin.data(), data_ptr, fatBinData->paddedPayloadSize);
                data_ptr += fatBinData->paddedPayloadSize;
            }

            // cout << "kind " << fatBinData->kind << endl;
            if (fatBinData->kind == 2) {
                if (memcmp(cubin.data(), ELF_MAGIC, ELF_MAGIC_SIZE) != 0) {
                    cerr << "*** Error: Invalid ELF magic number in fat binary" << endl;
                    return nullptr;  // Not a valid ELF file
                }
                Elf64_Ehdr *eh = (Elf64_Ehdr *)(cubin.data());

                Elf64_Shdr *sh_table = copySectionHeaders(eh);
                if (!sh_table) return nullptr;

                char *sh_str = copySectionHeaderStrTable(eh, sh_table);
                if (!sh_str) {
                    free(sh_table);
                    return nullptr;
                }

                parseNvInfoSections(eh, sh_table, sh_str, pThis);

                free(sh_str);
                free(sh_table);
            } else {
                // cout << "Ignoring PTX" << endl;
            }
            remaining_size -= (fatBinData->paddedPayloadSize + fatBinData->headerSize);
        }

        return std::make_shared<Result>(cudaSuccess);
    } catch (const std::exception &e) {
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
    } catch (const std::exception &e) {
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
    } catch (const std::exception &e) {
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
        __cudaRegisterFunction(fatCubinHandle, hostfun, deviceFun, deviceName, thread_limit, tid,
                               bid, bDim, gDim, wSize);

#ifdef DEBUG
        cudaError_t error = cudaGetLastError();
        if (error != 0) {
            cerr << "error executing RegisterFunction: " << cudaGetErrorString(error) << endl;
        }
#endif

        std::shared_ptr<Buffer> output_buffer = std::make_shared<Buffer>();

        output_buffer->AddString(deviceFun);
        output_buffer->Add(tid);
        output_buffer->Add(bid);
        output_buffer->Add(bDim);
        output_buffer->Add(gDim);
        output_buffer->Add(wSize);

        pThis->addHost2DeviceFunc((void *)hostfun, deviceFun);

        return std::make_shared<Result>(cudaSuccess, output_buffer);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(RegisterVar) {
    //   LOG4CPLUS_DEBUG(pThis->GetLogger(), "Entering in RegisterVar");
    try {
        char *handler = input_buffer->AssignString();
        // LOG4CPLUS_DEBUG(pThis->GetLogger(), "Handler: " << handler);
        void **fatCubinHandle = pThis->GetFatBinary(handler);
        // LOG4CPLUS_DEBUG(pThis->GetLogger(), "FatCubinHandle: " <<
        // fatCubinHandle);
        char *hostVar = input_buffer->AssignString();
        // LOG4CPLUS_DEBUG(pThis->GetLogger(), "HostVar: " << hostVar);
        char *deviceAddress = strdup(input_buffer->AssignString());
        // LOG4CPLUS_DEBUG(pThis->GetLogger(), "DeviceAddress: " <<
        // deviceAddress);
        const char *deviceName = strdup(input_buffer->AssignString());
        // LOG4CPLUS_DEBUG(pThis->GetLogger(), "DeviceName: " << deviceName);
        int ext = input_buffer->Get<int>();
        int size = input_buffer->Get<int>();
        int constant = input_buffer->Get<int>();
        int global = input_buffer->Get<int>();
        // LOG4CPLUS_DEBUG(pThis->GetLogger(), "Calling RegisterVar: " <<
        // deviceName << " for "
        //                   << fatCubinHandle << " with hostVar: " << hostVar
        //                   << ", deviceAddress: " << deviceAddress
        //                   << ", deviceName: " << deviceName
        //                   << ", ext: " << ext << ", size: " << size
        //                   << ", constant: " << constant
        //                   << ", global: " << global);
        __cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant,
                          global);
        cudaError_t error = cudaGetLastError();
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "RegisterVar Executed");
        if (error != cudaSuccess) {
            LOG4CPLUS_DEBUG(pThis->GetLogger(),
                            "error executing RegisterVar: " << cudaGetErrorString(error));
        }
    } catch (const std::exception &e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "Exception" << e.what() << " in RegisterVar");
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
        __cudaRegisterSharedVar(fatCubinHandle, devicePtr, size, alignment, storage);
#ifdef DEBUG
        cout << "Registered SharedVar " << (char *)devicePtr << endl;
        cudaError_t error = cudaGetLastError();
        if (error != 0)
            cerr << "error executing RegisterSharedVar: " << cudaGetErrorString(error) << endl;
#endif
    } catch (const std::exception &e) {
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
        cout << "Registerd Shared " << (char *)devPtr << " for " << fatCubinHandle << endl;
        cudaError_t error = cudaGetLastError();
        if (error != 0) {
            cerr << "error executing RegisterSharedVar: " << cudaGetErrorString(error) << endl;
        }
#endif

    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(cudaSuccess);
}

CUDA_ROUTINE_HANDLER(PushCallConfiguration) {
    try {
        dim3 gridDim = input_buffer->Get<dim3>();
        dim3 blockDim = input_buffer->Get<dim3>();
        size_t sharedMem = input_buffer->Get<size_t>();
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaError_t exit_code = static_cast<cudaError_t>(
            __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream));
        return std::make_shared<Result>(exit_code);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(PopCallConfiguration) {
    try {
        dim3 gridDim, blockDim;
        size_t sharedMem;
        cudaStream_t stream;
        cudaError_t exit_code = static_cast<cudaError_t>(
            __cudaPopCallConfiguration(&gridDim, &blockDim, &sharedMem, &stream));
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->Add(gridDim);
        out->Add(blockDim);
        out->AddMarshal(sharedMem);
        out->AddMarshal(stream);

        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception &e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}
