
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
 * Written By: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            School of Computer Science, University College Dublin
 */

#include <CudaRt_internal.h>
#include <lz4.h>

#include <cstdio>

#include "CudaRt.h"

// Helper: allocate and copy section headers table
Elf64_Shdr *copySectionHeaders(const Elf64_Ehdr *eh) {
    // cout << "Will malloc "<< eh->e_shnum << " section headers of " << eh->e_shentsize << " bytes
    // each." << endl;
    Elf64_Shdr *sh_table = (Elf64_Shdr *)malloc(eh->e_shentsize * eh->e_shnum);
    if (!sh_table) return nullptr;

    byte *baseAddr = (byte *)eh;
    for (uint32_t i = 0; i < eh->e_shnum; i++) {
        // cout << "Section header " << i << " starts at adress: " << hex << (baseAddr + eh->e_shoff
        // + i * eh->e_shentsize) << dec << endl;
        Elf64_Shdr *src = (Elf64_Shdr *)(baseAddr + eh->e_shoff + i * eh->e_shentsize);
        memcpy(&sh_table[i], src, eh->e_shentsize);
    }
    return sh_table;
}

// Helper: allocate and copy section header string table
char *copySectionHeaderStrTable(const Elf64_Ehdr *eh, Elf64_Shdr *sh_table) {
    uint8_t *sh_str_table_bytes = (uint8_t *)&sh_table[eh->e_shstrndx];
    // for (int i = 0; i < sizeof(Elf64_Shdr); i++) {
    //     printf("%02x ", sh_str_table_bytes[i]);
    // }
    // printf("\n");

    size_t offset = sh_table[eh->e_shstrndx].sh_offset;
    size_t size = sh_table[eh->e_shstrndx].sh_size;
    // cout << "sh_table address: " << hex << (void *)sh_table << dec << endl;
    // cout << "offset address: " << hex << (void *)(&sh_table[eh->e_shstrndx].sh_offset) << dec <<
    // endl; cout << "size address: " << hex << (void *)(&sh_table[eh->e_shstrndx].sh_size) << dec
    // << endl; cout << "section header string table index: " << eh->e_shstrndx << " has offset: "
    // << offset << " and size: " << size << endl;
    char *sh_str = (char *)malloc(size);
    if (!sh_str) return nullptr;

    byte *baseAddr = (byte *)eh;
    memcpy(sh_str, baseAddr + offset, size);
    return sh_str;
}

// Helper: parse NvInfo sections and register functions and their parameters
void parseNvInfoKParams(const Elf64_Ehdr *eh, Elf64_Shdr *sh_table, char *sh_str) {
    byte *baseAddr = (byte *)eh;
    // cout << "Processing " << eh->e_shnum << " sections." << endl;
    for (uint32_t i = 0; i < eh->e_shnum; i++) {
        // cout << "Processing section " << i + 1 << " of " << eh->e_shnum << endl;
        char *sectionName = sh_str + sh_table[i].sh_name;
        if (strncmp(".nv.info.", sectionName, strlen(".nv.info.")) != 0) {
            // cout << "Skipping section: " << sectionName << endl;
            continue;
        }

        char *funcName = sectionName + strlen(".nv.info.");
        // cout << "Found NvInfo section: " << funcName << endl;
        byte *sectionData = baseAddr + sh_table[i].sh_offset;

        NvInfoFunction infoFunction;

        NvInfoAttribute *pAttr = (NvInfoAttribute *)sectionData;
        byte *sectionEnd = sectionData + sh_table[i].sh_size;

        // cout << "Section data start at: " << hex << sectionData << " and end at: " << sectionEnd
        // << dec << endl;

        while ((byte *)pAttr < sectionEnd) {
            size_t size = sizeof(NvInfoAttribute);
            // cout << "Processing attribute: " << pAttr->attr << ", fmt: " << pAttr->fmt << ",
            // value: " << pAttr->value << endl;
            if (pAttr->fmt == EIFMT_SVAL) {
                // cout << "Attribute is a string value." << endl;
                size += pAttr->value;
            }
            if (pAttr->attr == EIATTR_KPARAM_INFO) {
                // cout << "Attribute is a KParam info." << endl;
                NvInfoKParam *nvInfoKParam = (NvInfoKParam *)pAttr;
                infoFunction.params.push_back(*nvInfoKParam);
                // cout << nvInfoKParam->index << ", "
                //      << nvInfoKParam->ordinal << ", "
                //      << nvInfoKParam->offset << ", "
                //      << nvInfoKParam->log_alignment() << ", "
                //      << nvInfoKParam->space() << ", "
                //      << nvInfoKParam->cbank() << ", "
                //      << nvInfoKParam->is_cbank() << ", "
                //      << nvInfoKParam->size_bytes() << endl;
            }
            pAttr = (NvInfoAttribute *)((byte *)pAttr + size);
        }
        CudaRtFrontend::addDeviceFunc2InfoFunc(funcName, infoFunction);
    }
}

void writeCudaFatBinaryToFile(const void *data, const unsigned long long int fatBinSize,
                              const std::string &filename) {
    FILE *file = fopen(filename.c_str(), "rb");
    if (file) {
        // File already exists, skip writing
        fclose(file);
        return;
    }
    file = fopen(filename.c_str(), "wb");
    if (!file) {
        perror("Failed to open file for writing");
        return;
    }
    size_t written = fwrite(data, 1, fatBinSize, file);
    if (written != fatBinSize) {
        perror("Failed to write fat binary to file");
    }
    fclose(file);
}

/*
 Routines not found in the cuda's header files.
 KEEP THEM WITH CARE
 */

extern "C" __host__ void **__cudaRegisterFatBinary(void *fatCubin) {
    /* Fake host pointer */
    __fatBinC_Wrapper_t *bin = (__fatBinC_Wrapper_t *)fatCubin;
    if (bin->magic != FATBINWRAPPER_MAGIC) {
        cerr << "*** Error: Invalid fat binary magic number" << endl;
        return nullptr;  // Not a valid fat binary
    }
    // cout << "Fat binary wrapper magic: " << hex << bin->magic << endl;
    struct fatBinaryHeader *fatBinHdr = (struct fatBinaryHeader *)bin->data;
    if (fatBinHdr->magic != FATBIN_MAGIC || fatBinHdr->version != 1) {
        cerr << "*** Error: Invalid fat binary" << endl;
        return nullptr;  // Not a valid fat binary
    }
    // cout << "Fat binary header size: " << fatBinHdr->headerSize << endl;
    // cout << "Fat binary size: " << fatBinHdr->fatSize << endl;

    // only for debugging purposes
    // writeCudaFatBinaryToFile(fatBinHdr, fatBinHdr->headerSize + fatBinHdr->fatSize,
    // "fat_binary.cubin");

    uint8_t *data_ptr = (uint8_t *)bin->data + fatBinHdr->headerSize;
    size_t remaining_size = fatBinHdr->fatSize;

    std::vector<char> cubin;
    while (remaining_size > 0) {
        fatBinData_t *fatBinData = (fatBinData_t *)data_ptr;
        if (fatBinData->version != 0x0101 || (fatBinData->kind != 1 && fatBinData->kind != 2)) {
            cerr << "*** Error: Invalid fat binary data version or kind" << endl;
            return nullptr;  // Not a valid fat binary data
        }

        // cout << "Processing fat binary data of kind: " << fatBinData->kind
        //     << " and smVersion: " << fatBinData->smVersion << endl;

        data_ptr += fatBinData->headerSize;

        if (fatBinData->uncompressedPayload != 0) {
            const char *compressed_data = (char *)data_ptr;
            int compressed_size = fatBinData->payloadSize;
            data_ptr += fatBinData->paddedPayloadSize;

            // Prepare output buffer with the expected decompressed size
            cubin.resize(fatBinData->uncompressedPayload);

            // Decompress - LZ4_decompress_safe returns decompressed size or < 0 on error
            int decompressed_size = LZ4_decompress_safe(
                compressed_data, cubin.data(), compressed_size, fatBinData->uncompressedPayload);

            if (decompressed_size < 0) {
                cerr << "*** Error: LZ4 decompression failed with code " << decompressed_size
                     << endl;
                return nullptr;  // Decompression failed
            }
        } else {
            cubin.resize(fatBinData->paddedPayloadSize);
            memcpy(cubin.data(), data_ptr, fatBinData->paddedPayloadSize);
            data_ptr += fatBinData->paddedPayloadSize;
        }

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

            parseNvInfoKParams(eh, sh_table, sh_str);

            free(sh_str);
            free(sh_table);
        }
        remaining_size -= (fatBinData->headerSize + fatBinData->paddedPayloadSize);
    }

    Buffer *input_buffer = new Buffer();
    input_buffer->AddString(CudaUtil::MarshalHostPointer((void **)bin));
    input_buffer = CudaUtil::MarshalFatCudaBinary(bin, input_buffer);

    CudaRtFrontend::Prepare();
    CudaRtFrontend::Execute("cudaRegisterFatBinary", input_buffer);
    if (CudaRtFrontend::Success()) return (void **)fatCubin;

    return nullptr;
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

extern "C" __host__ void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                                                char *deviceFun, const char *deviceName,
                                                int thread_limit, uint3 *tid, uint3 *bid,
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

    CudaRtFrontend::addHost2DeviceFunc((void *)hostFun, deviceFun);
}

extern "C" __host__ void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                           char *deviceAddress, const char *deviceName, int ext,
                                           int size, int constant, int global) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    CudaRtFrontend::AddStringForArguments(hostVar);
    CudaRtFrontend::AddStringForArguments(deviceAddress);
    CudaRtFrontend::AddStringForArguments(deviceName);
    CudaRtFrontend::AddVariableForArguments(ext);
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::AddVariableForArguments(constant);
    CudaRtFrontend::AddVariableForArguments(global);
    // cout << "RegisterVar: fatCubinHandle: " << fatCubinHandle
    //      << ", hostVar: " << hostVar
    //      << ", deviceAddress: " << deviceAddress
    //      << ", deviceName: " << deviceName
    //      << ", ext: " << ext
    //      << ", size: " << size
    //      << ", constant: " << constant
    //      << ", global: " << global << endl;
    CudaRtFrontend::Execute("cudaRegisterVar");
}

extern "C" __host__ void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    CudaRtFrontend::AddStringForArguments((char *)devicePtr);
    CudaRtFrontend::Execute("cudaRegisterShared");
}

extern "C" __host__ void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
                                                 size_t size, size_t alignment, int storage) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    CudaRtFrontend::AddStringForArguments((char *)devicePtr);
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::AddVariableForArguments(alignment);
    CudaRtFrontend::AddVariableForArguments(storage);
    CudaRtFrontend::Execute("cudaRegisterSharedVar");
}

extern "C" __host__ int __cudaSynchronizeThreads(void **x, void *y) {
    // FIXME: implement
    std::cerr << "*** Error: __cudaSynchronizeThreads() not yet implemented!" << std::endl;
    return 0;
}

extern "C" __host__ __device__ unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim,
                                                                              dim3 blockDim,
                                                                              size_t sharedMem,
                                                                              cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(gridDim);
    CudaRtFrontend::AddVariableForArguments(blockDim);
    CudaRtFrontend::AddVariableForArguments(sharedMem);
    CudaRtFrontend::AddDevicePointerForArguments(stream);

    CudaRtFrontend::Execute("cudaPushCallConfiguration");

    return CudaRtFrontend::GetExitCode();
}

extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
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
