//
// Created by Raffaele Montella on 06/11/21.
// Edited by Theodoros Aslanidis on 22/06/25.
//

#ifndef GVIRTUS_CUDART_INTERNAL_H
#define GVIRTUS_CUDART_INTERNAL_H

#include <driver_types.h>
#include <elf.h>

#include <string>
#include <vector>

// defined in <cuda_install_dir>/include/fatbinary_section.h
// typedef struct {
//   int magic;
//   int version;
//   const unsigned long long* data;
//   void *filename_or_fatbins;  /* version 1: offline filename,
//                                * version 2: array of prelinked fatbins */
// } __fatBinC_Wrapper_t;

// defined in <gvirtus_src_dir>/plugins/cudart/util/CudaUtil.h
//   struct __align__(8) fatBinaryHeader {
//       unsigned int           magic;
//       unsigned short         version;
//       unsigned short         headerSize;
//       unsigned long long int fatSize;
//   };

// right after the fatBinaryHeader comes the fatBinData_t

typedef struct {
    unsigned short kind;
    unsigned short version;
    unsigned int headerSize;  // size of the header
    unsigned int paddedPayloadSize;
    unsigned int unknown0;
    unsigned int payloadSize;
    unsigned int unknown1;
    unsigned int unknown2;
    unsigned int smVersion;
    unsigned int bitWidth;
    unsigned int unknown3;
    unsigned long unkown4;
    unsigned long unknown5;
    unsigned long uncompressedPayload;
} fatBinData_t;

#define FATBINWRAPPER_MAGIC 0x466243B1
#define FATBIN_MAGIC 0xBA55ED50
#define ELF_MAGIC "\177ELF"
#define ELF_MAGIC_SIZE 4

#define EIFMT_NVAL 0x01
#define EIFMT_BVAL 0x02
#define EIFMT_HVAL 0x03
#define EIFMT_SVAL 0x04

#define EIATTR_ERROR 0x00
#define EIATTR_PAD 0x01
#define EIATTR_IMAGE_SLOT 0x02
#define EIATTR_JUMPTABLE_RELOCS 0x03
#define EIATTR_CTAIDZ_USED 0x04
#define EIATTR_MAX_THREADS 0x05
#define EIATTR_IMAGE_OFFSET 0x06
#define EIATTR_IMAGE_SIZE 0x07
#define EIATTR_TEXTURE_NORMALIZED 0x08
#define EIATTR_SAMPLER_INIT 0x09
#define EIATTR_PARAM_CBANK 0x0a
#define EIATTR_SMEM_PARAM_OFFSETS 0x0b
#define EIATTR_CBANK_PARAM_OFFSETS 0x0c
#define EIATTR_SYNC_STACK 0x0d
#define EIATTR_TEXID_SAMPID_MAP 0x0e
#define EIATTR_EXTERNS 0x0f
#define EIATTR_REQNTID 0x10
#define EIATTR_FRAME_SIZE 0x11
#define EIATTR_MIN_STACK_SIZE 0x12
#define EIATTR_SAMPLER_FORCE_UNNORMALIZED 0x13
#define EIATTR_BINDLESS_IMAGE_OFFSETS 0x14
#define EIATTR_BINDLESS_TEXTURE_BANK 0x15
#define EIATTR_BINDLESS_SURFACE_BANK 0x16
#define EIATTR_KPARAM_INFO 0x17
#define EIATTR_SMEM_PARAM_SIZE 0x18
#define EIATTR_CBANK_PARAM_SIZE 0x19
#define EIATTR_QUERY_NUMATTRIB 0x1a
#define EIATTR_MAXREG_COUNT 0x1b
#define EIATTR_EXIT_INSTR_OFFSETS 0x1c
#define EIATTR_S2RCTAID_INSTR_OFFSETS 0x1d
#define EIATTR_CRS_STACK_SIZE 0x1e
#define EIATTR_NEED_CNP_WRAPPER 0x1f
#define EIATTR_NEED_CNP_PATCH 0x20
#define EIATTR_EXPLICIT_CACHING 0x21
#define EIATTR_ISTYPEP_USED 0x22
#define EIATTR_MAX_STACK_SIZE 0x23
#define EIATTR_SUQ_USED 0x24
#define EIATTR_LD_CACHEMOD_INSTR_OFFSETS 0x25
#define EIATTR_LOAD_CACHE_REQUEST 0x26
#define EIATTR_ATOM_SYS_INSTR_OFFSETS 0x27
#define EIATTR_COOP_GROUP_INSTR_OFFSETS 0x28
#define EIATTR_COOP_GROUP_MAX_REGIDS 0x29
#define EIATTR_SW1850030_WAR 0x2a
#define EIATTR_WMMA_USED 0x2b
#define EIATTR_HAS_PRE_V10_OBJECT 0x2c
#define EIATTR_ATOMF16_EMUL_INSTR_OFFSETS 0x2d
#define EIATTR_ATOM16_EMUL_INSTR_REG_MAP 0x2e
#define EIATTR_REGCOUNT 0x2f
#define EIATTR_SW2393858_WAR 0x30
#define EIATTR_INT_WARP_WIDE_INSTR_OFFSETS 0x31
#define EIATTR_SHARED_SCRATCH 0x32
#define EIATTR_STATISTICS 0x33

// New between cuda 10.2 and 11.6
#define EIATTR_INDIRECT_BRANCH_TARGETS 0x34
#define EIATTR_SW2861232_WAR 0x35
#define EIATTR_SW_WAR 0x36
#define EIATTR_CUDA_API_VERSION 0x37
#define EIATTR_NUM_MBARRIERS 0x38
#define EIATTR_MBARRIER_INSTR_OFFSETS 0x39
#define EIATTR_COROUTINE_RESUME_ID_OFFSETS 0x3a
#define EIATTR_SAM_REGION_STACK_SIZE 0x3b
#define EIATTR_PER_REG_TARGET_PERF_STATS 0x3c

// New between cuda 11.6 and 11.8
#define EIATTR_CTA_PER_CLUSTER 0x3d
#define EIATTR_EXPLICIT_CLUSTER 0x3e
#define EIATTR_MAX_CLUSTER_RANK 0x3f
#define EIATTR_INSTR_REG_MAP 0x40

// original gvirtus ones
typedef struct {
    uint8_t fmt;
    uint8_t attr;
    uint16_t value;
} NvInfoAttribute;

// see: https://github.com/VivekPanyam/cudaparsers
typedef struct {
    NvInfoAttribute nvInfoAttribute;
    uint32_t index;
    uint16_t ordinal;
    uint16_t offset;
    uint32_t tmp;
    uint8_t log_alignment() const { return tmp & 0xFF; }
    uint8_t space() const { return (tmp >> 8) & 0xF; }
    uint8_t cbank() const { return (tmp >> 12) & 0x1F; }
    bool is_cbank() const { return ((tmp >> 16) & 2) == 0; }
    uint16_t size_bytes() const { return (((tmp >> 16) & 0xFFFF) >> 2); }
} NvInfoKParam;

typedef struct {
    std::vector<NvInfoKParam> params;
} NvInfoFunction;

#endif  // GVIRTUS_CUDART_INTERNAL_H
