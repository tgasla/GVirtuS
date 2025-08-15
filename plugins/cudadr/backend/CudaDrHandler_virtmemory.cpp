/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDrHandler.h"

using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

size_t getHandleSize(CUmemAllocationHandleType handleType) {
    switch (handleType) {
        case CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR:
            return sizeof(int);  // 4
        case CU_MEM_HANDLE_TYPE_WIN32:
            return sizeof(void*);  // 8 on 64-bit
        case CU_MEM_HANDLE_TYPE_WIN32_KMT:
            return sizeof(unsigned int);  // 4
        case CU_MEM_HANDLE_TYPE_FABRIC:
            return sizeof(CUmemFabricHandle);
        case CU_MEM_HANDLE_TYPE_NONE:
        default:
            return 0;
    }
}

CUDA_DRIVER_HANDLER(MemCreate) {
    CUmemGenericAllocationHandle handle;
    size_t size = input_buffer->Get<size_t>();
    CUmemAllocationProp prop = input_buffer->Get<CUmemAllocationProp>();
    unsigned long long flags = input_buffer->Get<unsigned long long>();

    CUresult exit_code = cuMemCreate(&handle, size, &prop, flags);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cuMemCreate executed");
    if (exit_code != CUDA_SUCCESS) {
        return std::make_shared<Result>((cudaError_t)exit_code);
    }

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(handle);

    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

CUDA_DRIVER_HANDLER(MemExportToShareableHandle) {
    CUmemGenericAllocationHandle handle = input_buffer->Get<CUmemGenericAllocationHandle>();
    CUmemAllocationHandleType handleType = input_buffer->Get<CUmemAllocationHandleType>();
    unsigned long long flags = input_buffer->Get<unsigned long long>();
    char shareableHandle[getHandleSize(handleType)];  // Adjust size based on handle type

    CUresult exit_code = cuMemExportToShareableHandle(&shareableHandle, handle, handleType, flags);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cuMemExportToShareableHandle executed");
    if (exit_code != CUDA_SUCCESS) {
        return std::make_shared<Result>((cudaError_t)exit_code);
    }

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(shareableHandle, getHandleSize(handleType));

    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

CUDA_DRIVER_HANDLER(MemAddressReserve) {
    CUdeviceptr ptr;
    size_t size = input_buffer->Get<size_t>();
    size_t alignment = input_buffer->Get<size_t>();
    CUdeviceptr addr = input_buffer->Get<CUdeviceptr>();
    unsigned long long flags = input_buffer->Get<unsigned long long>();

    CUresult exit_code = cuMemAddressReserve(&ptr, size, alignment, addr, flags);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(ptr);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cuMemAddressReserve executed, ptr: " << ptr);
    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

CUDA_DRIVER_HANDLER(MemGetAllocationGranularity) {
    size_t granularity;
    const CUmemAllocationProp prop = input_buffer->Get<CUmemAllocationProp>();
    CUmemAllocationGranularity_flags option = input_buffer->Get<CUmemAllocationGranularity_flags>();

    CUresult exit_code = cuMemGetAllocationGranularity(&granularity, &prop, option);

    LOG4CPLUS_DEBUG(pThis->GetLogger(),
                    "cuMemGetAllocationGranularity executed, granularity: " << granularity);

    if (exit_code != CUDA_SUCCESS) {
        return std::make_shared<Result>((cudaError_t)exit_code);
    }

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(granularity);
    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

CUDA_DRIVER_HANDLER(MemImportFromShareableHandle) {
    CUmemGenericAllocationHandle handle;
    CUmemAllocationHandleType shHandleType = input_buffer->Get<CUmemAllocationHandleType>();
    void* osHandle = input_buffer->Assign<char>(getHandleSize(shHandleType));

    CUresult exit_code = cuMemImportFromShareableHandle(&handle, osHandle, shHandleType);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cuMemImportFromShareableHandle executed");
    if (exit_code != CUDA_SUCCESS) {
        return std::make_shared<Result>((cudaError_t)exit_code);
    }

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(handle);

    return std::make_shared<Result>((cudaError_t)exit_code, out);
}

CUDA_DRIVER_HANDLER(MemSetAccess) {
    CUdeviceptr ptr = input_buffer->Get<CUdeviceptr>();
    size_t size = input_buffer->Get<size_t>();
    size_t count = input_buffer->Get<size_t>();
    const CUmemAccessDesc* desc = input_buffer->Assign<CUmemAccessDesc>(count);

    CUresult exit_code = cuMemSetAccess(ptr, size, desc, count);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cuMemSetAccess executed for ptr: "
                                            << ptr << ", size: " << size << ", count: " << count);
    return std::make_shared<Result>((cudaError_t)exit_code);
}

CUDA_DRIVER_HANDLER(MemUnmap) {
    CUdeviceptr ptr = input_buffer->Get<CUdeviceptr>();
    size_t size = input_buffer->Get<size_t>();

    CUresult exit_code = cuMemUnmap(ptr, size);

    LOG4CPLUS_DEBUG(pThis->GetLogger(),
                    "cuMemUnmap executed for ptr: " << ptr << ", size: " << size);
    return std::make_shared<Result>((cudaError_t)exit_code);
}