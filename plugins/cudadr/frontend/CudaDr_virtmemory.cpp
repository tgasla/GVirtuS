/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDr.h"

using namespace std;

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

extern "C" CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size,
                                const CUmemAllocationProp* prop, unsigned long long flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(size);
    CudaDrFrontend::AddVariableForArguments<const CUmemAllocationProp>(*prop);
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::Execute("cuMemCreate");
    if (CudaDrFrontend::Success()) {
        *handle = CudaDrFrontend::GetOutputVariable<CUmemGenericAllocationHandle>();
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuMemExportToShareableHandle(void* shareableHandle,
                                                 CUmemGenericAllocationHandle handle,
                                                 CUmemAllocationHandleType handleType,
                                                 unsigned long long flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(handle);
    CudaDrFrontend::AddVariableForArguments(handleType);
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::Execute("cuMemExportToShareableHandle");
    if (CudaDrFrontend::Success()) {
        memcpy(shareableHandle,
               CudaDrFrontend::GetOutputHostPointer<char>(getHandleSize(handleType)),
               getHandleSize(handleType));
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment,
                                        CUdeviceptr addr, unsigned long long flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(size);
    CudaDrFrontend::AddVariableForArguments(alignment);
    CudaDrFrontend::AddVariableForArguments(addr);
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::Execute("cuMemAddressReserve");
    if (CudaDrFrontend::Success()) {
        *ptr = CudaDrFrontend::GetOutputVariable<CUdeviceptr>();
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuMemGetAllocationGranularity(size_t* granularity,
                                                  const CUmemAllocationProp* prop,
                                                  CUmemAllocationGranularity_flags option) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments<const CUmemAllocationProp>(prop);
    CudaDrFrontend::AddVariableForArguments(option);
    CudaDrFrontend::Execute("cuMemGetAllocationGranularity");
    if (CudaDrFrontend::Success()) {
        *granularity = CudaDrFrontend::GetOutputVariable<size_t>();
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle,
                                                   void* osHandle,
                                                   CUmemAllocationHandleType shHandleType) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments<char>((char*)osHandle, getHandleSize(shHandleType));
    CudaDrFrontend::AddVariableForArguments(shHandleType);
    CudaDrFrontend::Execute("cuMemImportFromShareableHandle");
    if (CudaDrFrontend::Success()) {
        *handle = CudaDrFrontend::GetOutputVariable<CUmemGenericAllocationHandle>();
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc,
                                   size_t count) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(ptr);
    CudaDrFrontend::AddVariableForArguments(size);
    CudaDrFrontend::AddVariableForArguments(count);
    CudaDrFrontend::AddHostPointerForArguments<const CUmemAccessDesc>(desc, count);
    CudaDrFrontend::Execute("cuMemSetAccess");
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(ptr);
    CudaDrFrontend::AddVariableForArguments(size);
    CudaDrFrontend::Execute("cuMemUnmap");
    return CudaDrFrontend::GetExitCode();
}