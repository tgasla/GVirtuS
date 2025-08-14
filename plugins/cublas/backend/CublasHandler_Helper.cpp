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
 *             School of Computer Science, University College Dublin
 */

#include "CublasHandler.h"

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

// cublasStatus_t is a typedef to cublasContext*
// "The cublasHandle_t type is a pointer type to an opaque structure holding the
// cuBLAS library context."

CUBLAS_ROUTINE_HANDLER(Create_v2) {
    cublasHandle_t handle;
    cublasStatus_t cs = cublasCreate(&handle);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasCreate_v2 executed with status: " << cs);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->AddMarshal(handle);  // equivalent
        // out->Add<uintptr_t>((uintptr_t)handle); // also equivalent
        // out->Add<cublasHandle_t>(handle); // this does not work as it tries
        // to do sizeof cublasHandle_t which is an incomplete type (opaque
        // struct)
        LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasCreate_v2 handle: " << handle);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(GetVersion_v2) {
    int version;
    cublasStatus_t cs = cublasGetVersion(NULL, &version);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasGetVersion_v2 executed with status: "
                                            << cs << " and version: " << version);
    try {
        out->Add(version);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(Destroy_v2) {
    cublasHandle_t handle = in->Get<cublasHandle_t>();  // here, we read the buffer which contains
                                                        // the cublasHandle_t as a memory address
                                                        // that points to the cublas context.
    // cublasHandle_t handle =
    // reinterpret_cast<cublasHandle_t>in->Get<uintptr_t>(); // you can also use
    // this if frontend sends the handle as a uintptr_t
    cublasStatus_t cs = cublasDestroy(handle);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasDestroy_v2 executed with status: " << cs);
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(SetVector) {
    int n = in->Get<int>();
    int elemSize = in->Get<int>();
    int incx = in->Get<int>();
    int incy = in->Get<int>();

    void* y = in->GetFromMarshal<void*>();
    const void* x = in->AssignAll<char>();

    cublasStatus_t cs = cublasSetVector(n, elemSize, x, incx, y, incy);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasSetVector executed");
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(SetMatrix) {
    int rows = in->Get<int>();
    int cols = in->Get<int>();
    int elemSize = in->Get<int>();
    void* B = in->GetFromMarshal<void*>();
    int ldb = in->Get<int>();
    int lda = in->Get<int>();

    void* A = in->AssignAll<char>();
    cublasStatus_t cs = cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasSetMatrix executed");
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(GetVector) {
    int n = in->Get<int>();
    int elemSize = in->Get<int>();
    int incx = in->Get<int>();
    int incy = in->Get<int>();

    void* x = in->GetFromMarshal<void*>();
    void* y = in->Assign<void>();

    cublasStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        cs = cublasGetVector(n, elemSize, x, incx, y, incy);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    out->Add<char>((char*)y, n * elemSize);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasGetVector executed");
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(GetMatrix) {
    int rows = in->Get<int>();
    int cols = in->Get<int>();
    int elemSize = in->Get<int>();
    void* A = in->GetFromMarshal<void*>();
    int lda = in->Get<int>();
    void* B = in->Assign<void>();
    int ldb = in->Get<int>();

    cublasStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        cs = cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    out->Add<char>((char*)B, rows * cols * elemSize);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasGetMatrix executed");
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(SetMathMode) {
    cublasHandle_t handle = in->Get<cublasHandle_t>();
    cublasMath_t mode = in->Get<cublasMath_t>();

    cublasStatus_t cs = cublasSetMathMode(handle, mode);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasSetMathMode executed with status: " << cs);

    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(GetMathMode) {
    cublasHandle_t handle = in->Get<cublasHandle_t>();
    cublasMath_t mode;

    cublasStatus_t cs = cublasGetMathMode(handle, &mode);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasGetMathMode executed with status: " << cs);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cublasMath_t>(mode);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(SetStream_v2) {
    cublasHandle_t handle = in->Get<cublasHandle_t>();
    cudaStream_t streamId = in->Get<cudaStream_t>();

    cublasStatus_t cs = cublasSetStream_v2(handle, streamId);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasSetStream_v2 executed");
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(GetStream_v2) {
    cublasHandle_t handle = in->Get<cublasHandle_t>();
    cudaStream_t streamId;
    cublasStatus_t cs = cublasGetStream_v2(handle, &streamId);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<cudaStream_t>(streamId);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasGetStream_v2 executed");
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(GetPointerMode_v2) {
    cublasHandle_t handle = in->Get<cublasHandle_t>();
    cublasPointerMode_t mode;
    cublasStatus_t cs = cublasGetPointerMode_v2(handle, &mode);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<cublasPointerMode_t>(mode);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasGetPointerMode_v2 executed");
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(SetPointerMode_v2) {
    cublasHandle_t handle = in->Get<cublasHandle_t>();
    cublasPointerMode_t mode = in->Get<cublasPointerMode_t>();
    cublasStatus_t cs = cublasSetPointerMode_v2(handle, mode);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<cublasPointerMode_t>(mode);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(pThis->GetLogger(), LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasSetPointerMode_v2 executed");
    return std::make_shared<Result>(cs, out);
}

CUBLAS_ROUTINE_HANDLER(SetWorkspace_v2) {
    cublasHandle_t handle = in->Get<cublasHandle_t>();
    void* workspace = in->GetFromMarshal<void*>();
    size_t workspaceSizeInBytes = in->Get<size_t>();

    cublasStatus_t cs = cublasSetWorkspace(handle, workspace, workspaceSizeInBytes);
    LOG4CPLUS_DEBUG(pThis->GetLogger(), "cublasSetWorkspace executed with status: " << cs);

    return std::make_shared<Result>(cs);
}