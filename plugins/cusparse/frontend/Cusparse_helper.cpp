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
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@uniparthenope.it>,
 *            School of Computer Science, University College Dublin
 */

#include "CusparseFrontend.h"

using namespace std;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

extern "C" cusparseStatus_t cusparseCreate(cusparseHandle_t* handle) {
    CusparseFrontend::Prepare();
    CusparseFrontend::Execute("cusparseCreate");
    if (CusparseFrontend::Success())
        *handle = CusparseFrontend::GetOutputVariable<cusparseHandle_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::Execute("cusparseDestroy");
    return CusparseFrontend::GetExitCode();
}
extern "C" cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddDevicePointerForArguments(streamId);
    CusparseFrontend::Execute("cusparseSetStream");
    return CusparseFrontend::GetExitCode();
}
extern "C" cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::Execute("cusparseGetStream");
    if (CusparseFrontend::Success())
        *streamId = CusparseFrontend::GetOutputVariable<cudaStream_t>();
    return CusparseFrontend::GetExitCode();
}

// TODO: test
extern "C" cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descr,
                                                   cusparseDiagType_t diagType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(descr);
    CusparseFrontend::AddVariableForArguments(diagType);
    CusparseFrontend::Execute("cusparseSetMatDiagType");
    return CusparseFrontend::GetExitCode();
}

// TODO: test
extern "C" cusparseStatus_t cusparseDestroyBsrsv2Info(bsrsv2Info_t info) {
    CusparseFrontend::Prepare();
    CusparseFrontend::Execute("cusparseDestroyBsrsv2Info");
    return CusparseFrontend::GetExitCode();
}

// TODO: test
extern "C" cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA,
                                                   cusparseFillMode_t fillMode) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments(fillMode);
    CusparseFrontend::Execute("cusparseSetMatFillMode");
    return CusparseFrontend::GetExitCode();
}

// TODO: test
extern "C" cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size,
                                                void* values, cudaDataType valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(dnVecDescr);
    CusparseFrontend::AddVariableForArguments(size);
    CusparseFrontend::AddDevicePointerForArguments(values);
    CusparseFrontend::AddVariableForArguments(valueType);
    CusparseFrontend::Execute("cusparseCreateDnVec");
    if (CusparseFrontend::Success())
        *dnVecDescr = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
    return CusparseFrontend::GetExitCode();
}

// TODO: test
extern "C" cusparseStatus_t cusparseCbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const cuComplex* alpha,
    const cusparseMatDescr_t descrA, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr,
    const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, const cuComplex* B, int ldb,
    cuComplex* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments(ldb);
    CusparseFrontend::AddDevicePointerForArguments(X);
    CusparseFrontend::AddVariableForArguments(ldx);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCbsrsm2_solve");
    return CusparseFrontend::GetExitCode();
}

// TODO: test
extern "C" cusparseStatus_t cusparseDestroyBsrsm2Info(bsrsm2Info_t info) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::Execute("cusparseDestroyBsrsm2Info");
    return CusparseFrontend::GetExitCode();
}

// TODO: test
extern "C" cusparseStatus_t cusparseSpMV(
    cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnVecDescr_t vecX,  // non-const descriptor supported
    const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg,
    void* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(vecX);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(vecY);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(bufferSize);
    CusparseFrontend::Execute("cusparseSpMV");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output vector
        vecY = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO: test
extern "C" cusparseStatus_t cusparseCbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
                                           cusparseOperation_t transA, cusparseOperation_t transB,
                                           int mb, int n, int kb, int nnzb, const cuComplex* alpha,
                                           const cusparseMatDescr_t descrA,
                                           const cuComplex* bsrValA, const int* bsrRowPtrA,
                                           const int* bsrColIndA, int blockDim, const cuComplex* B,
                                           int ldb, const cuComplex* beta, cuComplex* C, int ldc) {
    CusparseFrontend::Prepare();
    // Add the necessary arguments for cusparseCbsrmm
    // This is a placeholder as the actual implementation is not provided
    CusparseFrontend::Execute("cusparseCbsrmm");
    return CusparseFrontend::GetExitCode();
}

// TODO: tests
extern "C" cusparseStatus_t cusparseScsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA,
    int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
    const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(m);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(descrB);
    CusparseFrontend::AddVariableForArguments(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddDevicePointerForArguments(descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseScsrgeam2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseCcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA,
    int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB,
    const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const cusparseMatDescr_t descrC, const cuComplex* csrSortedValC, const int* csrSortedRowPtrC,
    const int* csrSortedColIndC, size_t* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(m);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(descrB);
    CusparseFrontend::AddVariableForArguments(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddDevicePointerForArguments(descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseCcsrgeam2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO: test
extern "C" cusparseStatus_t cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spMatDescr);
    CusparseFrontend::Execute("cusparseDestroySpMat");
    return CusparseFrontend::GetExitCode();
}

// TODO: test
extern "C" cusparseStatus_t cusparseCbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
    cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA,
    int blockDim, bsrsm2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseCbsrsm2_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseCreateBsrsm2Info(bsrsm2Info_t* info) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::Execute("cusparseCreateBsrsm2Info");
    if (CusparseFrontend::Success()) *info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseSpGEMM_workEstimation(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstSpMatDescr_t matB,  // non-const descriptor supported
    const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(spgemmDescr);
    CusparseFrontend::AddDevicePointerForArguments(bufferSize1);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer1);
    CusparseFrontend::Execute("cusparseSpGEMM_workEstimation");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *bufferSize1 = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseSbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
    float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
    bsrsm2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseSbsrsm2_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseZbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
    cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA,
    int blockDim, bsrsm2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseZbsrsm2_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA,
                                               cusparseMatrixType_t type) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments(type);
    CusparseFrontend::Execute("cusparseSetMatType");
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                           int nnz, const int* cooRowInd,
                                                           const int* cooColInd,
                                                           size_t* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(m);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnz);
    CusparseFrontend::AddDevicePointerForArguments(cooRowInd);
    CusparseFrontend::AddDevicePointerForArguments(cooColInd);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseXcoosort_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                           int nnz, const int* csrRowPtr,
                                                           const int* csrColInd,
                                                           size_t* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(m);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnz);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(csrColInd);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseXcsrsort_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseSpMV_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnVecDescr_t vecX,  // non-const descriptor supported
    const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg,
    size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(vecX);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(vecY);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(bufferSize);
    CusparseFrontend::Execute("cusparseSpMV_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *bufferSize = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseZcsrgeam2(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha,
    const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta,
    const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB, cusparseMatDescr_t descrC,
    cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC,
    void* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(m);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(descrB);
    CusparseFrontend::AddVariableForArguments(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddDevicePointerForArguments(descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseZcsrgeam2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        pBufferSizeInBytes = CusparseFrontend::GetOutputDevicePointer();
    }
    return CusparseFrontend::GetExitCode();
}

// TODO:
extern "C" cusparseStatus_t cusparseSbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
    const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim,
    bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSbsrsm2_analysis");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the info object
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle,
                                                   cusparsePointerMode_t mode) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(mode);
    CusparseFrontend::Execute("cusparseSetPointerMode");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the handle
        handle = CusparseFrontend::GetOutputVariable<cusparseHandle_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex* bsrValA, const int* bsrRowPtrA,
    const int* bsrColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseCbsrsv2_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t* spsmDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spsmDescr);
    CusparseFrontend::Execute("cusparseSpSM_createDescr");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        *spsmDescr = CusparseFrontend::GetOutputVariable<cusparseSpSMDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t spsmDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spsmDescr);
    CusparseFrontend::Execute("cusparseSpSM_destroyDescr");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSDDMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstDnMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnMatDescr_t matB,  // non-const descriptor supported
    const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg,
    size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(bufferSize);
    CusparseFrontend::Execute("cusparseSDDMM_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *bufferSize = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSDDMM(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstDnMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnMatDescr_t matB,  // non-const descriptor supported
    const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg,
    void* externalBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer);
    CusparseFrontend::Execute("cusparseSDDMM");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpSM_solve(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnMatDescr_t matB,  // non-const descriptor supported
    cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg,
    cusparseSpSMDescr_t spsmDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(spsmDescr);
    CusparseFrontend::Execute("cusparseSpSM_solve");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix
        matC = CusparseFrontend::GetOutputVariable<cusparseDnMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA,
    int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseDbsrsv2_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
    const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd,
    int blockDim, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDbsrsm2_analysis");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the info object
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr,
                                                   void* csrRowOffsets, void* csrColInd,
                                                   void* csrValues) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spMatDescr);
    CusparseFrontend::AddDevicePointerForArguments(csrRowOffsets);
    CusparseFrontend::AddDevicePointerForArguments(csrColInd);
    CusparseFrontend::AddDevicePointerForArguments(csrValues);
    CusparseFrontend::Execute("cusparseCsrSetPointers");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the spMatDescr
        spMatDescr = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                                                       int batchCount, int64_t offsetsBatchStride,
                                                       int64_t columnsValuesBatchStride) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spMatDescr);
    CusparseFrontend::AddVariableForArguments(batchCount);
    CusparseFrontend::AddVariableForArguments(offsetsBatchStride);
    CusparseFrontend::AddVariableForArguments(columnsValuesBatchStride);
    CusparseFrontend::Execute("cusparseCsrSetStridedBatch");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the spMatDescr
        spMatDescr = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
    const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA,
    int blockDim, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCbsrsm2_analysis");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the info object
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descr,
                                                    cusparseIndexBase_t base) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(descr);
    CusparseFrontend::AddVariableForArguments(base);
    CusparseFrontend::Execute("cusparseSetMatIndexBase");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        descr = CusparseFrontend::GetOutputVariable<cusparseMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnMatDescr_t matB,  // non-const descriptor supported
    const void* beta,
    cusparseDnMatDescr_t matC,  // non-const descriptor supported
    cudaDataType computeType, cusparseSpMMAlg_t alg, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(bufferSize);
    CusparseFrontend::Execute("cusparseSpMM_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *bufferSize = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpMM(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnMatDescr_t matB,  // non-const descriptor supported
    const void* beta,
    cusparseDnMatDescr_t matC,  // non-const descriptor supported
    cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer);
    CusparseFrontend::Execute("cusparseSpMM");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix
        matC = CusparseFrontend::GetOutputVariable<cusparseDnMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpGEMM(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstSpMatDescr_t matB,  // non-const descriptor supported
    const void* beta,
    cusparseSpMatDescr_t matC,  // non-const descriptor supported
    cudaDataType computeType, cusparseSpGEMMAlg_t alg, void* externalBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer);
    CusparseFrontend::Execute("cusparseSpGEMM");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix
        matC = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseZbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrValA, const int* bsrRowPtrA,
    const int* bsrColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy,
    void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZbsrsv2_analysis");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the info object
        info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t spsvDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spsvDescr);
    CusparseFrontend::Execute("cusparseSpSV_destroyDescr");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        spsvDescr = CusparseFrontend::GetOutputVariable<cusparseSpSVDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz,
                                             const cusparseMatDescr_t descrA, const int* csrRowPtr,
                                             int* csrColInd, int* P, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(m);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnz);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(csrColInd);
    CusparseFrontend::AddDevicePointerForArguments(P);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseXcsrsort");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the sorted indices
        csrColInd = CusparseFrontend::GetOutputHostPointer<int>();
        P = CusparseFrontend::GetOutputHostPointer<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpSV_solve(
    cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnVecDescr_t vecX,  // non-const descriptor supported
    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
    cusparseSpSVDescr_t spsvDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(vecX);
    CusparseFrontend::AddDevicePointerForArguments(vecY);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(spsvDescr);
    CusparseFrontend::Execute("cusparseSpSV_solve");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output vector
        vecY = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(dnVecDescr);
    CusparseFrontend::Execute("cusparseDestroyDnVec");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        dnVecDescr = CusparseFrontend::GetOutputVariable<cusparseConstDnVecDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpGEMM_copy(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstSpMatDescr_t matB,  // non-const descriptor supported
    const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(spgemmDescr);
    CusparseFrontend::Execute("cusparseSpGEMM_copy");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix
        matC = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA,
    const int* bsrColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy,
    void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDbsrsv2_analysis");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the info object
        info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
                                           cusparseOperation_t transA, cusparseOperation_t transB,
                                           int mb, int n, int kb, int nnzb, const double* alpha,
                                           const cusparseMatDescr_t descrA, const double* bsrValA,
                                           const int* bsrRowPtrA, const int* bsrColIndA,
                                           int blockDim, const double* B, int ldb,
                                           const double* beta, double* C, int ldc) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transB);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(kb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const double>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddHostPointerForArguments<const double>(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<const double>(beta);
    CusparseFrontend::AddHostPointerForArguments<double>(C);
    CusparseFrontend::AddVariableForArguments<int>(ldc);
    CusparseFrontend::Execute("cusparseDbsrmm");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix C
        C = CusparseFrontend::GetOutputHostPointer<double>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::Execute("cusparseDestroyMatDescr");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        descrA = CusparseFrontend::GetOutputVariable<cusparseMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr,
                                                         int batchCount, int64_t batchStride) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(dnMatDescr);
    CusparseFrontend::AddVariableForArguments(batchCount);
    CusparseFrontend::AddVariableForArguments(batchStride);
    CusparseFrontend::Execute("cusparseDnMatSetStridedBatch");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        dnMatDescr = CusparseFrontend::GetOutputVariable<cusparseDnMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA,
    int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseSbsrsv2_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrValA,
    const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrsv2Info_t info,
    const cuComplex* x, cuComplex* y, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddHostPointerForArguments<const cuComplex>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCbsrsv2_solve");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output vector y
        y = CusparseFrontend::GetOutputHostPointer<cuComplex>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseZbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const cuDoubleComplex* alpha,
    const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedVal,
    const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info,
    const cuDoubleComplex* B, int ldb, cuDoubleComplex* X, int ldx, cusparseSolvePolicy_t policy,
    void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(X);
    CusparseFrontend::AddVariableForArguments<int>(ldx);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZbsrsm2_solve");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix X
        X = CusparseFrontend::GetOutputHostPointer<cuDoubleComplex>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpSM_analysis(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnMatDescr_t matB,  // non-const descriptor supported
    cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg,
    cusparseSpSMDescr_t spsmDescr, void* externalBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(spsmDescr);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer);
    CusparseFrontend::Execute("cusparseSpSM_analysis");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output descriptor
        spsmDescr = CusparseFrontend::GetOutputVariable<cusparseSpSMDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCbsrmv(cusparseHandle_t handle, cusparseDirection_t dir,
                                           cusparseOperation_t trans, int mb, int nb, int nnzb,
                                           const cuComplex* alpha, const cusparseMatDescr_t descr,
                                           const cuComplex* bsrVal, const int* bsrRowPtr,
                                           const int* bsrColInd, int blockDim, const cuComplex* x,
                                           const cuComplex* beta, cuComplex* y) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dir);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const cuComplex>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descr);
    CusparseFrontend::AddDevicePointerForArguments(bsrVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddHostPointerForArguments<const cuComplex>(beta);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::Execute("cusparseCbsrmv");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output vector y
        y = CusparseFrontend::GetOutputHostPointer<cuComplex>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCcsrgeam2(
    cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA,
    int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB,
    const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC,
    int* csrSortedColIndC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments<const cuComplex>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments<const cuComplex>(beta);
    CusparseFrontend::AddDevicePointerForArguments(descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddDevicePointerForArguments(descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCcsrgeam2");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix C
        csrSortedValC = CusparseFrontend::GetOutputHostPointer<cuComplex>();
        csrSortedRowPtrC = CusparseFrontend::GetOutputHostPointer<int>();
        csrSortedColIndC = CusparseFrontend::GetOutputHostPointer<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSbsrmv(cusparseHandle_t handle, cusparseDirection_t dir,
                                           cusparseOperation_t trans, int mb, int nb, int nnzb,
                                           const float* alpha, const cusparseMatDescr_t descr,
                                           const float* bsrVal, const int* bsrRowPtr,
                                           const int* bsrColInd, int blockDim, const float* x,
                                           const float* beta, float* y) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dir);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const float>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descr);
    CusparseFrontend::AddDevicePointerForArguments(bsrVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(x);
    CusparseFrontend::AddHostPointerForArguments<const float>(beta);
    CusparseFrontend::AddDevicePointerForArguments(y);
    CusparseFrontend::Execute("cusparseSbsrmv");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output vector y
        y = CusparseFrontend::GetOutputHostPointer<float>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const double* alpha,
    const cusparseMatDescr_t descrA, const double* bsrSortedVal, const int* bsrSortedRowPtr,
    const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, const double* B, int ldb,
    double* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const double>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddHostPointerForArguments<const double>(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddDevicePointerForArguments(X);
    CusparseFrontend::AddVariableForArguments<int>(ldx);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDbsrsm2_solve");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix X
        X = CusparseFrontend::GetOutputHostPointer<double>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows,
                                                int64_t cols, int64_t ld, void* data,
                                                cudaDataType valueType, cusparseOrder_t order) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(dnMatDescr);
    CusparseFrontend::AddVariableForArguments(rows);
    CusparseFrontend::AddVariableForArguments(cols);
    CusparseFrontend::AddVariableForArguments(ld);
    CusparseFrontend::AddDevicePointerForArguments(data);
    CusparseFrontend::AddVariableForArguments(valueType);
    CusparseFrontend::AddVariableForArguments(order);
    CusparseFrontend::Execute("cusparseCreateDnMat");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        *dnMatDescr = CusparseFrontend::GetOutputVariable<cusparseDnMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
                                           cusparseOperation_t transA, cusparseOperation_t transB,
                                           int mb, int n, int kb, int nnzb, const float* alpha,
                                           const cusparseMatDescr_t descrA, const float* bsrValA,
                                           const int* bsrRowPtrA, const int* bsrColIndA,
                                           int blockDim, const float* B, int ldb, const float* beta,
                                           float* C, int ldc) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transB);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(kb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const float>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddHostPointerForArguments<const float>(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<const float>(beta);
    CusparseFrontend::AddHostPointerForArguments<float>(C);
    CusparseFrontend::AddVariableForArguments<int>(ldc);
    CusparseFrontend::Execute("cusparseSbsrmm");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix C
        C = CusparseFrontend::GetOutputHostPointer<float>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseZbsrmm(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuDoubleComplex* alpha,
    const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrValA, const int* bsrRowPtrA,
    const int* bsrColIndA, int blockDim, const cuDoubleComplex* B, int ldb,
    const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transB);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(kb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(beta);
    CusparseFrontend::AddHostPointerForArguments<cuDoubleComplex>(C);
    CusparseFrontend::AddVariableForArguments<int>(ldc);
    CusparseFrontend::Execute("cusparseZbsrmm");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix C
        C = CusparseFrontend::GetOutputHostPointer<cuDoubleComplex>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n,
                                                              int* p) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddDevicePointerForArguments(p);
    CusparseFrontend::Execute("cusparseCreateIdentityPermutation");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the permutation array
        p = CusparseFrontend::GetOutputHostPointer<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA,
    const int* bsrColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy,
    void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSbsrsv2_analysis");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the info object
        info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz,
                                                  int* cooRows, int* cooCols, int* P,
                                                  void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(m);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnz);
    CusparseFrontend::AddDevicePointerForArguments(cooRows);
    CusparseFrontend::AddDevicePointerForArguments(cooCols);
    CusparseFrontend::AddDevicePointerForArguments(P);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseXcoosortByRow");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the sorted COO arrays
        cooRows = CusparseFrontend::GetOutputHostPointer<int>();
        cooCols = CusparseFrontend::GetOutputHostPointer<int>();
        P = CusparseFrontend::GetOutputHostPointer<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpGEMM_compute(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstSpMatDescr_t matB,  // non-const descriptor supported
    const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(spgemmDescr);
    CusparseFrontend::AddDevicePointerForArguments(bufferSize2);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer2);
    CusparseFrontend::Execute("cusparseSpGEMM_compute");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output descriptor
        spgemmDescr = CusparseFrontend::GetOutputVariable<cusparseSpGEMMDescr_t>();
        *bufferSize2 = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex* bsrValA, const int* bsrRowPtrA,
    const int* bsrColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy,
    void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseCbsrsv2_analysis");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the info object
        info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz,
                                             int m, int* csrRowPtr, cusparseIndexBase_t idxBase) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddDevicePointerForArguments(cooRowInd);
    CusparseFrontend::AddVariableForArguments(nnz);
    CusparseFrontend::AddVariableForArguments(m);
    CusparseFrontend::AddDevicePointerForArguments(csrRowPtr);
    CusparseFrontend::AddVariableForArguments(idxBase);
    CusparseFrontend::Execute("cusparseXcoo2csr");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the CSR row pointer array
        csrRowPtr = CusparseFrontend::GetOutputHostPointer<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseZbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
    const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd,
    int blockDim, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZbsrsm2_analysis");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the info object
        info = CusparseFrontend::GetOutputVariable<bsrsm2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseZcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha,
    const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta,
    const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
    const cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(beta);
    CusparseFrontend::AddDevicePointerForArguments(descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddDevicePointerForArguments(descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseZcsrgeam2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCreateBsrsv2Info(bsrsv2Info_t* info) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::Execute("cusparseCreateBsrsv2Info");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the info object
        *info = CusparseFrontend::GetOutputVariable<bsrsv2Info_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(descr);
    CusparseFrontend::Execute("cusparseSpGEMM_destroyDescr");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        descr = CusparseFrontend::GetOutputVariable<cusparseSpGEMMDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSDDMM_preprocess(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstDnMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnMatDescr_t matB,  // non-const descriptor supported
    const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg,
    void* externalBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(opA);
    CusparseFrontend::AddVariableForArguments(opB);
    CusparseFrontend::AddDevicePointerForArguments(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(beta);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments(computeType);
    CusparseFrontend::AddVariableForArguments(alg);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer);
    CusparseFrontend::Execute("cusparseSDDMM_preprocess");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output descriptor
        matC = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle, bsrsv2Info_t info,
                                                      int* position) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(position);
    CusparseFrontend::Execute("cusparseXbsrsv2_zeroPivot");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the position of the zero pivot
        *position = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const float* alpha,
    const cusparseMatDescr_t descrA, const float* bsrSortedVal, const int* bsrSortedRowPtr,
    const int* bsrSortedColInd, int blockDim, bsrsm2Info_t info, const float* B, int ldb, float* X,
    int ldx, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const float>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColInd);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddHostPointerForArguments<const float>(B);
    CusparseFrontend::AddVariableForArguments<int>(ldb);
    CusparseFrontend::AddHostPointerForArguments<float>(X);
    CusparseFrontend::AddVariableForArguments<int>(ldx);
    CusparseFrontend::AddVariableForArguments(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSbsrsm2_solve");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix X
        X = CusparseFrontend::GetOutputHostPointer<float>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseXcsrgeam2Nnz(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB,
    int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr,
    void* workspace) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddDevicePointerForArguments(descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddDevicePointerForArguments(descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(nnzTotalDevHostPtr);
    CusparseFrontend::AddDevicePointerForArguments(workspace);
    CusparseFrontend::Execute("cusparseXcsrgeam2Nnz");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the CSR row pointer and nnz
        csrSortedRowPtrC = CusparseFrontend::GetOutputHostPointer<int>();
        *nnzTotalDevHostPtr = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA,
    cusparseOperation_t transX, int mb, int n, int nnzb, const cusparseMatDescr_t descrA,
    double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim,
    bsrsm2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(transX);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(n);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrSortedColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseDbsrsm2_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const double* alpha, const cusparseMatDescr_t descrA, const double* bsrValA,
    const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrsv2Info_t info, const double* x,
    double* y, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const double>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddHostPointerForArguments<const double>(x);
    CusparseFrontend::AddHostPointerForArguments<double>(y);
    CusparseFrontend::AddVariableForArguments<int>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDbsrsv2_solve");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output vector y
        y = CusparseFrontend::GetOutputHostPointer<double>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseScsrgeam2(
    cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA,
    int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
    float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments<const float>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments<const float>(beta);
    CusparseFrontend::AddDevicePointerForArguments(descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddDevicePointerForArguments(descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseScsrgeam2");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix C
        csrSortedValC = CusparseFrontend::GetOutputHostPointer<float>();
        csrSortedRowPtrC = CusparseFrontend::GetOutputHostPointer<int>();
        csrSortedColIndC = CusparseFrontend::GetOutputHostPointer<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDestroyDnMat(cusparseConstDnMatDescr_t mat) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(mat);
    CusparseFrontend::Execute("cusparseDestroyDnMat");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        mat = CusparseFrontend::GetOutputVariable<cusparseConstDnMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDbsrmv(cusparseHandle_t handle, cusparseDirection_t dir,
                                           cusparseOperation_t trans, int mb, int nb, int nnzb,
                                           const double* alpha, const cusparseMatDescr_t descr,
                                           const double* bsrVal, const int* bsrRowPtr,
                                           const int* bsrColInd, int blockDim, const double* x,
                                           const double* beta, double* y) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dir);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const double>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descr);
    CusparseFrontend::AddDevicePointerForArguments(bsrVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddHostPointerForArguments<const double>(x);
    CusparseFrontend::AddHostPointerForArguments<const double>(beta);
    CusparseFrontend::AddHostPointerForArguments<double>(y);
    CusparseFrontend::Execute("cusparseDbsrmv");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output vector y
        y = CusparseFrontend::GetOutputHostPointer<double>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* descrA) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::Execute("cusparseCreateMatDescr");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the matrix descriptor
        *descrA = CusparseFrontend::GetOutputVariable<cusparseMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows,
                                              int64_t cols, int64_t nnz, void* csrRowOffsets,
                                              void* csrColInd, void* csrValues,
                                              cusparseIndexType_t csrRowOffsetsType,
                                              cusparseIndexType_t csrColIndType,
                                              cusparseIndexBase_t idxBase, cudaDataType valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spMatDescr);
    CusparseFrontend::AddVariableForArguments<int64_t>(rows);
    CusparseFrontend::AddVariableForArguments<int64_t>(cols);
    CusparseFrontend::AddVariableForArguments<int64_t>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(csrRowOffsets);
    CusparseFrontend::AddDevicePointerForArguments(csrColInd);
    CusparseFrontend::AddDevicePointerForArguments(csrValues);
    CusparseFrontend::AddVariableForArguments(csrRowOffsetsType);
    CusparseFrontend::AddVariableForArguments(csrColIndType);
    CusparseFrontend::AddVariableForArguments(idxBase);
    CusparseFrontend::AddVariableForArguments(valueType);
    CusparseFrontend::Execute("cusparseCreateCsr");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the sparse matrix descriptor
        *spMatDescr = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr,
                                                      cusparseSpMatAttribute_t attribute,
                                                      void* data, size_t dataSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spMatDescr);
    CusparseFrontend::AddVariableForArguments(attribute);
    CusparseFrontend::AddDevicePointerForArguments(data);
    CusparseFrontend::AddVariableForArguments(dataSize);
    CusparseFrontend::Execute("cusparseSpMatSetAttribute");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the updated sparse matrix descriptor
        spMatDescr = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA,
    int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
    const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC,
    size_t* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments<const double>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments<const double>(beta);
    CusparseFrontend::AddDevicePointerForArguments(descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddDevicePointerForArguments(descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseDcsrgeam2_bufferSizeExt");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle, bsrsm2Info_t info,
                                                      int* position) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(position);
    CusparseFrontend::Execute("cusparseXbsrsm2_zeroPivot");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the position of the zero pivot
        *position = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(descr);
    CusparseFrontend::Execute("cusparseSpGEMM_createDescr");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        *descr = CusparseFrontend::GetOutputVariable<cusparseSpGEMMDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseDcsrgeam2(
    cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA,
    int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
    double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments<int>(m);
    CusparseFrontend::AddVariableForArguments<int>(n);
    CusparseFrontend::AddHostPointerForArguments<const double>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddVariableForArguments<int>(nnzA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndA);
    CusparseFrontend::AddHostPointerForArguments<const double>(beta);
    CusparseFrontend::AddDevicePointerForArguments(descrB);
    CusparseFrontend::AddVariableForArguments<int>(nnzB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrB);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndB);
    CusparseFrontend::AddDevicePointerForArguments(descrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedValC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedRowPtrC);
    CusparseFrontend::AddDevicePointerForArguments(csrSortedColIndC);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseDcsrgeam2");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output matrix C
        csrSortedValC = CusparseFrontend::GetOutputHostPointer<double>();
        csrSortedRowPtrC = CusparseFrontend::GetOutputHostPointer<int>();
        csrSortedColIndC = CusparseFrontend::GetOutputHostPointer<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr, int64_t rows,
                                              int64_t cols, int64_t nnz, void* cooRowInd,
                                              void* cooColInd, void* cooValues,
                                              cusparseIndexType_t cooIdxType,
                                              cusparseIndexBase_t idxBase, cudaDataType valueType) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spMatDescr);
    CusparseFrontend::AddVariableForArguments<int64_t>(rows);
    CusparseFrontend::AddVariableForArguments<int64_t>(cols);
    CusparseFrontend::AddVariableForArguments<int64_t>(nnz);
    CusparseFrontend::AddDevicePointerForArguments(cooRowInd);
    CusparseFrontend::AddDevicePointerForArguments(cooColInd);
    CusparseFrontend::AddDevicePointerForArguments(cooValues);
    CusparseFrontend::AddVariableForArguments(cooIdxType);
    CusparseFrontend::AddVariableForArguments(idxBase);
    CusparseFrontend::AddVariableForArguments(valueType);
    CusparseFrontend::Execute("cusparseCreateCoo");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the sparse matrix descriptor
        *spMatDescr = CusparseFrontend::GetOutputVariable<cusparseSpMatDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseZbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex* bsrValA, const int* bsrRowPtrA,
    const int* bsrColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddDevicePointerForArguments(pBufferSizeInBytes);
    CusparseFrontend::Execute("cusparseZbsrsv2_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the buffer size
        *pBufferSizeInBytes = CusparseFrontend::GetOutputVariable<int>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseZbsrmv(
    cusparseHandle_t handle, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb,
    int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descr,
    const cuDoubleComplex* bsrVal, const int* bsrRowPtr, const int* bsrColInd, int blockDim,
    const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dir);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(trans);
    CusparseFrontend::AddVariableForArguments<int>(mb);
    CusparseFrontend::AddVariableForArguments<int>(nb);
    CusparseFrontend::AddVariableForArguments<int>(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descr);
    CusparseFrontend::AddDevicePointerForArguments(bsrVal);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtr);
    CusparseFrontend::AddDevicePointerForArguments(bsrColInd);
    CusparseFrontend::AddVariableForArguments<int>(blockDim);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(x);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(beta);
    CusparseFrontend::AddHostPointerForArguments<cuDoubleComplex>(y);
    CusparseFrontend::Execute("cusparseZbsrmv");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output vector y
        y = CusparseFrontend::GetOutputHostPointer<cuDoubleComplex>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpSV_analysis(
    cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnVecDescr_t vecX,  // non-const descriptor supported
    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
    cusparseSpSVDescr_t spsvDescr, void* externalBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    CusparseFrontend::AddHostPointerForArguments<const void>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(vecX);
    CusparseFrontend::AddDevicePointerForArguments(vecY);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpSVAlg_t>(alg);
    CusparseFrontend::AddDevicePointerForArguments(spsvDescr);
    CusparseFrontend::AddDevicePointerForArguments(externalBuffer);
    CusparseFrontend::Execute("cusparseSpSV_analysis");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the sparse matrix descriptor
        matA = CusparseFrontend::GetOutputVariable<cusparseConstSpMatDescr_t>();
        vecX = CusparseFrontend::GetOutputVariable<cusparseConstDnVecDescr_t>();
        vecY = CusparseFrontend::GetOutputVariable<cusparseDnVecDescr_t>();
        spsvDescr = CusparseFrontend::GetOutputVariable<cusparseSpSVDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseZbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrValA,
    const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrsv2Info_t info,
    const cuDoubleComplex* x, cuDoubleComplex* y, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddHostPointerForArguments<const cuDoubleComplex>(x);
    CusparseFrontend::AddHostPointerForArguments<cuDoubleComplex>(y);
    CusparseFrontend::AddVariableForArguments<int>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseZbsrsv2_solve");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output vector y
        y = CusparseFrontend::GetOutputHostPointer<cuDoubleComplex>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpSM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnMatDescr_t matB,  // non-const descriptor supported
    cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg,
    cusparseSpSMDescr_t spsmDescr, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opB);
    CusparseFrontend::AddHostPointerForArguments<const void>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(matB);
    CusparseFrontend::AddDevicePointerForArguments(matC);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpSMAlg_t>(alg);
    CusparseFrontend::AddDevicePointerForArguments(spsmDescr);
    CusparseFrontend::AddDevicePointerForArguments(bufferSize);
    CusparseFrontend::Execute("cusparseSpSM_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the buffer size
        *bufferSize = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpSV_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha,
    cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
    cusparseConstDnVecDescr_t vecX,  // non-const descriptor supported
    cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
    cusparseSpSVDescr_t spsvDescr, size_t* bufferSize) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments<cusparseOperation_t>(opA);
    CusparseFrontend::AddHostPointerForArguments<const void>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(matA);
    CusparseFrontend::AddDevicePointerForArguments(vecX);
    CusparseFrontend::AddDevicePointerForArguments(vecY);
    CusparseFrontend::AddVariableForArguments<cudaDataType>(computeType);
    CusparseFrontend::AddVariableForArguments<cusparseSpSVAlg_t>(alg);
    CusparseFrontend::AddDevicePointerForArguments(spsvDescr);
    CusparseFrontend::AddDevicePointerForArguments(bufferSize);
    CusparseFrontend::Execute("cusparseSpSV_bufferSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the buffer size
        *bufferSize = CusparseFrontend::GetOutputVariable<size_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const float* alpha, const cusparseMatDescr_t descrA, const float* bsrValA,
    const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsrsv2Info_t info, const float* x,
    float* y, cusparseSolvePolicy_t policy, void* pBuffer) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::AddVariableForArguments(dirA);
    CusparseFrontend::AddVariableForArguments(transA);
    CusparseFrontend::AddVariableForArguments(mb);
    CusparseFrontend::AddVariableForArguments(nnzb);
    CusparseFrontend::AddHostPointerForArguments<const float>(alpha);
    CusparseFrontend::AddDevicePointerForArguments(descrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrValA);
    CusparseFrontend::AddDevicePointerForArguments(bsrRowPtrA);
    CusparseFrontend::AddDevicePointerForArguments(bsrColIndA);
    CusparseFrontend::AddVariableForArguments(blockDim);
    CusparseFrontend::AddDevicePointerForArguments(info);
    CusparseFrontend::AddHostPointerForArguments<const float>(x);
    CusparseFrontend::AddHostPointerForArguments<float>(y);
    CusparseFrontend::AddVariableForArguments<int>(policy);
    CusparseFrontend::AddDevicePointerForArguments(pBuffer);
    CusparseFrontend::Execute("cusparseSbsrsv2_solve");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the output vector y
        y = CusparseFrontend::GetOutputHostPointer<float>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpMatGetSize(
    cusparseConstSpMatDescr_t spMatDescr,  // non-const descriptor supported
    int64_t* rows, int64_t* cols, int64_t* nnz) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spMatDescr);
    CusparseFrontend::AddDevicePointerForArguments(rows);
    CusparseFrontend::AddDevicePointerForArguments(cols);
    CusparseFrontend::AddDevicePointerForArguments(nnz);
    CusparseFrontend::Execute("cusparseSpMatGetSize");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the size of the sparse matrix
        *rows = CusparseFrontend::GetOutputVariable<int64_t>();
        *cols = CusparseFrontend::GetOutputVariable<int64_t>();
        *nnz = CusparseFrontend::GetOutputVariable<int64_t>();
    }
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t* spsvDescr) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(spsvDescr);
    CusparseFrontend::Execute("cusparseSpSV_createDescr");
    if (CusparseFrontend::Success()) {
        // If the operation was successful, we can retrieve the descriptor
        *spsvDescr = CusparseFrontend::GetOutputVariable<cusparseSpSVDescr_t>();
    }
    return CusparseFrontend::GetExitCode();
}