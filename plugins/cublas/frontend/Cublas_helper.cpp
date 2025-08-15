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
 *             Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>,
 *             Department of Applied Science
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             School of Computer Science, University College Dublin
 */

#include "CublasFrontend.h"

using namespace std;

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t *handle) {
    CublasFrontend::Prepare();
    CublasFrontend::Execute("cublasCreate_v2");
    if (CublasFrontend::Success()) *handle = CublasFrontend::GetOutputVariable<cublasHandle_t>();
    // *handle =
    // reinterpret_cast<cublasHandle_t>(CublasFrontend::GetOutputDevicePointer());
    // // equivalent *handle =
    // reinterpret_cast<cublasHandle_t>(CublasFrontend::GetOutputVariable<uintptr_t>());
    // // also equivalent reinterpret_cast<cublasHandle_t> is equivalent to
    // (cublasHandle_t)
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetVector(int n, int elemSize, const void *x,
                                                                 int incx, void *y, int incy) {
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(elemSize);
    // CublasFrontend::AddHostPointerForArguments(x,sizeof(x));

    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddHostPointerForArguments<char>(static_cast<char *>(const_cast<void *>(x)),
                                                     n * elemSize);
    CublasFrontend::Execute("cublasSetVector");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetMatrix(int rows, int cols, int elemSize,
                                                                 const void *A, int lda, void *B,
                                                                 int ldb) {
    CublasFrontend::Prepare();

    CublasFrontend::AddVariableForArguments<int>(rows);
    CublasFrontend::AddVariableForArguments<int>(cols);
    CublasFrontend::AddVariableForArguments<int>(elemSize);
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddHostPointerForArguments<char>(static_cast<char *>(const_cast<void *>(A)),
                                                     rows * cols * elemSize);
    CublasFrontend::Execute("cublasSetMatrix");
    return CublasFrontend::GetExitCode();
}

/* This function copies n elements from a vector x in GPU memory space to a
 * vector y in host memory space. Elements in both vectors are assumed to have a
 * size of elemSize bytes. The storage spacing between consecutive elements is
 * given by incx for the source vector and incy for the destination vector y.
 */
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetVector(int n, int elemSize, const void *x,
                                                                 int incx, void *y, int incy) {
    CublasFrontend::Prepare();

    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(elemSize);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddVariableForArguments<int>(incy);

    // void * _x = const_cast<void *>(x);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddHostPointerForArguments<void>(y);

    CublasFrontend::Execute("cublasGetVector");

    if (CublasFrontend::Success()) {
        memmove(y, CublasFrontend::GetOutputHostPointer<char>(n * elemSize), n * elemSize);
    }
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetMatrix(int rows, int cols, int elemSize,
                                                                 const void *A, int lda, void *B,
                                                                 int ldb) {
    CublasFrontend::Prepare();

    CublasFrontend::AddVariableForArguments<int>(rows);
    CublasFrontend::AddVariableForArguments<int>(cols);
    CublasFrontend::AddVariableForArguments<int>(elemSize);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddHostPointerForArguments<void>(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);

    CublasFrontend::Execute("cublasGetMatrix");

    if (CublasFrontend::Success()) {
        memmove(B, CublasFrontend::GetOutputHostPointer<char>(rows * cols * elemSize),
                rows * cols * elemSize);
    }
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle) {
    CublasFrontend::Prepare();
    // CublasFrontend::AddVariableForArguments<long long int>((long long
    // int)handle); works until cublas < 12
    // CublasFrontend::AddVariableForArguments<uintptr_t>((uintptr_t)handle); //
    // this works if backend also reads the handle as a uintptr_t and then casts
    // it to cublasHandle_t
    CublasFrontend::AddDevicePointerForArguments(handle);
    // CublasFrontend::AddVariableForArguments<cublasHandle_t>(handle); // this
    // does not work as it tries to do sizeof cublasHandle_t which is an
    // incomplete type (opaque struct)
    CublasFrontend::Execute("cublasDestroy_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetVersion_v2(cublasHandle_t handle,
                                                                     int *version) {
    CublasFrontend::Prepare();
    CublasFrontend::Execute("cublasGetVersion_v2");
    if (CublasFrontend::Success()) *version = CublasFrontend::GetOutputVariable<int>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t handle,
                                                                    cudaStream_t streamId) {
    CublasFrontend::Prepare();

    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<long long int>((long long int)streamId);
    CublasFrontend::Execute("cublasSetStream_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetStream_v2(cublasHandle_t handle,
                                                                    cudaStream_t *streamId) {
    CublasFrontend::Prepare();

    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::Execute("cublasGetStream_v2");
    if (CublasFrontend::Success())
        *streamId = (cudaStream_t)CublasFrontend::GetOutputVariable<long long int>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode) {
    CublasFrontend::Prepare();

    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::Execute("cublasGetPointerMode_v2");
    if (CublasFrontend::Success()) *mode = CublasFrontend::GetOutputVariable<cublasPointerMode_t>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle,
                                                                         cublasPointerMode_t mode) {
    CublasFrontend::Prepare();

    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasPointerMode_t>(mode);
    CublasFrontend::Execute("cublasSetPointerMode_v2");
    return CublasFrontend::GetExitCode();
}

// TODO:
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                   cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha,
                   const float *const A[], int lda, float *const B[], int ldb, int batchCount) {
    CublasFrontend::Prepare();
    return CUBLAS_STATUS_NOT_SUPPORTED;  // Not implemented yet
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetMathMode(cublasHandle_t handle,
                                                                   cublasMath_t mode) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasMath_t>(mode);
    CublasFrontend::Execute("cublasSetMathMode");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetMathMode(cublasHandle_t handle,
                                                                   cublasMath_t *mode) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::Execute("cublasGetMathMode");
    if (CublasFrontend::Success()) *mode = CublasFrontend::GetOutputVariable<cublasMath_t>();
    return CublasFrontend::GetExitCode();
}

// TODO:
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrfBatched(cublasHandle_t handle, int n,
                                                                     float *const Aarray[], int lda,
                                                                     int *PivotArray,
                                                                     int *infoArray,
                                                                     int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddHostPointerForArguments<int>(PivotArray);  // Array of pivot indices
    CublasFrontend::AddHostPointerForArguments<int>(infoArray);   // Array of info values
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasSgetrfBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasSgetrfBatched");
    return CublasFrontend::GetExitCode();
}

// TODO:
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrsBatched(
    cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *const Aarray[],
    int lda, const int *devIpiv, float *const Barray[], int ldb, int *info, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(nrhs);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(devIpiv);  // Device pointer to pivot indices
    CublasFrontend::AddDevicePointerForArguments(Barray);   // Array of pointers to matrices B
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddHostPointerForArguments<int>(info);  // Host pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasSgetrsBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasSgetrsBatched");
    return CublasFrontend::GetExitCode();
}

// TODO:
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B,
    int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc,
    long long int strideC, int batchCount) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddHostPointerForArguments<const cuComplex>(alpha);
    CublasFrontend::AddHostPointerForArguments<const cuComplex>(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddVariableForArguments<long long int>(strideA);
    CublasFrontend::AddHostPointerForArguments<const cuComplex>(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddVariableForArguments<long long int>(strideB);
    CublasFrontend::AddHostPointerForArguments<const cuComplex>(beta);
    CublasFrontend::AddHostPointerForArguments<cuComplex>(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddVariableForArguments<long long int>(strideC);
    CublasFrontend::AddVariableForArguments<int>(batchCount);

    CublasFrontend::Execute("cublasCgemmStridedBatched");
    return CublasFrontend::GetExitCode();
}

// TODO:
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotEx(cublasHandle_t handle, int n,
                                                             const void *x, cudaDataType xType,
                                                             int incx, const void *y,
                                                             cudaDataType yType, int incy,
                                                             void *result, cudaDataType resultType,
                                                             cudaDataType executionType) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<cudaDataType>(xType);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<cudaDataType>(yType);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(result);
    CublasFrontend::AddVariableForArguments<cudaDataType>(resultType);
    CublasFrontend::AddVariableForArguments<cudaDataType>(executionType);
    CublasFrontend::Execute("cublasDotEx");
    if (CublasFrontend::Success()) {
        // If the operation was successful, we can retrieve the result
        // Note: The result is already in the output pointer provided
        // so no need to copy it again.
    }
    return CublasFrontend::GetExitCode();
}

// TODO:
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrfBatched(cublasHandle_t handle, int n,
                                                                     double *const Aarray[],
                                                                     int lda, int *PivotArray,
                                                                     int *infoArray,
                                                                     int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddHostPointerForArguments<int>(PivotArray);  // Array of pivot indices
    CublasFrontend::AddHostPointerForArguments<int>(infoArray);   // Array of info values
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasDgetrfBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasDgetrfBatched");
    return CublasFrontend::GetExitCode();
}

// TODO:
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsmBatched(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, int m, int n, const double *alpha, const double *const A[], int lda,
    double *const B[], int ldb, int batchCount) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<const double>(alpha);
    CublasFrontend::AddDevicePointerForArguments(A);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(B);  // Array of pointers to matrices B
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    // Add the necessary arguments for cublasDtrsmBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasDtrsmBatched");
    return CublasFrontend::GetExitCode();
}

// TODO:
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs,
                   cuComplex *const Aarray[], int lda, cuComplex *const Carray[], int ldc,
                   int *info, int *devInfoArray, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(nrhs);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(Carray);  // Array of pointers to matrices C
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddHostPointerForArguments<int>(info);       // Host pointer to info array
    CublasFrontend::AddDevicePointerForArguments(devInfoArray);  // Device pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasCgelsBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasCgelsBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddDevicePointerForArguments(workspace);
    CublasFrontend::AddVariableForArguments<size_t>(workspaceSizeInBytes);
    CublasFrontend::Execute("cublasSetWorkspace_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs,
                    const cuComplex *const Aarray[], int lda, const int *devIpiv,
                    cuComplex *const Barray[], int ldb, int *info, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(nrhs);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(devIpiv);  // Device pointer to pivot indices
    CublasFrontend::AddDevicePointerForArguments(Barray);   // Array of pointers to matrices B
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddHostPointerForArguments<int>(info);  // Host pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasCgetrsBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasCgetrsBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs,
                    const cuDoubleComplex *const Aarray[], int lda, const int *devIpiv,
                    cuDoubleComplex *const Barray[], int ldb, int *info, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(nrhs);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(devIpiv);  // Device pointer to pivot indices
    CublasFrontend::AddDevicePointerForArguments(Barray);   // Array of pointers to matrices B
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddHostPointerForArguments<int>(info);  // Host pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasZgetrsBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasZgetrsBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrfBatched(cublasHandle_t handle, int n,
                                                                     cuComplex *const Aarray[],
                                                                     int lda, int *PivotArray,
                                                                     int *infoArray,
                                                                     int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddHostPointerForArguments<int>(PivotArray);  // Array of pivot indices
    CublasFrontend::AddHostPointerForArguments<int>(infoArray);   // Array of info values
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasCgetrfBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasCgetrfBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                   cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                   const cuDoubleComplex *alpha, const cuDoubleComplex *const A[], int lda,
                   cuDoubleComplex *const B[], int ldb, int batchCount) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<const cuDoubleComplex>(alpha);
    CublasFrontend::AddDevicePointerForArguments(A);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(B);  // Array of pointers to matrices B
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    // Add the necessary arguments for cublasZtrsmBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasZtrsmBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeqrfBatched(cublasHandle_t handle, int m,
                                                                     int n, double *const Aarray[],
                                                                     int lda,
                                                                     double *const TauArray[],
                                                                     int *info, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(TauArray);  // Array of pointers to Tau
    CublasFrontend::AddHostPointerForArguments<int>(info);   // Host pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasDgeqrfBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasDgeqrfBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb,
    long long int strideB, const double *beta, double *C, int ldc, long long int strideC,
    int batchCount) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddHostPointerForArguments<const double>(alpha);
    CublasFrontend::AddHostPointerForArguments<const double>(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddVariableForArguments<long long int>(strideA);
    CublasFrontend::AddHostPointerForArguments<const double>(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddVariableForArguments<long long int>(strideB);
    CublasFrontend::AddHostPointerForArguments<const double>(beta);
    CublasFrontend::AddHostPointerForArguments<double>(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddVariableForArguments<long long int>(strideC);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    CublasFrontend::Execute("cublasDgemmStridedBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgelsBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double *const Aarray[],
    int lda, double *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(nrhs);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(Carray);  // Array of pointers to matrices C
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddHostPointerForArguments<int>(info);       // Host pointer to info array
    CublasFrontend::AddDevicePointerForArguments(devInfoArray);  // Device pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasDgelsBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasDgelsBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex *const Aarray[], int lda,
                    cuComplex *const TauArray[], int *info, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(TauArray);  // Array of pointers to Tau
    CublasFrontend::AddHostPointerForArguments<int>(info);   // Host pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasCgeqrfBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasCgeqrfBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex *const Aarray[], int lda,
                    int *PivotArray, int *infoArray, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddHostPointerForArguments<int>(PivotArray);  // Array of pivot indices
    CublasFrontend::AddHostPointerForArguments<int>(infoArray);   // Array of info values
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasZgetrfBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasZgetrfBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const float *alpha, const void *A, cudaDataType_t Atype, int lda, const void *B,
    cudaDataType_t Btype, int ldb, const float *beta, void *C, cudaDataType_t Ctype, int ldc) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddHostPointerForArguments<const float>(alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<cudaDataType_t>(Atype);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<cudaDataType_t>(Btype);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddHostPointerForArguments<const float>(beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<cudaDataType_t>(Ctype);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    // Add the necessary arguments for cublasSgemmEx
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasSgemmEx");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, long long int strideA,
    const cuDoubleComplex *B, int ldb, long long int strideB, const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc, long long int strideC, int batchCount) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddHostPointerForArguments<const cuDoubleComplex>(alpha);
    CublasFrontend::AddHostPointerForArguments<const cuDoubleComplex>(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddVariableForArguments<long long int>(strideA);
    CublasFrontend::AddHostPointerForArguments<const cuDoubleComplex>(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddVariableForArguments<long long int>(strideB);
    CublasFrontend::AddHostPointerForArguments<const cuDoubleComplex>(beta);
    CublasFrontend::AddHostPointerForArguments<cuDoubleComplex>(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddVariableForArguments<long long int>(strideC);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    CublasFrontend::Execute("cublasZgemmStridedBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex *const Aarray[], int lda,
                    cuDoubleComplex *const TauArray[], int *info, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(TauArray);  // Array of pointers to Tau
    CublasFrontend::AddHostPointerForArguments<int>(info);   // Host pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasZgeqrfBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasZgeqrfBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs,
                   cuDoubleComplex *const Aarray[], int lda, cuDoubleComplex *const Carray[],
                   int ldc, int *info, int *devInfoArray, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(nrhs);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(Carray);  // Array of pointers to matrices C
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddHostPointerForArguments<int>(info);       // Host pointer to info array
    CublasFrontend::AddDevicePointerForArguments(devInfoArray);  // Device pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasZgelsBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasZgelsBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsmBatched(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *const A[],
    int lda, cuComplex *const B[], int ldb, int batchCount) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<const cuComplex>(alpha);
    CublasFrontend::AddDevicePointerForArguments(A);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(B);  // Array of pointers to matrices B
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    // Add the necessary arguments for cublasCtrsmBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasCtrsmBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgelsBatched(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float *const Aarray[],
    int lda, float *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(nrhs);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(Carray);  // Array of pointers to matrices C
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddHostPointerForArguments<int>(info);       // Host pointer to info array
    CublasFrontend::AddDevicePointerForArguments(devInfoArray);  // Device pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasSgelsBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasSgelsBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeqrfBatched(cublasHandle_t handle, int m,
                                                                     int n, float *const Aarray[],
                                                                     int lda,
                                                                     float *const TauArray[],
                                                                     int *info, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(TauArray);  // Array of pointers to Tau
    CublasFrontend::AddHostPointerForArguments<int>(info);   // Host pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasSgeqrfBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasSgeqrfBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrsBatched(
    cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *const Aarray[],
    int lda, const int *devIpiv, double *const Barray[], int ldb, int *info, int batchSize) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(nrhs);
    CublasFrontend::AddDevicePointerForArguments(Aarray);  // Array of pointers to matrices A
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(devIpiv);  // Device pointer to pivot indices
    CublasFrontend::AddDevicePointerForArguments(Barray);   // Array of pointers to matrices B
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddHostPointerForArguments<int>(info);  // Host pointer to info array
    CublasFrontend::AddVariableForArguments<int>(batchSize);
    // Add the necessary arguments for cublasDgetrsBatched
    // This is a placeholder as the actual implementation is not provided
    CublasFrontend::Execute("cublasDgetrsBatched");
    return CublasFrontend::GetExitCode();
}