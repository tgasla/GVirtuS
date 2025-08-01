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
 * Written by: Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>,
 *             Department of Science and Technologies
 */

#include "CublasFrontend.h"

using namespace std;

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle, 
                                                    cublasOperation_t transa,
                                                    cublasOperation_t transb, 
                                                    int m,
                                                    int n,
                                                    int k,
                                                    const float *alpha, /* host or device pointer */  
                                                    const float *A, 
                                                    int lda,
                                                    const float *B,
                                                    int ldb, 
                                                    const float *beta, /* host or device pointer */  
                                                    float *C,
                                                    int ldc) {
    // TODO: For now, it only works if alpha, beta are host pointers.
    // If they are device pointers, we need to add a check in the frontend to handle them correctly.
    // If the function cublasSetPointerMode is called with CUBLAS_POINTER_MODE_DEVICE,
    // then alpha and beta should be device pointers, otherwise they should be host pointers.
    // We need to keep track of the pointer mode in the frontend and handle it accordingly.
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddHostPointerForArguments<const float>(alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddHostPointerForArguments<const float>(beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::Execute("cublasSgemm_v2");
    return CublasFrontend::GetExitCode();
}

#if CUDART_VERSION <= 9000
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched (cublasHandle_t handle,
                                                                cublasOperation_t transa, cublasOperation_t transb,
                                                                int m, int n, int k,
                                                                const float *alpha, const float *Aarray[], int lda,
                                                                const float *Barray[], int ldb,
                                                                const float *beta, float *Carray[], int ldc,
                                                                int batchCount){
#elif CUDART_VERSION > 9000
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched (cublasHandle_t handle,
                                                                cublasOperation_t transa, cublasOperation_t transb,
                                                                int m, int n, int k,
                                                                const float *alpha, const float *const Aarray[], int lda,
                                                                const float *const Barray[], int ldb,
                                                                const float *beta, float *const Carray[], int ldc,
                                                                int batchCount){
#endif

    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);

    CublasFrontend::AddDevicePointerForArguments(Aarray);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(Barray);
    CublasFrontend::AddVariableForArguments<int>(ldb);

    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);

    CublasFrontend::AddDevicePointerForArguments(Carray);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    CublasFrontend::Execute("cublasSgemmBatched_v2");
    if (CublasFrontend::Success())
        Carray = (float **)CublasFrontend::GetOutputDevicePointer();
    return CublasFrontend::GetExitCode();
}
  
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const double *alpha, /* host or device pointer */  
                                                      const double *A, 
                                                      int lda,
                                                      const double *B,
                                                      int ldb, 
                                                      const double *beta, /* host or device pointer */  
                                                      double *C,
                                                      int ldc) {
    CublasFrontend::Prepare();

    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddHostPointerForArguments<const double>(alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddHostPointerForArguments<const double>(beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasDgemm_v2");
    return CublasFrontend::GetExitCode();
}

#if CUDART_VERSION <= 9000

    extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched(cublasHandle_t handle,
                                                                    cublasOperation_t transa, cublasOperation_t transb,
                                                                    int m, int n, int k,
                                                                    const double *alpha, const double *Aarray[], int lda,
                                                                    const double *Barray[], int ldb,
                                                                    const double *beta, double *Carray[], int ldc,
                                                                    int batchCount) {

#elif CUDART_VERSION > 9000
    extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched(cublasHandle_t handle,
                                                                    cublasOperation_t transa, cublasOperation_t transb,
                                                                    int m, int n, int k,
                                                                    const double *alpha, const double *const Aarray[], int lda,
                                                                    const double *const Barray[], int ldb,
                                                                    const double *beta, double *const Carray[], int ldc,
                                                                    int batchCount) {
#endif

    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);

    CublasFrontend::AddDevicePointerForArguments(Aarray);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(Barray);
    CublasFrontend::AddVariableForArguments<int>(ldb);

    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);

    CublasFrontend::AddDevicePointerForArguments(Carray);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    CublasFrontend::Execute("cublasDgemmBatched_v2");
    if (CublasFrontend::Success())
        Carray = (double **)CublasFrontend::GetOutputDevicePointer();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha, /* host or device pointer */  
                                                      const cuComplex *A, 
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb, 
                                                      const cuComplex *beta, /* host or device pointer */  
                                                      cuComplex *C,
                                                      int ldc){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasCgemm_v2");
    if (CublasFrontend::Success())
        C = (cuComplex *)CublasFrontend::GetOutputDevicePointer();
    return CublasFrontend::GetExitCode();
}

#if CUDART_VERSION <= 9000

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa, cublasOperation_t transb,
                                                                int m, int n, int k,
                                                                const cuComplex *alpha, const cuComplex *Aarray[], int lda,
                                                                const cuComplex *Barray[], int ldb,
                                                                const cuComplex *beta, cuComplex *Carray[], int ldc,
                                                                int batchCount){
#elif CUDART_VERSION > 9000

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa, cublasOperation_t transb,
                                                                int m, int n, int k,
                                                                const cuComplex *alpha, const cuComplex *const Aarray[], int lda,
                                                                const cuComplex *const Barray[], int ldb,
                                                                const cuComplex *beta, cuComplex *const Carray[], int ldc,
                                                                int batchCount){

#endif

    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);

    CublasFrontend::AddDevicePointerForArguments(Aarray);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(Barray);
    CublasFrontend::AddVariableForArguments<int>(ldb);

    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);

    CublasFrontend::AddDevicePointerForArguments(Carray);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    CublasFrontend::Execute("cublasCgemmBatched_v2");
    if (CublasFrontend::Success())
        Carray = (cuComplex **)CublasFrontend::GetOutputDevicePointer();
    return CublasFrontend::GetExitCode();
} 

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm_v2 (cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha, /* host or device pointer */  
                                                      const cuDoubleComplex *A, 
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb, 
                                                      const cuDoubleComplex *beta, /* host or device pointer */  
                                                      cuDoubleComplex *C,
                                                      int ldc){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasZgemm_v2");
    if (CublasFrontend::Success())
        C = (cuDoubleComplex *)CublasFrontend::GetOutputDevicePointer();
    return CublasFrontend::GetExitCode();
}

#if CUDART_VERSION <= 9000

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa, cublasOperation_t transb,
                                                                int m, int n, int k,
                                                                const cuDoubleComplex *alpha, const cuDoubleComplex *Aarray[], int lda,
                                                                const cuDoubleComplex *Barray[], int ldb,
                                                                const cuDoubleComplex *beta, cuDoubleComplex *Carray[], int ldc,
                                                                int batchCount){
#elif CUDART_VERSION > 9000

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched(cublasHandle_t handle,
                                                                cublasOperation_t transa, cublasOperation_t transb,
                                                                int m, int n, int k,
                                                                const cuDoubleComplex *alpha, const cuDoubleComplex *const Aarray[], int lda,
                                                                const cuDoubleComplex *const Barray[], int ldb,
                                                                const cuDoubleComplex *beta, cuDoubleComplex *const Carray[], int ldc,
                                                                int batchCount){

#endif

    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);

    CublasFrontend::AddDevicePointerForArguments(Aarray);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(Barray);
    CublasFrontend::AddVariableForArguments<int>(ldb);

    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);

    CublasFrontend::AddDevicePointerForArguments(Carray);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    CublasFrontend::Execute("cublasZgemmBatched_v2");
    if (CublasFrontend::Success())
        Carray = (cuDoubleComplex **)CublasFrontend::GetOutputDevicePointer();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
    CublasFrontend::Prepare();
    
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasSnrm2_v2");
    if (CublasFrontend::Success())
        *result = CublasFrontend::GetOutputVariable<float>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
    CublasFrontend::Prepare();
    
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasDnrm2_v2");
    if(CublasFrontend::Success())
        *result = CublasFrontend::GetOutputVariable<double>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScnrm2_v2 (cublasHandle_t handle,int n, const cuComplex *x, int incx, float *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasScnrm2_v2");
    if(CublasFrontend::Success())
        *result = CublasFrontend::GetOutputVariable<float>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2 (cublasHandle_t handle,int n, const cuDoubleComplex *x, int incx, double *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasDznrm2_v2");
    if(CublasFrontend::Success())
        *result = CublasFrontend::GetOutputVariable<double>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasSsyrk_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasDsyrk_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, cuComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasCsyrk_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasZsyrk_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const cuComplex *A, int lda, const float *beta, cuComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasCherk_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const cuDoubleComplex *A, int lda, const double *beta, cuDoubleComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasZherk_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasSsyr2k_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasDsyr2k_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasCsyr2k_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasZsyr2k_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasCher2k_v2");
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasZher2k_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasSsymm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasDsymm_v2");
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasCsymm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasZsymm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasChemm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasZhemm_v2");
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    CublasFrontend::Execute("cublasStrsm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    CublasFrontend::Execute("cublasDtrsm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, cuComplex *B, int ldb){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    CublasFrontend::Execute("cublasCtrsm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    CublasFrontend::Execute("cublasZtrsm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, float *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasStrmm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, double *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasDtrmm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasCtrmm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasSideMode_t>(side);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    
    CublasFrontend::Execute("cublasZtrmm_v2");
    return CublasFrontend::GetExitCode();
}


// TODO: Add the missing cublasHgemm function implementation
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const __half *alpha,
                           const __half *A, int lda,
                           const __half *B, int ldb,
                           const __half *beta,
                           __half *C, int ldc) {
    CublasFrontend::Prepare();
    CublasFrontend::Execute("cublasHgemm");
    return CublasFrontend::GetExitCode();
}

// TODO: Add the missing cublasHgemmStridedBatched function implementation
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const __half           *alpha,
                                  const __half           *A, int lda,
                                  long long int          strideA,
                                  const __half           *B, int ldb,
                                  long long int          strideB,
                                  const __half           *beta,
                                  __half                 *C, int ldc,
                                  long long int          strideC,
                                  int batchCount) {
    CublasFrontend::Prepare();
    CublasFrontend::Execute("cublasHgemmStridedBatched");
    return CublasFrontend::GetExitCode();
}

// TODO: Add the missing cublasHgemmBatched function implementation
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const __half          *alpha,
                                  const __half          *const Aarray[], int lda,
                                  const __half          *const Barray[], int ldb,
                                  const __half          *beta,
                                  __half          *const Carray[], int ldc,
                                  int batchCount) {
    CublasFrontend::Prepare();
    CublasFrontend::Execute("cublasHgemmBatched");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float           *alpha,
                                  const float           *A, int lda,
                                  long long int          strideA,
                                  const float           *B, int ldb,
                                  long long int          strideB,
                                  const float           *beta,
                                  float                 *C, int ldc,
                                  long long int          strideC,
                                  int batchCount) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddHostPointerForArguments<const float>(alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddVariableForArguments<long long int>(strideA);
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddVariableForArguments<long long int>(strideB);
    CublasFrontend::AddHostPointerForArguments<const float>(beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddVariableForArguments<long long int>(strideC);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    CublasFrontend::Execute("cublasSgemmStridedBatched");
    return CublasFrontend::GetExitCode();
}