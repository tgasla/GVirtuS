/* GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Copyright (C) 2025
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             School of Computer Science, University College Dublin
 */

#include "CublasFrontend.h"

using namespace std;

// TODO: this only suppors the version where alpha, beta are float32 host
// pointers
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const void *alpha, const void *A, cudaDataType_t Atype, int lda, const void *B,
    cudaDataType_t Btype, int ldb, const void *beta, void *C, cudaDataType_t Ctype, int ldc,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddHostPointerForArguments<const float>((float *)alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<cudaDataType_t>(Atype);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<cudaDataType_t>(Btype);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddHostPointerForArguments<const float>((float *)beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<cudaDataType_t>(Ctype);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddVariableForArguments<cublasComputeType_t>(computeType);
    CublasFrontend::AddVariableForArguments<cublasGemmAlgo_t>(algo);
    CublasFrontend::Execute("cublasGemmEx");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const void *alpha, const void *A, cudaDataType_t Atype, int lda, long long int strideA,
    const void *B, cudaDataType_t Btype, int ldb, long long int strideB, const void *beta, void *C,
    cudaDataType_t Ctype, int ldc, long long int strideC, int batchCount,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transa);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(transb);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddHostPointerForArguments<const float>((float *)alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<cudaDataType_t>(Atype);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddVariableForArguments<long long int>(strideA);
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddVariableForArguments<cudaDataType_t>(Btype);
    CublasFrontend::AddVariableForArguments<int>(ldb);
    CublasFrontend::AddVariableForArguments<long long int>(strideB);
    CublasFrontend::AddHostPointerForArguments<const float>((float *)beta);
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddVariableForArguments<cudaDataType_t>(Ctype);
    CublasFrontend::AddVariableForArguments<int>(ldc);
    CublasFrontend::AddVariableForArguments<long long int>(strideC);
    CublasFrontend::AddVariableForArguments<int>(batchCount);
    CublasFrontend::AddVariableForArguments<cublasComputeType_t>(computeType);
    CublasFrontend::AddVariableForArguments<cublasGemmAlgo_t>(algo);
    CublasFrontend::Execute("cublasGemmStridedBatchedEx");
    return CublasFrontend::GetExitCode();
}