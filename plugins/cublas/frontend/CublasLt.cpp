/*
 * GVirtuS -- A GPGPU transparent virtualization component.
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             School of Computer Science, University College Dublin
 */

#include "CublasFrontend.h"

using namespace std;

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int *returnAlgoCount) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(lightHandle);
    CublasFrontend::AddDevicePointerForArguments(operationDesc);
    CublasFrontend::AddDevicePointerForArguments(Adesc);
    CublasFrontend::AddDevicePointerForArguments(Bdesc);
    CublasFrontend::AddDevicePointerForArguments(Cdesc);
    CublasFrontend::AddDevicePointerForArguments(Ddesc);
    CublasFrontend::AddDevicePointerForArguments(preference);
    CublasFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CublasFrontend::Execute("cublasLtMatmulAlgoGetHeuristic");
    if (CublasFrontend::Success()) {
        // Copy the results back to the output array
        cublasLtMatmulHeuristicResult_t *temp =
            CublasFrontend::GetOutputHostPointer<cublasLtMatmulHeuristicResult_t>(
                requestedAlgoCount);
        memcpy(heuristicResultsArray, temp,
               requestedAlgoCount * sizeof(cublasLtMatmulHeuristicResult_t));
        *returnAlgoCount = CublasFrontend::GetOutputVariable<int>();
    }
    return CublasFrontend::GetExitCode();
}

// TODO
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutSetAttribute(
    cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, const void *buf,
    size_t sizeInBytes) {
    CublasFrontend::Prepare();
    CublasFrontend::Execute("cublasLtMatrixLayoutSetAttribute");
    cout << "Hello from cublasLtMatrixLayoutSetAttribute" << endl;
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr,
                               const void *buf, size_t sizeInBytes) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(matmulDesc);
    CublasFrontend::AddVariableForArguments<cublasLtMatmulDescAttributes_t>(attr);
    CublasFrontend::AddVariableForArguments<size_t>(sizeInBytes);
    CublasFrontend::AddHostPointerForArguments(buf, sizeInBytes);
    CublasFrontend::Execute("cublasLtMatmulDescSetAttribute");
    if (CublasFrontend::Success()) {
        matmulDesc = CublasFrontend::GetOutputVariable<cublasLtMatmulDesc_t>();
    }
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, cudaDataType type, uint64_t rows,
                           uint64_t cols, int64_t ld) {
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<cudaDataType>(type);
    CublasFrontend::AddVariableForArguments<uint64_t>(rows);
    CublasFrontend::AddVariableForArguments<uint64_t>(cols);
    CublasFrontend::AddVariableForArguments<int64_t>(ld);
    CublasFrontend::Execute("cublasLtMatrixLayoutCreate");
    if (CublasFrontend::Success()) {
        *matLayout = CublasFrontend::GetOutputVariable<cublasLtMatrixLayout_t>();
    }
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(matLayout);
    CublasFrontend::Execute("cublasLtMatrixLayoutDestroy");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(
    cublasLtMatmulDesc_t *matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType) {
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<cublasComputeType_t>(computeType);
    CublasFrontend::AddVariableForArguments<cudaDataType_t>(scaleType);
    // cout << "Creating LtMatmulDesc " << *matmulDesc << " with computeType: "
    //  << computeType << ", scaleType: " << scaleType << endl;
    CublasFrontend::Execute("cublasLtMatmulDescCreate");
    if (CublasFrontend::Success()) {
        *matmulDesc = CublasFrontend::GetOutputVariable<cublasLtMatmulDesc_t>();
        // cout << "matmulDesc: " << *matmulDesc << endl;
    }
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(matmulDesc);
    CublasFrontend::Execute("cublasLtMatmulDescDestroy");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void *buf,
    size_t sizeInBytes) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(pref);
    CublasFrontend::AddVariableForArguments<cublasLtMatmulPreferenceAttributes_t>(attr);
    CublasFrontend::AddVariableForArguments<size_t>(sizeInBytes);
    CublasFrontend::AddHostPointerForArguments(buf, sizeInBytes);
    CublasFrontend::Execute("cublasLtMatmulPreferenceSetAttribute");
    if (CublasFrontend::Success()) {
        pref = CublasFrontend::GetOutputVariable<cublasLtMatmulPreference_t>();
    }
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *preference) {
    CublasFrontend::Prepare();
    CublasFrontend::Execute("cublasLtMatmulPreferenceCreate");
    if (CublasFrontend::Success()) {
        *preference = CublasFrontend::GetOutputVariable<cublasLtMatmulPreference_t>();
    }
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(pref);
    CublasFrontend::Execute("cublasLtMatmulPreferenceDestroy");
    return CublasFrontend::GetExitCode();
}

// TODO: now it only works only if alpha, beta are floats on host
// In reality, they can bei float, double or int on host or device
// This needs to be fixed in the future
extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasLtMatmul(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha,
    const void *A, cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc,
    const void *beta, const void *C, cublasLtMatrixLayout_t Cdesc, void *D,
    cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo, void *workspace,
    size_t workspaceSizeInBytes, cudaStream_t stream) {
    CublasFrontend::Prepare();
    CublasFrontend::AddDevicePointerForArguments(lightHandle);
    CublasFrontend::AddDevicePointerForArguments(computeDesc);
    CublasFrontend::AddHostPointerForArguments(alpha, sizeof(float));
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddDevicePointerForArguments(Adesc);
    CublasFrontend::AddDevicePointerForArguments(B);
    CublasFrontend::AddDevicePointerForArguments(Bdesc);
    CublasFrontend::AddHostPointerForArguments(beta, sizeof(float));
    CublasFrontend::AddDevicePointerForArguments(C);
    CublasFrontend::AddDevicePointerForArguments(Cdesc);
    CublasFrontend::AddDevicePointerForArguments(D);
    CublasFrontend::AddDevicePointerForArguments(Ddesc);
    CublasFrontend::AddHostPointerForArguments<const cublasLtMatmulAlgo_t>(algo);
    CublasFrontend::AddDevicePointerForArguments(workspace);
    CublasFrontend::AddVariableForArguments<size_t>(workspaceSizeInBytes);
    CublasFrontend::AddDevicePointerForArguments(stream);

    // cout << "Executing cublasLtMatmul with lightHandle: " << lightHandle
    //     << ", computeDesc: " << computeDesc
    //     << ", alpha: " << *(float*)alpha
    //     << ", A: " << A
    //     << ", Adesc: " << Adesc
    //     << ", B: " << B
    //     << ", Bdesc: " << Bdesc
    //     << ", beta: " << *(float*)beta
    //     << ", C: " << C
    //     << ", Cdesc: " << Cdesc
    //     << ", D: " << D
    //     << ", Ddesc: " << Ddesc
    //     << ", algo: " << algo
    //     << ", workspace: " << workspace
    //     << ", workspaceSizeInBytes: " << workspaceSizeInBytes
    //     << ", stream: " << stream << endl;
    CublasFrontend::Execute("cublasLtMatmul");
    return CublasFrontend::GetExitCode();
}