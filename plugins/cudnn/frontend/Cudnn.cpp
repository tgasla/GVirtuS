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
 *             School of Computer Science, University College Dublin
 */

#include <mutex>
#include <unordered_map>

#include "CudnnFrontend.h"

using namespace std;

static std::mutex desc_type_mutex;
static std::unordered_map<void *, bool> desc_is_float_map;

// Helper Functions

void printHex(const void *data, size_t length, const std::string &label) {
    auto bytes = reinterpret_cast<const unsigned char *>(data);
    std::cout << label << " (" << length << " bytes): ";
    for (size_t i = 0; i < length; i++) {
        printf("%02X ", bytes[i]);                               // print each byte as two-digit hex
        if ((i + 1) % 16 == 0) std::cout << std::endl << "   ";  // break lines for readability
    }
    std::cout << std::endl;
}

size_t getCudnnTypeSize(cudnnBackendAttributeType_t type) {
    switch (type) {
        case CUDNN_TYPE_HANDLE:
            return sizeof(cudnnHandle_t);
        case CUDNN_TYPE_DATA_TYPE:
            return sizeof(cudnnDataType_t);
        case CUDNN_TYPE_BOOLEAN:
            return sizeof(bool);
        case CUDNN_TYPE_INT64:
            return sizeof(int64_t);
        case CUDNN_TYPE_FLOAT:
            return sizeof(float);
        case CUDNN_TYPE_DOUBLE:
            return sizeof(double);
        case CUDNN_TYPE_VOID_PTR:
            return sizeof(void *);
        case CUDNN_TYPE_CONVOLUTION_MODE:
            return sizeof(cudnnConvolutionMode_t);
        case CUDNN_TYPE_HEUR_MODE:
            return sizeof(cudnnBackendHeurMode_t);
        case CUDNN_TYPE_KNOB_TYPE:
            return sizeof(cudnnBackendKnobType_t);
        case CUDNN_TYPE_NAN_PROPOGATION:
            return sizeof(cudnnNanPropagation_t);
        case CUDNN_TYPE_NUMERICAL_NOTE:
            return sizeof(cudnnBackendNumericalNote_t);
        case CUDNN_TYPE_LAYOUT_TYPE:
            return sizeof(cudnnBackendLayoutType_t);
        case CUDNN_TYPE_ATTRIB_NAME:
            return sizeof(cudnnBackendAttributeName_t);
        case CUDNN_TYPE_POINTWISE_MODE:
            return sizeof(cudnnPointwiseMode_t);
        case CUDNN_TYPE_BACKEND_DESCRIPTOR:
            return sizeof(cudnnBackendDescriptor_t);
        case CUDNN_TYPE_GENSTATS_MODE:
            return sizeof(cudnnGenStatsMode_t);
        case CUDNN_TYPE_BN_FINALIZE_STATS_MODE:
            return sizeof(cudnnBnFinalizeStatsMode_t);
        case CUDNN_TYPE_REDUCTION_OPERATOR_TYPE:
            return sizeof(cudnnReduceTensorOp_t);
        case CUDNN_TYPE_BEHAVIOR_NOTE:
            return sizeof(cudnnBackendBehaviorNote_t);
        case CUDNN_TYPE_TENSOR_REORDERING_MODE:
            return sizeof(cudnnBackendTensorReordering_t);
        case CUDNN_TYPE_RESAMPLE_MODE:
            return sizeof(cudnnResampleMode_t);
        case CUDNN_TYPE_PADDING_MODE:
            return sizeof(cudnnPaddingMode_t);
        case CUDNN_TYPE_INT32:
            return sizeof(int32_t);
        case CUDNN_TYPE_CHAR:
            return sizeof(char);
        case CUDNN_TYPE_SIGNAL_MODE:
            return sizeof(cudnnSignalMode_t);
        case CUDNN_TYPE_FRACTION:
            return sizeof(cudnnFraction_t);
        case CUDNN_TYPE_NORM_MODE:
            return sizeof(cudnnBackendNormMode_t);
        case CUDNN_TYPE_NORM_FWD_PHASE:
            return sizeof(cudnnBackendNormFwdPhase_t);
        case CUDNN_TYPE_RNG_DISTRIBUTION:
            return sizeof(cudnnRngDistribution_t);
        default:
            return 0;
    }
}

// Generic setter (used when you create a descriptor)
void registerDescriptorType(void *desc, const cudnnDataType_t dataType) {
    std::lock_guard<std::mutex> lock(desc_type_mutex);
    desc_is_float_map[desc] = (dataType != CUDNN_DATA_DOUBLE);
}

// Generic getter for descriptor type
bool isFloatDescriptor(const void *desc) {
    std::lock_guard<std::mutex> lock(desc_type_mutex);
    auto it = desc_is_float_map.find(const_cast<void *>(desc));
    if (it != desc_is_float_map.end()) {
        return it->second;
    }
    return true;  // Default if unknown, assume float
}

extern "C" size_t CUDNNWINAPI cudnnGetVersion() {
    CudnnFrontend::Prepare();
    CudnnFrontend::Execute("cudnnGetVersion");
    return CudnnFrontend::GetExitCode();  // Instead of exit code, we return the version directly!
}

extern "C" const char *CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<cudnnStatus_t>(status);
    CudnnFrontend::Execute("cudnnGetErrorString");
    return CudnnFrontend::GetOutputString();
}

extern "C" void CUDNNWINAPI cudnnGetLastErrorString(char *message, size_t max_size) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<size_t>(max_size);
    CudnnFrontend::Execute("cudnnGetLastErrorString");
    if (CudnnFrontend::Success()) {
        message = CudnnFrontend::GetOutputString();
        // this is safe because cuda promises to always return a null-terminated string
        // even if the message is longer than max_size
    }
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateTensorDescriptor");
    if (CudnnFrontend::Success()) {
        *tensorDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                                                cudnnTensorFormat_t format,
                                                                cudnnDataType_t dataType, int n,
                                                                int c, int h, int w) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(format);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(n);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::Execute("cudnnSetTensor4dDescriptor");
    if (CudnnFrontend::Success()) {
        // tensorDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
        registerDescriptorType(tensorDesc, dataType);
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(
    cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int n, int c, int h, int w,
    int nStride, int cStride, int hStride, int wStride) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(n);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::AddVariableForArguments<int>(nStride);
    CudnnFrontend::AddVariableForArguments<int>(cStride);
    CudnnFrontend::AddVariableForArguments<int>(hStride);
    CudnnFrontend::AddVariableForArguments<int>(wStride);

    CudnnFrontend::Execute("cudnnSetTensor4dDescriptorEx");
    if (CudnnFrontend::Success()) {
        registerDescriptorType(tensorDesc, dataType);
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(
    const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t *dataType, int *n, int *c, int *h,
    int *w, int *nStride, int *cStride, int *hStride, int *wStride) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
    CudnnFrontend::Execute("cudnnGetTensor4dDescriptor");

    if (CudnnFrontend::Success()) {
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *n = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
        *nStride = CudnnFrontend::GetOutputVariable<int>();
        *cStride = CudnnFrontend::GetOutputVariable<int>();
        *hStride = CudnnFrontend::GetOutputVariable<int>();
        *wStride = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                                                cudnnDataType_t dataType,
                                                                int nbDims, const int *dimA,
                                                                const int *strideA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<const int>(dimA, nbDims);
    CudnnFrontend::AddHostPointerForArguments<const int>(strideA, nbDims);

    CudnnFrontend::Execute("cudnnSetTensorNdDescriptor");
    if (CudnnFrontend::Success()) {
        cout << "Successfully set tensor descriptor " << tensorDesc << " with data type "
             << dataType << endl;
        registerDescriptorType(tensorDesc, dataType);
        cout << "isFloatDescriptor: " << isFloatDescriptor(tensorDesc) << endl;
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format,
                             cudnnDataType_t dataType, int nbDims, const int *dimA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(format);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)dimA);

    if (CudnnFrontend::Success()) {
        registerDescriptorType(tensorDesc, dataType);
    }
    return CudnnFrontend::GetExitCode();
}

// Method F1: This method is compatible with the method B1 of the backend.
// In this method, we do not pass uninitialized pointers from frontend (e.g., dataType, nbDims,
// dimA, strideA), as these are all outputs. For these variables, the backend uses arrays and
// variables allocated in the stack. The Backend sends the dataType and nbDims as values, so we need
// to read the buffer as values as well (e.g., GetOutputVariable) and not as pointers!. Note: If we
// try to read these variables as pointers and dereference them it will not work. For example,
// doing: *dataType = *CudnnFrontend::GetOutputHostVariable<cudnnDataType_t>() will not work. The
// same applies for nbDims as well. extern "C" cudnnStatus_t CUDNNWINAPI
// cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
//                             int nbDimsRequested,
//                             cudnnDataType_t *dataType,
//                             int *nbDims,
//                             int dimA[],
//                             int strideA[]) {

//     CudnnFrontend::Prepare();

//     CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
//     CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
//     CudnnFrontend::Execute("cudnnGetTensorNdDescriptor");
//     if (CudnnFrontend::Success()) {
//         *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
//         *nbDims = CudnnFrontend::GetOutputVariable<int>();
//         int *dimA_backend = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
//         std::memcpy(dimA, dimA_backend, nbDimsRequested * sizeof(int));
//         int *strideA_backend = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
//         std::memcpy(strideA, strideA_backend, nbDimsRequested * sizeof(int));
//     }
//     return CudnnFrontend::GetExitCode();
// }

// Method F2: This method is compatible with the method B2 of the backend.
// In this method, again, we do not pass any uninitialized pointers from frontend (e.g., dataType,
// nbDims, dimA, strideA), as these are all outputs. For these variables, the backend uses arrays
// and variables allocated in the heap. More specifically, the backend uses the Delegate Method to
// dynamically allocate memory for these variables and directly write them into the output buffer.
// The Backend sends the dataType and nbDims as pointers, so we need to
// read the buffer as host pointers (e.g., GetOutputHostPointer) and dereference them to get the
// values. For example, we do: *dataType = *CudnnFrontend::GetOutputHostPointer<cudnnDataType_t>();
// and the same applies for nbDims as well.
// Note: If we try to read these variables as variables directly, it will not work.
// For example, doing:
// *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>()
// will not work. The same applies for nbDims as well.
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
                           cudnnDataType_t *dataType, int *nbDims, int dimA[], int strideA[]) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::Execute("cudnnGetTensorNdDescriptor");
    if (CudnnFrontend::Success()) {
        *dataType = *CudnnFrontend::GetOutputHostPointer<cudnnDataType_t>();  // Important!
        *nbDims = *CudnnFrontend::GetOutputHostPointer<int>();                // Important!
        int *dimA_backend = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
        std::memcpy(dimA, dimA_backend, nbDimsRequested * sizeof(int));
        int *strideA_backend = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
        std::memcpy(strideA, strideA_backend, nbDimsRequested * sizeof(int));
    }
    return CudnnFrontend::GetExitCode();
}

// Method F3: This method is compatible with the method B3 of the backend.
// In this method, we pass uninitialized pointers for all output arguments (e.g., dataType, nbDims,
// dimA, strideA) For these variables, the backend uses Get function to allocate memory for these
// variables, then the function manipulates their values and writes them into the output buffer.
// Because the buffer writes the values of these variables, we need to read them as variables and
// not as pointers, similar to Method F1. extern "C" cudnnStatus_t CUDNNWINAPI
// cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
//                             int nbDimsRequested,
//                             cudnnDataType_t *dataType,
//                             int *nbDims,
//                             int dimA[],
//                             int strideA[]) {

//     CudnnFrontend::Prepare();

//     CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
//     CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
//     CudnnFrontend::AddHostPointerForArguments<cudnnDataType_t>(dataType);
//     CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
//     CudnnFrontend::AddHostPointerForArguments<int>(dimA, nbDimsRequested);
//     CudnnFrontend::AddHostPointerForArguments<int>(strideA, nbDimsRequested);
//     CudnnFrontend::Execute("cudnnGetTensorNdDescriptor");
//     if (CudnnFrontend::Success()) {
//         *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
//         *nbDims = CudnnFrontend::GetOutputVariable<int>();
//         int *dimA_backend = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
//         std::memcpy(dimA, dimA_backend, nbDimsRequested * sizeof(int));
//         int *strideA_backend = CudnnFrontend::GetOutputHostPointer<int>(nbDimsRequested);
//         std::memcpy(strideA, strideA_backend, nbDimsRequested * sizeof(int));
//     }
//     return CudnnFrontend::GetExitCode();
// }

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t *size) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(tensorDesc);

    CudnnFrontend::Execute("cudnnGetTensorSizeInBytes");
    if (CudnnFrontend::Success()) {
        *size = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(tensorDesc);
    CudnnFrontend::Execute("cudnnDestroyTensorDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnInitTransformDest(
    const cudnnTensorTransformDescriptor_t transformDesc, const cudnnTensorDescriptor_t srcDesc,
    cudnnTensorDescriptor_t destDesc, size_t *destSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(transformDesc);
    CudnnFrontend::AddDevicePointerForArguments(srcDesc);

    CudnnFrontend::Execute("cudnnInitTransformDest");

    if (CudnnFrontend::Success()) {
        destDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
        *destSizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t *transformDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateTensorTransformDescriptor");

    if (CudnnFrontend::Success()) {
        *transformDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc, const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat, const int32_t *padBeforeA, const int32_t *padAfterA,
    const uint32_t *foldA, const cudnnFoldingDirection_t direction) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<uint32_t>(nbDims);
    CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(destFormat);
    CudnnFrontend::AddHostPointerForArguments<const int32_t>(padBeforeA);
    CudnnFrontend::AddHostPointerForArguments<const int32_t>(padAfterA);
    CudnnFrontend::AddHostPointerForArguments<const uint32_t>(foldA);
    CudnnFrontend::AddVariableForArguments<cudnnFoldingDirection_t>(direction);

    CudnnFrontend::Execute("cudnnSetTensorTransformDescriptor");
    // if (CudnnFrontend::Success()) {
    //     transformDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
    // }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc, uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat, int32_t *padBeforeA, int32_t *padAfterA, uint32_t *foldA,
    cudnnFoldingDirection_t *direction) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(transformDesc);
    CudnnFrontend::AddVariableForArguments<uint32_t>(nbDimsRequested);

    CudnnFrontend::Execute("cudnnGetTensorTransformDescriptor");
    if (CudnnFrontend::Success()) {
        *destFormat = CudnnFrontend::GetOutputVariable<cudnnTensorFormat_t>();
        *padBeforeA = CudnnFrontend::GetOutputVariable<int32_t>();
        *padAfterA = CudnnFrontend::GetOutputVariable<int32_t>();
        *foldA = CudnnFrontend::GetOutputVariable<uint32_t>();
        *direction = CudnnFrontend::GetOutputVariable<cudnnFoldingDirection_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(transformDesc);

    CudnnFrontend::Execute("cudnnDestroyTensorTransformDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(cudnnHandle_t handle, const void *alpha,
                                                          const cudnnTensorDescriptor_t xDesc,
                                                          const void *x, const void *beta,
                                                          const cudnnTensorDescriptor_t yDesc,
                                                          void *y) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(y);

    CudnnFrontend::Execute("cudnnTransformTensor");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnTransformTensorEx(
    cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc, const void *alpha,
    const cudnnTensorDescriptor_t srcDesc, const void *srcData, const void *beta,
    const cudnnTensorDescriptor_t destDesc, void *destData) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(transDesc);
    CudnnFrontend::AddDevicePointerForArguments(srcDesc);
    isFloatDescriptor(srcDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                     reinterpret_cast<const float *>(alpha))
                               : CudnnFrontend::AddHostPointerForArguments<const double>(
                                     reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddHostPointerForArguments(srcData);
    CudnnFrontend::AddDevicePointerForArguments(destDesc);
    isFloatDescriptor(destDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                      reinterpret_cast<const float *>(beta))
                                : CudnnFrontend::AddHostPointerForArguments<const double>(
                                      reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(destData);

    CudnnFrontend::Execute("cudnnTransformTensorEx");
    if (CudnnFrontend::Success()) {
        destData = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFoldedConvBackwardDataDescriptors(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc, cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc, cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);

    CudnnFrontend::Execute("cudnnGetFoldedConvBackwardDataDescriptors");

    if (CudnnFrontend::Success()) {
        foldedFilterDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
        paddedDiffDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
        foldedConvDesc = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>();
        foldedGradDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
        filterFoldTransDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
        diffPadTransDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
        gradFoldTransDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
        gradUnfoldTransDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t handle, const void *alpha,
                                                    const cudnnTensorDescriptor_t aDesc,
                                                    const void *A, const void *beta,
                                                    const cudnnTensorDescriptor_t cDesc, void *C) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(aDesc);
    isFloatDescriptor(aDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(A);
    CudnnFrontend::AddDevicePointerForArguments(cDesc);
    isFloatDescriptor(cDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(C);

    CudnnFrontend::Execute("cudnnAddTensor");
    if (CudnnFrontend::Success()) {
        C = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateOpTensorDescriptor");
    if (CudnnFrontend::Success()) {
        *opTensorDesc = CudnnFrontend::GetOutputVariable<cudnnOpTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp,
                           cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnOpTensorOp_t>(opTensorOp);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(opTensorCompType);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(opTensorNanOpt);

    CudnnFrontend::Execute("cudnnSetOpTensorDescriptor");
    if (CudnnFrontend::Success()) {
        opTensorDesc = CudnnFrontend::GetOutputVariable<cudnnOpTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetOpTensorDescriptor(
    const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(opTensorDesc);

    CudnnFrontend::Execute("cudnnGetOpTensorDescriptor");
    if (CudnnFrontend::Success()) {
        *opTensorOp = CudnnFrontend::GetOutputVariable<cudnnOpTensorOp_t>();
        *opTensorCompType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *opTensorNanOpt = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(opTensorDesc);

    CudnnFrontend::Execute("cudnnDestroyOpTensorDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnOpTensor(cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
              const void *alpha1, const cudnnTensorDescriptor_t aDesc, const void *A,
              const void *alpha2, const cudnnTensorDescriptor_t bDesc, const void *B,
              const void *beta, const cudnnTensorDescriptor_t cDesc, void *C) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(opTensorDesc);
    CudnnFrontend::AddDevicePointerForArguments(aDesc);
    isFloatDescriptor(aDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha1))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha1));
    CudnnFrontend::AddDevicePointerForArguments(A);
    CudnnFrontend::AddDevicePointerForArguments(bDesc);
    isFloatDescriptor(bDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha2))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha2));
    CudnnFrontend::AddDevicePointerForArguments(B);
    CudnnFrontend::AddDevicePointerForArguments(cDesc);
    isFloatDescriptor(cDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(C);

    CudnnFrontend::Execute("cudnnOpTensor");
    if (CudnnFrontend::Success()) {
        C = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateReduceTensorDescriptor");
    if (CudnnFrontend::Success()) {
        *reduceTensorDesc = CudnnFrontend::GetOutputVariable<cudnnReduceTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t reduceTensorOp,
    cudnnDataType_t reduceTensorCompType, cudnnNanPropagation_t reduceTensorNanOpt,
    cudnnReduceTensorIndices_t reduceTensorIndices, cudnnIndicesType_t reduceTensorIndicesType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnReduceTensorOp_t>(reduceTensorOp);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(reduceTensorCompType);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(reduceTensorNanOpt);
    CudnnFrontend::AddVariableForArguments<cudnnReduceTensorIndices_t>(reduceTensorIndices);
    CudnnFrontend::AddVariableForArguments<cudnnIndicesType_t>(reduceTensorIndicesType);

    CudnnFrontend::Execute("cudnnSetReduceTensorDescriptor");
    if (CudnnFrontend::Success()) {
        reduceTensorDesc = CudnnFrontend::GetOutputVariable<cudnnReduceTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType, cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices, cudnnIndicesType_t *reduceTensorIndicesType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(reduceTensorDesc);
    CudnnFrontend::AddDevicePointerForArguments(reduceTensorNanOpt);

    CudnnFrontend::Execute("cudnnGetReduceTensorDescriptor");
    if (CudnnFrontend::Success()) {
        *reduceTensorOp = CudnnFrontend::GetOutputVariable<cudnnReduceTensorOp_t>();
        *reduceTensorCompType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *reduceTensorIndices = CudnnFrontend::GetOutputVariable<cudnnReduceTensorIndices_t>();
        *reduceTensorIndicesType = CudnnFrontend::GetOutputVariable<cudnnIndicesType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc) {
    CudnnFrontend::AddDevicePointerForArguments(reduceTensorDesc);

    CudnnFrontend::Execute("cudnnDestroyReduceTensorDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(reduceTensorDesc);
    CudnnFrontend::AddDevicePointerForArguments(aDesc);
    CudnnFrontend::AddDevicePointerForArguments(cDesc);

    CudnnFrontend::Execute("cudnnGetReductionIndicesSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(reduceTensorDesc);
    CudnnFrontend::AddDevicePointerForArguments(aDesc);
    CudnnFrontend::AddDevicePointerForArguments(cDesc);

    CudnnFrontend::Execute("cudnnGetReductionWorkspaceSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, void *indices,
    size_t indicesSizeInBytes, void *workspace, size_t workspaceSizeInBytes, const void *alpha,
    const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta,
    const cudnnTensorDescriptor_t cDesc, void *C) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<size_t>(indicesSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workspaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(aDesc);
    isFloatDescriptor(aDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(A);
    CudnnFrontend::AddDevicePointerForArguments(cDesc);
    isFloatDescriptor(cDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(C);

    CudnnFrontend::Execute("cudnnReduceTensor");
    if (CudnnFrontend::Success()) {
        indices = CudnnFrontend::GetOutputDevicePointer();
        C = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor(cudnnHandle_t handle,
                                                    const cudnnTensorDescriptor_t yDesc, void *y,
                                                    const void *valuePtr) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddHostPointerForArguments((void *)valuePtr);

    CudnnFrontend::Execute("cudnnSetTensor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(cudnnHandle_t handle,
                                                      const cudnnTensorDescriptor_t yDesc, void *y,
                                                      const void *alpha) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));

    CudnnFrontend::Execute("cudnnScaleTensor");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateFilterDescriptor");
    if (CudnnFrontend::Success()) {
        *filterDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                                                cudnnDataType_t dataType,
                                                                cudnnTensorFormat_t format, int k,
                                                                int c, int h, int w) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(format);

    CudnnFrontend::AddVariableForArguments<int>(k);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::Execute("cudnnSetFilter4dDescriptor");
    if (CudnnFrontend::Success()) {
        registerDescriptorType(filterDesc, dataType);
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                                                cudnnDataType_t *dataType,
                                                                cudnnTensorFormat_t *format, int *k,
                                                                int *c, int *h, int *w) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);

    CudnnFrontend::Execute("cudnnGetFilter4dDescriptor");

    if (CudnnFrontend::Success()) {
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *format = CudnnFrontend::GetOutputVariable<cudnnTensorFormat_t>();
        *k = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }

    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 6000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor_v3(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, int k, int c, int h, int w) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::AddDevicePointerForArguments(dataType);

    CudnnFrontend::AddVariableForArguments<int>(k);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::Execute("cudnnSetFilter4dDescriptor_v3");
    if (CudnnFrontend::Success()) registerDescriptorType(filterDesc, dataType);
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilter4dDescriptor_v3(
    cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType, int *k, int *c, int *h, int *w) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);

    CudnnFrontend::Execute("cudnnGetFilter4dDescriptor_v3");

    if (CudnnFrontend::Success()) {
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *k = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetFilter4dDescriptor_v4(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
                              cudnnTensorFormat_t format, int k, int c, int h, int w) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::AddDevicePointerForArguments(dataType);
    CudnnFrontend::AddDevicePointerForArguments(format);

    CudnnFrontend::AddVariableForArguments<int>(k);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::Execute("cudnnSetFilter4dDescriptor_v4");
    if (CudnnFrontend::Success()) {
        registerDescriptorType(filterDesc, dataType);
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetFilter4dDescriptor_v4(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType,
                              cudnnTensorFormat_t *format, int *k, int *c, int *h, int *w) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);

    CudnnFrontend::Execute("cudnnGetFilter4dDescriptor_v4");

    if (CudnnFrontend::Success()) {
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *format = CudnnFrontend::GetOutputVariable<cudnnTensorFormat_t>();
        *k = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }

    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                                                                cudnnDataType_t dataType,
                                                                cudnnTensorFormat_t format,
                                                                int nbDims, const int *filterDimA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(format);

    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<const int>(filterDimA);

    CudnnFrontend::Execute("cudnnSetFilterNdDescriptor");
    if (CudnnFrontend::Success()) registerDescriptorType(filterDesc, dataType);
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t wDesc,
                                                                int nbDimsRequested,
                                                                cudnnDataType_t *dataType,
                                                                cudnnTensorFormat_t *format,
                                                                int *nbDims, int *filterDimA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments(dataType);

    CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>(filterDimA);

    CudnnFrontend::Execute("cudnnGetFilterNdDescriptor");
    if (CudnnFrontend::Success()) {
        *format = CudnnFrontend::GetOutputVariable<cudnnTensorFormat_t>();
    }

    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 6000
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetFilterNdDescriptor_v3(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
                              int nbDims, const int *filterDimA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::AddDevicePointerForArguments(dataType);

    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)filterDimA);

    CudnnFrontend::Execute("cudnnSetFilterNdDescriptor_v3");
    if (CudnnFrontend::Success()) registerDescriptorType(filterDesc, dataType);
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetFilterNdDescriptor_v3(const cudnnFilterDescriptor_t wDesc, int nbDimsRequested,
                              cudnnDataType_t *dataType, int *nbDims, int *filterDimA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments(dataType);

    CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>(filterDimA);

    CudnnFrontend::Execute("cudnnGetFilterNdDescriptor_v3");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetFilterNdDescriptor_v4(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType,
                              cudnnTensorFormat_t format, int nbDims, const int *filterDimA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::AddDevicePointerForArguments(dataType);
    CudnnFrontend::AddDevicePointerForArguments(format);

    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)filterDimA);

    CudnnFrontend::Execute("cudnnSetFilterNdDescriptor_v4");
    if (CudnnFrontend::Success()) registerDescriptorType(filterDesc, dataType);
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor_v4(
    const cudnnFilterDescriptor_t wDesc, int nbDimsRequested, cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format, int *nbDims, int *filterDimA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments(dataType);

    CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>(filterDimA);

    CudnnFrontend::Execute("cudnnGetFilterNdDescriptor_v4");
    if (CudnnFrontend::Success()) {
        *format = CudnnFrontend::GetOutputVariable<cudnnTensorFormat_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t *size) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);

    CudnnFrontend::Execute("cudnnGetFilterSizeInBytes");
    if (CudnnFrontend::Success()) {
        *size = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::Execute("cudnnDestroyFilterDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnTransformFilter(cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc,
                     const void *alpha, const cudnnFilterDescriptor_t srcDesc, const void *srcData,
                     const void *beta, const cudnnFilterDescriptor_t destDesc, void *destData) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(transDesc);
    CudnnFrontend::AddDevicePointerForArguments(srcDesc);
    isFloatDescriptor(srcDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                     reinterpret_cast<const float *>(alpha))
                               : CudnnFrontend::AddHostPointerForArguments<const double>(
                                     reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(srcData);
    CudnnFrontend::AddDevicePointerForArguments(destDesc);
    isFloatDescriptor(destDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                      reinterpret_cast<const float *>(beta))
                                : CudnnFrontend::AddHostPointerForArguments<const double>(
                                      reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(destData);

    CudnnFrontend::Execute("cudnnTransformFilter");
    if (CudnnFrontend::Success()) {
        destData = CudnnFrontend::GetOutputDevicePointer();
    }

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnReorderFilterAndBias(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, cudnnReorderType_t reorderType,
    const void *filterData, void *reorderedFilterData, int reorderBias, const void *biasData,
    void *reorderedBiasData) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::AddVariableForArguments<cudnnReorderType_t>(reorderType);
    CudnnFrontend::AddDevicePointerForArguments(filterData);
    CudnnFrontend::AddDevicePointerForArguments(reorderedFilterData);
    CudnnFrontend::AddVariableForArguments<int>(reorderBias);
    CudnnFrontend::AddDevicePointerForArguments(biasData);
    CudnnFrontend::AddDevicePointerForArguments(reorderedBiasData);

    CudnnFrontend::Execute("cudnnReorderFilterAndBias");

    if (CudnnFrontend::Success()) {
        reorderedFilterData = CudnnFrontend::GetOutputDevicePointer();
        reorderedBiasData = CudnnFrontend::GetOutputDevicePointer();
    }

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateConvolutionDescriptor");
    if (CudnnFrontend::Success()) {
        *convDesc = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnMathType_t>(mathType);

    CudnnFrontend::Execute("cudnnSetConvolutionMathType");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);

    CudnnFrontend::Execute("cudnnGetConvolutionMathType");
    if (CudnnFrontend::Success()) {
        *mathType = CudnnFrontend::GetOutputVariable<cudnnMathType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddVariableForArguments<int>(groupCount);

    CudnnFrontend::Execute("cudnnSetConvolutionGroupCount");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int *groupCount) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);

    CudnnFrontend::Execute("cudnnGetConvolutionGroupCount");
    if (CudnnFrontend::Success()) {
        *groupCount = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionReorderType(
    cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnReorderType_t>(reorderType);

    CudnnFrontend::Execute("cudnnSetConvolutionReorderType");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionReorderType(
    cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);

    CudnnFrontend::Execute("cudnnGetConvolutionReorderType");
    if (CudnnFrontend::Success()) {
        *reorderType = CudnnFrontend::GetOutputVariable<cudnnReorderType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v, int upscalex,
    int upscaley, cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddVariableForArguments<int>(pad_h);
    CudnnFrontend::AddVariableForArguments<int>(pad_w);
    CudnnFrontend::AddVariableForArguments<int>(u);
    CudnnFrontend::AddVariableForArguments<int>(v);
    CudnnFrontend::AddVariableForArguments<int>(upscalex);
    CudnnFrontend::AddVariableForArguments<int>(upscaley);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(computeType);

    CudnnFrontend::Execute("cudnnSetConvolution2dDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dDescriptor(
    const cudnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_w, int *u, int *v,
    int *upscalex, int *upscaley, cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);

    CudnnFrontend::Execute("cudnnGetConvolution2dDescriptor");
    if (CudnnFrontend::Success()) {
        *pad_h = CudnnFrontend::GetOutputVariable<int>();
        *pad_w = CudnnFrontend::GetOutputVariable<int>();
        *u = CudnnFrontend::GetOutputVariable<int>();
        *v = CudnnFrontend::GetOutputVariable<int>();
        *upscalex = CudnnFrontend::GetOutputVariable<int>();
        *upscaley = CudnnFrontend::GetOutputVariable<int>();
        *mode = CudnnFrontend::GetOutputVariable<cudnnConvolutionMode_t>();
        *computeType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(inputTensorDesc);
    CudnnFrontend::AddDevicePointerForArguments(filterDesc);

    CudnnFrontend::Execute("cudnnGetConvolution2dForwardOutputDim");
    if (CudnnFrontend::Success()) {
        *n = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc, int arrayLength,
                                const int *padA, const int *filterStrideA, const int *dilationA,
                                cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddVariableForArguments<int>(arrayLength);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)padA);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)filterStrideA);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)dilationA);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(computeType);

    CudnnFrontend::Execute("cudnnSetConvolutionNdDescriptor");
    if (CudnnFrontend::Success()) {
        convDesc = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(
    const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested, int *arrayLength,
    int *padA, int *strideA, int *dilationA, cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddVariableForArguments<int>(arrayLengthRequested);

    CudnnFrontend::Execute("cudnnGetConvolutionNdDescriptor");
    if (CudnnFrontend::Success()) {
        arrayLengthRequested = CudnnFrontend::GetOutputVariable<int>();
        *arrayLength = CudnnFrontend::GetOutputVariable<int>();
        *strideA = CudnnFrontend::GetOutputVariable<int>();
        *dilationA = CudnnFrontend::GetOutputVariable<int>();
        *mode = CudnnFrontend::GetOutputVariable<cudnnConvolutionMode_t>();
        *computeType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims, int *tensorOuputDimA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(inputTensorDesc);
    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);

    CudnnFrontend::Execute("cudnnGetConvolutionNdForwardOutputDim");
    if (CudnnFrontend::Success()) {
        int *outputDims = CudnnFrontend::GetOutputHostPointer<int>(nbDims);
        std::memcpy(tensorOuputDimA, outputDims, nbDims * sizeof(int));
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(convDesc);

    CudnnFrontend::Execute("cudnnDestroyConvolutionDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);

    CudnnFrontend::Execute("cudnnGetConvolutionForwardAlgorithmMaxCount");
    if (CudnnFrontend::Success()) {
        *count = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);

    CudnnFrontend::Execute("cudnnFindConvolutionForwardAlgorithm");
    if (CudnnFrontend::Success()) {
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        *perfResults = CudnnFrontend::GetOutputVariable<cudnnConvolutionFwdAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, void *y, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnConvolutionFwdAlgoPerf_t>(perfResults,
                                                                             requestedAlgoCount);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindConvolutionForwardAlgorithmEx");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnConvolutionFwdAlgoPerf_t *backend_perfResults =
            CudnnFrontend::GetOutputHostPointer<cudnnConvolutionFwdAlgoPerf_t>(*returnedAlgoCount);
        std::memcpy(perfResults, backend_perfResults,
                    sizeof(cudnnConvolutionFwdAlgoPerf_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 8000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionFwdAlgo_t *algo) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdPreference_t>(preference);
    CudnnFrontend::AddVariableForArguments<size_t>(memoryLimitInBytes);

    CudnnFrontend::Execute("cudnnGetConvolutionForwardAlgorithm");
    if (CudnnFrontend::Success()) {
        *algo = CudnnFrontend::GetOutputVariable<cudnnConvolutionFwdAlgo_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

#if CUDNN_VERSION >= 7000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t *perfResults) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(srcDesc);
    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(destDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnConvolutionFwdAlgoPerf_t>(perfResults,
                                                                             requestedAlgoCount);

    CudnnFrontend::Execute("cudnnGetConvolutionForwardAlgorithm_v7");
    if (CudnnFrontend::Success()) {
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnConvolutionFwdAlgoPerf_t *backend_perfResults =
            CudnnFrontend::GetOutputHostPointer<cudnnConvolutionFwdAlgoPerf_t>(*returnedAlgoCount);
        std::memcpy(perfResults, backend_perfResults,
                    sizeof(cudnnConvolutionFwdAlgoPerf_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t algo, size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdAlgo_t>(algo);

    CudnnFrontend::Execute("cudnnGetConvolutionForwardWorkspaceSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(
    cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments(alpha, sizeof(float))
                             : CudnnFrontend::AddHostPointerForArguments(alpha, sizeof(double));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(y);

    CudnnFrontend::Execute("cudnnConvolutionForward");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle, const void *alpha1, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes,
    const void *alpha2, const cudnnTensorDescriptor_t zDesc, const void *z,
    const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc,
    void *y) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha1))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha1));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(zDesc);
    isFloatDescriptor(zDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha2))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha2));
    CudnnFrontend::AddDevicePointerForArguments(z);
    CudnnFrontend::AddDevicePointerForArguments(biasDesc);
    CudnnFrontend::AddDevicePointerForArguments(bias);
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);

    CudnnFrontend::Execute("cudnnConvolutionBiasActivationForward");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(
    cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const void *beta, const cudnnTensorDescriptor_t dbDesc, void *db) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    isFloatDescriptor(dyDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(alpha))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(dbDesc);
    isFloatDescriptor(dbDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(beta))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(db);

    CudnnFrontend::Execute("cudnnConvolutionBackwardBias");
    if (CudnnFrontend::Success()) {
        db = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
    if (CudnnFrontend::Success()) *count = CudnnFrontend::GetOutputVariable<int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(dwDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnConvolutionBwdFilterAlgoPerf_t>(
        perfResults, requestedAlgoCount);

    CudnnFrontend::Execute("cudnnFindConvolutionBackwardFilterAlgorithm");
    if (CudnnFrontend::Success()) {
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnConvolutionBwdFilterAlgoPerf_t *perfResults_backend =
            CudnnFrontend::GetOutputHostPointer<cudnnConvolutionBwdFilterAlgoPerf_t>(
                *returnedAlgoCount);
        std::memcpy(perfResults, perfResults_backend,
                    sizeof(cudnnConvolutionBwdFilterAlgoPerf_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *y,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(dwDesc);
    CudnnFrontend::AddDevicePointerForArguments(dw);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnConvolutionBwdFilterAlgoPerf_t>(
        perfResults, requestedAlgoCount);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindConvolutionBackwardFilterAlgorithmEx");
    if (CudnnFrontend::Success()) {
        dw = CudnnFrontend::GetOutputDevicePointer();
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnConvolutionBwdFilterAlgoPerf_t *perfResults_backend =
            CudnnFrontend::GetOutputHostPointer<cudnnConvolutionBwdFilterAlgoPerf_t>(
                *returnedAlgoCount);
        std::memcpy(perfResults, perfResults,
                    sizeof(cudnnConvolutionBwdFilterAlgoPerf_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 8000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(dwDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdFilterPreference_t>(preference);
    CudnnFrontend::AddVariableForArguments<size_t>(memoryLimitInBytes);

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterAlgorithm");
    if (CudnnFrontend::Success()) {
        *algo = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdFilterAlgo_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

#if CUDNN_VERSION >= 7000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(srcDesc);
    CudnnFrontend::AddDevicePointerForArguments(diffDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(gradDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnConvolutionBwdFilterAlgoPerf_t>(
        perfResults, requestedAlgoCount);

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterAlgorithm_v7");
    if (CudnnFrontend::Success()) {
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnConvolutionBwdFilterAlgoPerf_t *perfResults_backend =
            CudnnFrontend::GetOutputHostPointer<cudnnConvolutionBwdFilterAlgoPerf_t>(
                *returnedAlgoCount);
        std::memcpy(perfResults, perfResults_backend,
                    sizeof(cudnnConvolutionBwdFilterAlgoPerf_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(gradDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdFilterAlgo_t>(algo);

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterWorkspaceSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdFilterAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(dwDesc);
    isFloatDescriptor(dwDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(beta))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(dw);

    CudnnFrontend::Execute("cudnnConvolutionBackwardFilter");
    if (CudnnFrontend::Success()) {
        dw = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
    if (CudnnFrontend::Success()) *count = CudnnFrontend::GetOutputVariable<int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnConvolutionBwdDataAlgoPerf_t>(
        perfResults, requestedAlgoCount);

    CudnnFrontend::Execute("cudnnFindConvolutionBackwardDataAlgorithm");
    if (CudnnFrontend::Success()) {
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnConvolutionBwdDataAlgoPerf_t *perfResults_backend =
            CudnnFrontend::GetOutputHostPointer<cudnnConvolutionBwdDataAlgoPerf_t>(
                *returnedAlgoCount);
        std::memcpy(perfResults, perfResults_backend,
                    sizeof(cudnnConvolutionBwdDataAlgoPerf_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace, size_t workSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dx);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnConvolutionBwdDataAlgoPerf_t>(
        perfResults, requestedAlgoCount);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindConvolutionBackwardDataAlgorithmEx");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnConvolutionBwdDataAlgoPerf_t *perfResults_backend =
            CudnnFrontend::GetOutputHostPointer<cudnnConvolutionBwdDataAlgoPerf_t>(
                *returnedAlgoCount);
        std::memcpy(perfResults, perfResults_backend,
                    sizeof(cudnnConvolutionBwdDataAlgoPerf_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 8000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdDataAlgo_t *algo) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdDataPreference_t>(preference);
    CudnnFrontend::AddVariableForArguments<size_t>(memoryLimitInBytes);

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataAlgorithm");
    if (CudnnFrontend::Success()) {
        *algo = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdDataAlgo_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

#if CUDNN_VERSION >= 7000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(filterDesc);
    CudnnFrontend::AddDevicePointerForArguments(diffDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(gradDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnConvolutionBwdDataAlgoPerf_t>(
        perfResults, requestedAlgoCount);

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataAlgorithm_v7");
    if (CudnnFrontend::Success()) {
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnConvolutionBwdDataAlgoPerf_t *perfResults_backend =
            CudnnFrontend::GetOutputHostPointer<cudnnConvolutionBwdDataAlgoPerf_t>(
                *returnedAlgoCount);
        std::memcpy(perfResults, perfResults_backend,
                    sizeof(cudnnConvolutionBwdDataAlgoPerf_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo, size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdDataAlgo_t>(algo);

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataWorkspaceSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha, const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    isFloatDescriptor(wDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdDataAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    isFloatDescriptor(dxDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(beta))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(dx);

    CudnnFrontend::Execute("cudnnConvolutionBackwardData");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnIm2Col(cudnnHandle_t handle,
                                                 const cudnnTensorDescriptor_t xDesc, const void *x,
                                                 const cudnnFilterDescriptor_t wDesc,
                                                 const cudnnConvolutionDescriptor_t convDesc,
                                                 void *colBuffer) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(convDesc);

    CudnnFrontend::Execute("cudnnIm2Col");
    if (CudnnFrontend::Success()) {
        colBuffer = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSoftmaxForward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
                    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxAlgorithm_t>(algo);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxMode_t>(mode);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(y);

    CudnnFrontend::Execute("cudnnSoftmaxForward");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxAlgorithm_t>(algo);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxMode_t>(mode);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    isFloatDescriptor(dxDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(beta))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(beta));

    CudnnFrontend::Execute("cudnnSoftmaxBackward");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreatePoolingDescriptor");
    if (CudnnFrontend::Success())
        *poolingDesc = CudnnFrontend::GetOutputVariable<cudnnPoolingDescriptor_t>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(
    cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth, int verticalPadding,
    int horizontalPadding, int verticalStride, int horizontalStride) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(poolingDesc);
    CudnnFrontend::AddVariableForArguments<cudnnPoolingMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(maxpoolingNanOpt);
    CudnnFrontend::AddVariableForArguments<int>(windowHeight);
    CudnnFrontend::AddVariableForArguments<int>(windowWidth);
    CudnnFrontend::AddVariableForArguments<int>(verticalPadding);
    CudnnFrontend::AddVariableForArguments<int>(horizontalPadding);
    CudnnFrontend::AddVariableForArguments<int>(verticalStride);
    CudnnFrontend::AddVariableForArguments<int>(horizontalStride);

    CudnnFrontend::Execute("cudnnSetPooling2dDescriptor");

    if (CudnnFrontend::Success()) {
        poolingDesc = CudnnFrontend::GetOutputVariable<cudnnPoolingDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescripto(
    const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight, int *windowWidth,
    int *verticalPadding, int *horizontalPadding, int *verticalStride, int *horizontalStride) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(poolingDesc);

    CudnnFrontend::Execute("cudnnGetPooling2dDescripto");
    if (CudnnFrontend::Success()) {
        *mode = CudnnFrontend::GetOutputVariable<cudnnPoolingMode_t>();
        *maxpoolingNanOpt = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
        *windowHeight = CudnnFrontend::GetOutputVariable<int>();
        *windowWidth = CudnnFrontend::GetOutputVariable<int>();
        *verticalPadding = CudnnFrontend::GetOutputVariable<int>();
        *horizontalPadding = CudnnFrontend::GetOutputVariable<int>();
        *verticalStride = CudnnFrontend::GetOutputVariable<int>();
        *horizontalStride = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode,
                            const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
                            const int *windowDimA, const int *paddingA, const int *strideA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(poolingDesc);
    CudnnFrontend::AddVariableForArguments<cudnnPoolingMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(maxpoolingNanOpt);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)windowDimA);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)paddingA);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)strideA);

    CudnnFrontend::Execute("cudnnSetPoolingNdDescriptor");
    if (CudnnFrontend::Success()) {
        poolingDesc = CudnnFrontend::GetOutputVariable<cudnnPoolingDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
                            cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
                            int *nbDims, int *windowDimA, int *paddingA, int *strideA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(poolingDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments<cudnnNanPropagation_t>(
        (cudnnNanPropagation_t *)maxpoolingNanOpt);

    CudnnFrontend::Execute("cudnnGetPoolingNdDescriptor");
    if (CudnnFrontend::Success()) {
        *mode = CudnnFrontend::GetOutputVariable<cudnnPoolingMode_t>();
        *nbDims = CudnnFrontend::GetOutputVariable<int>();
        *windowDimA = CudnnFrontend::GetOutputVariable<int>();
        *paddingA = CudnnFrontend::GetOutputVariable<int>();
        *strideA = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdForwardOutputDim(
    const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    int nbDims, int *outputTensorDimA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(poolingDesc);
    CudnnFrontend::AddDevicePointerForArguments(inputTensorDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);

    CudnnFrontend::Execute("cudnnGetPoolingNdForwardOutputDim");
    if (CudnnFrontend::Success()) {
        *outputTensorDimA = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dForwardOutputDim(
    const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    int *n, int *c, int *h, int *w) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(poolingDesc);
    CudnnFrontend::AddDevicePointerForArguments(inputTensorDesc);

    CudnnFrontend::Execute("cudnnGetPooling2dForwardOutputDim");
    if (CudnnFrontend::Success()) {
        *n = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(poolingDesc);

    CudnnFrontend::Execute("cudnnDestroyPoolingDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnPoolingForward(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
                    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(poolingDesc);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(y);

    CudnnFrontend::Execute("cudnnPoolingForward");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(poolingDesc);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    isFloatDescriptor(dxDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(beta))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(dx);

    CudnnFrontend::Execute("cudnnPoolingBackward");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateActivationDescriptor");
    if (CudnnFrontend::Success())
        *activationDesc = CudnnFrontend::GetOutputVariable<cudnnActivationDescriptor_t>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode,
                             cudnnNanPropagation_t reluNanOpt, double coef) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);
    CudnnFrontend::AddVariableForArguments<cudnnActivationMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(reluNanOpt);
    CudnnFrontend::AddVariableForArguments<double>(coef);
    CudnnFrontend::Execute("cudnnSetActivationDescriptor");

    if (CudnnFrontend::Success()) {
        activationDesc = CudnnFrontend::GetOutputVariable<cudnnActivationDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetActivationDescriptor(
    const cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t *mode,
    cudnnNanPropagation_t *reluNanOpt, double *coef) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);
    CudnnFrontend::Execute("cudnnGetActivationDescriptor");
    if (CudnnFrontend::Success()) {
        *mode = CudnnFrontend::GetOutputVariable<cudnnActivationMode_t>();
        *reluNanOpt = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
        *coef = CudnnFrontend::GetOutputVariable<double>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);
    CudnnFrontend::Execute("cudnnDestroyActivationDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnActivationForward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
                       const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
                       const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(y);

    CudnnFrontend::Execute("cudnnActivationForward");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    isFloatDescriptor(dxDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(beta))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(dx);

    CudnnFrontend::Execute("cudnnActivationBackward");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateLRNDescriptor");
    if (CudnnFrontend::Success()) {
        *normDesc = CudnnFrontend::GetOutputVariable<cudnnLRNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                                           unsigned lrnN, double lrnAlpha,
                                                           double lrnBeta, double lrnK) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(normDesc);
    CudnnFrontend::AddVariableForArguments<unsigned>(lrnN);
    CudnnFrontend::AddVariableForArguments<double>(lrnAlpha);
    CudnnFrontend::AddVariableForArguments<double>(lrnBeta);
    CudnnFrontend::AddVariableForArguments<double>(lrnK);

    CudnnFrontend::Execute("cudnnSetLRNDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                                           unsigned *lrnN, double *lrnAlpha,
                                                           double *lrnBeta, double *lrnK) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(normDesc);

    CudnnFrontend::Execute("cudnnGetLRNDescriptor");
    if (CudnnFrontend::Success()) {
        *lrnN = CudnnFrontend::GetOutputVariable<unsigned>();
        *lrnAlpha = CudnnFrontend::GetOutputVariable<double>();
        *lrnBeta = CudnnFrontend::GetOutputVariable<double>();
        *lrnK = CudnnFrontend::GetOutputVariable<double>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(lrnDesc);

    CudnnFrontend::Execute("cudnnDestroyLRNDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(normDesc);
    CudnnFrontend::AddVariableForArguments<cudnnLRNMode_t>(lrnMode);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::Execute("cudnnLRNCrossChannelForward");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(normDesc);
    CudnnFrontend::AddVariableForArguments<cudnnLRNMode_t>(lrnMode);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    isFloatDescriptor(dxDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(beta))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(dx);

    CudnnFrontend::Execute("cudnnLRNCrossChannelBackward");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *means, void *temp, void *temp2,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(normDesc);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(means);
    CudnnFrontend::AddDevicePointerForArguments(temp);
    CudnnFrontend::AddDevicePointerForArguments(temp2);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));

    CudnnFrontend::Execute("cudnnDivisiveNormalizationForward");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *means, const void *dy,
    void *temp, void *temp2, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx,
    void *dMeans) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(normDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDivNormMode_t>(mode);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(means);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(temp);
    CudnnFrontend::AddDevicePointerForArguments(temp2);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    isFloatDescriptor(dxDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(beta))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(beta));

    CudnnFrontend::Execute("cudnnDivisiveNormalizationBackward");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
        dMeans = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                              const cudnnTensorDescriptor_t xDesc, cudnnBatchNormMode_t mode) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);

    CudnnFrontend::Execute("cudnnDeriveBNTensorDescriptor");
    if (CudnnFrontend::Success()) {
        derivedBnDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormOps_t>(bnOps);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(zDesc);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(bnScaleBiasMeanVarDesc);
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);

    CudnnFrontend::Execute("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormOps_t>(bnOps);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dzDesc);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dBnScaleBiasDesc);
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);

    CudnnFrontend::Execute("cudnnGetBatchNormalizationBackwardExWorkspaceSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc,
    size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormOps_t>(bnOps);
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);

    CudnnFrontend::Execute("cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc,
    void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor, void *resultRunningMean,
    void *resultRunningVariance, double epsilon, void *resultSaveMean,
    void *resultSaveInvVariance) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(bnScaleBiasMeanVarDesc);
    CudnnFrontend::AddDevicePointerForArguments(bnScale);
    CudnnFrontend::AddDevicePointerForArguments(bnBias);
    CudnnFrontend::AddVariableForArguments<double>(exponentialAverageFactor);
    CudnnFrontend::AddDevicePointerForArguments(resultRunningMean);
    CudnnFrontend::AddDevicePointerForArguments(resultRunningVariance);
    CudnnFrontend::AddVariableForArguments<double>(epsilon);
    CudnnFrontend::AddDevicePointerForArguments(resultSaveMean);
    CudnnFrontend::AddDevicePointerForArguments(resultSaveInvVariance);

    CudnnFrontend::Execute("cudnnBatchNormalizationForwardTraining");
    if (CudnnFrontend::Success()) {
        resultRunningMean = CudnnFrontend::GetOutputDevicePointer();
        resultRunningVariance = CudnnFrontend::GetOutputDevicePointer();
        resultSaveMean = CudnnFrontend::GetOutputDevicePointer();
        resultSaveInvVariance = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t zDesc, const void *zData, const cudnnTensorDescriptor_t yDesc,
    void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor, void *resultRunningMean,
    void *resultRunningVariance, double epsilon, void *saveMean, void *saveInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workspace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormOps_t>(bnOps);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(xData);
    CudnnFrontend::AddDevicePointerForArguments(zDesc);
    CudnnFrontend::AddDevicePointerForArguments(zData);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(bnScaleBiasMeanVarDesc);
    CudnnFrontend::AddDevicePointerForArguments(bnScale);
    CudnnFrontend::AddDevicePointerForArguments(bnBias);
    CudnnFrontend::AddVariableForArguments<double>(exponentialAverageFactor);
    CudnnFrontend::AddDevicePointerForArguments(resultRunningMean);
    CudnnFrontend::AddDevicePointerForArguments(resultRunningVariance);
    CudnnFrontend::AddVariableForArguments<double>(epsilon);
    CudnnFrontend::AddDevicePointerForArguments(saveMean);
    CudnnFrontend::AddDevicePointerForArguments(saveInvVariance);
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);
    CudnnFrontend::AddDevicePointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(reserveSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnBatchNormalizationForwardTrainingEx");
    if (CudnnFrontend::Success()) {
        resultRunningMean = CudnnFrontend::GetOutputDevicePointer();
        resultRunningVariance = CudnnFrontend::GetOutputDevicePointer();
        saveMean = CudnnFrontend::GetOutputDevicePointer();
        saveInvVariance = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc,
    void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    // cout << "xDesc: " << xDesc << endl;
    // cout << "isFloatDescriptor(xDesc): " << isFloatDescriptor(xDesc) << endl;
    if (isFloatDescriptor(xDesc)) {
        CudnnFrontend::AddHostPointerForArguments<char>((char *)alpha, sizeof(float));
        CudnnFrontend::AddHostPointerForArguments<char>((char *)beta, sizeof(float));
    } else {
        CudnnFrontend::AddHostPointerForArguments<char>((char *)alpha, sizeof(double));
        CudnnFrontend::AddHostPointerForArguments<char>((char *)beta, sizeof(double));
    }
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(bnScaleBiasMeanVarDesc);
    CudnnFrontend::AddDevicePointerForArguments(bnScale);
    CudnnFrontend::AddDevicePointerForArguments(bnBias);
    CudnnFrontend::AddDevicePointerForArguments(estimatedMean);
    CudnnFrontend::AddDevicePointerForArguments(estimatedVariance);
    CudnnFrontend::AddVariableForArguments<double>(epsilon);

    CudnnFrontend::Execute("cudnnBatchNormalizationForwardInference");
    // if (CudnnFrontend::Success()) {
    //     y = CudnnFrontend::GetOutputDevicePointer();
    // }

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alphaDataDiff,
    const void *betaDataDiff, const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale, void *dBnScaleResult,
    void *dBnBiasResult, double epsilon, const void *savedMean, const void *savedInvVariance) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alphaDataDiff))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alphaDataDiff));
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(betaDataDiff))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(betaDataDiff));
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alphaParamDiff))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alphaParamDiff));
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(betaParamDiff))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(betaParamDiff));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dx);
    CudnnFrontend::AddDevicePointerForArguments(dBnScaleBiasDesc);
    CudnnFrontend::AddDevicePointerForArguments(bnScale);
    CudnnFrontend::AddDevicePointerForArguments(dBnScaleResult);
    CudnnFrontend::AddDevicePointerForArguments(dBnBiasResult);
    CudnnFrontend::AddVariableForArguments<double>(epsilon);
    CudnnFrontend::AddDevicePointerForArguments(savedMean);
    CudnnFrontend::AddDevicePointerForArguments(savedInvVariance);

    CudnnFrontend::Execute("cudnnBatchNormalizationBackward");
    if (CudnnFrontend::Success()) {
        dBnScaleResult = CudnnFrontend::GetOutputDevicePointer();
        dBnBiasResult = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const void *alphaDataDiff, const void *betaDataDiff, const void *alphaParamDiff,
    const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t yDesc, const void *yData, const cudnnTensorDescriptor_t dyDesc,
    const void *dyData, const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScaleData, const void *bnBiasData,
    void *dBnScaleData, void *dBnBiasData, double epsilon, const void *savedMean,
    const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormOps_t>(bnOps);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alphaDataDiff))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alphaDataDiff));
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(betaDataDiff))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(betaDataDiff));
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alphaParamDiff))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alphaParamDiff));
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(betaParamDiff))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(betaParamDiff));
    CudnnFrontend::AddDevicePointerForArguments(xData);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(yData);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dyData);
    CudnnFrontend::AddDevicePointerForArguments(dzDesc);
    CudnnFrontend::AddDevicePointerForArguments(dzData);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    CudnnFrontend::AddHostPointerForArguments(dxData);
    CudnnFrontend::AddDevicePointerForArguments(dBnScaleBiasDesc);
    CudnnFrontend::AddDevicePointerForArguments(bnScaleData);
    CudnnFrontend::AddDevicePointerForArguments(bnBiasData);
    CudnnFrontend::AddDevicePointerForArguments(dBnScaleData);
    CudnnFrontend::AddDevicePointerForArguments(dBnBiasData);
    CudnnFrontend::AddVariableForArguments<double>(epsilon);
    CudnnFrontend::AddDevicePointerForArguments(savedMean);
    CudnnFrontend::AddDevicePointerForArguments(savedInvVariance);
    CudnnFrontend::AddDevicePointerForArguments(activationDesc);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(reserveSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnBatchNormalizationBackwardEx");
    if (CudnnFrontend::Success()) {
        dzData = CudnnFrontend::GetOutputDevicePointer();
        dxData = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateSpatialTransformerDescriptor");
    if (CudnnFrontend::Success()) {
        *stDesc = CudnnFrontend::GetOutputVariable<cudnnSpatialTransformerDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetSpatialTransformerNdDescriptor(
    cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType, const int nbDims, const int *dimA) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(stDesc);
    CudnnFrontend::AddVariableForArguments<cudnnSamplerType_t>(samplerType);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)dimA);

    CudnnFrontend::Execute("cudnnSetSpatialTransformerNdDescriptor");
    if (CudnnFrontend::Success()) {
        stDesc = CudnnFrontend::GetOutputVariable<cudnnSpatialTransformerDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(stDesc);

    CudnnFrontend::Execute("cudnnDestroySpatialTransformerDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorForward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc, const void *theta,
    void *grid) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(stDesc);
    CudnnFrontend::AddHostPointerForArguments(theta);

    CudnnFrontend::Execute("cudnnSpatialTfGridGeneratorForward");
    if (CudnnFrontend::Success()) {
        grid = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorBackward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc, const void *dgrid,
    void *dtheta) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(stDesc);
    CudnnFrontend::AddHostPointerForArguments(dgrid);

    CudnnFrontend::Execute("cudnnSpatialTfGridGeneratorBackward");
    if (CudnnFrontend::Success()) {
        dtheta = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerForward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *grid, const void *beta,
    cudnnTensorDescriptor_t yDesc, void *y) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(stDesc);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(grid);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    isFloatDescriptor(yDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(beta))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(y);

    CudnnFrontend::Execute("cudnnSpatialTfSamplerForward");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerBackward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx, const void *alphaDgrid,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *grid, const void *betaDgrid,
    void *dgrid) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(stDesc);
    isFloatDescriptor(xDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                   reinterpret_cast<const float *>(alpha))
                             : CudnnFrontend::AddHostPointerForArguments<const double>(
                                   reinterpret_cast<const double *>(alpha));
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    isFloatDescriptor(dxDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(beta))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(beta));
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    isFloatDescriptor(dyDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(alphaDgrid))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(alphaDgrid));
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(grid);
    isFloatDescriptor(dyDesc) ? CudnnFrontend::AddHostPointerForArguments<const float>(
                                    reinterpret_cast<const float *>(betaDgrid))
                              : CudnnFrontend::AddHostPointerForArguments<const double>(
                                    reinterpret_cast<const double *>(betaDgrid));

    CudnnFrontend::Execute("cudnnSpatialTfSamplerBackward");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
        dgrid = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateDropoutDescriptor");
    if (CudnnFrontend::Success()) {
        *dropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(dropoutDesc);

    CudnnFrontend::Execute("cudnnDestroyDropoutDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle,
                                                               size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);

    CudnnFrontend::Execute("cudnnDropoutGetStatesSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xDesc,
                                                                     size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(xDesc);

    CudnnFrontend::Execute("cudnnDropoutGetReserveSpaceSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                                               cudnnHandle_t handle, float dropout,
                                                               void *states,
                                                               size_t stateSizeInBytes,
                                                               unsigned long long seed) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(dropoutDesc);
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<float>(dropout);
    CudnnFrontend::AddDevicePointerForArguments(states);
    CudnnFrontend::AddVariableForArguments<size_t>(stateSizeInBytes);
    CudnnFrontend::AddVariableForArguments<unsigned long long>(seed);

    CudnnFrontend::Execute("cudnnSetDropoutDescriptor");
    if (CudnnFrontend::Success()) {
        dropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
        states = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRestoreDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout, void *states,
    size_t stateSizeInBytes, unsigned long long seed) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(dropoutDesc);
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<float>(dropout);
    CudnnFrontend::AddHostPointerForArguments(states);
    CudnnFrontend::AddVariableForArguments<size_t>(stateSizeInBytes);
    CudnnFrontend::AddVariableForArguments<unsigned long long>(seed);

    CudnnFrontend::Execute("cudnnRestoreDropoutDescriptor");
    if (CudnnFrontend::Success()) {
        dropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                                               cudnnHandle_t handle, float *dropout,
                                                               void **states,
                                                               unsigned long long *seed) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(dropoutDesc);
    CudnnFrontend::AddDevicePointerForArguments(handle);

    CudnnFrontend::Execute("cudnnGetDropoutDescriptor");
    if (CudnnFrontend::Success()) {
        *dropout = CudnnFrontend::GetOutputVariable<float>();
        *states = CudnnFrontend::GetOutputDevicePointer();
        *seed = CudnnFrontend::GetOutputVariable<unsigned long long>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t xdesc, const void *x, const cudnnTensorDescriptor_t ydesc,
    void *y, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(dropoutDesc);
    CudnnFrontend::AddDevicePointerForArguments(xdesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(ydesc);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnDropoutForward");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(
    cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc,
    const cudnnTensorDescriptor_t dydesc, const void *dy, const cudnnTensorDescriptor_t dxdesc,
    void *dx, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(dropoutDesc);
    CudnnFrontend::AddDevicePointerForArguments(dydesc);
    CudnnFrontend::AddHostPointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(dxdesc);
    CudnnFrontend::AddHostPointerForArguments(reserveSpace);
    CudnnFrontend::AddVariableForArguments<size_t>((reserveSpaceSizeInBytes));

    CudnnFrontend::Execute("cudnnDropoutBackward");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateRNNDescriptor");
    if (CudnnFrontend::Success()) {
        *rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnDestroyRNNDescriptor");

    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 8000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v5(
    cudnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
    cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnDataType_t mathPrec) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(hiddenSize);
    CudnnFrontend::AddVariableForArguments<int>(numLayers);
    CudnnFrontend::AddDevicePointerForArguments(dropoutDesc);
    CudnnFrontend::AddVariableForArguments<cudnnRNNInputMode_t>(inputMode);
    CudnnFrontend::AddVariableForArguments<cudnnDirectionMode_t>(direction);
    CudnnFrontend::AddVariableForArguments<cudnnRNNMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(mathPrec);

    CudnnFrontend::Execute("cudnnSetRNNDescriptor_v5");
    if (CudnnFrontend::Success()) {
        rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

#if CUDNN_VERSION >= 8000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v8(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t algo, cudnnRNNMode_t cellMode,
    cudnnRNNBiasMode_t biasMode, cudnnDirectionMode_t dirMode, cudnnRNNInputMode_t inputMode,
    cudnnDataType_t dataType, cudnnDataType_t mathPrec, cudnnMathType_t mathType, int32_t inputSize,
    int32_t hiddenSize, int32_t projSize, int32_t numLayers, cudnnDropoutDescriptor_t dropoutDesc,
    uint32_t auxFlags) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnRNNAlgo_t>(algo);
    CudnnFrontend::AddVariableForArguments<cudnnRNNMode_t>(cellMode);
    CudnnFrontend::AddVariableForArguments<cudnnRNNBiasMode_t>(biasMode);
    CudnnFrontend::AddVariableForArguments<cudnnDirectionMode_t>(dirMode);
    CudnnFrontend::AddVariableForArguments<cudnnRNNInputMode_t>(inputMode);

    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(mathPrec);
    CudnnFrontend::AddVariableForArguments<cudnnMathType_t>(mathType);

    CudnnFrontend::AddVariableForArguments<int32_t>(inputSize);
    CudnnFrontend::AddVariableForArguments<int32_t>(hiddenSize);
    CudnnFrontend::AddVariableForArguments<int32_t>(projSize);
    CudnnFrontend::AddVariableForArguments<int32_t>(numLayers);
    CudnnFrontend::AddDevicePointerForArguments(dropoutDesc);
    CudnnFrontend::AddVariableForArguments<uint32_t>(auxFlags);

    CudnnFrontend::Execute("cudnnSetRNNDescriptor_v8");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor_v8(
    cudnnRNNDescriptor_t rnnDesc, cudnnRNNAlgo_t *algo, cudnnRNNMode_t *cellMode,
    cudnnRNNBiasMode_t *biasMode, cudnnDirectionMode_t *dirMode, cudnnRNNInputMode_t *inputMode,
    cudnnDataType_t *dataType, cudnnDataType_t *mathPrec, cudnnMathType_t *mathType,
    int32_t *inputSize, int32_t *hiddenSize, int32_t *projSize, int32_t *numLayers,
    cudnnDropoutDescriptor_t *dropoutDesc, uint32_t *auxFlags) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnGetRNNDescriptor_v8");
    if (CudnnFrontend::Success()) {
        *algo = CudnnFrontend::GetOutputVariable<cudnnRNNAlgo_t>();
        *cellMode = CudnnFrontend::GetOutputVariable<cudnnRNNMode_t>();
        *biasMode = CudnnFrontend::GetOutputVariable<cudnnRNNBiasMode_t>();
        *dirMode = CudnnFrontend::GetOutputVariable<cudnnDirectionMode_t>();
        *inputMode = CudnnFrontend::GetOutputVariable<cudnnRNNInputMode_t>();

        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *mathPrec = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *mathType = CudnnFrontend::GetOutputVariable<cudnnMathType_t>();

        *inputSize = CudnnFrontend::GetOutputVariable<int>();
        *hiddenSize = CudnnFrontend::GetOutputVariable<int>();
        *projSize = CudnnFrontend::GetOutputVariable<int>();
        *numLayers = CudnnFrontend::GetOutputVariable<int>();
        *dropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
        *auxFlags = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnRNNForward(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnForwardMode_t fwdMode,
                const int32_t devSeqLengths[], cudnnRNNDataDescriptor_t xDesc, const void *x,
                cudnnRNNDataDescriptor_t yDesc, void *y, cudnnTensorDescriptor_t hDesc,
                const void *hx, void *hy, cudnnTensorDescriptor_t cDesc, const void *cx, void *cy,
                size_t weightSpaceSize, const void *weightSpace, size_t workSpaceSize,
                void *workSpace, size_t reserveSpaceSize, void *reserveSpace) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnForwardMode_t>(fwdMode);
    CudnnFrontend::AddDevicePointerForArguments(devSeqLengths);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(hDesc);
    CudnnFrontend::AddDevicePointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(hy);
    CudnnFrontend::AddDevicePointerForArguments(cDesc);
    CudnnFrontend::AddDevicePointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(cy);
    CudnnFrontend::AddVariableForArguments<size_t>(weightSpaceSize);
    CudnnFrontend::AddDevicePointerForArguments(weightSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSize);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSize);
    CudnnFrontend::AddDevicePointerForArguments(reserveSpace);

    CudnnFrontend::Execute("cudnnRNNForward");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc) {
    CudnnFrontend::Prepare();
    CudnnFrontend::Execute("cudnnCreateRNNDataDescriptor");
    if (CudnnFrontend::Success()) {
        *rnnDataDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDataDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(rnnDataDesc);
    CudnnFrontend::Execute("cudnnDestroyRNNDataDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType, cudnnRNNDataLayout_t layout,
    int maxSeqLength, int batchSize, int vectorSize, const int *seqLengthArray, void *paddingFill) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDataDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<cudnnRNNDataLayout_t>(layout);
    CudnnFrontend::AddVariableForArguments<int>(maxSeqLength);
    CudnnFrontend::AddVariableForArguments<int>(batchSize);
    CudnnFrontend::AddVariableForArguments<int>(vectorSize);
    CudnnFrontend::AddHostPointerForArguments<const int>(seqLengthArray, batchSize);
    isFloatDescriptor(rnnDataDesc)
        ? CudnnFrontend::AddHostPointerForArguments<float>(reinterpret_cast<float *>(paddingFill))
        : CudnnFrontend::AddHostPointerForArguments<double>(
              reinterpret_cast<double *>(paddingFill));

    CudnnFrontend::Execute("cudnnSetRNNDataDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType, cudnnRNNDataLayout_t *layout,
    int *maxSeqLength, int *batchSize, int *vectorSize, int arrayLengthRequested,
    int seqLengthArray[], void *paddingFill) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDataDesc);
    CudnnFrontend::AddVariableForArguments<int>(arrayLengthRequested);

    CudnnFrontend::Execute("cudnnGetRNNDataDescriptor");
    if (CudnnFrontend::Success()) {
        *dataType = *CudnnFrontend::GetOutputHostPointer<cudnnDataType_t>();
        *layout = *CudnnFrontend::GetOutputHostPointer<cudnnRNNDataLayout_t>();
        *maxSeqLength = *CudnnFrontend::GetOutputHostPointer<int>();
        *batchSize = *CudnnFrontend::GetOutputHostPointer<int>();
        *vectorSize = *CudnnFrontend::GetOutputHostPointer<int>();
        int *seqLengthArray_ptr = CudnnFrontend::GetOutputHostPointer<int>(arrayLengthRequested);
        memcpy(seqLengthArray, seqLengthArray_ptr, sizeof(int) * arrayLengthRequested);
        if (isFloatDescriptor(rnnDataDesc)) {
            void *ptr = static_cast<void *>(CudnnFrontend::GetOutputHostPointer<float>());
            memcpy(paddingFill, ptr, sizeof(float));
        } else {
            void *ptr = static_cast<void *>(CudnnFrontend::GetOutputHostPointer<double>());
            memcpy(paddingFill, ptr, sizeof(double));
        }
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNWeightSpaceSize(cudnnHandle_t handle,
                                                                cudnnRNNDescriptor_t rnnDesc,
                                                                size_t *weightSpaceSize) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnGetRNNWeightSpaceSize");
    if (CudnnFrontend::Success()) *weightSpaceSize = CudnnFrontend::GetOutputVariable<size_t>();

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNTempSpaceSizes(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnForwardMode_t fMode,
    cudnnRNNDataDescriptor_t xDesc, size_t *workSpaceSize, size_t *reserveSpaceSize) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnForwardMode_t>(fMode);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);

    CudnnFrontend::Execute("cudnnGetRNNTempSpaceSizes");
    if (CudnnFrontend::Success()) {
        *workSpaceSize = CudnnFrontend::GetOutputVariable<size_t>();
        *reserveSpaceSize = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateSeqDataDescriptor");
    if (CudnnFrontend::Success()) {
        *seqDataDesc = CudnnFrontend::GetOutputVariable<cudnnSeqDataDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(seqDataDesc);

    CudnnFrontend::Execute("cudnnDestroySeqDataDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType,
                          int nbDims, const int *dimA, const cudnnSeqDataAxis_t *axes,
                          size_t seqLengthArraySize, const int *seqLengthArray, void *paddingFill) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)dimA);
    CudnnFrontend::AddHostPointerForArguments<cudnnSeqDataAxis_t>((cudnnSeqDataAxis_t *)axes);
    CudnnFrontend::AddVariableForArguments<size_t>(seqLengthArraySize);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)seqLengthArray);
    CudnnFrontend::AddHostPointerForArguments(paddingFill);

    CudnnFrontend::Execute("cudnnSetSeqDataDescriptor");
    if (CudnnFrontend::Success()) {
        seqDataDesc = CudnnFrontend::GetOutputVariable<cudnnSeqDataDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetSeqDataDescriptor(
    const cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t *dataType, int *nbDims,
    int nbDimsRequested, int *dimA, cudnnSeqDataAxis_t *axes, size_t *seqLengthArraySize,
    size_t seqLengthSizeRequested, int *seqLengthArray, void *paddingFill) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(seqDataDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddVariableForArguments<size_t>(seqLengthSizeRequested);

    CudnnFrontend::Execute("cudnnGetSeqDataDescriptor");
    if (CudnnFrontend::Success()) {
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *nbDims = CudnnFrontend::GetOutputVariable<int>();
        *dimA = CudnnFrontend::GetOutputVariable<int>();
        *axes = CudnnFrontend::GetOutputVariable<cudnnSeqDataAxis_t>();
        *seqLengthArraySize = CudnnFrontend::GetOutputVariable<size_t>();
        *seqLengthArray = CudnnFrontend::GetOutputVariable<int>();
        paddingFill = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateAttnDescriptor");
    if (CudnnFrontend::Success()) {
        *attnDesc = CudnnFrontend::GetOutputVariable<cudnnAttnDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(attnDesc);

    CudnnFrontend::Execute("cudnnDestroyAttnDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, unsigned attnMode, int nHeads, double smScaler,
    cudnnDataType_t dataType, cudnnDataType_t computePrec, cudnnMathType_t mathType,
    cudnnDropoutDescriptor_t attnDropoutDesc, cudnnDropoutDescriptor_t postDropoutDesc, int qSize,
    int kSize, int vSize, int qProjSize, int kProjSize, int vProjSize, int oProjSize,
    int qoMaxSeqLength, int kvMaxSeqLength, int maxBatchSize, int maxBeamSize) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<unsigned>(attnMode);
    CudnnFrontend::AddVariableForArguments<int>(nHeads);
    CudnnFrontend::AddVariableForArguments<double>(smScaler);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(computePrec);
    CudnnFrontend::AddVariableForArguments<cudnnMathType_t>(mathType);
    CudnnFrontend::AddDevicePointerForArguments(attnDropoutDesc);
    CudnnFrontend::AddDevicePointerForArguments(postDropoutDesc);
    CudnnFrontend::AddVariableForArguments<int>(qSize);
    CudnnFrontend::AddVariableForArguments<int>(kSize);
    CudnnFrontend::AddVariableForArguments<int>(vSize);
    CudnnFrontend::AddVariableForArguments<int>(qProjSize);
    CudnnFrontend::AddVariableForArguments<int>(kProjSize);
    CudnnFrontend::AddVariableForArguments<int>(vProjSize);
    CudnnFrontend::AddVariableForArguments<int>(oProjSize);
    CudnnFrontend::AddVariableForArguments<int>(qoMaxSeqLength);
    CudnnFrontend::AddVariableForArguments<int>(kvMaxSeqLength);
    CudnnFrontend::AddVariableForArguments<int>(maxBatchSize);
    CudnnFrontend::AddVariableForArguments<int>(maxBeamSize);

    CudnnFrontend::Execute("cudnnSetAttnDescriptor");
    if (CudnnFrontend::Success()) {
        attnDesc = CudnnFrontend::GetOutputVariable<cudnnAttnDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, unsigned *attnMode, int *nHeads, double *smScaler,
    cudnnDataType_t *dataType, cudnnDataType_t *computePrec, cudnnMathType_t *mathType,
    cudnnDropoutDescriptor_t *attnDropoutDesc, cudnnDropoutDescriptor_t *postDropoutDesc,
    int *qSize, int *kSize, int *vSize, int *qProjSize, int *kProjSize, int *vProjSize,
    int *oProjSize, int *qoMaxSeqLength, int *kvMaxSeqLength, int *maxBatchSize, int *maxBeamSize) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(attnDesc);

    CudnnFrontend::Execute("cudnnGetAttnDescriptor");
    if (CudnnFrontend::Success()) {
        *attnMode = CudnnFrontend::GetOutputVariable<unsigned>();
        *nHeads = CudnnFrontend::GetOutputVariable<int>();
        *smScaler = CudnnFrontend::GetOutputVariable<double>();
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *computePrec = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *mathType = CudnnFrontend::GetOutputVariable<cudnnMathType_t>();
        *attnDropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
        *postDropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
        *qSize = CudnnFrontend::GetOutputVariable<int>();
        *kSize = CudnnFrontend::GetOutputVariable<int>();
        *vSize = CudnnFrontend::GetOutputVariable<int>();
        *qProjSize = CudnnFrontend::GetOutputVariable<int>();
        *vProjSize = CudnnFrontend::GetOutputVariable<int>();
        *oProjSize = CudnnFrontend::GetOutputVariable<int>();
        *qoMaxSeqLength = CudnnFrontend::GetOutputVariable<int>();
        *kvMaxSeqLength = CudnnFrontend::GetOutputVariable<int>();
        *maxBatchSize = CudnnFrontend::GetOutputVariable<int>();
        *maxBeamSize = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnBuffers(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, size_t *weightSizeInBytes,
    size_t *workSpaceSizeInBytes, size_t *reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(attnDesc);

    CudnnFrontend::Execute("cudnnGetMultiHeadAttnBuffers");
    if (CudnnFrontend::Success()) {
        *weightSizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
        *workSpaceSizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
        *reserveSpaceSizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
                             cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes,
                             const void *weights, cudnnTensorDescriptor_t wDesc, void **wAddr) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(attnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnMultiHeadAttnWeightKind_t>(wKind);
    CudnnFrontend::AddVariableForArguments<size_t>(weightSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(weights);

    CudnnFrontend::Execute("cudnnGetMultiHeadAttnWeights");
    if (CudnnFrontend::Success()) {
        wDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
        *wAddr = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnForward(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, int currIdx, const int *loWinIdx,
    const int *hiWinIdx, const int *seqLengthArrayQRO, const int *seqLengthArrayKV,
    const cudnnSeqDataDescriptor_t qDesc, const void *queries, const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc, const void *keys, const cudnnSeqDataDescriptor_t vDesc,
    const void *values, const cudnnSeqDataDescriptor_t oDesc, void *out, size_t weightSizeInBytes,
    const void *weights, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(attnDesc);
    CudnnFrontend::AddVariableForArguments<int>(currIdx);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)loWinIdx);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)hiWinIdx);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)seqLengthArrayQRO);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)seqLengthArrayKV);
    CudnnFrontend::AddDevicePointerForArguments(qDesc);
    CudnnFrontend::AddHostPointerForArguments(queries);
    CudnnFrontend::AddHostPointerForArguments(residuals);
    CudnnFrontend::AddDevicePointerForArguments(kDesc);
    CudnnFrontend::AddHostPointerForArguments(keys);
    CudnnFrontend::AddDevicePointerForArguments(vDesc);
    CudnnFrontend::AddHostPointerForArguments(values);
    CudnnFrontend::AddDevicePointerForArguments(oDesc);
    CudnnFrontend::AddVariableForArguments<size_t>(weightSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(weights);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(reserveSpace);

    CudnnFrontend::Execute("cudnnMultiHeadAttnForward");
    if (CudnnFrontend::Success()) {
        out = CudnnFrontend::GetOutputDevicePointer();
        workSpace = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, const int *loWinIdx,
    const int *hiWinIdx, const int *seqLengthArrayDQDO, const int *seqLengthArrayDKDV,
    const cudnnSeqDataDescriptor_t doDesc, const void *dout, const cudnnSeqDataDescriptor_t dqDesc,
    void *dqueries, const void *queries, const cudnnSeqDataDescriptor_t dkDesc, void *dkeys,
    const void *keys, const cudnnSeqDataDescriptor_t dvDesc, void *dvalues, const void *values,
    size_t weightSizeInBytes, const void *weights, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(attnDesc);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)loWinIdx);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)hiWinIdx);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)seqLengthArrayDQDO);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)seqLengthArrayDKDV);
    CudnnFrontend::AddDevicePointerForArguments(doDesc);
    CudnnFrontend::AddHostPointerForArguments(dout);
    CudnnFrontend::AddDevicePointerForArguments(dqDesc);
    CudnnFrontend::AddHostPointerForArguments(queries);
    CudnnFrontend::AddDevicePointerForArguments(dkDesc);
    CudnnFrontend::AddHostPointerForArguments(keys);
    CudnnFrontend::AddDevicePointerForArguments(dvDesc);
    CudnnFrontend::AddHostPointerForArguments(values);
    CudnnFrontend::AddVariableForArguments<size_t>(weightSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(weights);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(reserveSpace);

    CudnnFrontend::Execute("cudnnMultiHeadAttnBackwardData");
    if (CudnnFrontend::Success()) {
        dout = CudnnFrontend::GetOutputDevicePointer();
        dqueries = CudnnFrontend::GetOutputDevicePointer();
        dqueries = CudnnFrontend::GetOutputDevicePointer();
        dkeys = CudnnFrontend::GetOutputDevicePointer();
        dvalues = CudnnFrontend::GetOutputDevicePointer();
        workSpace = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, cudnnWgradMode_t addGrad,
    const cudnnSeqDataDescriptor_t qDesc, const void *queries, const cudnnSeqDataDescriptor_t kDesc,
    const void *keys, const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t doDesc, const void *dout, size_t weightSizeInBytes,
    const void *weights, void *dweights, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(attnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnWgradMode_t>(addGrad);
    CudnnFrontend::AddDevicePointerForArguments(qDesc);
    CudnnFrontend::AddHostPointerForArguments(queries);
    CudnnFrontend::AddDevicePointerForArguments(kDesc);
    CudnnFrontend::AddHostPointerForArguments(keys);
    CudnnFrontend::AddDevicePointerForArguments(vDesc);
    CudnnFrontend::AddHostPointerForArguments(values);
    CudnnFrontend::AddDevicePointerForArguments(doDesc);
    CudnnFrontend::AddHostPointerForArguments(dout);
    CudnnFrontend::AddVariableForArguments<size_t>(weightSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(weights);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(reserveSpace);

    CudnnFrontend::Execute("cudnnMultiHeadAttnBackwardWeights");
    if (CudnnFrontend::Success()) {
        dweights = CudnnFrontend::GetOutputDevicePointer();
        workSpace = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateCTCLossDescriptor");
    if (CudnnFrontend::Success()) {
        *ctcLossDesc = CudnnFrontend::GetOutputVariable<cudnnCTCLossDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                                               cudnnDataType_t compType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(compType);

    CudnnFrontend::Execute("cudnnSetCTCLossDescriptor");
    if (CudnnFrontend::Success()) {
        ctcLossDesc = CudnnFrontend::GetOutputVariable<cudnnCTCLossDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
                            cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(compType);
    CudnnFrontend::AddVariableForArguments<cudnnLossNormalizationMode_t>(normMode);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(gradMode);

    CudnnFrontend::Execute("cudnnSetCTCLossDescriptorEx");
    if (CudnnFrontend::Success()) {
        ctcLossDesc = CudnnFrontend::GetOutputVariable<cudnnCTCLossDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                                               cudnnDataType_t *compType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(ctcLossDesc);

    CudnnFrontend::Execute("cudnnGetCTCLossDescriptor");
    if (CudnnFrontend::Success()) {
        *compType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptorEx(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(ctcLossDesc);

    CudnnFrontend::Execute("cudnnGetCTCLossDescriptorEx");
    if (CudnnFrontend::Success()) {
        *compType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *normMode = CudnnFrontend::GetOutputVariable<cudnnLossNormalizationMode_t>();
        *gradMode = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(ctcLossDesc);

    CudnnFrontend::Execute("cudnnDestroyCTCLossDescriptor");

    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 8204
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCTCLoss(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const void *probs,
    const int *labels, const int *labelLengths, const int *inputLengths, void *costs,
    const cudnnTensorDescriptor_t gradientsDesc, const void *gradients, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace, size_t workSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(probsDesc);
    CudnnFrontend::AddHostPointerForArguments(probs);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)labels);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)labelLengths);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)inputLengths);
    CudnnFrontend::AddDevicePointerForArguments(gradientsDesc);
    CudnnFrontend::AddVariableForArguments<cudnnCTCLossAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(ctcLossDesc);
    CudnnFrontend::AddHostPointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnCTCLoss");
    if (CudnnFrontend::Success()) {
        costs = CudnnFrontend::GetOutputDevicePointer();
        gradients = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, const int *labels, const int *labelLengths,
    const int *inputLengths, cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(probsDesc);
    CudnnFrontend::AddDevicePointerForArguments(gradientsDesc);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)labels);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)labelLengths);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)inputLengths);
    CudnnFrontend::AddVariableForArguments<cudnnCTCLossAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(ctcLossDesc);

    CudnnFrontend::Execute("cudnnGetCTCLossWorkspaceSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetCallback(unsigned mask, void *udata,
                                                      cudnnCallback_t fptr) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<unsigned>(mask);
    CudnnFrontend::AddHostPointerForArguments(udata);
    CudnnFrontend::AddVariableForArguments<cudnnCallback_t>(fptr);

    CudnnFrontend::Execute("cudnnSetCallback");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetCallback(unsigned *mask, void **udata,
                                                      cudnnCallback_t *fptr) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnGetCallback");
    if (CudnnFrontend::Success()) {
        *mask = CudnnFrontend::GetOutputVariable<unsigned>();
        *udata = CudnnFrontend::GetOutputDevicePointer();
        *fptr = CudnnFrontend::GetOutputVariable<cudnnCallback_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddHostPointerForArguments<cudnnFusedOpsConstParamPack_t>(
        (cudnnFusedOpsConstParamPack_t *)constPack);
    CudnnFrontend::AddVariableForArguments<cudnnFusedOps_t>(ops);

    CudnnFrontend::Execute("cudnnCreateFusedOpsConstParamPack");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(constPack);

    CudnnFrontend::Execute("cudnnDestroyFusedOpsConstParamPack");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFusedOpsConstParamPackAttribute(
    cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel,
    const void *param) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(constPack);
    CudnnFrontend::AddVariableForArguments<cudnnFusedOpsConstParamLabel_t>(paramLabel);
    CudnnFrontend::AddHostPointerForArguments(param);

    CudnnFrontend::Execute("cudnnSetFusedOpsConstParamPackAttribute");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFusedOpsConstParamPackAttribute(
    const cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel,
    void *param, int *isNULL) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(constPack);
    CudnnFrontend::AddVariableForArguments<cudnnFusedOpsConstParamLabel_t>(paramLabel);
    CudnnFrontend::AddHostPointerForArguments(param);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)isNULL);

    CudnnFrontend::Execute("cudnnGetFusedOpsConstParamPackAttribute");
    if (CudnnFrontend::Success()) {
        *isNULL = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnFusedOps_t>(ops);

    CudnnFrontend::Execute("cudnnCreateFusedOpsVariantParamPack");
    if (CudnnFrontend::Success()) {
        *varPack = CudnnFrontend::GetOutputVariable<cudnnFusedOpsVariantParamPack_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(varPack);

    CudnnFrontend::Execute("cudnnDestroyFusedOpsVariantParamPack");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t varPack,
                                          cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(varPack);
    CudnnFrontend::AddVariableForArguments<cudnnFusedOpsVariantParamLabel_t>(paramLabel);
    CudnnFrontend::AddHostPointerForArguments(ptr);

    CudnnFrontend::Execute("cudnnSetFusedOpsVariantParamPackAttribute");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t varPack,
                                          cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(varPack);
    CudnnFrontend::AddVariableForArguments<cudnnFusedOpsVariantParamLabel_t>(paramLabel);

    CudnnFrontend::Execute("cudnnGetFusedOpsVariantParamPackAttribute");
    if (CudnnFrontend::Success()) {
        ptr = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan,
                                                             cudnnFusedOps_t ops) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddHostPointerForArguments<cudnnFusedOpsPlan_t>((cudnnFusedOpsPlan_t *)plan);
    CudnnFrontend::AddVariableForArguments<cudnnFusedOps_t>(ops);

    CudnnFrontend::Execute("cudnnCreateFusedOpsPlan");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(plan);

    CudnnFrontend::Execute("cudnnDestroyFusedOpsPlan");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnMakeFusedOpsPlan(cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
                      const cudnnFusedOpsConstParamPack_t constPack, size_t *workspaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(plan);
    CudnnFrontend::AddDevicePointerForArguments(constPack);

    CudnnFrontend::Execute("cudnnMakeFusedOpsPlan");
    if (CudnnFrontend::Success()) {
        *workspaceSizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFusedOpsExecute(cudnnHandle_t handle,
                                                          const cudnnFusedOpsPlan_t plan,
                                                          cudnnFusedOpsVariantParamPack_t varPack) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(plan);
    CudnnFrontend::AddDevicePointerForArguments(varPack);

    CudnnFrontend::Execute("cudnnFusedOpsExecute");

    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 9000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc,
                                                               cudnnMathType_t mType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnMathType_t>(mType);

    CudnnFrontend::Execute("cudnnSetRNNMatrixMathType");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc,
                                                               cudnnMathType_t *mType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnGetRNNMatrixMathType");
    if (CudnnFrontend::Success()) {
        *mType = CudnnFrontend::GetOutputVariable<cudnnMathType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc,
                                                         cudnnRNNBiasMode_t biasMode) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnRNNBiasMode_t>((biasMode));

    CudnnFrontend::Execute("cudnnSetRNNBiasMode");
    if (CudnnFrontend::Success()) {
        rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc,
                                                         cudnnRNNBiasMode_t *biasMode) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnGetRNNBiasMode");
    if (CudnnFrontend::Success()) {
        *biasMode = CudnnFrontend::GetOutputVariable<cudnnRNNBiasMode_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNSetClip(cudnnHandle_t handle,
                                                     cudnnRNNDescriptor_t rnnDesc,
                                                     cudnnRNNClipMode_t clipMode,
                                                     cudnnNanPropagation_t clipNanOpt, double lclip,
                                                     double rclip) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnRNNClipMode_t>(clipMode);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(clipNanOpt);
    CudnnFrontend::AddVariableForArguments<double>(lclip);
    CudnnFrontend::AddVariableForArguments<double>(rclip);

    CudnnFrontend::Execute("cudnnRNNSetClip");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNGetClip(cudnnHandle_t handle,
                                                     cudnnRNNDescriptor_t rnnDesc,
                                                     cudnnRNNClipMode_t *clipMode,
                                                     cudnnNanPropagation_t *clipNanOpt,
                                                     double *lclip, double *rclip) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnRNNGetClip");
    if (CudnnFrontend::Success()) {
        *clipMode = CudnnFrontend::GetOutputVariable<cudnnRNNClipMode_t>();
        *clipNanOpt = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
        *lclip = CudnnFrontend::GetOutputVariable<double>();
        *rclip = CudnnFrontend::GetOutputVariable<double>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
                                                                 cudnnRNNDescriptor_t rnnDesc,
                                                                 const int recProjSize,
                                                                 const int outProjSize) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(recProjSize);
    CudnnFrontend::AddVariableForArguments<int>(outProjSize);

    CudnnFrontend::Execute("cudnnSetRNNProjectionLayers");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
                                                                 const cudnnRNNDescriptor_t rnnDesc,
                                                                 int *recProjSize,
                                                                 int *outProjSize) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnGetRNNProjectionLayers");
    if (CudnnFrontend::Success()) {
        *recProjSize = CudnnFrontend::GetOutputVariable<int>();
        *outProjSize = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                                                  const int minibatch,
                                                                  const cudnnDataType_t dataType,
                                                                  cudnnPersistentRNNPlan_t *plan) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(minibatch);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);

    CudnnFrontend::Execute("cudnnCreatePersistentRNNPlan");
    if (CudnnFrontend::Success()) {
        *plan = CudnnFrontend::GetOutputVariable<cudnnPersistentRNNPlan_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(plan);

    CudnnFrontend::Execute("cudnnDestroyPersistentRNNPlan");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                                               cudnnPersistentRNNPlan_t plan) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddDevicePointerForArguments(plan);

    CudnnFrontend::Execute("cudnnSetPersistentRNNPlan");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
                                                              const cudnnRNNDescriptor_t rnnDesc,
                                                              const int seqLength,
                                                              const cudnnTensorDescriptor_t *xDesc,
                                                              size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddHostPointerForArguments<cudnnTensorDescriptor_t>(
        (cudnnTensorDescriptor_t *)xDesc);

    CudnnFrontend::Execute("cudnnGetRNNWorkspaceSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNTrainingReserveSize(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength,
    const cudnnTensorDescriptor_t *xDesc, size_t *sizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddHostPointerForArguments<cudnnTensorDescriptor_t>(
        (cudnnTensorDescriptor_t *)xDesc);

    CudnnFrontend::Execute("cudnnGetRNNTrainingReserveSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNParamsSize(cudnnHandle_t handle,
                                                           const cudnnRNNDescriptor_t rnnDesc,
                                                           const cudnnTensorDescriptor_t xDesc,
                                                           size_t *sizeInBytes,
                                                           cudnnDataType_t dataType) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);

    CudnnFrontend::Execute("cudnnGetRNNParamsSize");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerMatrixParams(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const void *w,
    const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc, void **linLayerMat) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(pseudoLayer);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddVariableForArguments<int>(linLayerID);

    CudnnFrontend::Execute("cudnnGetRNNLinLayerMatrixParams");
    if (CudnnFrontend::Success()) {
        linLayerMatDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
        *linLayerMat = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerBiasParams(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const void *w,
    const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc, void **linLayerBias) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(pseudoLayer);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddVariableForArguments<int>(linLayerID);

    CudnnFrontend::Execute("cudnnGetRNNLinLayerBiasParams");
    if (CudnnFrontend::Success()) {
        linLayerBiasDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
        *linLayerBias = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInference(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength,
    const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, void *workspace, size_t workSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(cxDesc);
    CudnnFrontend::AddHostPointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(hyDesc);
    CudnnFrontend::AddDevicePointerForArguments(cyDesc);
    CudnnFrontend::AddHostPointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnRNNForwardInference");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
        hy = CudnnFrontend::GetOutputDevicePointer();
        cy = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength,
    const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddHostPointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(cxDesc);
    CudnnFrontend::AddHostPointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(hyDesc);
    CudnnFrontend::AddDevicePointerForArguments(cyDesc);
    CudnnFrontend::AddHostPointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnRNNForwardTraining");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
        hy = CudnnFrontend::GetOutputDevicePointer();
        cy = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength,
    const cudnnTensorDescriptor_t *yDesc, const void *y, const cudnnTensorDescriptor_t *dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnTensorDescriptor_t *dxDesc,
    void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, void *workspace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddHostPointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(dhyDesc);
    CudnnFrontend::AddHostPointerForArguments(dhy);
    CudnnFrontend::AddDevicePointerForArguments(dcyDesc);
    CudnnFrontend::AddHostPointerForArguments(dcy);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddHostPointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(cxDesc);
    CudnnFrontend::AddHostPointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dhxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dcxDesc);
    CudnnFrontend::AddHostPointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnRNNBackwardData");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
        dhx = CudnnFrontend::GetOutputDevicePointer();
        dcx = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength,
    const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t *yDesc, const void *y, const void *workspace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw,
    const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddHostPointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddHostPointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(dwDesc);
    CudnnFrontend::AddHostPointerForArguments(dw);
    CudnnFrontend::AddHostPointerForArguments(reserveSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnRNNBackwardWeights");
    if (CudnnFrontend::Success()) {
        dw = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc,
                                                            cudnnRNNPaddingMode_t paddingMode) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnRNNPaddingMode_t>(paddingMode);

    CudnnFrontend::Execute("cudnnSetRNNPaddingMode");
    if (CudnnFrontend::Success()) {
        rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc,
                                                            cudnnRNNPaddingMode_t *paddingMode) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddHostPointerForArguments<cudnnRNNPaddingMode_t>(
        (cudnnRNNPaddingMode_t *)paddingMode);

    CudnnFrontend::Execute("cudnnGetRNNPaddingMode");
    if (CudnnFrontend::Success()) {
        rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTrainingEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys, const cudnnRNNDataDescriptor_t cDesc,
    void *cAttn, const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddHostPointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(cxDesc);
    CudnnFrontend::AddHostPointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(hyDesc);
    CudnnFrontend::AddDevicePointerForArguments(cyDesc);
    CudnnFrontend::AddDevicePointerForArguments(kDesc);
    CudnnFrontend::AddHostPointerForArguments(keys);
    CudnnFrontend::AddDevicePointerForArguments(cDesc);
    CudnnFrontend::AddHostPointerForArguments(cAttn);
    CudnnFrontend::AddDevicePointerForArguments(iDesc);
    CudnnFrontend::AddHostPointerForArguments(iAttn);
    CudnnFrontend::AddDevicePointerForArguments(qDesc);
    CudnnFrontend::AddHostPointerForArguments(queries);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(reserveSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnRNNForwardTrainingEx");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
        hy = CudnnFrontend::GetOutputDevicePointer();
        cy = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInferenceEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys, const cudnnRNNDataDescriptor_t cDesc,
    void *cAttn, const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries, void *workSpace,
    size_t workSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddHostPointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(cxDesc);
    CudnnFrontend::AddHostPointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(hyDesc);
    CudnnFrontend::AddDevicePointerForArguments(cyDesc);
    CudnnFrontend::AddDevicePointerForArguments(kDesc);
    CudnnFrontend::AddHostPointerForArguments(keys);
    CudnnFrontend::AddDevicePointerForArguments(cDesc);
    CudnnFrontend::AddHostPointerForArguments(cAttn);
    CudnnFrontend::AddDevicePointerForArguments(iDesc);
    CudnnFrontend::AddHostPointerForArguments(iAttn);
    CudnnFrontend::AddDevicePointerForArguments(qDesc);
    CudnnFrontend::AddHostPointerForArguments(queries);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnRNNForwardInferenceEx");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
        hy = CudnnFrontend::GetOutputDevicePointer();
        cy = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardDataEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t yDesc,
    const void *y, const cudnnRNNDataDescriptor_t dyDesc, const void *dy,
    const cudnnRNNDataDescriptor_t dcDesc, const void *dcAttn,
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
    const void *dcy, const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnRNNDataDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx, const cudnnTensorDescriptor_t dcxDesc,
    void *dcx, const cudnnRNNDataDescriptor_t dkDesc, void *dkeys, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddHostPointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(dcDesc);
    CudnnFrontend::AddHostPointerForArguments(dcAttn);
    CudnnFrontend::AddDevicePointerForArguments(dhyDesc);
    CudnnFrontend::AddHostPointerForArguments(dhy);
    CudnnFrontend::AddDevicePointerForArguments(dcyDesc);
    CudnnFrontend::AddHostPointerForArguments(dcy);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddHostPointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(cxDesc);
    CudnnFrontend::AddHostPointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dhxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dcxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dkDesc);
    CudnnFrontend::AddHostPointerForArguments(dkeys);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(reserveSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnRNNBackwardDataEx");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
        dhx = CudnnFrontend::GetOutputDevicePointer();
        dcx = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeightsEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnRNNDataDescriptor_t yDesc, const void *y, void *workSpace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddHostPointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(dwDesc);
    CudnnFrontend::AddHostPointerForArguments(dw);
    CudnnFrontend::AddHostPointerForArguments(reserveSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnRNNBackwardWeightsEx");
    if (CudnnFrontend::Success()) {
        dw = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNAlgorithmDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnAlgorithmDescriptor_t algoDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddDevicePointerForArguments(algoDesc);

    CudnnFrontend::Execute("cudnnSetRNNAlgorithmDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnGetRNNForwardInferenceAlgorithmMaxCount");
    if (CudnnFrontend::Success()) {
        *count = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardInferenceAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength,
    const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddDevicePointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(cxDesc);
    CudnnFrontend::AddDevicePointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(hyDesc);
    CudnnFrontend::AddDevicePointerForArguments(cyDesc);
    CudnnFrontend::AddVariableForArguments<float>(findIntensity);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnAlgorithmPerformance_t>(perfResults,
                                                                           requestedAlgoCount);
    CudnnFrontend::AddDevicePointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindRNNForwardInferenceAlgorithmEx");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
        hy = CudnnFrontend::GetOutputDevicePointer();
        cy = CudnnFrontend::GetOutputDevicePointer();
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnAlgorithmPerformance_t *perfResults_backend =
            CudnnFrontend::GetOutputHostPointer<cudnnAlgorithmPerformance_t>(*returnedAlgoCount);
        std::memcpy(perfResults, perfResults_backend,
                    sizeof(cudnnAlgorithmPerformance_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardTrainingAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnGetRNNForwardTrainingAlgorithmMaxCount");
    if (CudnnFrontend::Success()) {
        *count = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardTrainingAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength,
    const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc,
    void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc,
    void *cy, const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddDevicePointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(cxDesc);
    CudnnFrontend::AddDevicePointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(hyDesc);
    CudnnFrontend::AddDevicePointerForArguments(cyDesc);
    CudnnFrontend::AddVariableForArguments<float>(findIntensity);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnAlgorithmPerformance_t>(perfResults,
                                                                           requestedAlgoCount);
    CudnnFrontend::AddDevicePointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(reserveSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindRNNForwardTrainingAlgorithmEx");
    if (CudnnFrontend::Success()) {
        y = CudnnFrontend::GetOutputDevicePointer();
        hy = CudnnFrontend::GetOutputDevicePointer();
        cy = CudnnFrontend::GetOutputDevicePointer();
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnAlgorithmPerformance_t *perfResults_backend =
            CudnnFrontend::GetOutputHostPointer<cudnnAlgorithmPerformance_t>(*returnedAlgoCount);
        std::memcpy(perfResults, perfResults_backend,
                    sizeof(cudnnAlgorithmPerformance_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnGetRNNBackwardDataAlgorithmMaxCount");
    if (CudnnFrontend::Success()) {
        *count = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength,
    const cudnnTensorDescriptor_t *yDesc, const void *y, const cudnnTensorDescriptor_t *dyDesc,
    const void *dy, const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnTensorDescriptor_t *dxDesc,
    void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults,
    void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(dhyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dhy);
    CudnnFrontend::AddDevicePointerForArguments(dcyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dcy);
    CudnnFrontend::AddDevicePointerForArguments(wDesc);
    CudnnFrontend::AddDevicePointerForArguments(w);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddDevicePointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(cxDesc);
    CudnnFrontend::AddDevicePointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(dxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dhxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dcxDesc);
    CudnnFrontend::AddVariableForArguments<float>(findIntensity);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnAlgorithmPerformance_t>(perfResults,
                                                                           requestedAlgoCount);
    CudnnFrontend::AddDevicePointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(reserveSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindRNNBackwardDataAlgorithmEx");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
        dhx = CudnnFrontend::GetOutputDevicePointer();
        dcx = CudnnFrontend::GetOutputDevicePointer();
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnAlgorithmPerformance_t *perfResults_backend =
            CudnnFrontend::GetOutputHostPointer<cudnnAlgorithmPerformance_t>(*returnedAlgoCount);
        std::memcpy(perfResults, perfResults_backend,
                    sizeof(cudnnAlgorithmPerformance_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::Execute("cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
    if (CudnnFrontend::Success()) {
        *count = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardWeightsAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength,
    const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc,
    const void *hx, const cudnnTensorDescriptor_t *yDesc, const void *y, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults,
    const void *workspace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(hxDesc);
    CudnnFrontend::AddDevicePointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<float>(findIntensity);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments<cudnnAlgorithmPerformance_t>(perfResults,
                                                                           requestedAlgoCount);
    CudnnFrontend::AddDevicePointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(dwDesc);
    CudnnFrontend::AddDevicePointerForArguments(dw);
    CudnnFrontend::AddDevicePointerForArguments(reserveSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindRNNBackwardWeightsAlgorithmEx");
    if (CudnnFrontend::Success()) {
        dw = CudnnFrontend::GetOutputDevicePointer();
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        cudnnAlgorithmPerformance_t *perfResults_backend =
            CudnnFrontend::GetOutputHostPointer<cudnnAlgorithmPerformance_t>(*returnedAlgoCount);
        std::memcpy(perfResults, perfResults_backend,
                    sizeof(cudnnAlgorithmPerformance_t) * (*returnedAlgoCount));
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateAlgorithmDescriptor");
    if (CudnnFrontend::Success()) {
        *algoDesc = CudnnFrontend::GetOutputVariable<cudnnAlgorithmDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(algoDesc);
    CudnnFrontend::AddVariableForArguments<cudnnAlgorithm_t>(algorithm);

    CudnnFrontend::Execute("cudnnSetAlgorithmDescriptor");
    if (CudnnFrontend::Success()) {
        algoDesc = CudnnFrontend::GetOutputVariable<cudnnAlgorithmDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmDescriptor(
    const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(algoDesc);
    CudnnFrontend::AddHostPointerForArguments<cudnnAlgorithm_t>((cudnnAlgorithm_t *)algorithm);

    CudnnFrontend::Execute("cudnnGetAlgorithmDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCopyAlgorithmDescriptor(
    const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(src);
    CudnnFrontend::AddDevicePointerForArguments(dest);

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(algoDesc);

    CudnnFrontend::Execute("cudnnDestroyAlgorithmDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<int>(numberToCreate);

    CudnnFrontend::Execute("cudnnCreateAlgorithmPerformance");
    if (CudnnFrontend::Success()) {
        *algoPerf = CudnnFrontend::GetOutputVariable<cudnnAlgorithmPerformance_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmPerformance(
    cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t algoDesc, cudnnStatus_t status,
    float time, size_t memory) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(algoPerf);
    CudnnFrontend::AddDevicePointerForArguments(algoDesc);
    CudnnFrontend::AddVariableForArguments<cudnnStatus_t>(status);
    CudnnFrontend::AddVariableForArguments<float>(time);
    CudnnFrontend::AddVariableForArguments<size_t>(memory);

    CudnnFrontend::Execute("cudnnSetAlgorithmPerformance");
    if (CudnnFrontend::Success()) {
        algoPerf = CudnnFrontend::GetOutputVariable<cudnnAlgorithmPerformance_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmPerformance(
    const cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t *algoDesc,
    cudnnStatus_t *status, float *time, size_t *memory) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(algoPerf);

    CudnnFrontend::Execute("cudnnGetAlgorithmPerformance");
    if (CudnnFrontend::Success()) {
        *algoDesc = CudnnFrontend::GetOutputVariable<cudnnAlgorithmDescriptor_t>();
        *status = CudnnFrontend::GetOutputVariable<cudnnStatus_t>();
        *memory = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddHostPointerForArguments<cudnnAlgorithmPerformance_t>(algoPerf);

    CudnnFrontend::Execute("cudnnDestroyAlgorithmPerformance");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle,
                                                                cudnnAlgorithmDescriptor_t algoDesc,
                                                                size_t *algoSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(algoDesc);

    CudnnFrontend::Execute("cudnnGetAlgorithmSpaceSize");
    if (CudnnFrontend::Success()) {
        *algoSpaceSizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSaveAlgorithm(cudnnHandle_t handle,
                                                        cudnnAlgorithmDescriptor_t algoDesc,
                                                        void *algoSpace,
                                                        size_t algoSpaceSizeInBytes) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(algoDesc);
    CudnnFrontend::AddHostPointerForArguments(algoSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(algoSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnSaveAlgorithm");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRestoreAlgorithm(cudnnHandle_t handle, void *algoSpace,
                                                           size_t algoSpaceSizeInBytes,
                                                           cudnnAlgorithmDescriptor_t algoDesc) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddHostPointerForArguments(algoSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(algoSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(algoDesc);

    CudnnFrontend::Execute("cudnnRestoreAlgorithm");

    return CudnnFrontend::GetExitCode();
}
#endif

#if CUDNN_VERSION >= 6000 && CUDNN_VERSION < 9000
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetRNNDescriptor_v6(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize,
                         const int numLayers, cudnnDropoutDescriptor_t dropoutDesc,
                         cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction,
                         cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(hiddenSize);
    CudnnFrontend::AddVariableForArguments<int>(numLayers);
    CudnnFrontend::AddDevicePointerForArguments(dropoutDesc);
    CudnnFrontend::AddVariableForArguments<cudnnRNNInputMode_t>(inputMode);
    CudnnFrontend::AddVariableForArguments<cudnnDirectionMode_t>(direction);
    CudnnFrontend::AddVariableForArguments<cudnnRNNMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnRNNAlgo_t>(algo);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(mathPrec);

    CudnnFrontend::Execute("cudnnSetRNNDescriptor_v6");
    if (CudnnFrontend::Success()) {
        rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetRNNDescriptor_v6(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int *hiddenSize,
                         int *numLayers, cudnnDropoutDescriptor_t *dropoutDesc,
                         cudnnRNNInputMode_t *inputMode, cudnnDirectionMode_t *direction,
                         cudnnRNNMode_t *mode, cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);

    CudnnFrontend::Execute("cudnnGetRNNDescriptor_v6");
    if (CudnnFrontend::Success()) {
        *hiddenSize = CudnnFrontend::GetOutputVariable<int>();
        *numLayers = CudnnFrontend::GetOutputVariable<int>();
        *dropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
        *inputMode = CudnnFrontend::GetOutputVariable<cudnnRNNInputMode_t>();
        *direction = CudnnFrontend::GetOutputVariable<cudnnDirectionMode_t>();
        *mode = CudnnFrontend::GetOutputVariable<cudnnRNNMode_t>();
        *algo = CudnnFrontend::GetOutputVariable<cudnnRNNAlgo_t>();
        *mathPrec = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

// TODO:
extern "C" size_t CUDNNWINAPI cudnnGetCudartVersion() {
    CudnnFrontend::Prepare();
    CudnnFrontend::Execute("cudnnGetCudartVersion");
    if (CudnnFrontend::Success()) {
        return CudnnFrontend::GetOutputVariable<cudnnStatus_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendCreateDescriptor(
    cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<cudnnBackendDescriptorType_t>(descriptorType);
    cout << "cudnnBackendCreateDescriptor called with descriptorType: " << descriptorType << endl;
    CudnnFrontend::Execute("cudnnBackendCreateDescriptor");
    if (CudnnFrontend::Success()) {
        *descriptor = CudnnFrontend::GetOutputVariable<cudnnBackendDescriptor_t>();
        cout << "cudnnBackendCreateDescriptor created descriptor: " << *descriptor << endl;
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendSetAttribute(
    cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t elementCount, const void *arrayOfElements) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(descriptor);
    CudnnFrontend::AddVariableForArguments<cudnnBackendAttributeName_t>(attributeName);
    CudnnFrontend::AddVariableForArguments<cudnnBackendAttributeType_t>(attributeType);
    CudnnFrontend::AddVariableForArguments<int64_t>(elementCount);

    int64_t byteCount = elementCount * getCudnnTypeSize(attributeType);
    CudnnFrontend::AddHostPointerForArguments<char>((char *)arrayOfElements, byteCount);

    if (attributeName == CUDNN_ATTR_VARIANT_PACK_WORKSPACE &&
        attributeType == CUDNN_TYPE_VOID_PTR) {
        cout << "SETTING WORKSPACE POINTER TO" << endl;
    }

    cout << "cudnnBackendSetAttribute called with:" << endl;
    cout << "descriptor: " << descriptor << endl;
    cout << "attribute name: " << attributeName << endl;
    cout << "attribute type: " << attributeType << endl;
    cout << "element count: " << elementCount << endl;
    cout << "byte count: " << byteCount << endl;
    if (byteCount > 0) {
        printHex(arrayOfElements, byteCount,
                 "[FRONTEND] Sending bytes for cudnnBackendSetAttribute");
    } else {
        cout << "[FRONTEND] No bytes to send for cudnnBackendSetAttribute" << endl;
    }
    CudnnFrontend::Execute("cudnnBackendSetAttribute");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendGetAttribute(
    cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t requestedElementCount, int64_t *elementCount,
    void *arrayOfElements) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(descriptor);
    CudnnFrontend::AddVariableForArguments<cudnnBackendAttributeName_t>(attributeName);
    CudnnFrontend::AddVariableForArguments<cudnnBackendAttributeType_t>(attributeType);
    CudnnFrontend::AddVariableForArguments<int64_t>(requestedElementCount);
    // CudnnFrontend::AddHostPointerForArguments<int64_t>(elementCount);
    CudnnFrontend::AddHostPointerForArguments<char>(
        (char *)arrayOfElements, requestedElementCount * getCudnnTypeSize(attributeType));
    cout << "cudnnBackendGetAttribute called with:" << endl;
    cout << "descriptor: " << descriptor << endl;
    cout << "attribute name: " << attributeName << endl;
    cout << "attribute type: " << attributeType << endl;
    cout << "requested element count: " << requestedElementCount << endl;
    CudnnFrontend::Execute("cudnnBackendGetAttribute");
    if (CudnnFrontend::Success()) {
        cout << "cudnnBackendGetAttribute succeeded" << endl;
        auto val = CudnnFrontend::GetOutputVariable<int64_t>();
        cout << "val: " << val << endl;
        int64_t elementsToWrite = std::min(val, requestedElementCount);
        if (elementCount != nullptr) {
            *elementCount = val;
            cout << "setting elementCount to: " << val << endl;
            cout << "elementCount: " << *elementCount << endl;
        }
        int64_t bytesToWrite = elementsToWrite * getCudnnTypeSize(attributeType);
        cout << "elementsToWrite: " << elementsToWrite << endl;
        cout << "bytesToWrite: " << bytesToWrite << endl;
        if (bytesToWrite > 0 && arrayOfElements != nullptr) {
            std::memcpy(arrayOfElements, CudnnFrontend::GetOutputHostPointer<char>(bytesToWrite),
                        bytesToWrite);
        }
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendExecute(cudnnHandle_t handle,
                                                         cudnnBackendDescriptor_t executionPlan,
                                                         cudnnBackendDescriptor_t variantPack) {
    cout << "cudnnBackendExecute called" << endl;
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(executionPlan);
    CudnnFrontend::AddDevicePointerForArguments(variantPack);
    CudnnFrontend::Execute("cudnnBackendExecute");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(descriptor);
    CudnnFrontend::Execute("cudnnBackendFinalize");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t desc) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(desc);
    CudnnFrontend::Execute("cudnnBackendDestroyDescriptor");
    return CudnnFrontend::GetExitCode();
}

// TODO:
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCTCLoss(
    cudnnHandle_t handle, cudnnTensorDescriptor_t probsDesc, const void *probs,
    const int hostLabels[], const int hostLabelLengths[], const int hostInputLengths[], void *costs,
    cudnnTensorDescriptor_t gradientsDesc, void *gradients, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace, size_t workSpaceSizeInBytes) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(probsDesc);
    CudnnFrontend::AddDevicePointerForArguments(probs);
    CudnnFrontend::AddHostPointerForArguments<const int>(hostLabels);
    CudnnFrontend::AddHostPointerForArguments<const int>(hostLabelLengths);
    CudnnFrontend::AddHostPointerForArguments<const int>(hostInputLengths);
    CudnnFrontend::AddDevicePointerForArguments(costs);
    CudnnFrontend::AddDevicePointerForArguments(gradientsDesc);
    CudnnFrontend::AddDevicePointerForArguments(gradients);
    CudnnFrontend::AddVariableForArguments<cudnnCTCLossAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(ctcLossDesc);
    CudnnFrontend::AddDevicePointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::Execute("cudnnCTCLoss");
    if (CudnnFrontend::Success()) {
        costs = CudnnFrontend::GetOutputDevicePointer();
        gradients = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

// TODO:
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData_v8(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int32_t devSeqLengths[],
    cudnnRNNDataDescriptor_t yDesc, const void *y, const void *dy, cudnnRNNDataDescriptor_t xDesc,
    void *dx, cudnnTensorDescriptor_t hDesc, const void *hx, const void *dhy, void *dhx,
    cudnnTensorDescriptor_t cDesc, const void *cx, const void *dcy, void *dcx,
    size_t weightSpaceSize, const void *weightSpace, size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddHostPointerForArguments<const int32_t>(devSeqLengths);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(dx);
    CudnnFrontend::AddDevicePointerForArguments(hDesc);
    CudnnFrontend::AddDevicePointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(dhy);
    CudnnFrontend::AddDevicePointerForArguments(dhx);
    CudnnFrontend::AddDevicePointerForArguments(cDesc);
    CudnnFrontend::AddDevicePointerForArguments(cx);
    CudnnFrontend::AddDevicePointerForArguments(dcy);
    CudnnFrontend::AddDevicePointerForArguments(dcx);
    CudnnFrontend::AddVariableForArguments<size_t>(weightSpaceSize);
    CudnnFrontend::AddDevicePointerForArguments(weightSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSize);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSize);
    CudnnFrontend::AddDevicePointerForArguments(reserveSpace);
    CudnnFrontend::Execute("cudnnRNNBackwardData_v8");
    if (CudnnFrontend::Success()) {
        dx = CudnnFrontend::GetOutputDevicePointer();
        dhx = CudnnFrontend::GetOutputDevicePointer();
        dcx = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCTCLoss_v8(cudnnHandle_t handle, cudnnCTCLossAlgo_t algo,
                const cudnnCTCLossDescriptor_t ctcLossDesc, const cudnnTensorDescriptor_t probsDesc,
                const void *probs, const int labels[], const int labelLengths[],
                const int inputLengths[], void *costs, const cudnnTensorDescriptor_t gradientsDesc,
                void *gradients, size_t workSpaceSizeInBytes, void *workspace) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnCTCLossAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(ctcLossDesc);
    CudnnFrontend::AddDevicePointerForArguments(probsDesc);
    CudnnFrontend::AddDevicePointerForArguments(probs);
    CudnnFrontend::AddHostPointerForArguments<const int>(labels);
    CudnnFrontend::AddHostPointerForArguments<const int>(labelLengths);
    CudnnFrontend::AddHostPointerForArguments<const int>(inputLengths);
    CudnnFrontend::AddDevicePointerForArguments(costs);
    CudnnFrontend::AddDevicePointerForArguments(gradientsDesc);
    CudnnFrontend::AddDevicePointerForArguments(gradients);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddDevicePointerForArguments(workspace);
    CudnnFrontend::Execute("cudnnCTCLoss_v8");
    if (CudnnFrontend::Success()) {
        costs = CudnnFrontend::GetOutputDevicePointer();
        gradients = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor_v9(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType,
    cudnnLossNormalizationMode_t normMode, cudnnCTCGradMode_t ctcGradMode, int maxLabelLength) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(ctcLossDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(compType);
    CudnnFrontend::AddVariableForArguments<cudnnLossNormalizationMode_t>(normMode);
    CudnnFrontend::AddVariableForArguments<cudnnCTCGradMode_t>(ctcGradMode);
    CudnnFrontend::AddVariableForArguments<int>(maxLabelLength);
    CudnnFrontend::Execute("cudnnSetCTCLossDescriptor_v9");
    if (CudnnFrontend::Success()) {
        ctcLossDesc = CudnnFrontend::GetOutputVariable<cudnnCTCLossDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNWeightParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int32_t pseudoLayer, size_t weightSpaceSize,
    const void *weightSpace, int32_t linLayerID, cudnnTensorDescriptor_t mDesc, void **mAddr,
    cudnnTensorDescriptor_t bDesc, void **bAddr) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<int32_t>(pseudoLayer);
    CudnnFrontend::AddVariableForArguments<size_t>(weightSpaceSize);
    CudnnFrontend::AddDevicePointerForArguments(weightSpace);
    CudnnFrontend::AddVariableForArguments<int32_t>(linLayerID);
    CudnnFrontend::AddDevicePointerForArguments(mDesc);
    CudnnFrontend::AddDevicePointerForArguments(mAddr);
    CudnnFrontend::AddDevicePointerForArguments(bDesc);
    CudnnFrontend::AddDevicePointerForArguments(bAddr);
    CudnnFrontend::Execute("cudnnGetRNNWeightParams");
    if (CudnnFrontend::Success()) {
        *mAddr = CudnnFrontend::GetOutputDevicePointer();
        *bAddr = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights_v8(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnWgradMode_t addGrad,
    const int32_t devSeqLengths[], cudnnRNNDataDescriptor_t xDesc, const void *x,
    cudnnTensorDescriptor_t hDesc, const void *hx, cudnnRNNDataDescriptor_t yDesc, const void *y,
    size_t weightSpaceSize, void *dweightSpace, size_t workSpaceSize, void *workSpace,
    size_t reserveSpaceSize, void *reserveSpace) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(rnnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnWgradMode_t>(addGrad);
    CudnnFrontend::AddHostPointerForArguments<const int32_t>(devSeqLengths);
    CudnnFrontend::AddDevicePointerForArguments(xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddDevicePointerForArguments(hDesc);
    CudnnFrontend::AddDevicePointerForArguments(hx);
    CudnnFrontend::AddDevicePointerForArguments(yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<size_t>(weightSpaceSize);
    CudnnFrontend::AddDevicePointerForArguments(dweightSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSize);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSize);
    CudnnFrontend::AddDevicePointerForArguments(reserveSpace);
    CudnnFrontend::Execute("cudnnRNNBackwardWeights_v8");
    if (CudnnFrontend::Success()) {
        dweightSpace = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossWorkspaceSize_v8(
    cudnnHandle_t handle, cudnnCTCLossAlgo_t algo, const cudnnCTCLossDescriptor_t ctcLossDesc,
    const cudnnTensorDescriptor_t probsDesc, const cudnnTensorDescriptor_t gradientsDesc,
    size_t *sizeInBytes) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddVariableForArguments<cudnnCTCLossAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(ctcLossDesc);
    CudnnFrontend::AddDevicePointerForArguments(probsDesc);
    CudnnFrontend::AddDevicePointerForArguments(gradientsDesc);
    CudnnFrontend::AddHostPointerForArguments<size_t>(sizeInBytes);
    CudnnFrontend::Execute("cudnnGetCTCLossWorkspaceSize_v8");
    if (CudnnFrontend::Success()) {
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType type, int *value) {
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<libraryPropertyType>(type);

    CudnnFrontend::Execute("cudnnGetProperty");
    if (CudnnFrontend::Success()) {
        *value = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}