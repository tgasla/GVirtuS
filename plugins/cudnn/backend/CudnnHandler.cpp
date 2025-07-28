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
 *
*/

#include "CudnnHandler.h"

using namespace std;
using namespace log4cplus;

std::map<string, CudnnHandler::CudnnRoutineHandler> * CudnnHandler::mspHandlers = NULL;

static std::mutex desc_type_mutex;
static std::unordered_map<void*, bool> desc_is_float_map;

// Helper Functions

// Generic setter (used when you create a descriptor)
void registerDescriptorType(void* desc, const cudnnDataType_t dataType) {
    std::lock_guard<std::mutex> lock(desc_type_mutex);
    desc_is_float_map[desc] = (dataType != CUDNN_DATA_DOUBLE);
}

// Generic getter for descriptor type
bool isFloatDescriptor(const void* desc) {
    std::lock_guard<std::mutex> lock(desc_type_mutex);
    auto it = desc_is_float_map.find(const_cast<void*>(desc));
    if (it != desc_is_float_map.end()) {
        return it->second;
    }
    return true; // Default if unknown, assume float
}

extern "C" std::shared_ptr<CudnnHandler> create_t() {
    return std::make_shared<CudnnHandler>();
}

extern "C" int HandlerInit() {
    return 0;
}

CudnnHandler::CudnnHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("CudnnHandler"));
    setLogLevel(&logger);
    Initialize();
}

CudnnHandler::~CudnnHandler() {}

void CudnnHandler::setLogLevel(Logger *logger) {
	log4cplus::LogLevel logLevel = log4cplus::INFO_LOG_LEVEL;
    char *val = getenv("GVIRTUS_LOGLEVEL");
	std::string logLevelString =(val == NULL ? std::string("") : std::string(val));
	if (logLevelString != "") {
        try {
            logLevel = std::stoi(logLevelString);
        } catch (const std::exception &e) {
            std::cerr << "[WARNING] Invalid GVIRTUS_LOGLEVEL='" << logLevelString
                      << "', defaulting to INFO. Reason: " << e.what() << std::endl;
            logLevel = log4cplus::INFO_LOG_LEVEL;
        }
    }
    logger->setLogLevel(logLevel);
}

bool CudnnHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();

}

std::shared_ptr<Result> CudnnHandler::Execute(std::string routine, std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger,"Called " << routine);
    map<string, CudnnHandler::CudnnRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw runtime_error("No handler for '" + routine + "' found!");
    try {
        return it->second(this, input_buffer);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
    }
    return NULL;
}

void CudnnHandler::Initialize() {
   if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CudnnHandler::CudnnRoutineHandler> ();

    /* CublasHandler Query Platform Info */
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetVersion));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetErrorString));   
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Create));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Destroy));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetStream));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetStream)); 
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor4dDescriptorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensor4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensorNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensorNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensorSizeInBytes));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(InitTransformDest));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyTensorTransformDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(TransformTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(TransformTensorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFoldedConvBackwardDataDescriptors));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(AddTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyOpTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(OpTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyReduceTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetReductionIndicesSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetReductionWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ReduceTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ScaleTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFilterDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor));
    #if CUDNN_VERSION < 6000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor_v4));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor_v4));
    #endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor));
    #if CUDNN_VERSION < 6000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor_v4));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor_v4));
    #endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterSizeInBytes));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFilterDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(TransformFilter));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ReorderFilterAndBias));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateConvolutionDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionGroupCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionGroupCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionReorderType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionReorderType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolution2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolution2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolution2dForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolutionNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionNdForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyConvolutionDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionForwardAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionForwardAlgorithmEx));
#if CUDNN_VERSION < 8000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardAlgorithm));
#endif
#if CUDNN_VERSION >= 7000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardAlgorithm_v7));
#endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionForwardWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBiasActivationForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBackwardBias));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardFilterAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardFilterAlgorithmEx));
#if CUDNN_VERSION < 8000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterAlgorithm));
#endif
#if CUDNN_VERSION >= 7000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterAlgorithm_v7));
#endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardFilterWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBackwardFilter));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardDataAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindConvolutionBackwardDataAlgorithmEx));
#if CUDNN_VERSION < 8000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataAlgorithm));
#endif
#if CUDNN_VERSION >= 7000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataAlgorithm_v7));
#endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolutionBackwardDataWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ConvolutionBackwardData));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Im2Col));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SoftmaxForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SoftmaxBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreatePoolingDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetPooling2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPooling2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetPoolingNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPoolingNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPoolingNdForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetPooling2dForwardOutputDim));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyPoolingDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(PoolingForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(PoolingBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyActivationDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ActivationForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ActivationBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyLRNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(LRNCrossChannelForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(LRNCrossChannelBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DivisiveNormalizationForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DivisiveNormalizationBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DeriveBNTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetBatchNormalizationForwardTrainingExWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetBatchNormalizationBackwardExWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetBatchNormalizationTrainingExReserveSpaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationForwardTraining));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationForwardTrainingEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationForwardInference));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(BatchNormalizationBackwardEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateSpatialTransformerDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetSpatialTransformerNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroySpatialTransformerDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfGridGeneratorForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfGridGeneratorBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfSamplerForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SpatialTfSamplerBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutGetStatesSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutGetReserveSpaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RestoreDropoutDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetDropoutDescriptor)); 
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DropoutBackward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateRNNDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyRNNDescriptor));
#if CUDNN_VERSION < 8000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDescriptor_v5));
    //mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNDescriptor_v5));
#endif
#if CUDNN_VERSION >= 8000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDescriptor_v8));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNDescriptor_v8));
#endif
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNWeightSpaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNTempSpaceSizes));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateSeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroySeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetSeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetSeqDataDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAttnDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetMultiHeadAttnBuffers));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetMultiHeadAttnWeights));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MultiHeadAttnForward));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MultiHeadAttnBackwardData));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MultiHeadAttnBackwardWeights));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetCTCLossDescriptorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCTCLossDescriptorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyCTCLossDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CTCLoss));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCTCLossWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetCallback));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetCallback));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFusedOpsConstParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFusedOpsConstParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFusedOpsConstParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFusedOpsConstParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFusedOpsVariantParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFusedOpsVariantParamPack));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFusedOpsVariantParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFusedOpsVariantParamPackAttribute));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFusedOpsPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFusedOpsPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(MakeFusedOpsPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FusedOpsExecute));
#if CUDNN_VERSION < 9000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNMatrixMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNMatrixMathType));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNBiasMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNBiasMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNSetClip));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNGetClip));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNProjectionLayers));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNProjectionLayers));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreatePersistentRNNPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyPersistentRNNPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetPersistentRNNPlan));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNWorkspaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNTrainingReserveSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNParamsSize));   
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNLinLayerMatrixParams));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNLinLayerBiasParams));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardInference));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardTraining));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardData));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardWeights));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNPaddingMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNPaddingMode));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardTrainingEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNForwardInferenceEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardDataEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RNNBackwardWeightsEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNForwardInferenceAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNForwardInferenceAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNForwardTrainingAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNForwardTrainingAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNBackwardDataAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNBackwardDataAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNBackwardWeightsAlgorithmMaxCount));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(FindRNNBackwardWeightsAlgorithmEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CopyAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyAlgorithmDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyAlgorithmPerformance));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetAlgorithmSpaceSize));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SaveAlgorithm));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(RestoreAlgorithm));
#endif
#if CUDNN_VERSION >= 6000 && CUDNN_VERSION < 9000
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetRNNDescriptor_v6));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetRNNDescriptor_v6));
#endif
}

CUDNN_ROUTINE_HANDLER(GetConvolutionMathType) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionMathType"));

     cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
     cudnnMathType_t mathType;

     cudnnStatus_t cs = cudnnGetConvolutionMathType(convDesc, &mathType);

     LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionMathType Executed");
     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
     try {
        out->Add<cudnnMathType_t>(mathType);
     } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
     }
     return std::make_shared<Result>(cs, out);
 }

CUDNN_ROUTINE_HANDLER(SetConvolutionReorderType) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionReorderType"));

    cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    cudnnReorderType_t reorderType = in->Get<cudnnReorderType_t>();

    cudnnStatus_t cs = cudnnSetConvolutionReorderType(convDesc, reorderType);
    LOG4CPLUS_DEBUG(logger, "cudnnSetConvolutionReorderType Executed");

    return std::make_shared<Result>(cs); 
 }

CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardFilterAlgorithm) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionBackwardFilterAlgorithm"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const cudnnTensorDescriptor_t DyDesc = in->Get<cudnnTensorDescriptor_t>();
    const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    const cudnnFilterDescriptor_t dwDesc = in->Get<cudnnFilterDescriptor_t>();
    const int requestedAlgoCount = in->Get<int>();
    int returnedAlgoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults = in->Get<cudnnConvolutionBwdFilterAlgoPerf_t>(requestedAlgoCount);
    
    cudnnStatus_t cs = cudnnFindConvolutionBackwardFilterAlgorithm(handle, xDesc, DyDesc, convDesc, dwDesc, requestedAlgoCount, &returnedAlgoCount, perfResults);
    LOG4CPLUS_DEBUG(logger, "cudnnFindConvolutionBackwardFilterAlgorithm Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionBwdFilterAlgoPerf_t>(perfResults, returnedAlgoCount);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);  
 }

CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithmMaxCount) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithmMaxCount"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    int count;

    cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &count);
    LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionForwardAlgorithmMaxCount Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(count);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);
 }

CUDNN_ROUTINE_HANDLER(SetConvolutionNdDescriptor) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionNdDescriptor"));

     cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
     int arrayLength = in->Get<int>();
     int *padA = in->Assign<int>();
     int *filterStrideA = in->Assign<int>();
     int *dilationA = in->Assign<int>();
     cudnnConvolutionMode_t mode = in->Get<cudnnConvolutionMode_t>();
     cudnnDataType_t computeType = in->Get<cudnnDataType_t>();

     cudnnStatus_t cs = cudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);

     LOG4CPLUS_DEBUG(logger, "cudnnSetConvolutionNdDescriptor Executed");   
     
     return std::make_shared<Result>(cs);
 }

CUDNN_ROUTINE_HANDLER(GetConvolutionNdForwardOutputDim) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionNdForwardOutputDim"));

    cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    cudnnTensorDescriptor_t inputTensorDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
    int nbDims = in->Get<int>();

    int tensorOutputDimA[nbDims];

    cudnnStatus_t cs = cudnnGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOutputDimA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(tensorOutputDimA, nbDims);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionNdForwardOutputDim Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardFilterAlgorithmEx) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionBackwardFilterAlgorithmEx"));
    
    cudnnHandle_t handle = in->Get<cudnnHandle_t>(); //INPUT
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>(); //INPUT
    const void *x = in->GetFromMarshal<void*>(); //INPUT
    const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>(); //INPUT
    const void *y = in->GetFromMarshal<void*>(); //INPUT
    const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>(); //INPUT
    const cudnnFilterDescriptor_t dwDesc = in->Get<cudnnFilterDescriptor_t>(); //INPUT
    void *dw = in->GetFromMarshal<void*>(); //INPUT/OUTPUT
    const int requestedAlgoCount = in->Get<int>(); //INPUT
    int returnedAlgoCount; //OUTPUT
    cudnnConvolutionBwdFilterAlgoPerf_t* perfResults = in->Get<cudnnConvolutionBwdFilterAlgoPerf_t>(requestedAlgoCount); //OUTPUT
    void *workSpace = in->GetFromMarshal<void*>(); //INPUT
    size_t workSpaceSizeInBytes = in->Get<size_t>(); //INPUT
   
    cudnnStatus_t cs = cudnnFindConvolutionBackwardFilterAlgorithmEx(handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, &returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
    LOG4CPLUS_DEBUG(logger, "cudnnFindConvolutionBackwardFilterAlgorithmEx Executed");
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(dw);
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionBwdFilterAlgoPerf_t>(perfResults, returnedAlgoCount); 
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);
}



CUDNN_ROUTINE_HANDLER(GetConvolution2dDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolution2dDescriptor"));

     cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
     int padh,padw,u,v,upscalex,upscaley;
     cudnnConvolutionMode_t mode;
     cudnnDataType_t computeType = CUDNN_DATA_FLOAT;

     cudnnStatus_t cs = cudnnGetConvolution2dDescriptor(convDesc,&padh,&padw,&u,&v,&upscalex,&upscaley,&mode,&computeType);
     LOG4CPLUS_DEBUG(logger, "cudnnGetConvolution2dDescriptor Executed");

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add(padh);
        out->Add(padw);
        out->Add(u);
        out->Add(v);
        out->Add(upscalex);
        out->Add(upscaley);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs,out);
 }

CUDNN_ROUTINE_HANDLER(SetConvolutionGroupCount) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionGroupCount"));

     cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
     int groupCount = in->Get<int>();

     cudnnStatus_t cs = cudnnSetConvolutionGroupCount(convDesc, groupCount);

     LOG4CPLUS_DEBUG(logger, "cudnnSetConvolutionGroupCount Executed");
     return std::make_shared<Result>(cs);
 }

CUDNN_ROUTINE_HANDLER(FindConvolutionForwardAlgorithmEx) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionForwardAlgorithmEx"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *x = in->GetFromMarshal<void *>();
    const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
    const void *w = in->GetFromMarshal<void *>();
    const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    void *y = in->GetFromMarshal<void *>();
    const int requestedAlgoCount = in->Get<int>();
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t *perfResults = in->Get<cudnnConvolutionFwdAlgoPerf_t>(requestedAlgoCount);
    void *workSpace = in->GetFromMarshal<void *>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();

    cudnnStatus_t cs = cudnnFindConvolutionForwardAlgorithmEx(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, &returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
    LOG4CPLUS_DEBUG(logger, "cudnnFindConvolutionForwardAlgorithmEx Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void *>(y);
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionFwdAlgoPerf_t>(perfResults, returnedAlgoCount);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionNdDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionNdDescriptor"));
    
    cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    int arrayLengthRequested = in->Get<int>();
    int arrayLength;
    int *padA = in->Assign<int>();
    int *filterStrideA = in->Assign<int>();
    int *dilationA = in->Assign<int>();
    cudnnConvolutionMode_t mode;
    cudnnDataType_t dataType;

    cudnnStatus_t cs = cudnnGetConvolutionNdDescriptor(convDesc, arrayLengthRequested, &arrayLength,padA, filterStrideA, dilationA, &mode, &dataType);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
        out->Add<int>(arrayLength);
        out->Add<int>(padA);
        out->Add<int>(filterStrideA);
        out->Add<int>(dilationA);
        out->Add<cudnnConvolutionMode_t>(mode);
        out->Add<cudnnDataType_t>(dataType);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionNdDescriptor Executed");
    return std::make_shared<Result>(cs,out);   
}

#if CUDNN_VERSION < 8000
CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithm) {
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithm"));

  cudnnHandle_t handle = in->Get<cudnnHandle_t>();
  cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
  cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
  cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
  cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
  cudnnConvolutionFwdPreference_t preference = in->Get<cudnnConvolutionFwdPreference_t>();
    size_t memoryLimitInBytes = (size_t)in->Get<int>();

    cudnnConvolutionFwdAlgo_t algo;



  cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, &algo);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnConvolutionFwdAlgo_t>(algo);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionForwardAlgorithm  Executed");
    return std::make_shared<Result>(cs, out);
}
#endif

#if CUDNN_VERSION >= 7000
CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithm_v7) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithm_v7"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
    cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    int requestedAlgoCount = in->Get<int>();
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t *perfResults = in->Get<cudnnConvolutionFwdAlgoPerf_t>(requestedAlgoCount);

    cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, &returnedAlgoCount, perfResults);
    LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionForwardAlgorithm_v7  Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionFwdAlgoPerf_t>(perfResults, returnedAlgoCount);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);
}
#endif

CUDNN_ROUTINE_HANDLER(GetConvolutionReorderType) {
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionReorderType"));
 
  cudnnConvolutionDescriptor_t convDesc= in->Get<cudnnConvolutionDescriptor_t>();
  cudnnReorderType_t reorderType;

  cudnnStatus_t cs = cudnnGetConvolutionReorderType(convDesc, &reorderType);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnReorderType_t>(reorderType);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionReorderType  Executed");
    return std::make_shared<Result>(cs, out); 
}

CUDNN_ROUTINE_HANDLER(SetConvolutionMathType) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionMathType"));

   cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
   cudnnMathType_t mathType = in->Get<cudnnMathType_t>();

   cudnnStatus_t cs = cudnnSetConvolutionMathType(convDesc, mathType);

    LOG4CPLUS_DEBUG(logger,"cudnnSetConvolutionMathType  Executed");
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(ConvolutionBiasActivationForward) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBiasActivationForward"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *alpha1 = isFloatDescriptor(xDesc)
    ? static_cast<const void *>(in->Assign<const float>())
    : static_cast<const void *>(in->Assign<const double>());
   const void *x = in->GetFromMarshal<void *>();
   const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
   const void *w = in->GetFromMarshal<void *>();
   const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
   const cudnnConvolutionFwdAlgo_t algo = in->Get<cudnnConvolutionFwdAlgo_t>();
   void *workSpace = in->GetFromMarshal<void *>();
   const size_t workSpaceSizeInBytes = in->Get<size_t>();
   const cudnnTensorDescriptor_t zDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *alpha2 = isFloatDescriptor(zDesc)
    ? static_cast<const void *>(in->Assign<const float>())
    : static_cast<const void *>(in->Assign<const double>());
   const void *z = in->GetFromMarshal<void *>();
   const cudnnTensorDescriptor_t biasDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *bias = in->GetFromMarshal<void *>();
   const cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
   const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
   void *y = in->GetFromMarshal<void *>();

   cudnnStatus_t cs = cudnnConvolutionBiasActivationForward(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc, y);
   LOG4CPLUS_DEBUG(logger,"cudnnConvolutionBiasActivationForward  Executed");

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->AddMarshal<void *>(y);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithmMaxCount) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterAlgorithmMaxCount"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    int count;

    cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle, &count);
    LOG4CPLUS_DEBUG(logger,"GetConvolutionBackwardFilterAlgorithmMaxCount  Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(count);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);
}

#if CUDNN_VERSION < 8000
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithm) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterAlgorithm"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    cudnnFilterDescriptor_t dwDesc = in->Get<cudnnFilterDescriptor_t>();
    int requestedAlgoCount = in->Get<int>();
    int returnedAlgoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults;

    cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, &returnedAlgoCount, &perfResults);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<int>(returnedAlgoCount);
         out->Add<cudnnConvolutionBwdFilterAlgoPerf_t>(perfResults);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"GetConvolutionBackwardFilterAlgorithm  Executed");
    return std::make_shared<Result>(cs,out);
}
#endif

#if CUDNN_VERSION >= 7000
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithm_v7) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterAlgorithm_v7"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
    const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    const cudnnFilterDescriptor_t dwDesc = in->Get<cudnnFilterDescriptor_t>();
    const int requestedAlgoCount = in->Get<int>();
    int returnedAlgoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t* perfResults = in->Get<cudnnConvolutionBwdFilterAlgoPerf_t>(requestedAlgoCount);
    
    cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, &returnedAlgoCount, perfResults);
    LOG4CPLUS_DEBUG(logger,"GetConvolutionBackwardFilterAlgorithm_v7  Executed");
  
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionBwdFilterAlgoPerf_t>(perfResults, returnedAlgoCount);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);   
}
#endif

CUDNN_ROUTINE_HANDLER(FindConvolutionForwardAlgorithm) {
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionForwardAlgorithm"));

  cudnnHandle_t handle = in->Get<cudnnHandle_t>();
  cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
  cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
  cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
  cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
  int requestedAlgoCount = in->Get<int>();
  int returnedAlgoCount;
  cudnnConvolutionFwdAlgoPerf_t perfResults;

  cudnnStatus_t cs = cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, &returnedAlgoCount, &perfResults);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<int>(returnedAlgoCount);
         out->Add<cudnnConvolutionFwdAlgoPerf_t>(perfResults);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"FindConvolutionForwardAlgorithm  Executed");
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionGroupCount) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionGroupCount"));

   cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
   int groupCount;

   cudnnStatus_t cs = cudnnGetConvolutionGroupCount(convDesc, &groupCount);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<int>(groupCount);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionGroupCount  Executed");
    return std::make_shared<Result>(cs,out); 
}


CUDNN_ROUTINE_HANDLER(DestroyConvolutionDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyConvolutionDescriptor"));
    cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();

    cudnnStatus_t cs = cudnnDestroyConvolutionDescriptor(convDesc);

    //LOG4CPLUS_DEBUG(logger,"cudnnDestroyConvolutionDescriptor  Executed");
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(ConvolutionBackwardBias) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBackwardBias"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alpha = isFloatDescriptor(dyDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *dy = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t dbDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *beta = isFloatDescriptor(dbDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    void *db = in->GetFromMarshal<void *>(); 

    cudnnStatus_t cs = cudnnConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta,  dbDesc, db);
    LOG4CPLUS_DEBUG(logger,"cudnnConvolutionBackwardBias  Executed");
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->AddMarshal<void *>(db);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs,out);
}


CUDNN_ROUTINE_HANDLER(ConvolutionForward) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionForward"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void* alpha = isFloatDescriptor(xDesc) ? in->Assign<void>(sizeof(float)) : in->Assign<void>(sizeof(double));
    const void* x = in->GetFromMarshal<void *>();
    const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
    const void* w = in->GetFromMarshal<void *>();
    const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    cudnnConvolutionFwdAlgo_t algo = in->Get<cudnnConvolutionFwdAlgo_t>();
    void* workSpace = in->GetFromMarshal<void *>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    const void* beta = isFloatDescriptor(yDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    void* y = in->GetFromMarshal<void *>();

    cudnnStatus_t cs = cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);

    LOG4CPLUS_DEBUG(logger,"cudnnConvolutionForward  Executed");
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
            out->AddMarshal<void*>(y);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);  
}


CUDNN_ROUTINE_HANDLER(ConvolutionBackwardFilter) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBackwardFilter"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void* alpha = isFloatDescriptor(xDesc)
    ? static_cast<const void*>(in->Assign<const float>())
    : static_cast<const void*>(in->Assign<const double>());
    const void* x = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
    const void* dy = in->GetFromMarshal<void *>();
    const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    cudnnConvolutionBwdFilterAlgo_t algo = in->Get<cudnnConvolutionBwdFilterAlgo_t>();
    void* workSpace = in->GetFromMarshal<void *>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    const cudnnFilterDescriptor_t dwDesc = in->Get<cudnnFilterDescriptor_t>();
    const void* beta = isFloatDescriptor(dwDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    void* dw = in->GetFromMarshal<void *>();

    cudnnStatus_t cs = cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta,dwDesc, dw);
    LOG4CPLUS_DEBUG(logger,"cudnnConvolutionBackwardFilter  Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void *>(dw);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs,out);  
}

CUDNN_ROUTINE_HANDLER(GetConvolution2dForwardOutputDim) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolution2dForwardOutputDim"));
   
   const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
   const cudnnTensorDescriptor_t inputTensor = in->Get<cudnnTensorDescriptor_t>();
   const cudnnFilterDescriptor_t filterDesc  = in->Get<cudnnFilterDescriptor_t>();
   int n;
   int c;
   int h;
   int w;

   cudnnStatus_t cs = cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensor, filterDesc, &n ,&c, &h, &w);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<int>(n);
         out->Add<int>(c);
         out->Add<int>(h);
         out->Add<int>(w);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetConvolution2dForwardOutputDim  Executed");
    return std::make_shared<Result>(cs,out);  
}


CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterWorkspaceSize) {
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterWorkspaceSize"));

  cudnnHandle_t handle = in->Get<cudnnHandle_t>();
  const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
  const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
  const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
  const cudnnFilterDescriptor_t dwDesc = in->Get<cudnnFilterDescriptor_t>();
  cudnnConvolutionBwdFilterAlgo_t algo = in->Get<cudnnConvolutionBwdFilterAlgo_t>();
  size_t sizeInBytes;

  cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, dyDesc, convDesc, dwDesc, algo, &sizeInBytes);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<size_t>(sizeInBytes);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionBackwardFilterWorkspaceSize  Executed");
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(CreateConvolutionDescriptor) {
  Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateConvolutionDescriptor"));

  cudnnConvolutionDescriptor_t convDesc;

  cudnnStatus_t cs = cudnnCreateConvolutionDescriptor(&convDesc);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnCreateConvolutionDescriptor  Executed");
    return std::make_shared<Result>(cs,out); 
}

#if CUDNN_VERSION < 8204
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardFilterAlgorithm) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardFilterAlgorithm"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
   const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
   const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
   const cudnnFilterDescriptor_t dwDesc = in->Get<cudnnFilterDescriptor_t>();
   cudnnConvolutionBwdFilterPreference_t preference = in->Get<cudnnConvolutionBwdFilterPreference_t>();
   size_t memoryLimitInBytes = in->Get<size_t>();
   cudnnConvolutionBwdFilterAlgo_t algo;

   cudnnStatus_t cs = cudnnGetConvolutionBackwardFilterAlgorithm(handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, &algo);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnConvolutionBwdFilterAlgo_t>(algo);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionBackwardFilterAlgorithm  Executed");
    return std::make_shared<Result>(cs,out);   
}
#endif
 
CUDNN_ROUTINE_HANDLER(SetConvolution2dDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolution2dDescriptor"));

    cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    int pad_h = in->Get<int>();
    int pad_w = in->Get<int>();
    int u = in->Get<int>();
    int v = in->Get<int>();
    int dilation_h = in->Get<int>();
    int dilation_w = in->Get<int>();
    cudnnConvolutionMode_t mode = in->Get<cudnnConvolutionMode_t>();
    cudnnDataType_t computeType = in->Get<cudnnDataType_t>();
   
    cudnnStatus_t cs = cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
    LOG4CPLUS_DEBUG(logger,"cudnnSetConvolution2dDescriptor  Executed");
    return std::make_shared<Result>(cs);
}

#if CUDNN_VERSION < 8204
CUDNN_ROUTINE_HANDLER(GetConvolutionForwardAlgorithm) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardAlgorithm"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
   const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
   const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
   const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
   cudnnConvolutionFwdPreference_t preference = in->Get<cudnnConvolutionFwdPreference_t>();
   size_t memoryLimitInBytes = in->Get<size_t>();
   cudnnConvolutionFwdAlgo_t algo;  
 
   cudnnStatus_t cs = cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, &algo);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnConvolutionFwdAlgo_t>(algo);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionForwardAlgorithm  Executed");
    return std::make_shared<Result>(cs,out);
}
#endif

CUDNN_ROUTINE_HANDLER(GetConvolutionForwardWorkspaceSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionForwardWorkspaceSize"));
   
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
    const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnConvolutionFwdAlgo_t algo = in->Get<cudnnConvolutionFwdAlgo_t>();
    size_t sizeInBytes;

    cudnnStatus_t cs = cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, &sizeInBytes);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<size_t>(sizeInBytes);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    //LOG4CPLUS_DEBUG(logger,"cudnnGetConvolutionForwardWorkspaceSize  Executed");
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetVersion) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetVersion"));

    size_t version = cudnnGetVersion();
    LOG4CPLUS_DEBUG(logger,"cudnnGetVersion Executed, version: " << version);
    return std::make_shared<Result>(version); // Return the version as an exit code!
}

CUDNN_ROUTINE_HANDLER(GetErrorString) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetErrorString"));
    cudnnStatus_t cs = in->Get<cudnnStatus_t>();
    const char* s = cudnnGetErrorString(cs);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddString(s);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(CUDNN_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnGetErrorString Executed");
    return std::make_shared<Result>(CUDNN_STATUS_SUCCESS, out);
}

CUDNN_ROUTINE_HANDLER(Create) {

    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Create"));
    cudnnHandle_t handle;
    cudnnStatus_t cs = cudnnCreate(&handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnHandle_t>(handle);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(CUDNN_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnCreate Executed");
    return std::make_shared<Result>(cs, out);

}

CUDNN_ROUTINE_HANDLER(Destroy) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Destroy"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnStatus_t cs = cudnnDestroy(handle);
    
    LOG4CPLUS_DEBUG(logger,"cudnnDestroy Executed");
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetStream) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetStream"));
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudaStream_t streamId = in->Get<cudaStream_t>();

    cudnnStatus_t cs = cudnnSetStream(handle, streamId);
    
     LOG4CPLUS_DEBUG(logger," cudnnSetStream Executed");
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetStream) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetStream"));
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudaStream_t streamId;
    cudnnStatus_t cs = cudnnGetStream(handle, &streamId);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudaStream_t>(streamId);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetStream Executed");
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(CreateTensorDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateTensorDescriptor"));
    cudnnTensorDescriptor_t tensorDesc;
    cudnnStatus_t cs = cudnnCreateTensorDescriptor(&tensorDesc);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger,"cudnnCreateTensorDescriptor Executed");
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor4dDescriptor"));
    cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();                                                                                          
    int n = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();

    cudnnStatus_t cs = cudnnSetTensor4dDescriptor(tensorDesc,format,dataType,n,c,h,w);
    LOG4CPLUS_DEBUG(logger,"cudnnSetTensor4dDescriptor Executed");
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (const std::exception& e) {
         LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }                      
    registerDescriptorType(tensorDesc, dataType);
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptorEx) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor4dDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();

    int n = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();

    int nStride = in->Get<int>();
    int cStride = in->Get<int>();
    int hStride = in->Get<int>();
    int wStride = in->Get<int>();

    cudnnStatus_t cs = cudnnSetTensor4dDescriptorEx(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride);
    LOG4CPLUS_DEBUG(logger, "cudnnSetTensor4dDescriptor Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (const std::exception& e) {
         LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    registerDescriptorType(tensorDesc, dataType);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetTensor4dDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensor4dDescriptor"));
    cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();

    cudnnDataType_t dataType;
    int n,c,h,w;
    int nStride,cStride,hStride,wStride;

    cudnnStatus_t cs = cudnnGetTensor4dDescriptor(tensorDesc,&dataType,&n,&c,&h,&w,&nStride,&cStride,&hStride,&wStride);
    LOG4CPLUS_DEBUG(logger, "cudnnGetTensor4dDescriptor Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnDataType_t>(dataType);
        out->Add<int>(n);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
        out->Add<int>(nStride);
        out->Add<int>(cStride);
        out->Add<int>(hStride);
        out->Add<int>(wStride);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorNdDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>(nbDims);
    int *strideA = in->Assign<int>(nbDims);

    cudnnStatus_t cs = cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA);
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetTensorNdDescriptor Executed");
    registerDescriptorType(tensorDesc, dataType);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptorEx) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorNdDescriptorEx"));

    cudnnTensorDescriptor_t tensorDesc;
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>();

    cudnnStatus_t cs = cudnnSetTensorNdDescriptorEx(tensorDesc, format, dataType, nbDims, dimA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (const std::exception& e) {
         LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetTensorNdDescriptorEx Executed");
    registerDescriptorType(tensorDesc, dataType);
    return std::make_shared<Result>(cs);  
}

// Method B1: This method is compatible with Method F1 of frontend.
// Use static arrays for all host pointers. Do not pass uninitialized pointers from frontend.
// For dataType and nbDims which are simple types, we pass their values directly,
// So, the frontend reads the variables directly from the output buffer.
// CUDNN_ROUTINE_HANDLER(GetTensorNdDescriptor) {
//     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorNdDescriptor"));

//     const cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
//     int nbDimsRequested = in->Get<int>();
//     cudnnDataType_t dataType;
//     int nbDims;
//     int dimA[nbDimsRequested];
//     int strideA[nbDimsRequested];

//     cudnnStatus_t cs = cudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, &dataType, &nbDims, dimA, strideA);
//     LOG4CPLUS_DEBUG(logger, "cudnnGetTensorNdDescriptor: nbDims = " << nbDims);
//     LOG4CPLUS_DEBUG(logger, "cudnnGetTensorNdDescriptor Executed");

//     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
//     try {
//         out->Add<cudnnDataType_t>(dataType);
//         out->Add<int>(nbDims);
//         out->Add<int>(dimA, nbDimsRequested);
//         out->Add<int>(strideA, nbDimsRequested);
//     } catch (const std::exception& e) {
//         LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
//         return std::make_shared<Result>(cs);
//     }
    
//     return std::make_shared<Result>(cs, out);
// }

// Method B2: This method is compatible with Method F2 of frontend.
// Use dynamic arrays for all host pointers. Do not pass uninitialized pointers from frontend.
// Use the "Delegate" function to allocate memory for pointers.
// However, Delegate allocates memory to the output buffer, so 
// the frontend needs to appropriately read the pointers and dereference them.
CUDNN_ROUTINE_HANDLER(GetTensorNdDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorNdDescriptor"));
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    const cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t* dataType = out->Delegate<cudnnDataType_t>();
    int* nbDims = out->Delegate<int>();
    int* dimA = out->Delegate<int>(nbDimsRequested);
    int* strideA = out->Delegate<int>(nbDimsRequested);

    cudnnStatus_t cs = cudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
    LOG4CPLUS_DEBUG(logger, "cudnnGetTensorNdDescriptor: nbDims = " << nbDims);
    LOG4CPLUS_DEBUG(logger, "cudnnGetTensorNdDescriptor Executed");

    return std::make_shared<Result>(cs, out);
}

// Method B3: This method is compatible with Method F3 of frontend.
// In this method, we allocate memory for the host pointers in the backend using Get from the input buffer.
// This means that the frontend should pass the pointers to the backend, but the pointers do not contain valid data.
// It is just a trick to allocate the amount of memory needed to store values for the arguments that are initially non-valid.
// We then send the values back to the frontend in the output buffer similar to Method B1.
// CUDNN_ROUTINE_HANDLER(GetTensorNdDescriptor) {
//     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorNdDescriptor"));

//     const cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
//     int nbDimsRequested = in->Get<int>();
//     cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
//     int nbDims = in->Get<int>();
//     int* dimA = in->Assign<int>(nbDimsRequested);
//     int* strideA = in->Assign<int>(nbDimsRequested);

//     cudnnStatus_t cs = cudnnGetTensorNdDescriptor(tensorDesc, nbDimsRequested, &dataType, &nbDims, dimA, strideA);
//     LOG4CPLUS_DEBUG(logger, "cudnnGetTensorNdDescriptor: nbDims = " << nbDims);
//     LOG4CPLUS_DEBUG(logger, "cudnnGetTensorNdDescriptor Executed");

//     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
//     try {
//         out->Add<cudnnDataType_t>(dataType);
//         out->Add<int>(nbDims);
//         out->Add<int>(dimA, nbDimsRequested);
//         out->Add<int>(strideA, nbDimsRequested);
//     } catch (const std::exception& e) {
//         LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
//         return std::make_shared<Result>(cs);
//     }
    
//     return std::make_shared<Result>(cs, out);
// }

CUDNN_ROUTINE_HANDLER(GetTensorSizeInBytes) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorSizeInBytes"));

    cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
    size_t size = in->Get<size_t>();

    cudnnStatus_t cs = cudnnGetTensorSizeInBytes(tensorDesc, &size);
  
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<size_t>(size);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }

    LOG4CPLUS_DEBUG(logger, "cudnnGetTensorSizeInBytes Executed");

    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyTensorDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyTensorDescriptor"));

    cudnnTensorDescriptor_t tensorDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnStatus_t cs = cudnnDestroyTensorDescriptor(tensorDesc);
    
    //LOG4CPLUS_DEBUG(logger, "DestroyTensorDescriptor Executed");
    
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(InitTransformDest) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("InitTransformDest"));
    
    cudnnTensorTransformDescriptor_t transformDesc = in->Get<cudnnTensorTransformDescriptor_t>();
    cudnnTensorDescriptor_t srcDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnTensorDescriptor_t destDesc;
    size_t destSizeInBytes;
  
    cudnnStatus_t cs = cudnnInitTransformDest(transformDesc, srcDesc, destDesc, &destSizeInBytes);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnTensorDescriptor_t>(destDesc);
        out->Add<size_t>(destSizeInBytes);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnInitTransformDest Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateTensorTransformDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateTensorTransformDescriptor"));

    cudnnTensorTransformDescriptor_t transformDesc;
    
    cudnnStatus_t cs = cudnnCreateTensorTransformDescriptor(&transformDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnTensorTransformDescriptor_t>(transformDesc);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, " cudnnCreateTensorTransformDescriptor Execute");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetTensorTransformDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorTransformDescriptor"));
    
    cudnnTensorTransformDescriptor_t transformDesc;
    uint32_t nbDims = in->Get<uint32_t>();
    cudnnTensorFormat_t destFormat = in->Get<cudnnTensorFormat_t>();
    int32_t *padBeforeA = in->Assign<int32_t>();
    int32_t *padAfterA = in->Assign<int32_t>();
    uint32_t *foldA = in->Assign<uint32_t>();
    cudnnFoldingDirection_t direction = in->Get<cudnnFoldingDirection_t>();

    cudnnStatus_t cs = cudnnSetTensorTransformDescriptor(transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction);
     
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnTensorTransformDescriptor_t>(transformDesc);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger,  "cudnnSetTensorTransformDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetTensorTransformDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorTransformDescriptor"));

    cudnnTensorTransformDescriptor_t transformDesc = in->Get<cudnnTensorTransformDescriptor_t>();
    uint32_t nbDimsRequested = in->Get<uint32_t>();
    cudnnTensorFormat_t destFormat;
    int32_t padBeforeA;
    int32_t padAfterA;
    uint32_t foldA;
    cudnnFoldingDirection_t direction;
   
    cudnnStatus_t cs = cudnnGetTensorTransformDescriptor(transformDesc, nbDimsRequested, &destFormat, &padBeforeA, &padAfterA, &foldA, &direction);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnTensorFormat_t>(destFormat);
        out->Add<int32_t>(padBeforeA);
        out->Add<int32_t>(padAfterA);
        out->Add<uint32_t>(foldA);
        out->Add<cudnnFoldingDirection_t>(direction);   
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetTensorTransformDescriptor Executed"); 
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyTensorTransformDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyTensorTransformDescriptor"));
    
    cudnnTensorTransformDescriptor_t transformDesc = in->Get<cudnnTensorTransformDescriptor_t>();
    
    cudnnStatus_t cs = cudnnDestroyTensorTransformDescriptor(transformDesc);
    
    //LOG4CPLUS_DEBUG(logger, " cudnnDestroyTensorTransformDescriptor Execute");
    
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(TransformTensor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("TransformTensor"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void* alpha = isFloatDescriptor(xDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    const void* x = in->GetFromMarshal<void*>();
    cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    const void* beta = isFloatDescriptor(yDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    void* y = in->GetFromMarshal<void*>();

    cudnnStatus_t cs = cudnnTransformTensor(handle, alpha, xDesc, x, beta, yDesc, y);
    LOG4CPLUS_DEBUG(logger, "cudnnTransformTensor Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(y);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return make_shared<Result>(cs);
    }
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(TransformTensorEx) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("TransformTensorEx"));
   
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnTensorTransformDescriptor_t transDesc = in->Get<cudnnTensorTransformDescriptor_t>();
    const cudnnTensorDescriptor_t srcDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alpha = isFloatDescriptor(srcDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    const void *srcData = in->GetFromMarshal<void*>();
    const cudnnTensorDescriptor_t destDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *beta = isFloatDescriptor(destDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    void *destData = in->GetFromMarshal<void*>();

    cudnnStatus_t cs = cudnnTransformTensorEx(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
    LOG4CPLUS_DEBUG(logger, "cuddTransformTensorEx Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
            out->AddMarshal<void*>(destData);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }

    return std::make_shared<Result>(cs, out);
}

// NON SONO SICURO DI QUESTA FUNZIONE DA FAR VEDERE A MONTELLA!!! 
CUDNN_ROUTINE_HANDLER(GetFoldedConvBackwardDataDescriptors) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFoldedConvBackwardDataDescriptors"));
   
   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   cudnnFilterDescriptor_t filterDesc;
   cudnnTensorDescriptor_t diffDesc;
   cudnnConvolutionDescriptor_t convDesc;
   cudnnTensorDescriptor_t gradDesc;
   cudnnTensorFormat_t transformFormat;
   cudnnFilterDescriptor_t foldedFilterDesc;
   cudnnTensorDescriptor_t paddedDiffDesc;
   cudnnConvolutionDescriptor_t foldedConvDesc = in->Get<cudnnConvolutionDescriptor_t>();
   cudnnTensorDescriptor_t foldedGradDesc;
   cudnnTensorTransformDescriptor_t filterFoldTransDesc;
   cudnnTensorTransformDescriptor_t diffPadTransDesc;
   cudnnTensorTransformDescriptor_t gradFoldTransDesc;
   cudnnTensorTransformDescriptor_t gradUnfoldTransDesc;


   cudnnStatus_t cs = cudnnGetFoldedConvBackwardDataDescriptors(handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat, foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc, gradUnfoldTransDesc);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnFilterDescriptor_t>(filterDesc);
        out->Add<cudnnTensorDescriptor_t>(diffDesc);
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
        out->Add<cudnnTensorDescriptor_t>(gradDesc);
        out->Add<cudnnTensorFormat_t>(transformFormat);
        out->Add<cudnnFilterDescriptor_t>(foldedFilterDesc);
        out->Add<cudnnTensorDescriptor_t>(paddedDiffDesc);
        out->Add<cudnnConvolutionDescriptor_t>(foldedConvDesc);
        out->Add<cudnnTensorDescriptor_t>(foldedGradDesc);
        out->Add<cudnnTensorTransformDescriptor_t>(filterFoldTransDesc);
        out->Add<cudnnTensorTransformDescriptor_t>(diffPadTransDesc);
        out->Add<cudnnTensorTransformDescriptor_t>(gradFoldTransDesc);
        out->Add<cudnnTensorTransformDescriptor_t>(gradUnfoldTransDesc);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFoldedConvBackwardDataDescriptors Executed");
    
    return std::make_shared<Result>(cs, out);
    
}

CUDNN_ROUTINE_HANDLER(AddTensor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("AddTensor"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnTensorDescriptor_t aDesc = in->Get<cudnnTensorDescriptor_t>();
    const void* alpha = isFloatDescriptor(aDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    const void* A = in->GetFromMarshal<void*>();
    const cudnnTensorDescriptor_t cDesc = in->Get<cudnnTensorDescriptor_t>();
    const void* beta = isFloatDescriptor(cDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    void* C = in->GetFromMarshal<void*>();

    cudnnStatus_t cs = cudnnAddTensor(handle, alpha, aDesc,A, beta, cDesc, C);
    LOG4CPLUS_DEBUG(logger, "cudnnAddTensor Executed");
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(C);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }

    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateOpTensorDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateOpTensorDescriptor"));
   
    cudnnOpTensorDescriptor_t opTensorDesc;
    
    cudnnStatus_t cs = cudnnCreateOpTensorDescriptor(&opTensorDesc);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnOpTensorDescriptor_t>(opTensorDesc);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateOpTensorDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetOpTensorDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetOpTensorDescriptor"));

    cudnnOpTensorDescriptor_t opTensorDesc;
    cudnnOpTensorOp_t opTensorOp = in->Get<cudnnOpTensorOp_t>();
    cudnnDataType_t opTensorCompType = in->Get<cudnnDataType_t>();
    cudnnNanPropagation_t opTensorNanOpt = in->Get<cudnnNanPropagation_t>();

   cudnnStatus_t cs = cudnnSetOpTensorDescriptor(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->Add<cudnnOpTensorDescriptor_t>(opTensorDesc);
   } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
   

   LOG4CPLUS_DEBUG(logger, "cudnnSetOpTensorDescriptor Executed");
   
   return std::make_shared<Result>(cs, out); 
}

CUDNN_ROUTINE_HANDLER(GetOpTensorDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetOpTensorDescriptor"));
   
    cudnnOpTensorDescriptor_t opTensorDesc = in->Get<cudnnOpTensorDescriptor_t>();
    cudnnOpTensorOp_t opTensorOp;
    cudnnDataType_t opTensorCompType;
    cudnnNanPropagation_t opTensorNanOpt;

    cudnnStatus_t cs = cudnnGetOpTensorDescriptor(opTensorDesc, &opTensorOp, &opTensorCompType, &opTensorNanOpt);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnOpTensorOp_t>(opTensorOp);
        out->Add<cudnnDataType_t>(opTensorCompType);
        out->Add<cudnnNanPropagation_t>(opTensorNanOpt);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
   
   LOG4CPLUS_DEBUG(logger, "cudnnGetOpTensorDescriptor");
   
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyOpTensorDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyOpTensorDescriptor"));

   cudnnOpTensorDescriptor_t opTensorDesc = in->Get<cudnnOpTensorDescriptor_t>();
   
   cudnnStatus_t cs = cudnnDestroyOpTensorDescriptor(opTensorDesc);
   
   LOG4CPLUS_DEBUG(logger, "cudnnDestroyOpTensorDescriptor Executed");
   
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(OpTensor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("OpTensor"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnOpTensorDescriptor_t opTensorDesc = in->Get<cudnnOpTensorDescriptor_t>();
    cudnnTensorDescriptor_t aDesc = in->Get<cudnnTensorDescriptor_t>();
    const void * alpha1 = isFloatDescriptor(aDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    void * A = in->GetFromMarshal<void*>();
    cudnnTensorDescriptor_t bDesc = in->Get<cudnnTensorDescriptor_t>();
    void * alpha2 = isFloatDescriptor(bDesc)
        ? static_cast<void*>(in->Assign<void>())
        : static_cast<void*>(in->Assign<void>());
    void * B = in->GetFromMarshal<void*>();
    cudnnTensorDescriptor_t cDesc = in->Get<cudnnTensorDescriptor_t>();
    void * beta = isFloatDescriptor(cDesc)
        ? static_cast<void*>(in->Assign<void>())
        : static_cast<void*>(in->Assign<void>());
    void * C = in->GetFromMarshal<void*>();

    cudnnStatus_t cs = cudnnOpTensor(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
    LOG4CPLUS_DEBUG(logger, "cudnnOpTensor Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(C);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }

    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateReduceTensorDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateReduceTensorDescriptor"));

    cudnnReduceTensorDescriptor_t reduceTensorDesc;
   
    cudnnStatus_t cs = cudnnCreateReduceTensorDescriptor(& reduceTensorDesc);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnReduceTensorDescriptor_t>(reduceTensorDesc);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateReduceTensorDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}
 
CUDNN_ROUTINE_HANDLER(SetReduceTensorDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetReduceTensorDescriptor"));

    cudnnReduceTensorDescriptor_t reduceTensorDesc = in->Get<cudnnReduceTensorDescriptor_t>();
    cudnnReduceTensorOp_t reduceTensorOp = in->Get<cudnnReduceTensorOp_t>();
    cudnnDataType_t reduceTensorCompType = in->Get<cudnnDataType_t>();
    cudnnNanPropagation_t reduceTensorNanOpt = in->Get<cudnnNanPropagation_t>();
    cudnnReduceTensorIndices_t reduceTensorIndices = in->Get<cudnnReduceTensorIndices_t>();
    cudnnIndicesType_t reduceTensorIndicesType = in->Get<cudnnIndicesType_t>();

    cudnnStatus_t cs = cudnnSetReduceTensorDescriptor(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnReduceTensorDescriptor_t>(reduceTensorDesc);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetReduceTensorDescriptor");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetReduceTensorDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetReduceTensorDescriptor"));
 
    cudnnReduceTensorDescriptor_t reduceTensorDesc = in->Get<cudnnReduceTensorDescriptor_t>(); //INPUT
    cudnnReduceTensorOp_t reduceTensorOp; //OUTPUT
    cudnnDataType_t reduceTensorCompType; //OUTPUT
    cudnnNanPropagation_t reduceTensorNanOpt = in->Get<cudnnNanPropagation_t>(); //INPUT
    cudnnReduceTensorIndices_t reduceTensorIndices; //OUTPUT
    cudnnIndicesType_t reduceTensorIndicesType; //OUTPUT
  
    cudnnStatus_t cs = cudnnGetReduceTensorDescriptor(reduceTensorDesc, &reduceTensorOp, &reduceTensorCompType, &reduceTensorNanOpt, &reduceTensorIndices, &reduceTensorIndicesType);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnReduceTensorOp_t>(reduceTensorOp);
        out->Add<cudnnDataType_t>(reduceTensorCompType);
        out->Add<cudnnReduceTensorIndices_t>(reduceTensorIndices);
        out->Add<cudnnIndicesType_t>(reduceTensorIndicesType);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
 
    LOG4CPLUS_DEBUG(logger, "cudnnGetReduceTensorDescriptor Executed");   
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyReduceTensorDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyReduceTensorDescriptor"));

    cudnnReduceTensorDescriptor_t reduceTensorDesc = in->Get<cudnnReduceTensorDescriptor_t>();
    cudnnStatus_t cs = cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
    
    LOG4CPLUS_DEBUG(logger, "cudnnDestroyReduceTensorDescriptor Executed");
    
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetReductionIndicesSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetReductionIndicesSize"));
    
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnReduceTensorDescriptor_t reduceTensorDesc = in->Get<cudnnReduceTensorDescriptor_t>();
    cudnnTensorDescriptor_t aDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnTensorDescriptor_t cDesc = in->Get<cudnnTensorDescriptor_t>();
    size_t *sizeInBytes;
   
    cudnnStatus_t cs = cudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
     
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<size_t>(sizeInBytes);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
   }
   
    LOG4CPLUS_DEBUG(logger, "cuddGetReductionIndicesSize Executed");
   
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetReductionWorkspaceSize) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetReductionWorkspaceSize"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   cudnnReduceTensorDescriptor_t reduceTensorDesc = in->Get<cudnnReduceTensorDescriptor_t>();
   cudnnTensorDescriptor_t aDesc = in->Get<cudnnTensorDescriptor_t>();
   cudnnTensorDescriptor_t cDesc = in->Get<cudnnTensorDescriptor_t>();
   size_t *sizeInBytes;

   cudnnStatus_t cs = cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
    
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->Add<size_t>(sizeInBytes);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
    LOG4CPLUS_DEBUG(logger, "cudnnGetReductionWorkspaceSize Executed");
   
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ReduceTensor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ReduceTensor"));
 
   cudnnHandle_t handle = in->Get<cudnnHandle_t>(); //INPUT
   const cudnnReduceTensorDescriptor_t reduceTensorDesc = in->Get<cudnnReduceTensorDescriptor_t>(); //INPUT
   void *indices = in->GetFromMarshal<void*>();  //OUTPUT
   size_t indicesSizeInBytes = in->Get<size_t>(); //INPUT
   void *workspace = in->GetFromMarshal<void*>(); //INPUT
   size_t workspaceSizeInBytes = in->Get<size_t>(); //INPUT
   const cudnnTensorDescriptor_t aDesc = in->Get<cudnnTensorDescriptor_t>(); //INPUT
   const void *alpha = isFloatDescriptor(aDesc)
       ? static_cast<const void*>(in->Assign<const float>())
       : static_cast<const void*>(in->Assign<const double>()); //INPUT
   const void *A = in->GetFromMarshal<void*>(); //INPUT
   const cudnnTensorDescriptor_t cDesc = in->Get<cudnnTensorDescriptor_t>(); //INPUT
   const void *beta = isFloatDescriptor(cDesc)
       ? static_cast<const void*>(in->Assign<const float>())
       : static_cast<const void*>(in->Assign<const double>()); //INPUT
   void *C = in->GetFromMarshal<void*>(); //INPUT/OUTPUT

   cudnnStatus_t cs = cudnnReduceTensor(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha,  aDesc, A, beta, cDesc, C);
   LOG4CPLUS_DEBUG(logger, "cudnnReduceTensor Execute");
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->AddMarshal<void*>(indices);
       out->AddMarshal<void*>(C);
   } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
      return std::make_shared<Result>(cs);
   }
   
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetTensor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    void * y = in->Assign<void>();
    void * valuePtr = in->Assign<void>();

    cudnnStatus_t cs = cudnnSetTensor(handle,yDesc,y,valuePtr);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<void>(y);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
   }
    
     LOG4CPLUS_DEBUG(logger, "cudnnSetTensor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ScaleTensor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ScaleTensor"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    void* y = in->GetFromMarshal<void*>();
    const void* alpha = isFloatDescriptor(yDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());

    cudnnStatus_t cs = cudnnScaleTensor(handle, yDesc, y, alpha);
    LOG4CPLUS_DEBUG(logger, "cudnnScaleTensor Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(y);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
   }

    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateFilterDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateFilterDescriptor"));
    
    cudnnFilterDescriptor_t filterDesc;

    cudnnStatus_t cs = cudnnCreateFilterDescriptor(&filterDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnFilterDescriptor_t>(filterDesc);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger,"cudnnCreateFilterDescriptor Executed");
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor"));
   
    cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    int k = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();

    cudnnStatus_t cs = cudnnSetFilter4dDescriptor(filterDesc, dataType, format, k, c, h, w);
    LOG4CPLUS_DEBUG(logger,"cudnnSetFilter4dDescriptor Executed");

    registerDescriptorType(filterDesc, dataType);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilter4dDescriptor"));

   cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
   cudnnDataType_t dataType;
   cudnnTensorFormat_t format;
   int k;
   int c;
   int h;
   int w;

   cudnnStatus_t cs = cudnnGetFilter4dDescriptor(filterDesc, &dataType, &format, &k, &c, &h, &w);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->Add<cudnnDataType_t>(dataType);
       out->Add<cudnnTensorFormat_t>(format);
       out->Add<int>(k);
       out->Add<int>(c);
       out->Add<int>(h);
       out->Add<int>(w);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnGetFilter4dDescriptor Executed");
   return std::make_shared<Result>(cs, out);
}

#if CUDNN_VERSION < 6000
    CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v3) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor_v3"));

        cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
        cudnnDataType_t dataType = in->Get<cudnnDataType_t>();

        int k = in->Get<int>();
        int h = in->Get<int>();
        int w = in->Get<int>();

        cudnnStatus_t cs = cudnnSetFilter4dDescriptor_v3(filterDesc, dataType, k, c, h, w);
    
        LOG4CPLUS_DEBUG(logger, "cudnnSetFilter4dDescriptor_v3 Executed");
        registerDescriptorType(filterDesc, dataType);
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v3) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilter4dDescriptor_v3"));

        cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
        cudnnDataType_t dataType;

        int k,c,h,w;

        cudnnStatus_t cs = cudnnGetFilter4dDescriptor_v3(filterDesc,&dataType,&k,&c,&h,&w);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        try {
            out->Add<cudnnDataType_t>(dataType);
            out->Add<int>(k);
            out->Add<int>(c);
            out->Add<int>(h);
            out->Add<int>(w);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger,e);
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetFilter4dDescriptor_v3 Executed");
        
        return std::make_shared<Result>(cs,out);
    }

    CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v4) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor_v4"));

        cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
        cudnnDataType_t dataType =  in->Get<cudnnDataType_t>();
        cudnnTensorFormat_t  format = in->Get<cudnnTensorFormat_tlong>();

        int k = in->Get<int>();
        int c = in->Get<int>();
        int h = in->Get<int>();
        int w = in->Get<int>();

        cudnnStatus_t cs = cudnnSetFilter4dDescriptor_v4(filterDesc, dataType, format, k, c, h, w);
        
        LOG4CPLUS_DEBUG(logger, "cudnnSetFilter4dDescriptor_v4 Executed");
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v4) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilter4dDescriptor_v4"));

        cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
        cudnnDataType_t dataType;
        cudnnTensorFormat_t  format;

        int k,c,h,w;

        cudnnStatus_t cs = cudnnGetFilter4dDescriptor_v4(filterDesc,&dataType,&format,&k,&c,&h,&w);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        try {
            out->Add<cudnnDataType_t>(dataType);
            out->Add<cudnnTensorFormat_t>(format);
            out->Add<int>(k);
            out->Add<int>(c);
            out->Add<int>(h);
            out->Add<int>(w);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger,e);
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetFilter4dDescriptor_v4 Executed");
        
        return std::make_shared<Result>(cs,out);
    }
#endif

CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor"));
    
    cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    int nbDims = in->Get<int>();
    const int *filterDimA = in->Assign<const int>();
    
    cudnnStatus_t cs = cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA);
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetFilterNdDescriptor Executed");
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor"));

    cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();

    cudnnTensorFormat_t  format;

    cudnnStatus_t cs = cudnnGetFilterNdDescriptor(wDesc,nbDimsRequested,dataType,&format,nbDims,filterDimA);

    std:shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<cudnnTensorFormat_t>(format);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFilterNdDescriptor Executed");
    
    return std::make_shared<Result>(cs,out);
}

#if CUDNN_VERSION < 6000
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v3) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor_v3"));

    cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();

    int nbDims = in->Get<int>();
    const int * filterDimA = in->Assign<const int>();

    cudnnStatus_t cs = cudnnSetFilterNdDescriptor_v3(filterDesc, dataType, nbDims, filterDimA);
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetFilterNdDescriptor_v3 Executed");
    registerDescriptorType(filterDesc, dataType);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v3) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor"));

    cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();


    cudnnStatus_t cs = cudnnGetFilterNdDescriptor_v3(wDesc,nbDimsRequested,dataType,nbDims,filterDimA);
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFilterNdDescriptor Executed");
    
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v4) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor_v4"));

    cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
    cudnnDataType_t dataType =  in->Get<cudnnDataType_t>();
    cudnnTensorFormat_t  format = in->Get<cudnnTensorFormat_t>();

    int nbDims = in->Get<int>();
    const int* filterDimA = in->Assign<const int>();

    cudnnStatus_t cs = cudnnSetFilterNdDescriptor_v4(filterDesc, dataType, format, nbDims, filterDimA);
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetFilterDescriptor_v4 Executed");
    registerDescriptorType(filterDesc, dataType);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v4) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor_v4"));

    cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();

    cudnnTensorFormat_t format;

    cudnnStatus_t cs = cudnnGetFilterNdDescriptor_v4(wDesc,nbDimsRequested,dataType,&format,nbDims,filterDimA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<cudnnTensorFormat_t>(format);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFilterDescriptor_v4 Executed");
    
    return std::make_shared<Result>(cs,out);
}
#endif

CUDNN_ROUTINE_HANDLER(GetFilterSizeInBytes) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterSizeInBytes"));
    
    cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();

    size_t size = in->Get<size_t>();
    
    cudnnStatus_t cs = cudnnGetFilterSizeInBytes(filterDesc, &size);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<size_t>(size);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger,  "cudnnGetFilterSizeInBytes Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyFilterDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestoryFilterDescriptor"));

    cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();

    cudnnStatus_t cs = cudnnDestroyFilterDescriptor(filterDesc);
    
    //LOG4CPLUS_DEBUG(logger,  "cudnnDestroyFilterDescriptor Executed");
    
    return make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(TransformFilter) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("TransformFilter"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>(); //INPUT
   const cudnnTensorTransformDescriptor_t transDesc = in->Get<cudnnTensorTransformDescriptor_t>(); //INPUT
   const cudnnFilterDescriptor_t srcDesc = in->Get<cudnnFilterDescriptor_t>(); //INPUT
   const void* alpha = isFloatDescriptor(srcDesc)
       ? static_cast<const void*>(in->Assign<const float>())
       : static_cast<const void*>(in->Assign<const double>()); //INPUT
   const void* srcData = in->GetFromMarshal<void*>(); //INPUT
   const cudnnFilterDescriptor_t destDesc = in->Get<cudnnFilterDescriptor_t>(); //INPUT
   const void* beta = isFloatDescriptor(destDesc)
       ? static_cast<const void*>(in->Assign<const float>())
       : static_cast<const void*>(in->Assign<const double>()); //INPUT
   void* destData = in->GetFromMarshal<void*>(); //INPUT/OUTPUT
   
   cudnnStatus_t cs = cudnnTransformFilter(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
   LOG4CPLUS_DEBUG(logger, "cudnnTransformFilter Executed");
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->AddMarshal<void*>(destData);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
   
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ReorderFilterAndBias) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ReorderFilterAndBias"));
   
   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   const cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
   const cudnnReorderType_t reorderType = in->Get<cudnnReorderType_t>();
   const void *filterData = in->GetFromMarshal<void*>();
   void *reorderedFilterData = in->GetFromMarshal<void*>();
   const int reorderBias = in->Get<int>();
   const void *biasData =  in->GetFromMarshal<void *>();
   void *reorderedBiasData = in->GetFromMarshal<void*>();
  
   cudnnStatus_t cs = cudnnReorderFilterAndBias(handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData, reorderedBiasData);
   LOG4CPLUS_DEBUG(logger, "cudnnReorderFilterAndBias Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->AddMarshal<void*>(reorderedFilterData);
         out->AddMarshal<void*>(reorderedBiasData);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
  return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataAlgorithmMaxCount) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetonvolutionBackwardData"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    int count;

    cudnnStatus_t cs = cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &count);
    LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionBackwardDataAlgorithmMaxCount Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(count);
    }  catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }

    return std::make_shared<Result>(cs, out);
}  

CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardDataAlgorithm) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionBackwardDataAlgorithm"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
    const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
    const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
    const int requestedAlgoCount = in->Get<int>();
    int returnedAlgoCount;
    cudnnConvolutionBwdDataAlgoPerf_t* perfResults = in->Get<cudnnConvolutionBwdDataAlgoPerf_t>(requestedAlgoCount);

    cudnnStatus_t cs = cudnnFindConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, &returnedAlgoCount, perfResults);
    LOG4CPLUS_DEBUG(logger, "cudnnFindConvolutionBackwardDataAlgorithm Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionBwdDataAlgoPerf_t>(perfResults, returnedAlgoCount);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
 
  return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(FindConvolutionBackwardDataAlgorithmEx) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindConvolutionBackwardDataAlgorithmEx"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>(); //INPUT
    const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>(); //INPUT
    const void *w = in->GetFromMarshal<void*>(); //INPUT
    const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>(); //INPUT
    const void *dy = in->GetFromMarshal<void*>(); //INPUT
    const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>(); //INPUT
    const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>(); //INPUT
    void *dx = in->GetFromMarshal<void*>(); //INPUT/OUTPUT
    const int requestedAlgoCount = in->Get<int>(); //INPUT
    int returnedAlgoCount; //OUTPUT
    cudnnConvolutionBwdDataAlgoPerf_t* perfResults = in->Get<cudnnConvolutionBwdDataAlgoPerf_t>(requestedAlgoCount); //OUTPUT
    void *workSpace = in->GetFromMarshal<void*>(); //INPUT
    size_t workSpaceSizeInBytes = in->Get<size_t>(); //INPUT

    cudnnStatus_t cs = cudnnFindConvolutionBackwardDataAlgorithmEx(handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, &returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
    LOG4CPLUS_DEBUG(logger, "cudnnFindConvolutionBackwardDataAlgorithmEx Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(dx);
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionBwdDataAlgoPerf_t>(perfResults, returnedAlgoCount);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }

    return std::make_shared<Result>(cs, out);  
}

#if CUDNN_VERSION < 8000
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataAlgorithm) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardDataAlgorithm"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
   const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
   const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
   const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
   cudnnConvolutionBwdDataPreference_t preference = in->Get<cudnnConvolutionBwdDataPreference_t>();
   size_t memoryLimitInBytes = in->Get<size_t>();
   cudnnConvolutionBwdDataAlgo_t algo;

   cudnnStatus_t cs = cudnnGetConvolutionBackwardDataAlgorithm(handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, &algo);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->Add<cudnnConvolutionBwdDataAlgo_t>(algo);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionBackwardDataAlgorithm Executed");
   return std::make_shared<Result>(cs, out);  
}
#endif

#if CUDNN_VERSION >= 7000
CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataAlgorithm_v7) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardDataAlgorithm_v7"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnFilterDescriptor_t filterDesc = in->Get<cudnnFilterDescriptor_t>();
    cudnnTensorDescriptor_t diffDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
    cudnnTensorDescriptor_t gradDesc = in->Get<cudnnTensorDescriptor_t>();
    int requestedAlgoCount = in->Get<int>();
    int returnedAlgoCount;
    cudnnConvolutionBwdDataAlgoPerf_t* perfResults = in->Get<cudnnConvolutionBwdDataAlgoPerf_t>(requestedAlgoCount);

    cudnnStatus_t cs = cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, &returnedAlgoCount, perfResults);
    LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionBackwardDataAlgorithm_v7 Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(returnedAlgoCount);
        out->Add<cudnnConvolutionBwdDataAlgoPerf_t>(perfResults, returnedAlgoCount);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }

    return std::make_shared<Result>(cs, out); 
}
#endif

CUDNN_ROUTINE_HANDLER(GetConvolutionBackwardDataWorkspaceSize) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolutionBackwardDataWorkspaceSize"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
   const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
   const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
   const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
   cudnnConvolutionBwdDataAlgo_t algo = in->Get<cudnnConvolutionBwdDataAlgo_t>();
   size_t sizeInBytes;

   cudnnStatus_t cs = cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, dyDesc, convDesc, dxDesc, algo, &sizeInBytes);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->Add<size_t>(sizeInBytes);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnGetConvolutionBackwardDataWorkspaceSize Executed");
   return std::make_shared<Result>(cs, out);             
}

CUDNN_ROUTINE_HANDLER(ConvolutionBackwardData) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ConvolutionBackwardData"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
   const void *alpha = isFloatDescriptor(wDesc)
       ? static_cast<const void *>(in->Assign<const float>())
       : static_cast<const void *>(in->Assign<const double>());
   const void *w = in->GetFromMarshal<void *>();
   const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *dy = in->GetFromMarshal<void *>();
   const cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
   cudnnConvolutionBwdDataAlgo_t algo = in->Get<cudnnConvolutionBwdDataAlgo_t>();
   void *workSpace = in->GetFromMarshal<void *>();
   size_t workSpaceSizeInBytes = in->Get<size_t>();
   const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *beta = isFloatDescriptor(dxDesc)
       ? static_cast<const void *>(in->Assign<const float>())
       : static_cast<const void *>(in->Assign<const double>());
   void *dx = in->GetFromMarshal<void *>();

   cudnnStatus_t cs = cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
   LOG4CPLUS_DEBUG(logger, "cudnnConvolutionBackwardData Executed");

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->AddMarshal<void *>(dx);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
      return std::make_shared<Result>(cs);
   }
   return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(Im2Col) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Im2Col"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
   void *x = in->Assign<void>();
   cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
   cudnnConvolutionDescriptor_t convDesc = in->Get<cudnnConvolutionDescriptor_t>();
   void *colBuffer;

   cudnnStatus_t cs = cudnnIm2Col(handle, xDesc, x, wDesc, convDesc, colBuffer);
   LOG4CPLUS_DEBUG(logger, "cudnnIm2Col Executed");

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
        out->Add<void>(colBuffer);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
   }
   
   return std::make_shared<Result>(cs, out);     
}

CUDNN_ROUTINE_HANDLER(SoftmaxForward) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SoftmaxForward"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   cudnnSoftmaxAlgorithm_t algo = in->Get<cudnnSoftmaxAlgorithm_t>();
   cudnnSoftmaxMode_t mode = in->Get<cudnnSoftmaxMode_t>();
   const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *alpha = isFloatDescriptor(xDesc)
       ? static_cast<const void *>(in->Assign<const float>())
       : static_cast<const void *>(in->Assign<const double>());
   const void *x = in->GetFromMarshal<void *>();
   const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *beta = isFloatDescriptor(yDesc)
       ? static_cast<const void *>(in->Assign<const float>())
       : static_cast<const void *>(in->Assign<const double>());
   void *y = in->GetFromMarshal<void *>();

   cudnnStatus_t cs = cudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
   LOG4CPLUS_DEBUG(logger, "cudnnSoftmaxForward Executed");

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
        out->AddMarshal<void *>(y);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
   }
   
   return std::make_shared<Result>(cs, out);     
}

CUDNN_ROUTINE_HANDLER(SoftmaxBackward) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SoftmaxBackward"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   cudnnSoftmaxAlgorithm_t algo = in->Get<cudnnSoftmaxAlgorithm_t>();
   cudnnSoftmaxMode_t mode = in->Get<cudnnSoftmaxMode_t>();
   const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *alpha = isFloatDescriptor(yDesc)
       ? static_cast<const void *>(in->Assign<const float>())
       : static_cast<const void *>(in->Assign<const double>());
   const void *y = in->GetFromMarshal<void *>();
   const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
   void *dy = in->Assign<void>();
   const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *beta = isFloatDescriptor(dxDesc)
       ? static_cast<const void *>(in->Assign<const float>())
       : static_cast<const void *>(in->Assign<const double>());
   void *dx = in->Assign<void>();

   cudnnStatus_t cs = cudnnSoftmaxBackward(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
   LOG4CPLUS_DEBUG(logger, "cudnnSoftmaxBackward Executed");
 
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->AddMarshal<void *>(dx);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
   
   return std::make_shared<Result>(cs, out);     
}

CUDNN_ROUTINE_HANDLER(CreatePoolingDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreatePoolingDescriptor"));

   cudnnPoolingDescriptor_t poolingDesc;

   cudnnStatus_t cs = cudnnCreatePoolingDescriptor(&poolingDesc);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
        out->Add<cudnnPoolingDescriptor_t>(poolingDesc);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnCreatePoolingDescriptor Executed");
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetPooling2dDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetPooling2dDescriptor"));

    cudnnPoolingDescriptor_t poolingDesc = in->Get<cudnnPoolingDescriptor_t>();
    cudnnPoolingMode_t mode = in->Get<cudnnPoolingMode_t>();
    cudnnNanPropagation_t maxpoolingNanOpt = in->Get<cudnnNanPropagation_t>();
    int windowHeight = in->Get<int>();
    int windowWidth = in->Get<int>();
    int verticalPadding = in->Get<int>();
    int horizontalPadding = in->Get<int>();
    int verticalStride = in->Get<int>();
    int horizontalStride = in->Get<int>();

    cudnnStatus_t cs = cudnnSetPooling2dDescriptor(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride);
    LOG4CPLUS_DEBUG(logger, "cudnnSetPooling2dDescriptor Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnPoolingDescriptor_t>(poolingDesc);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
   return std::make_shared<Result>(cs, out); 
}

CUDNN_ROUTINE_HANDLER(GetPooling2dDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetPooling2dDescriptor"));

   cudnnPoolingDescriptor_t poolingDesc = in->Get<cudnnPoolingDescriptor_t>();
   cudnnPoolingMode_t mode;
   cudnnNanPropagation_t maxpoolingNanOpt;
   int windowHeight;
   int windowWidth;
   int verticalPadding;
   int horizontalPadding;
   int verticalStride;
   int horizontalStride;

   cudnnStatus_t cs = cudnnGetPooling2dDescriptor(poolingDesc, &mode, &maxpoolingNanOpt, &windowHeight, &windowWidth, &verticalPadding, &horizontalPadding, &verticalStride, &horizontalStride);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->Add<cudnnPoolingMode_t>(mode);
       out->Add<cudnnNanPropagation_t>(maxpoolingNanOpt);
       out->Add<int>(windowHeight);
       out->Add<int>(windowWidth);
       out->Add<int>(verticalPadding);
       out->Add<int>(horizontalPadding);
       out->Add<int>(verticalStride);
       out->Add<int>(horizontalStride);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
   
    LOG4CPLUS_DEBUG(logger, "cudnnGetPooling2dDescriptor Executed");
   
   return std::make_shared<Result>(cs, out);     
}

CUDNN_ROUTINE_HANDLER(SetPoolingNdDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetPoolingNdDescriptor"));
   
   cudnnPoolingDescriptor_t poolingDesc = in->Get<cudnnPoolingDescriptor_t>();
   cudnnPoolingMode_t mode = in->Get<cudnnPoolingMode_t>();
   cudnnNanPropagation_t maxpoolingNanOpt = in->Get<cudnnNanPropagation_t>();
   int nbDims = in->Get<int>();
   int *windowDimA = in->Assign<int>();
   int *paddingA = in->Assign<int>();
   int *strideA = in->Assign<int>();

   cudnnStatus_t cs = cudnnSetPoolingNdDescriptor(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);

   
   LOG4CPLUS_DEBUG(logger, "cudnnSetPoolingNdDescriptor Executed");
   
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetPoolingNdDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetPoolingNdDescriptor"));

    cudnnPoolingDescriptor_t poolingDesc = in->Get<cudnnPoolingDescriptor_t>();
    int nbDimsRequested = in->Get<int>();
    cudnnPoolingMode_t mode;
    cudnnNanPropagation_t maxpoolingNanOpt = in->Get<cudnnNanPropagation_t>();
    int nbDims;
    int *windowDimA = in->Assign<int>();
    int *paddingA = in->Assign<int>();
    int *strideA = in->Assign<int>();

    cudnnStatus_t cs = cudnnGetPoolingNdDescriptor(poolingDesc, nbDimsRequested, &mode, &maxpoolingNanOpt, &nbDims, windowDimA, paddingA, strideA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnPoolingMode_t>(mode);
        out->Add<int>(nbDims);
        out->Add<int>(windowDimA);
        out->Add<int>(paddingA);
        out->Add<int>(strideA);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetPoolingNdDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(GetPoolingNdForwardOutputDim) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetPoolingNdForwardOutputDim"));
    
    cudnnPoolingDescriptor_t poolingDesc = in->Get<cudnnPoolingDescriptor_t>();
    cudnnTensorDescriptor_t inputTensorDesc = in->Get<cudnnTensorDescriptor_t>();
    int nbDims = in->Get<int>();
    int *outputTensorDimA = in->Assign<int>();

    cudnnStatus_t cs = cudnnGetPoolingNdForwardOutputDim(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<int>(outputTensorDimA);
    }catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetPoolingNdForwardOutputDim Executed");
    
    return std::make_shared<Result>(cs, out);           
}

CUDNN_ROUTINE_HANDLER(GetPooling2dForwardOutputDim) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetPooling2dForwardOutputDim"));

   cudnnPoolingDescriptor_t poolingDesc = in->Get<cudnnPoolingDescriptor_t>();
   cudnnTensorDescriptor_t inputTensorDesc = in->Get<cudnnTensorDescriptor_t>();
   int n;
   int c;
   int h;
   int w;
 
   cudnnStatus_t cs = cudnnGetPooling2dForwardOutputDim(poolingDesc, inputTensorDesc, &n, &c, &h, &w);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->Add<int>(n);
       out->Add<int>(c);
       out->Add<int>(h);
       out->Add<int>(w);
   } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
   
   LOG4CPLUS_DEBUG(logger, "cudnnGetPooling2dForwardOutputDim Executed");
   
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyPoolingDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyPoolingDescriptor"));
   
   cudnnPoolingDescriptor_t poolingDesc = in->Get<cudnnPoolingDescriptor_t>();

   cudnnStatus_t cs = cudnnDestroyPoolingDescriptor(poolingDesc);

   //LOG4CPLUS_DEBUG(logger, "cudnnDestroyPoolingDescriptor Executed");
   
   return std::make_shared<Result>(cs);   
}

CUDNN_ROUTINE_HANDLER(PoolingForward) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("PoolingForward"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnPoolingDescriptor_t poolingDesc = in->Get<cudnnPoolingDescriptor_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alpha = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *x = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t yDesc  = in->Get<cudnnTensorDescriptor_t>();
    const void *beta = isFloatDescriptor(yDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    void *y = in->GetFromMarshal<void *>();

    cudnnStatus_t cs = cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
    LOG4CPLUS_DEBUG(logger, "cudnnPoolingForward Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
            out->AddMarshal<void *>(y);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
    }  
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(PoolingBackward) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("PoolingBackward"));   

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   const cudnnPoolingDescriptor_t poolingDesc = in->Get<cudnnPoolingDescriptor_t>();
   const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *alpha = isFloatDescriptor(yDesc)
       ? static_cast<const void *>(in->Assign<const float>())
       : static_cast<const void *>(in->Assign<const double>());
   const void *y = in->GetFromMarshal<void *>();
   const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *dy = in->GetFromMarshal<void *>();
   const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *x = in->GetFromMarshal<void *>();
   const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *beta = isFloatDescriptor(dxDesc)
       ? static_cast<const void *>(in->Assign<const float>())
       : static_cast<const void *>(in->Assign<const double>());
   void *dx = in->GetFromMarshal<void *>();

   cudnnStatus_t cs = cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
   LOG4CPLUS_DEBUG(logger, "cudnnPoolingBackward Executed");

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->AddMarshal<void *>(dx);  
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateActivationDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateActivationDescriptor"));

   cudnnActivationDescriptor_t activationDesc;
 
   cudnnStatus_t cs = cudnnCreateActivationDescriptor(&activationDesc);
   
   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
       out->Add<cudnnActivationDescriptor_t>(activationDesc);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
       return std::make_shared<Result>(cs);
   }
    //LOG4CPLUS_DEBUG(logger, "cudnnCreateActivationDescriptor Executed");
   return std::make_shared<Result>(cs, out);

}

CUDNN_ROUTINE_HANDLER(SetActivationDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetActivationDescriptor"));

   cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
   cudnnActivationMode_t mode = in->Get<cudnnActivationMode_t>();
   cudnnNanPropagation_t reluNanOpt = in->Get<cudnnNanPropagation_t>();
   double coef = in->Get<double>();

   cudnnStatus_t cs = cudnnSetActivationDescriptor(activationDesc, mode, reluNanOpt, coef);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
        out->Add<cudnnActivationDescriptor_t>(activationDesc);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
   }
   //LOG4CPLUS_DEBUG(logger, "cudnnSetActivationDescriptor Executed");
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetActivationDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetActivationDescriptor"));

   cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
   cudnnActivationMode_t mode;
   cudnnNanPropagation_t reluNanOpt;
   double coef;

   cudnnStatus_t cs = cudnnGetActivationDescriptor(activationDesc, &mode, &reluNanOpt, &coef);
   LOG4CPLUS_DEBUG(logger, "cudnnGetActivationDescriptor Executed");

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
   try {
        out->Add<cudnnActivationMode_t>(mode);
        out->Add<cudnnNanPropagation_t>(reluNanOpt);
        out->Add<double>(coef);
   } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
   }
   
   return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyActivationDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyActivationDescriptor"));
   
   cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
   
   cudnnStatus_t cs = cudnnDestroyActivationDescriptor(activationDesc);

   //LOG4CPLUS_DEBUG(logger, "cudnnDestroyActivationDescriptor Executed");
   
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(ActivationForward) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ActivationForward"));
    
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alpha = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *x = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *beta = isFloatDescriptor(yDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    void *y = in->GetFromMarshal<void *>();
    cudnnStatus_t cs = cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
    LOG4CPLUS_DEBUG(logger, "cudnnActivationForward Executed");
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void *>(y);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(ActivationBackward) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ActivationBackward"));

     cudnnHandle_t handle = in->Get<cudnnHandle_t>();
     cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
     const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
     const void *alpha = isFloatDescriptor(yDesc)
         ? static_cast<const void *>(in->Assign<const float>())
         : static_cast<const void *>(in->Assign<const double>());
     const void *y = in->GetFromMarshal<void *>();
     const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
     const void *dy = in->GetFromMarshal<void *>();
     const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
     const void *x = in->GetFromMarshal<void *>();
     cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
     const void *beta = isFloatDescriptor(dxDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
     void *dx = in->GetFromMarshal<void *>();

     cudnnStatus_t cs = cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
     LOG4CPLUS_DEBUG(logger, "cudnnActivationBackward Executed"); 

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
     try {
         out->AddMarshal<void *>(dx);
     } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
     }

     return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateLRNDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateLRNDescriptor"));

    cudnnLRNDescriptor_t normDesc;
     
    cudnnStatus_t cs = cudnnCreateLRNDescriptor(&normDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnLRNDescriptor_t>(normDesc);
     } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
     }
     
     LOG4CPLUS_DEBUG(logger, "cudnnCreateLRNDescriptor Executed");
     
     return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetLRNDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetLRNDescriptor"));

    cudnnLRNDescriptor_t normDesc = in->Get<cudnnLRNDescriptor_t>();
    unsigned lrnN = in->Get<unsigned>();
    double lrnAlpha = in->Get<double>();
    double lrnBeta = in->Get<double>();
    double lrnK = in->Get<double>();

    cudnnStatus_t cs = cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
    LOG4CPLUS_DEBUG(logger, "cudnnSetLRNDescriptor Executed");

    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetLRNDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetLRNDescriptor"));

    cudnnLRNDescriptor_t normDesc = in->Get<cudnnLRNDescriptor_t>();
    unsigned lrnN;
    double lrnAlpha;
    double lrnBeta;
    double lrnK;

    cudnnStatus_t cs = cudnnGetLRNDescriptor(normDesc, &lrnN, &lrnAlpha, &lrnBeta, &lrnK);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<unsigned>(lrnN);
         out->Add<double>(lrnAlpha);  
         out->Add<double>(lrnBeta);
         out->Add<double>(lrnK);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetLRNDescriptor Executed");
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyLRNDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyLRNDescriptor"));

    cudnnLRNDescriptor_t lrnDesc = in->Get<cudnnLRNDescriptor_t>();

    cudnnStatus_t cs = cudnnDestroyLRNDescriptor(lrnDesc);

    
    LOG4CPLUS_DEBUG(logger, "cudnnDestroyLRNDescriptor Executed");
    
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(LRNCrossChannelForward) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("LRNCrossChannelForward"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnLRNDescriptor_t normDesc = in->Get<cudnnLRNDescriptor_t>();
    cudnnLRNMode_t lrnMode = in->Get<cudnnLRNMode_t>();
    cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void* alpha = isFloatDescriptor(xDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    const void* x = in->GetFromMarshal<void*>();
    cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    const void* beta = isFloatDescriptor(yDesc)
        ? static_cast<const void*>(in->Assign<const float>())
        : static_cast<const void*>(in->Assign<const double>());
    void* y = in->GetFromMarshal<void*>();

    cudnnStatus_t cs = cudnnLRNCrossChannelForward(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
    LOG4CPLUS_DEBUG(logger, "cudnnLRNCrossChannelForward Executed"); 

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(y);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
 
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(LRNCrossChannelBackward) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("LRNCrossChannelBackward"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnLRNDescriptor_t normDesc = in->Get<cudnnLRNDescriptor_t>();
    cudnnLRNMode_t lrnMode = in->Get<cudnnLRNMode_t>();
    cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alpha = isFloatDescriptor(yDesc)
        ? static_cast<void *>(in->Assign<float>())
        : static_cast<void *>(in->Assign<double>());
    const void *y = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *dy = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    void *x = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *beta = isFloatDescriptor(dxDesc)
        ? static_cast<void *>(in->Assign<float>())
        : static_cast<void *>(in->Assign<double>());
    void *dx = in->GetFromMarshal<void *>();

    cudnnStatus_t cs = cudnnLRNCrossChannelBackward(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
    LOG4CPLUS_DEBUG(logger, "cudnnLRNCrossChannelBackward Executed");

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
         out->AddMarshal<void*>(dx);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DivisiveNormalizationForward) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DivisiveNormalizationForward"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   cudnnLRNDescriptor_t normDesc = in->Get<cudnnLRNDescriptor_t>();
   cudnnDivNormMode_t mode = in->Get<cudnnDivNormMode_t>();
   cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *alpha = isFloatDescriptor(xDesc)
       ? static_cast<void *>(in->Assign<float>())
       : static_cast<void *>(in->Assign<double>());
   const void *x = in->GetFromMarshal<void*>();
   const void *means = in->GetFromMarshal<void*>();
   void *temp = in->GetFromMarshal<void*>();
   void *temp2 = in->GetFromMarshal<void*>();
   const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
   const void *beta = isFloatDescriptor(yDesc)
       ? static_cast<const void *>(in->Assign<const float>())
       : static_cast<const void *>(in->Assign<const double>());
   void *y = in->GetFromMarshal<void*>();

   cudnnStatus_t cs = cudnnDivisiveNormalizationForward(handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y);
   LOG4CPLUS_DEBUG(logger, "cudnnDivisiveNormalizationForward Executed");

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
         out->AddMarshal<void*>(y);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DivisiveNormalizationBackward) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DivisiveNormalizationBackward"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnLRNDescriptor_t normDesc = in->Get<cudnnLRNDescriptor_t>();
    cudnnDivNormMode_t mode = in->Get<cudnnDivNormMode_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alpha = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *x = in->GetFromMarshal<void *>();
    const void *means = in->GetFromMarshal<void *>();
    const void *dy = in->GetFromMarshal<void *>();
    void *temp = in->GetFromMarshal<void *>();
    void *temp2 = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *beta = isFloatDescriptor(dxDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    void *dx = in->GetFromMarshal<void *>();
    void *dMeans = in->GetFromMarshal<void *>();

    cudnnStatus_t cs = cudnnDivisiveNormalizationBackward(handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dxDesc, dx, dMeans);
    LOG4CPLUS_DEBUG(logger, "cudnnDivisiveNormalizationBackward Executed");
     
     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
         out->AddMarshal<void*>(dx);
         out->AddMarshal<void*>(dMeans);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
      
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DeriveBNTensorDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeriveBNTensorDescriptor"));

    cudnnTensorDescriptor_t derivedBnDesc;
    cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();

    cudnnStatus_t cs = cudnnDeriveBNTensorDescriptor(derivedBnDesc, xDesc, mode);
  
     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
         out->Add<cudnnTensorDescriptor_t>(derivedBnDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnDeriveBNTensorDescriptor");
    
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(GetBatchNormalizationForwardTrainingExWorkspaceSize) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetBatchNormalizationForwardTrainingExWorkspaceSize"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
   cudnnBatchNormOps_t bnOps = in->Get<cudnnBatchNormOps_t>();
   cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
   cudnnTensorDescriptor_t zDesc = in->Get<cudnnTensorDescriptor_t>();
   cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
   cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc = in->Get<cudnnTensorDescriptor_t>();
   cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
   size_t sizeInBytes;

   cudnnStatus_t cs = cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc, &sizeInBytes);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
         out->Add<size_t>(sizeInBytes);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnDeriveBNTensorDescriptor");
    
    return std::make_shared<Result>(cs, out); 
}

CUDNN_ROUTINE_HANDLER(GetBatchNormalizationBackwardExWorkspaceSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetBatchNormalizationBackwardExWorkspaceSize"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
    cudnnBatchNormOps_t bnOps = in->Get<cudnnBatchNormOps_t>();
    cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnTensorDescriptor_t dzDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnTensorDescriptor_t dBnScaleBiasDesc = in->Get<cudnnTensorDescriptor_t>();
    cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
    size_t sizeInBytes;

    cudnnStatus_t cs = cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc, activationDesc, &sizeInBytes);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
         out->Add<size_t>(sizeInBytes);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
   
    LOG4CPLUS_DEBUG(logger, "cudnnDeriveBNTensorDescriptor");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetBatchNormalizationTrainingExReserveSpaceSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetBatchNormalizationTrainingExReserveSpaceSize"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
    cudnnBatchNormOps_t bnOps = in->Get<cudnnBatchNormOps_t>();
    cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
    cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    size_t sizeInBytes;

    cudnnStatus_t cs = cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, bnOps, activationDesc, xDesc, &sizeInBytes);
    LOG4CPLUS_DEBUG(logger, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<size_t>(sizeInBytes);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(BatchNormalizationForwardTraining) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BatchNormalizationForwardTraining"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alpha = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *beta = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *x = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    void *y = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *bnScale = in->GetFromMarshal<void *>();
    const void *bnBias = in->GetFromMarshal<void *>();
    double exponentialAverageFactor = in->Get<double>();
    
    void *resultRunningMean = in->GetFromMarshal<void*>(); //INPUT/OUTPUT
    void *resultRunningVariance = in->GetFromMarshal<void*>(); //INPUT/OUTPUT    
    double epsilon = in->Get<double>();
    void *resultSaveMean = in->GetFromMarshal<void*>(); //OUTPUT
    void *resultSaveInvVariance = in->GetFromMarshal<void*>(); //OUTPUT

    cudnnStatus_t cs = cudnnBatchNormalizationForwardTraining(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);
    LOG4CPLUS_DEBUG(logger, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(resultRunningMean);
        out->AddMarshal<void*>(resultRunningVariance);
        out->AddMarshal<void*>(resultSaveMean);
        out->AddMarshal<void*>(resultSaveInvVariance);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(BatchNormalizationForwardTrainingEx) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BatchNormalizationForwardTrainingEx"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
        cudnnBatchNormOps_t bnOps = in->Get<cudnnBatchNormOps_t>();
        const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *alpha = isFloatDescriptor(xDesc)
            ? static_cast<const void *>(in->Assign<const float>())
            : static_cast<const void *>(in->Assign<const double>());
        const void *beta = isFloatDescriptor(xDesc)
            ? static_cast<const void *>(in->Assign<const float>())
            : static_cast<const void *>(in->Assign<const double>());
        const void *xData = in->GetFromMarshal<void *>();
        const cudnnTensorDescriptor_t zDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *zData = in->GetFromMarshal<void *>();
        const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
        void *yData = in->GetFromMarshal<void *>(); //INPUT
        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *bnScaleData = in->GetFromMarshal<void *>(); //INPUT
        const void *bnBiasData = in->GetFromMarshal<void*>(); //INPUT
        double exponentialAverageFactor = in->Get<double>();
        void *resultRunningMean = in->GetFromMarshal<void*>(); //INPUT/OUTPUT
        void *resultRunningVariance = in->GetFromMarshal<void*>(); //INPUT/OUTPUT
        double epsilon = in->Get<double>();
        void *saveMean = in->GetFromMarshal<void*>(); //OUTPUT
        void *saveInvVariance = in->GetFromMarshal<void*>(); //OUTPUT
        const cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
        void *workspace = in->GetFromMarshal<void*>(); //INPUT
        size_t workSpaceSizeInBytes = in->Get<size_t>();
        void *reserveSpace = in->GetFromMarshal<void*>(); //INPUT
        size_t reserveSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnBatchNormalizationForwardTrainingEx(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData, bnScaleBiasMeanVarDesc, bnScaleData, bnBiasData, exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, saveMean, saveInvVariance, activationDesc, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
        LOG4CPLUS_DEBUG(logger, "cudnnBatchNormalizationForwardTrainingEx Executed");

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->AddMarshal<void*>(resultRunningMean);
            out->AddMarshal<void*>(resultRunningVariance);
            out->AddMarshal<void*>(saveMean);
            out->AddMarshal<void*>(saveInvVariance);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }

    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(BatchNormalizationForwardInference) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BatchNormalizationForwardInference"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alpha = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *beta = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *x = in->Assign<void>();
    const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    void *y = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *bnScale = in->GetFromMarshal<void *>();
    const void *bnBias = in->GetFromMarshal<void *>();
    const void *estimatedMean = in->GetFromMarshal<void *>();
    const void *estimatedVariance = in->GetFromMarshal<void *>();
    double epsilon = in->Get<double>();

    cudnnStatus_t cs = cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
    LOG4CPLUS_DEBUG(logger, "cudnnBatchNormalizationForwardInference Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(y);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
  
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(BatchNormalizationBackward) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BatchNormalizationBackward"));
     
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alphaDataDiff = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *betaDataDiff  = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *alphaParamDiff = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *betaParamDiff  = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *x = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *dy = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
    void *dx = in->GetFromMarshal<void *>();
    const cudnnTensorDescriptor_t bnScaleBiasDiffDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *bnScale = in->GetFromMarshal<void *>();
    void *resultBnScaleDiff = in->GetFromMarshal<void*>(); //OUTPUT
    void *resultBnBiasDiff = in->GetFromMarshal<void*>(); //OUTPUT
    double epsilon = in->Get<double>();
    const void *savedMean = in->GetFromMarshal<void *>();
    const void *savedInvVariance = in->GetFromMarshal<void *>();

    cudnnStatus_t cs = cudnnBatchNormalizationBackward(handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean, savedInvVariance);
    LOG4CPLUS_DEBUG(logger, "cudnnBatchNormalizationBackward Executed");

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->AddMarshal<void*>(resultBnScaleDiff);
          out->AddMarshal<void*>(resultBnBiasDiff);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(BatchNormalizationBackwardEx) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("BatchNormalizationBackwardEx"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnBatchNormMode_t mode = in->Get<cudnnBatchNormMode_t>();
    cudnnBatchNormOps_t bnOps = in->Get<cudnnBatchNormOps_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alphaDataDiff = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *betaDataDiff  = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *alphaParamDiff = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *betaParamDiff  = isFloatDescriptor(xDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *xData = in->GetFromMarshal<void*>();
    const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *yData = in->GetFromMarshal<void*>();
    const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *dyData = in->GetFromMarshal<void*>();
    const cudnnTensorDescriptor_t dzDesc = in->Get<cudnnTensorDescriptor_t>();
    void *dzData = in->GetFromMarshal<void*>(); //OUTPUT
    const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
    void *dxData = in->GetFromMarshal<void*>(); //OUTPUT  
    const cudnnTensorDescriptor_t dBnScaleBiasDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *bnScaleData = in->GetFromMarshal<void*>();
    const void *bnBiasData = in->GetFromMarshal<void*>();
    void *dBnScaleData = in->GetFromMarshal<void*>();
    void *dBnBiasData = in->GetFromMarshal<void*>();
    double epsilon = in->Get<double>();
    const void *savedMean = in->GetFromMarshal<void*>();
    const void *savedInvVariance = in->GetFromMarshal<void*>();
    const cudnnActivationDescriptor_t activationDesc = in->Get<cudnnActivationDescriptor_t>();
    void *workSpace = in->GetFromMarshal<void*>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    void *reserveSpace = in->GetFromMarshal<void*>();
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();
 
     cudnnStatus_t cs = cudnnBatchNormalizationBackwardEx(handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
     LOG4CPLUS_DEBUG(logger, "cudnnBatchNormalizationBackwardEx Executed");

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->AddMarshal<void*>(dzData);
          out->AddMarshal<void*>(dxData);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateSpatialTransformerDescriptor) {
      Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateSpatialTransformerDescriptor"));

      cudnnSpatialTransformerDescriptor_t stDesc;
    
      cudnnStatus_t cs = cudnnCreateSpatialTransformerDescriptor(&stDesc);

       std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnSpatialTransformerDescriptor_t>(stDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateSpatialTransformerDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetSpatialTransformerNdDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetSpatialTransformerNdDescriptor"));
    
    cudnnSpatialTransformerDescriptor_t stDesc = in->Get<cudnnSpatialTransformerDescriptor_t>();
    cudnnSamplerType_t samplerType = in->Get<cudnnSamplerType_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>();


    cudnnStatus_t cs = cudnnSetSpatialTransformerNdDescriptor(stDesc, samplerType, dataType, nbDims, dimA);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnSpatialTransformerDescriptor_t>(stDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    

     LOG4CPLUS_DEBUG(logger, "cudnnSetSpatialTransformerNdDescriptor Executed");
     
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(DestroySpatialTransformerDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroySpatialTransformerDescriptor"));

    cudnnSpatialTransformerDescriptor_t stDesc = in->Get<cudnnSpatialTransformerDescriptor_t>();

    cudnnStatus_t cs = cudnnDestroySpatialTransformerDescriptor(stDesc);

    
    LOG4CPLUS_DEBUG(logger, "cudnnDestroySpatialTransformerDescriptor Executed");
    
    return std::make_shared<Result>(cs); 
}

CUDNN_ROUTINE_HANDLER(SpatialTfGridGeneratorForward) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpatialTfGridGeneratorForward"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   cudnnSpatialTransformerDescriptor_t stDesc = in->Get<cudnnSpatialTransformerDescriptor_t>();
   void *theta = in->Assign<void>();
   void *grid = in->Assign<void>();

   cudnnStatus_t cs = cudnnSpatialTfGridGeneratorForward(handle, stDesc, theta, grid);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<void>(grid);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSpatialTfGridGeneratorForward Executed");
    
    return std::make_shared<Result>(cs, out);
}



CUDNN_ROUTINE_HANDLER(SpatialTfGridGeneratorBackward) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpatialTfGridGeneratorBackward"));

   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   cudnnSpatialTransformerDescriptor_t stDesc = in->Get<cudnnSpatialTransformerDescriptor_t>();
   void *dgrid = in->Assign<void>();
   void *dtheta = in->Assign<void>();

   cudnnStatus_t cs = cudnnSpatialTfGridGeneratorBackward(handle, stDesc, dgrid, dtheta);

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<void>(dtheta);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSpatialTfGridGeneratorBackward Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SpatialTfSamplerForward) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpatialTfSamplerForward"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnSpatialTransformerDescriptor_t stDesc = in->Get<cudnnSpatialTransformerDescriptor_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alpha = isFloatDescriptor(stDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *x = in->GetFromMarshal<void*>(); //INPUT
    const void *grid = in->GetFromMarshal<void*>(); //INPUT
    cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *beta = isFloatDescriptor(yDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    void *y = in->GetFromMarshal<void*>(); //OUTPUT

    cudnnStatus_t cs = cudnnSpatialTfSamplerForward(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);
    LOG4CPLUS_DEBUG(logger, "cudnnSpatialTfSamplerForward Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(y);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SpatialTfSamplerBackward) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SpatialTfSamplerBackward"));
  
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    const cudnnSpatialTransformerDescriptor_t stDesc = in->Get<cudnnSpatialTransformerDescriptor_t>();
    const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alpha = isFloatDescriptor(xDesc)
        ? static_cast<void *>(in->Assign<float>())
        : static_cast<void *>(in->Assign<double>());
    const void *x = in->GetFromMarshal<void*>(); // INPUT
    const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *beta = isFloatDescriptor(dxDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    void *dx = in->GetFromMarshal<void*>(); // OUTPUT
    const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *alphaDgrid = isFloatDescriptor(stDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    const void *dy = in->GetFromMarshal<void*>(); // INPUT
    const void *grid = in->GetFromMarshal<void*>();
    const void *betaDgrid = isFloatDescriptor(stDesc)
        ? static_cast<const void *>(in->Assign<const float>())
        : static_cast<const void *>(in->Assign<const double>());
    void *dgrid = in->GetFromMarshal<void*>(); // OUTPUT

    cudnnStatus_t cs = cudnnSpatialTfSamplerBackward(handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid);
    LOG4CPLUS_DEBUG(logger, "cudnnSpatialTfSamplerBackward Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->AddMarshal<void*>(dx);
        out->AddMarshal<void*>(dgrid);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
    }

    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateDropoutDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateDropoutDescriptor"));

    cudnnDropoutDescriptor_t dropoutDesc;

    cudnnStatus_t cs = cudnnCreateDropoutDescriptor(&dropoutDesc);
    
     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnDropoutDescriptor_t>(dropoutDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateDropoutDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyDropoutDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyDropoutDescriptor"));

   cudnnDropoutDescriptor_t dropoutDesc = in->Get<cudnnDropoutDescriptor_t>();
   
   cudnnStatus_t cs = cudnnDestroyDropoutDescriptor(dropoutDesc);

    LOG4CPLUS_DEBUG(logger, "cudnnDestroyDropoutDescriptor Executed");
    
    return std::make_shared<Result>(cs);
  
}

CUDNN_ROUTINE_HANDLER(DropoutGetStatesSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DropoutGetStatesSize"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    size_t sizeInBytes;

    cudnnStatus_t cs = cudnnDropoutGetStatesSize(handle, &sizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<size_t>(sizeInBytes);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger, "cudnnDropoutGetStatesSize Executed");
    
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(DropoutGetReserveSpaceSize) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DropoutGetReserveSpaceSize"));

     cudnnTensorDescriptor_t xdesc = in->Get<cudnnTensorDescriptor_t>();
     size_t sizeInBytes;
    
     cudnnStatus_t cs = cudnnDropoutGetReserveSpaceSize(xdesc, &sizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<size_t>(sizeInBytes);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    LOG4CPLUS_DEBUG(logger, "cudnnDropoutGetReserveSpaceSize Executed");
    
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(SetDropoutDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetDropoutDescriptor"));

   cudnnDropoutDescriptor_t dropoutDesc = in->Get<cudnnDropoutDescriptor_t>(); //INPUT/OUTPUT
   cudnnHandle_t handle = in->Get<cudnnHandle_t>();
   float dropout = in->Get<float>();
   void *states = in->GetFromMarshal<void*>();
   size_t stateSizeInBytes = in->Get<size_t>();
   unsigned long long seed = in->Get<unsigned long long>();

   cudnnStatus_t cs = cudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
   LOG4CPLUS_DEBUG(logger, "cudnnSetDropoutDescriptor Executed");

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnDropoutDescriptor_t>(dropoutDesc);
        out->AddMarshal<void*>(states);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
    }
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(RestoreDropoutDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RestoreDropoutDescriptor"));

    cudnnDropoutDescriptor_t dropoutDesc = in->Get<cudnnDropoutDescriptor_t>();
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    float dropout = in->Get<float>();
    void *states = in->Assign<void>();
    size_t stateSizeInBytes = in->Get<size_t>();
    unsigned long long seed = in->Get<unsigned long long>();

    cudnnStatus_t cs = cudnnRestoreDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnDropoutDescriptor_t>(dropoutDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnRestoreDropoutDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}


CUDNN_ROUTINE_HANDLER(GetDropoutDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetDropoutDescriptor"));

    cudnnDropoutDescriptor_t dropoutDesc = in->Get<cudnnDropoutDescriptor_t>();
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    float dropout;
    void *states;
    unsigned long long seed;

    cudnnStatus_t cs = cudnnGetDropoutDescriptor(dropoutDesc, handle, &dropout, &states, &seed);
    LOG4CPLUS_DEBUG(logger, "cudnnGetDropoutDescriptor Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<float>(dropout);
        out->AddMarshal<void*>(states);
        out->Add<unsigned long long>(seed);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DropoutForward) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DropoutForward"));

     cudnnHandle_t handle = in->Get<cudnnHandle_t>();
     cudnnDropoutDescriptor_t dropoutDesc = in->Get<cudnnDropoutDescriptor_t>();
     cudnnTensorDescriptor_t xdesc = in->Get<cudnnTensorDescriptor_t>();
     void *x = in->Assign<void>();
     cudnnTensorDescriptor_t ydesc = in->Get<cudnnTensorDescriptor_t>();
     void *y = in->Assign<void>(); //OUTPUT
     void *reserveSpace = in->Assign<void>(); //OUTPUT
     size_t reserveSpaceSizeInBytes = in->Get<size_t>();

     cudnnStatus_t cs = cudnnDropoutForward(handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<void>(y);
          out->Add<void>(reserveSpace);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnDropoutForward Executed");
    
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(DropoutBackward) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DropoutBackward"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnDropoutDescriptor_t dropoutDesc = in->Get<cudnnDropoutDescriptor_t>();
    cudnnTensorDescriptor_t dydesc = in->Get<cudnnTensorDescriptor_t>();
    void *dy = in->Assign<void>();
    cudnnTensorDescriptor_t dxdesc = in->Get<cudnnTensorDescriptor_t>();
    void *dx = in->Assign<void>(); //OUTPUT
    void *reserveSpace = in->Assign<void>();
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();

    cudnnStatus_t cs = cudnnDropoutBackward(handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<void>(dx);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnDropoutBackward Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateRNNDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateRNNDescriptor"));

   cudnnRNNDescriptor_t rnnDesc;

   cudnnStatus_t cs = cudnnCreateRNNDescriptor(&rnnDesc); 

   std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnRNNDescriptor_t>(rnnDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
   
    LOG4CPLUS_DEBUG(logger, "cudnnCreateRNNDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(DestroyRNNDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyRNNDescriptor"));

    cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();

    cudnnStatus_t cs = cudnnDestroyRNNDescriptor(rnnDesc);

    
    LOG4CPLUS_DEBUG(logger, "cudnnDestroyRNNDescriptor Executed");
    
    return std::make_shared<Result>(cs);
}

#if CUDNN_VERSION < 8000
    CUDNN_ROUTINE_HANDLER(SetRNNDescriptor_v5) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNDescriptor_v5"));

        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int hiddenSize = in->Get<int>();
        int numLayers = in->Get<int>();
        cudnnDropoutDescriptor_t dropoutDesc = in->Get<cudnnDropoutDescriptor_t>();
        cudnnRNNInputMode_t inputMode = in->Get<cudnnRNNInputMode_t>();
        cudnnDirectionMode_t direction = in->Get<cudnnDirectionMode_t>();
        cudnnRNNMode_t mode = in->Get<cudnnRNNMode_t>();
        cudnnDataType_t mathPrec = in->Get<cudnnDataType_t>();

        cudnnStatus_t cs = cudnnSetRNNDescriptor_v5(rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, mathPrec);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnRNNDescriptor_t>(rnnDesc);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }

        LOG4CPLUS_DEBUG(logger, "cudnnSetRNNDescriptor_v5 Executed");
        
        return std::make_shared<Result>(cs, out);
    }
#endif

#if CUDNN_VERSION >= 8000
    CUDNN_ROUTINE_HANDLER(SetRNNDescriptor_v8) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNDescriptor_v8"));

        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();

        cudnnRNNAlgo_t algo = in->Get<cudnnRNNAlgo_t>();
        cudnnRNNMode_t cellMode= in->Get<cudnnRNNMode_t>();
        cudnnRNNBiasMode_t biasMode= in->Get<cudnnRNNBiasMode_t>();
        cudnnDirectionMode_t dirMode= in->Get<cudnnDirectionMode_t>();
        cudnnRNNInputMode_t inputMode= in->Get<cudnnRNNInputMode_t>();
        cudnnDataType_t dataType= in->Get<cudnnDataType_t>();
        cudnnDataType_t mathPrec= in->Get<cudnnDataType_t>();
        cudnnMathType_t mathType= in->Get<cudnnMathType_t>();
        int32_t inputSize= in->Get<int32_t>();
        int32_t hiddenSize= in->Get<int32_t>();
        int32_t projSize= in->Get<int32_t>();
        int32_t numLayers= in->Get<int32_t>();
        cudnnDropoutDescriptor_t dropoutDesc= in->Get<cudnnDropoutDescriptor_t>();
        uint32_t auxFlags= in->Get<uint32_t>();

        cudnnStatus_t cs = cudnnSetRNNDescriptor_v8(rnnDesc, algo,
                cellMode,
                biasMode,
                dirMode,
                inputMode,
                dataType,
                mathPrec,
                mathType,
                inputSize,
                hiddenSize,
                projSize,
                numLayers,
                dropoutDesc,
                auxFlags);

        LOG4CPLUS_DEBUG(logger, "cudnnSetRNNDescriptor_v8 Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNDescriptor_v8) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNDescriptor_v8"));

        cudnnRNNDescriptor_t rnnDesc =  in->Get<cudnnRNNDescriptor_t>();

        cudnnRNNAlgo_t algo;
        cudnnRNNMode_t cellMode;
        cudnnRNNBiasMode_t biasMode;
        cudnnDirectionMode_t dirMode;
        cudnnRNNInputMode_t inputMode;
        cudnnDataType_t dataType;
        cudnnDataType_t mathPrec;
        cudnnMathType_t mathType;
        int32_t inputSize;
        int32_t hiddenSize;
        int32_t projSize;
        int32_t numLayers;
        cudnnDropoutDescriptor_t dropoutDesc;
        uint32_t auxFlags;

        cudnnStatus_t cs = cudnnGetRNNDescriptor_v8(rnnDesc,
                                                    &algo,
                                                    &cellMode, &biasMode, &dirMode, &inputMode,
                                                    &dataType, &mathPrec, &mathType,
                                                    &inputSize, &hiddenSize, &projSize, &numLayers,
                                                    &dropoutDesc, &auxFlags);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnRNNAlgo_t>(algo);
            out->Add<cudnnRNNMode_t>(cellMode);
            out->Add<cudnnRNNBiasMode_t>(biasMode);
            out->Add<cudnnDirectionMode_t>(dirMode);
            out->Add<cudnnRNNInputMode_t>(inputMode);
            out->Add<cudnnDataType_t>(dataType);
            out->Add<cudnnDataType_t>(mathPrec);
            out->Add<cudnnMathType_t>(mathType);
            out->Add<int>(inputSize);
            out->Add<int>(hiddenSize);
            out->Add<int>(projSize);
            out->Add<int>(numLayers);
            out->Add<cudnnDropoutDescriptor_t>(dropoutDesc);
            out->Add<int>(auxFlags);

        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }

        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNDescriptor_v8 Executed");
        
        return std::make_shared<Result>(cs, out);
    }
#endif

CUDNN_ROUTINE_HANDLER(RNNForward) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNForward"));
  
    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
    cudnnForwardMode_t fwdMode = in->Get<cudnnForwardMode_t>();
    const int32_t *devSeqLengths = in->Get<int32_t*>();
    cudnnRNNDataDescriptor_t xDesc = in->Get<cudnnRNNDataDescriptor_t>();
    const void *x = in->GetFromMarshal<const void*>();
    cudnnRNNDataDescriptor_t yDesc = in->Get<cudnnRNNDataDescriptor_t>();
    void *y = in->GetFromMarshal<void*>();
    cudnnTensorDescriptor_t hDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *hx = in->GetFromMarshal<const void*>();
    void *hy = in->GetFromMarshal<void*>();
    cudnnTensorDescriptor_t cDesc = in->Get<cudnnTensorDescriptor_t>();
    const void *cx = in->GetFromMarshal<const void*>();
    void *cy = in->GetFromMarshal<void*>();
    size_t weightSpaceSize = in->Get<size_t>();
    const void *weightSpace = in->GetFromMarshal<const void*>();
    size_t workSpaceSize = in->Get<size_t>(); 
    void *workSpace = in->GetFromMarshal<void*>();
    size_t reserveSpaceSize = in->Get<size_t>();
    void *reserveSpace = in->GetFromMarshal<void*>();

    cudnnStatus_t cs = cudnnRNNForward(handle, rnnDesc, fwdMode, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy,  weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CreateRNNDataDescriptor) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateRNNDataDescriptor"));

     cudnnRNNDataDescriptor_t rnnDataDesc;

     cudnnStatus_t cs = cudnnCreateRNNDataDescriptor(&rnnDataDesc);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnRNNDataDescriptor_t>(rnnDataDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnCreateRNNDataDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyRNNDataDescriptor) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyRNNDataDescriptor"));

     cudnnRNNDataDescriptor_t rnnDataDesc = in->Get<cudnnRNNDataDescriptor_t>();
     
     cudnnStatus_t cs = cudnnDestroyRNNDataDescriptor(rnnDataDesc);
     
     
     LOG4CPLUS_DEBUG(logger, "cudnnDestroyRNNDataDescriptor Executed");
     
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetRNNDataDescriptor) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNDataDescriptor"));

     cudnnRNNDataDescriptor_t rnnDataDesc = in->Get<cudnnRNNDataDescriptor_t>();
     cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
     cudnnRNNDataLayout_t layout = in->Get<cudnnRNNDataLayout_t>();
     int maxSeqLength = in->Get<int>();
     int batchSize = in->Get<int>();
     int vectorSize = in->Get<int>();
     int *seqLengthArray = in->Assign<int>(batchSize);
     void *paddingFill = isFloatDescriptor(rnnDataDesc)
        ? static_cast<void*>(in->Assign<float>())
        : static_cast<void*>(in->Assign<double>());
    
     cudnnStatus_t cs = cudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill);
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetRNNDataDescriptor Executed");
    
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetRNNDataDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNDataDescriptor"));
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    cudnnRNNDataDescriptor_t rnnDataDesc = in->Get<cudnnRNNDataDescriptor_t>();
    cudnnDataType_t *dataType = out->Delegate<cudnnDataType_t>();
    cudnnRNNDataLayout_t *layout = out->Delegate<cudnnRNNDataLayout_t>();
    int *maxSeqLength = out->Delegate<int>();
    int *batchSize = out->Delegate<int>();
    int *vectorSize = out->Delegate<int>();
    int arrayLengthRequested = in->Get<int>();
    int *seqLengthArray = out->Delegate<int>(arrayLengthRequested);
    void *paddingFill = isFloatDescriptor(rnnDataDesc)
        ? static_cast<void*>(out->Delegate<float>())
        : static_cast<void*>(out->Delegate<double>());

    cudnnStatus_t cs = cudnnGetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, arrayLengthRequested, seqLengthArray, paddingFill);
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNDataDescriptor Executed");
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetRNNWeightSpaceSize) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNWeightSpaceSize"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
    size_t weightSpaceSize;

    cudnnStatus_t cs = cudnnGetRNNWeightSpaceSize(handle, rnnDesc, &weightSpaceSize);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<size_t>(weightSpaceSize);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNWeightSpaceSize Executed");
    
    return std::make_shared<Result>(cs, out);
}
CUDNN_ROUTINE_HANDLER(GetRNNTempSpaceSizes) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNTempSpaceSizes"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
    cudnnForwardMode_t fMode = in->Get<cudnnForwardMode_t>();
    cudnnRNNDataDescriptor_t xDesc = in->Get<cudnnRNNDataDescriptor_t>();
    size_t workSpaceSize;
    size_t reserveSpaceSize;

    cudnnStatus_t cs = cudnnGetRNNTempSpaceSizes(handle, rnnDesc, fMode, xDesc, &workSpaceSize, &reserveSpaceSize);
    LOG4CPLUS_DEBUG(logger, "cudnnGetRNNTempSpaceSizes Executed");

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<size_t>(workSpaceSize);
        out->Add<size_t>(reserveSpaceSize);
    } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateSeqDataDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateSeqDataDescriptor"));

    cudnnSeqDataDescriptor_t seqDataDesc;

    cudnnStatus_t cs = cudnnCreateSeqDataDescriptor(&seqDataDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnSeqDataDescriptor_t>(seqDataDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnCreateSeqDataDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(DestroySeqDataDescriptor) {
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroySeqDataDescriptor"));

   cudnnSeqDataDescriptor_t seqDataDesc = in->Get<cudnnSeqDataDescriptor_t>();

   cudnnStatus_t cs = cudnnDestroySeqDataDescriptor(seqDataDesc);

   
   LOG4CPLUS_DEBUG(logger, "cudnnDestroySeqDataDescriptor Executed");
   
   return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetSeqDataDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetSeqDataDescriptor"));

    cudnnSeqDataDescriptor_t seqDataDesc;
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>();
    cudnnSeqDataAxis_t *axes = in->Assign<cudnnSeqDataAxis_t>();
    size_t seqLengthArraySize = in->Get<size_t>();
    int *seqLengthArray = in->Assign<int>();
    void *paddingFill = in->Assign<void>();

    cudnnStatus_t cs = cudnnSetSeqDataDescriptor(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill);
   
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnSeqDataDescriptor_t>(seqDataDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetSeqDataDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetSeqDataDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetSeqDataDescriptor"));

    cudnnSeqDataDescriptor_t seqDataDesc = in->Get<cudnnSeqDataDescriptor_t>();
    cudnnDataType_t dataType; //OUTPUT
    int nbDims; //OUTPUT
    int nbDimsRequested = in->Get<int>();
    int *dimA = in->Assign<int>(); //OUTPUT
    cudnnSeqDataAxis_t *axes = in->Assign<cudnnSeqDataAxis_t>(); //OUTPUT
    size_t seqLengthArraySize; //OUTPUT
    size_t seqLengthSizeRequested = in->Get<size_t>();
    int *seqLengthArray = in->Assign<int>(); //OUTPUT
    void *paddingFill = in->Assign<void>(); //OUTPUT

    cudnnStatus_t cs = cudnnGetSeqDataDescriptor(seqDataDesc, &dataType, &nbDims, nbDimsRequested, dimA, axes, &seqLengthArraySize, seqLengthSizeRequested, seqLengthArray, paddingFill);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnDataType_t>(dataType);
          out->Add<int>(nbDims);
          out->Add<int>(dimA);
          out->Add<cudnnSeqDataAxis_t>(axes);
          out->Add<int>(seqLengthArraySize);
          out->Add<int>(seqLengthArray);
          out->Add<void>(paddingFill);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetSeqDataDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateAttnDescriptor) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateAttnDescriptor"));

     cudnnAttnDescriptor_t attnDesc;

     cudnnStatus_t cs = cudnnCreateAttnDescriptor(& attnDesc);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnAttnDescriptor_t>(attnDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnCreateAttnDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(DestroyAttnDescriptor) {

     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyAttnDescriptor"));
   
     cudnnAttnDescriptor_t attnDesc = in->Get<cudnnAttnDescriptor_t>();
     
     cudnnStatus_t cs = cudnnDestroyAttnDescriptor(attnDesc); 

     
      LOG4CPLUS_DEBUG(logger, "cudnnDestroyAttnDescriptor Executed");
     
     return std::make_shared<Result>(cs); 
}

CUDNN_ROUTINE_HANDLER(SetAttnDescriptor) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetAttnDescriptor"));

     cudnnAttnDescriptor_t attnDesc; //OUTPUT
     unsigned attnMode = in->Get<unsigned>();
     int nHeads = in->Get<int>();
     double smScaler = in->Get<double>();
     cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
     cudnnDataType_t computePrec = in->Get<cudnnDataType_t>();
     cudnnMathType_t mathType = in->Get<cudnnMathType_t>();
     cudnnDropoutDescriptor_t attnDropoutDesc = in->Get<cudnnDropoutDescriptor_t>();
     cudnnDropoutDescriptor_t postDropoutDesc = in->Get<cudnnDropoutDescriptor_t>();
     int qSize = in->Get<int>();
     int kSize = in->Get<int>();
     int vSize = in->Get<int>();
     int qProjSize = in->Get<int>();
     int kProjSize = in->Get<int>();
     int vProjSize = in->Get<int>();
     int oProjSize = in->Get<int>();
     int qoMaxSeqLength = in->Get<int>();
     int kvMaxSeqLength = in->Get<int>();
     int maxBatchSize = in->Get<int>();
     int maxBeamSize = in->Get<int>();

     cudnnStatus_t cs = cudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnAttnDescriptor_t>(attnDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnSetAttnDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetAttnDescriptor) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetAttnDescriptor"));

     cudnnAttnDescriptor_t attnDesc = in->Get<cudnnAttnDescriptor_t>();
     unsigned attnMode;
     int nHeads;
     double smScaler;
     cudnnDataType_t dataType;
     cudnnDataType_t computePrec;
     cudnnMathType_t mathType;
     cudnnDropoutDescriptor_t attnDropoutDesc;
     cudnnDropoutDescriptor_t postDropoutDesc;
     int qSize;
     int kSize;
     int vSize;
     int qProjSize;
     int kProjSize;
     int vProjSize;
     int oProjSize;
     int qoMaxSeqLength;
     int kvMaxSeqLength;
     int maxBatchSize;
     int maxBeamSize;

     cudnnStatus_t cs = cudnnSetAttnDescriptor(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength, kvMaxSeqLength, maxBatchSize, maxBeamSize);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<unsigned>(attnMode);
          out->Add<int>(nHeads);
          out->Add<double>(smScaler);
          out->Add<cudnnDataType_t>(dataType);
          out->Add<cudnnDataType_t>(computePrec);
          out->Add<cudnnMathType_t>(mathType);
          out->Add<cudnnDropoutDescriptor_t>(attnDropoutDesc);
          out->Add<cudnnDropoutDescriptor_t>(postDropoutDesc);
          out->Add<int>(qSize);
          out->Add<int>(kSize);
          out->Add<int>(vSize);
          out->Add<int>(qProjSize);
          out->Add<int>(kProjSize);
          out->Add<int>(vProjSize);
          out->Add<int>(oProjSize);
          out->Add<int>(qoMaxSeqLength);
          out->Add<int>(kvMaxSeqLength);
          out->Add<int>(maxBatchSize);
          out->Add<int>(maxBeamSize);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetAttnDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetMultiHeadAttnBuffers) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetMultiHeadAttnBuffers"));
     
     cudnnHandle_t handle = in->Get<cudnnHandle_t>();
     cudnnAttnDescriptor_t attnDesc = in->Get<cudnnAttnDescriptor_t>();
     size_t weightSizeInBytes;
     size_t workSpaceSizeInBytes;
     size_t reserveSpaceSizeInBytes;

     cudnnStatus_t cs = cudnnGetMultiHeadAttnBuffers(handle, attnDesc, &weightSizeInBytes, &workSpaceSizeInBytes, &reserveSpaceSizeInBytes);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<size_t>(weightSizeInBytes);
          out->Add<size_t>(workSpaceSizeInBytes);
          out->Add<size_t>(reserveSpaceSizeInBytes);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetMultiHeadAttnBuffers Executed");
    
    return std::make_shared<Result>(cs, out); 
}

CUDNN_ROUTINE_HANDLER(GetMultiHeadAttnWeights) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetMultiHeadAttnWeights"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnAttnDescriptor_t attnDesc = in->Get<cudnnAttnDescriptor_t>();
    cudnnMultiHeadAttnWeightKind_t wKind = in->Get<cudnnMultiHeadAttnWeightKind_t>();
    size_t weightSizeInBytes = in->Get<size_t>();
    void *weights = in->Assign<void>();
    cudnnTensorDescriptor_t wDesc;
    void *wAddr = in->Assign<void>();

    cudnnStatus_t cs = cudnnGetMultiHeadAttnWeights(handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, &wAddr);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnTensorDescriptor_t>(wDesc);
          out->Add<void>(wAddr);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetMultiHeadAttnWeights Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(MultiHeadAttnForward) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MultiHeadAttnForward"));

     cudnnHandle_t handle = in->Get<cudnnHandle_t>();
     cudnnAttnDescriptor_t attnDesc = in->Get<cudnnAttnDescriptor_t>();
     int currIdx = in->Get<int>();
     int *loWinIdx = in->Assign<int>();
     int *hiWinIdx = in->Assign<int>();
     int *seqLengthArrayQRO = in->Assign<int>();
     int *seqLengthArrayKV = in->Assign<int>();
     cudnnSeqDataDescriptor_t qDesc = in->Get<cudnnSeqDataDescriptor_t>();
     void *queries = in->Assign<void>();
     void *residuals = in->Assign<void>();
     cudnnSeqDataDescriptor_t kDesc = in->Get<cudnnSeqDataDescriptor_t>();
     void *keys = in->Assign<void>();
     cudnnSeqDataDescriptor_t vDesc = in->Get<cudnnSeqDataDescriptor_t>();
     void *values = in->Assign<void>();
     cudnnSeqDataDescriptor_t oDesc = in->Get<cudnnSeqDataDescriptor_t>();
     void *output = in->Assign<void>(); //OUTPUT
     size_t weightSizeInBytes = in->Get<size_t>();
     void *weights = in->Assign<void>();
     size_t workSpaceSizeInBytes = in->Get<size_t>();
     void *workSpace = in->Assign<void>(); //INPUT/OUTPUT
     size_t reserveSpaceSizeInBytes = in->Get<size_t>();
     void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT

     cudnnStatus_t cs = cudnnMultiHeadAttnForward(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, seqLengthArrayQRO, seqLengthArrayKV, qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, output, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<void>(output);
          out->Add<void>(workSpace);
          out->Add<void>(reserveSpace);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnMultiHeadAttnForward Executed");
    
    return std::make_shared<Result>(cs, out);     
}

CUDNN_ROUTINE_HANDLER(MultiHeadAttnBackwardData) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MultiHeadAttnBackwardData"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnAttnDescriptor_t attnDesc = in->Get<cudnnAttnDescriptor_t>();
    int *loWinIdx = in->Assign<int>();
    int *hiWinIdx = in->Assign<int>();
    int *seqLengthArrayDQDO = in->Assign<int>();
    int *seqLengthArrayDKDV = in->Assign<int>();
    cudnnSeqDataDescriptor_t doDesc = in->Get<cudnnSeqDataDescriptor_t>();
    void *dout = in->Assign<void>();
    cudnnSeqDataDescriptor_t dqDesc = in->Get<cudnnSeqDataDescriptor_t>();
    void *dqueries = in->Assign<void>(); //OUTPUT
    void *queries = in->Assign<void>();
    cudnnSeqDataDescriptor_t dkDesc = in->Get<cudnnSeqDataDescriptor_t>();
    void *dkeys = in->Assign<void>(); //OUTPUT
    void *keys = in->Assign<void>();
    cudnnSeqDataDescriptor_t dvDesc = in->Get<cudnnSeqDataDescriptor_t>();
    void *dvalues = in->Assign<void>(); //OUTPUT
    void *values = in->Assign<void>();
    size_t weightSizeInBytes = in->Get<size_t>();
    void *weights = in->Assign<void>();
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    void *workSpace = in->Assign<void>(); //INPUT/OUTPUT
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();
    void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT

    cudnnStatus_t cs = cudnnMultiHeadAttnBackwardData(handle, attnDesc, loWinIdx, hiWinIdx, seqLengthArrayDQDO, seqLengthArrayDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<void>(dqueries);
          out->Add<void>(dkeys);
          out->Add<void>(dvalues);
          out->Add<void>(workSpace);
          out->Add<void>(reserveSpace);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnMultiHeadAttnBackwardData Executed");
    
    return std::make_shared<Result>(cs, out);        
}

CUDNN_ROUTINE_HANDLER(MultiHeadAttnBackwardWeights) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MultiHeadAttnBackwardWeights"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnAttnDescriptor_t attnDesc = in->Get<cudnnAttnDescriptor_t>();
    cudnnWgradMode_t addGrad = in->Get<cudnnWgradMode_t>();
    cudnnSeqDataDescriptor_t qDesc = in->Get<cudnnSeqDataDescriptor_t>();
    void *queries = in->Assign<void>();
    cudnnSeqDataDescriptor_t kDesc = in->Get<cudnnSeqDataDescriptor_t>();
    void *keys = in->Assign<void>();
    cudnnSeqDataDescriptor_t vDesc = in->Get<cudnnSeqDataDescriptor_t>();
    void *values = in->Assign<void>();
    cudnnSeqDataDescriptor_t doDesc = in->Get<cudnnSeqDataDescriptor_t>();
    void *dout = in->Assign<void>();
    size_t weightSizeInBytes = in->Get<size_t>();
    void *weights = in->Assign<void>();
    void *dweights = in->Assign<void>(); //OUTPUT
    size_t workSpaceSizeInBytes = in->Get<size_t>();
    void *workSpace = in->Assign<void>(); //INPUT/OUTPUT
    size_t reserveSpaceSizeInBytes = in->Get<size_t>();
    void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT

    cudnnStatus_t cs = cudnnMultiHeadAttnBackwardWeights(handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout, weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<void>(dweights);
          out->Add<void>(workSpace);
          out->Add<void>(reserveSpace);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
      
    LOG4CPLUS_DEBUG(logger, "cudnnMultiHeadAttnBackwardWeights Executed");
    
    return std::make_shared<Result>(cs, out);  
}

CUDNN_ROUTINE_HANDLER(CreateCTCLossDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateCTCLossDescriptor"));

    cudnnCTCLossDescriptor_t ctcLossDesc;
    
    cudnnStatus_t cs = cudnnCreateCTCLossDescriptor(&ctcLossDesc);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnCTCLossDescriptor_t>(ctcLossDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateCTCLossDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetCTCLossDescriptor) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetCTCLossDescriptor"));

    cudnnCTCLossDescriptor_t ctcLossDesc;
    cudnnDataType_t compType = in->Get<cudnnDataType_t>();

    cudnnStatus_t cs = cudnnSetCTCLossDescriptor(ctcLossDesc, compType);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnCTCLossDescriptor_t>(ctcLossDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, " cudnnSetCTCLossDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(SetCTCLossDescriptorEx) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetCTCLossDescriptorEx"));

    cudnnCTCLossDescriptor_t ctcLossDesc;
    cudnnDataType_t compType = in->Get<cudnnDataType_t>();
    cudnnLossNormalizationMode_t normMode = in->Get<cudnnLossNormalizationMode_t>();
    cudnnNanPropagation_t gradMode = in->Get<cudnnNanPropagation_t>();

    cudnnStatus_t cs = cudnnSetCTCLossDescriptorEx(ctcLossDesc, compType, normMode, gradMode);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnCTCLossDescriptor_t>(ctcLossDesc);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnSetCTCLossDescriptorEx Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetCTCLossDescriptor) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetCTCLossDescriptor"));

     cudnnCTCLossDescriptor_t ctcLossDesc = in->Get<cudnnCTCLossDescriptor_t>();
     cudnnDataType_t compType;

     cudnnStatus_t cs = cudnnGetCTCLossDescriptor(ctcLossDesc, &compType);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnDataType_t>(compType);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
     LOG4CPLUS_DEBUG(logger, "cudnnGetCTCLossDescriptor Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetCTCLossDescriptorEx) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetCTCLossDescriptorEx"));

     cudnnCTCLossDescriptor_t ctcLossDesc = in->Get<cudnnCTCLossDescriptor_t>();
     cudnnDataType_t compType;
     cudnnLossNormalizationMode_t normMode;
     cudnnNanPropagation_t gradMode;
    
     cudnnStatus_t cs = cudnnGetCTCLossDescriptorEx(ctcLossDesc, &compType, &normMode, &gradMode);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<cudnnDataType_t>(compType);
          out->Add<cudnnLossNormalizationMode_t>(normMode);
          out->Add<cudnnNanPropagation_t>(gradMode);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    

     LOG4CPLUS_DEBUG(logger, "cudnnGetCTCLossDescriptorEx Executed");
     
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(DestroyCTCLossDescriptor) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyCTCLossDescriptor"));

     cudnnCTCLossDescriptor_t ctcLossDesc = in->Get<cudnnCTCLossDescriptor_t>();

     cudnnStatus_t cs = cudnnDestroyCTCLossDescriptor(ctcLossDesc);

     
     LOG4CPLUS_DEBUG(logger, "cudnnGetCTCLossDescriptorEx Executed");
     
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CTCLoss) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CTCLoss"));

     cudnnHandle_t handle = in->Get<cudnnHandle_t>();
     cudnnTensorDescriptor_t probsDesc = in->Get<cudnnTensorDescriptor_t>();
     void *probs = in->Assign<void>();
     int labels = in->Get<int>();
     int labelLengths = in->Get<int>();
     int inputLengths = in->Get<int>();
     void *costs = in->Assign<void>(); //OUTPUT
     cudnnTensorDescriptor_t gradientsDesc = in->Get<cudnnTensorDescriptor_t>();
     void *gradients = in->Assign<void>(); //OUTPUT
     cudnnCTCLossAlgo_t algo = in->Get<cudnnCTCLossAlgo_t>();
     cudnnCTCLossDescriptor_t ctcLossDesc = in->Get<cudnnCTCLossDescriptor_t>();
     void *workspace = in->Assign<void>();
     size_t workSpaceSizeInBytes = in->Get<size_t>();

     cudnnStatus_t cs = cudnnCTCLoss(handle, probsDesc, probs, &labels, &labelLengths, &inputLengths, costs, gradientsDesc, gradients, algo, ctcLossDesc, workspace, workSpaceSizeInBytes);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<void>(costs);
          out->Add<void>(gradients);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
     
    LOG4CPLUS_DEBUG(logger, "cudnnCTCLoss Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(GetCTCLossWorkspaceSize) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetCTCLossWorkspaceSize"));

     cudnnHandle_t handle = in->Get<cudnnHandle_t>();
     cudnnTensorDescriptor_t probsDesc = in->Get<cudnnTensorDescriptor_t>();
     cudnnTensorDescriptor_t gradientsDesc = in->Get<cudnnTensorDescriptor_t>();
     int labels = in->Get<int>();
     int labelLengths = in->Get<int>();
     int inputLengths = in->Get<int>();
     cudnnCTCLossAlgo_t algo = in->Get<cudnnCTCLossAlgo_t>();
     cudnnCTCLossDescriptor_t ctcLossDesc = in->Get<cudnnCTCLossDescriptor_t>();
     size_t sizeInBytes;

     cudnnStatus_t cs = cudnnGetCTCLossWorkspaceSize(handle, probsDesc, gradientsDesc, &labels, &labelLengths, &inputLengths, algo, ctcLossDesc, &sizeInBytes);

       std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<size_t>(sizeInBytes);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
     
     LOG4CPLUS_DEBUG(logger, "cudnnGetCTCLossWorkspaceSize Executed");
    
    return std::make_shared<Result>(cs, out);    
}

CUDNN_ROUTINE_HANDLER(SetCallback) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetCallback"));

     unsigned mask = in->Get<unsigned>();
     void *udata = in->Assign<void>();
     cudnnCallback_t fptr = in->Get<cudnnCallback_t>();

     cudnnStatus_t cs = cudnnSetCallback(mask, udata, fptr);

    
    LOG4CPLUS_DEBUG(logger, "cudnnSetCallback Executed");
    
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetCallback) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetCallback"));

     unsigned mask;
     void *udata;
     cudnnCallback_t fptr;

     cudnnStatus_t cs = cudnnSetCallback( mask, &udata, fptr);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<unsigned>(mask);
          out->Add<void>(udata);
          out->Add<cudnnCallback_t>(fptr);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetCallback Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateFusedOpsConstParamPack) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateFusedOpsConstParamPack"));

     cudnnFusedOpsConstParamPack_t constPack = in->Get<cudnnFusedOpsConstParamPack_t>();
     cudnnFusedOps_t ops = in->Get<cudnnFusedOps_t>();

     cudnnStatus_t cs = cudnnCreateFusedOpsConstParamPack(&constPack, ops);

     
     LOG4CPLUS_DEBUG(logger, "cudnnCreateFusedOpsConstParamPack Executed");
     
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(DestroyFusedOpsConstParamPack) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyFusedOpsConstParamPack"));

     cudnnFusedOpsConstParamPack_t constPack = in->Get<cudnnFusedOpsConstParamPack_t>();
    
     cudnnStatus_t cs = cudnnDestroyFusedOpsConstParamPack(constPack);

     LOG4CPLUS_DEBUG(logger, "cudnnDestroyFusedOpsConstParamPack Executed");
     
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetFusedOpsConstParamPackAttribute) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFusedOpsConstParamPackAttribute"));

    cudnnFusedOpsConstParamPack_t constPack = in->Get<cudnnFusedOpsConstParamPack_t>();
    cudnnFusedOpsConstParamLabel_t paramLabel = in->Get<cudnnFusedOpsConstParamLabel_t>();
    void *param = in->Assign<void>();

    cudnnStatus_t cs = cudnnSetFusedOpsConstParamPackAttribute(constPack, paramLabel, param);

    
     LOG4CPLUS_DEBUG(logger, "cudnnSetFusedOpsConstParamPackAttribute Executed");
    
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFusedOpsConstParamPackAttribute) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFusedOpsConstParamPackAttribute"));

     cudnnFusedOpsConstParamPack_t constPack = in->Get<cudnnFusedOpsConstParamPack_t>();
     cudnnFusedOpsConstParamLabel_t paramLabel = in->Get<cudnnFusedOpsConstParamLabel_t>();
     void *param = in->Assign<void>();
     int isNULL = in->Get<int>();

     cudnnStatus_t cs = cudnnGetFusedOpsConstParamPackAttribute(constPack, paramLabel, param, &isNULL);

      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<int>(isNULL);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFusedOpsConstParamPackAttribute Executed");
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateFusedOpsVariantParamPack) {
      Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateFusedOpsVariantParamPack"));

      cudnnFusedOpsVariantParamPack_t varPack = in->Get<cudnnFusedOpsVariantParamPack_t>();
      cudnnFusedOps_t ops = in->Get<cudnnFusedOps_t>();

      cudnnStatus_t cs = cudnnCreateFusedOpsVariantParamPack(&varPack, ops);

      
       LOG4CPLUS_DEBUG(logger, "cudnnCreateFusedOpsVariantParamPack Executed");
      
      return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(DestroyFusedOpsVariantParamPack) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyFusedOpsVariantParamPack"));

     cudnnFusedOpsVariantParamPack_t varPack = in->Get<cudnnFusedOpsVariantParamPack_t>();

     cudnnStatus_t cs = cudnnDestroyFusedOpsVariantParamPack(varPack);

     
     LOG4CPLUS_DEBUG(logger, "cudnnDestroyFusedOpsVariantParamPack Executed");
     
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetFusedOpsVariantParamPackAttribute) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFusedOpsVariantParamPackAttribute"));

    cudnnFusedOpsVariantParamPack_t varPack = in->Get<cudnnFusedOpsVariantParamPack_t>();
    cudnnFusedOpsVariantParamLabel_t paramLabel = in->Get<cudnnFusedOpsVariantParamLabel_t>();
    void *ptr = in->Assign<void>();

    cudnnStatus_t cs = cudnnSetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr);

     LOG4CPLUS_DEBUG(logger, "cudnnSetFusedOpsVariantParamPackAttribute Executed");   
    
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetFusedOpsVariantParamPackAttribute) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFusedOpsVariantParamPackAttribute"));

    cudnnFusedOpsVariantParamPack_t varPack = in->Get<cudnnFusedOpsVariantParamPack_t>();
    cudnnFusedOpsVariantParamLabel_t paramLabel = in->Get<cudnnFusedOpsVariantParamLabel_t>();
    void *ptr;

    cudnnStatus_t cs = cudnnGetFusedOpsVariantParamPackAttribute(varPack, paramLabel, ptr);

     std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
      try {
          out->Add<void>(ptr);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
         return std::make_shared<Result>(cs);
    }
    
    LOG4CPLUS_DEBUG(logger, "cudnnGetFusedOpsVariantParamPackAttribute Executed");   
    
    return std::make_shared<Result>(cs, out);
}

CUDNN_ROUTINE_HANDLER(CreateFusedOpsPlan) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateFusedOpsPlan"));

     cudnnFusedOpsPlan_t plan = in->Get<cudnnFusedOpsPlan_t>();
     cudnnFusedOps_t ops = in->Get<cudnnFusedOps_t>();

     cudnnStatus_t cs = cudnnCreateFusedOpsPlan(&plan, ops);

     LOG4CPLUS_DEBUG(logger, "cudnnCreateFusedOpsPlan Executed");    
     
     return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(DestroyFusedOpsPlan) {
     Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyFusedOpsPlan"));

     cudnnFusedOpsPlan_t plan = in->Get<cudnnFusedOpsPlan_t>();
 
    cudnnStatus_t cs = cudnnDestroyFusedOpsPlan(plan);

    
    LOG4CPLUS_DEBUG(logger, "cudnnCreateFusedOpsPlan Executed"); 
    
    return std::make_shared<Result>(cs);   
}

CUDNN_ROUTINE_HANDLER(MakeFusedOpsPlan) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MakeFusedOpsPlan"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnFusedOpsPlan_t plan = in->Get<cudnnFusedOpsPlan_t>();
    cudnnFusedOpsConstParamPack_t constPack = in->Get<cudnnFusedOpsConstParamPack_t>();
    size_t workspaceSizeInBytes;

    cudnnStatus_t cs = cudnnMakeFusedOpsPlan(handle, plan, constPack, &workspaceSizeInBytes);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<size_t>(workspaceSizeInBytes);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
        return std::make_shared<Result>(cs);
    }

    LOG4CPLUS_DEBUG(logger, "cudnnMakeFusedOpsPlan Executed");

return std::make_shared<Result>(cs, out);   
}

CUDNN_ROUTINE_HANDLER(FusedOpsExecute) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FusedOpsExecute"));

    cudnnHandle_t handle = in->Get<cudnnHandle_t>();
    cudnnFusedOpsPlan_t plan = in->Get<cudnnFusedOpsPlan_t>();
    cudnnFusedOpsVariantParamPack_t varPack = in->Get<cudnnFusedOpsVariantParamPack_t>();

    cudnnStatus_t cs = cudnnFusedOpsExecute(handle, plan, varPack);

    LOG4CPLUS_DEBUG(logger, "cudnnFusedOpsExecute Executed"); 
    
    return std::make_shared<Result>(cs);   
}

#if CUDNN_VERSION < 9000
    CUDNN_ROUTINE_HANDLER(SetRNNMatrixMathType) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNMatrixMathType"));
        
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnMathType_t mType = in->Get<cudnnMathType_t>();
        
        cudnnStatus_t cs = cudnnSetRNNMatrixMathType(rnnDesc, mType);

        
        LOG4CPLUS_DEBUG(logger, "cudnnSetRNNMatrixMathType Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNMatrixMathType) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNMatrixMathType"));

        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnMathType_t mType;

        cudnnStatus_t cs = cudnnGetRNNMatrixMathType(rnnDesc, &mType);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnMathType_t>(mType);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNMatrixMathType Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(SetRNNBiasMode) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNBiasMode"));

        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>(); 
        cudnnRNNBiasMode_t biasMode = in->Get<cudnnRNNBiasMode_t>();
        
        cudnnStatus_t cs = cudnnSetRNNBiasMode(rnnDesc, biasMode);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnRNNDescriptor_t>(rnnDesc);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnSetRNNBiasMode Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNBiasMode) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNBiasMode"));

        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnRNNBiasMode_t biasMode;

        cudnnStatus_t  cs = cudnnGetRNNBiasMode(rnnDesc, &biasMode);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnRNNBiasMode_t>(biasMode);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNBiasMode Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(RNNSetClip) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNSetClip"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnRNNClipMode_t clipMode = in->Get<cudnnRNNClipMode_t>();
        cudnnNanPropagation_t clipNanOpt = in->Get<cudnnNanPropagation_t>();
        double lclip = in->Get<double>();
        double rclip = in->Get<double>();

        cudnnStatus_t cs = cudnnRNNSetClip(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
        
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnRNNDescriptor_t>(rnnDesc);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnRNNSetClip Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(RNNGetClip) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNGetClip"));
    
        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnRNNClipMode_t clipMode;
        cudnnNanPropagation_t clipNanOpt;
        double lclip;
        double rclip;

        cudnnStatus_t cs = cudnnRNNGetClip(handle, rnnDesc, &clipMode, &clipNanOpt, &lclip, &rclip);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnRNNClipMode_t>(clipMode);
            out->Add<cudnnNanPropagation_t>(clipNanOpt);
            out->Add<double>(lclip);
            out->Add<double>(rclip);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnRNNGetClip Execute");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(SetRNNProjectionLayers) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNProjectionLayers"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int recProjSize = in->Get<int>();
        int outProjSize = in->Get<int>();

        cudnnStatus_t cs = cudnnSetRNNProjectionLayers(handle, rnnDesc, recProjSize, outProjSize);

        

        LOG4CPLUS_DEBUG(logger, "cudnnSetRNNProjectionLayers Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNProjectionLayers) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNProjectionLayers"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int recProjSize;
        int outProjSize;

        cudnnStatus_t cs = cudnnGetRNNProjectionLayers(handle, rnnDesc, &recProjSize, &outProjSize);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<int>(recProjSize);
            out->Add<int>(outProjSize);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNProjectionLayers Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(CreatePersistentRNNPlan) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreatePersistentRNNPlan"));

        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int minibatch = in->Get<int>();
        cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
        cudnnPersistentRNNPlan_t plan;

        cudnnStatus_t cs = cudnnCreatePersistentRNNPlan(rnnDesc, minibatch, dataType, &plan);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnPersistentRNNPlan_t>(plan);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnCreatePersistentRNNPlan Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(DestroyPersistentRNNPlan) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyPersistentRNNPlan"));

        cudnnPersistentRNNPlan_t plan;
        
        cudnnStatus_t cs = cudnnDestroyPersistentRNNPlan(plan);

        
        LOG4CPLUS_DEBUG(logger, "cudnnDestroyPersistentRNNPlan Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(SetPersistentRNNPlan) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetPersistentRNNPlan"));
    
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnPersistentRNNPlan_t plan = in->Get<cudnnPersistentRNNPlan_t>();
    
        cudnnStatus_t cs = cudnnSetPersistentRNNPlan(rnnDesc, plan);

        
        LOG4CPLUS_DEBUG(logger, "cudnnSetPersistentRNNPlan Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNWorkspaceSize) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNWorkspaceSize"));
        
        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int seqLength = in->Get<int>();
        cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        size_t sizeInBytes;

        cudnnStatus_t cs = cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, &xDesc, &sizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<size_t>(sizeInBytes);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNWorkspaceSize Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNTrainingReserveSize) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNTrainingReserveSize"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int seqLength = in->Get<int>();
        cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        size_t sizeInBytes;

        cudnnStatus_t cs = cudnnGetRNNTrainingReserveSize(handle, rnnDesc, seqLength, &xDesc, &sizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<size_t>(sizeInBytes);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNTrainingReserveSize Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNParamsSize) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNParamsSize"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        size_t sizeInBytes;
        cudnnDataType_t dataType = in->Get<cudnnDataType_t>();

        cudnnStatus_t cs = cudnnGetRNNParamsSize(handle, rnnDesc, xDesc, &sizeInBytes, dataType);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<size_t>(sizeInBytes);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNParamsSize Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNLinLayerMatrixParams) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNLinLayerMatrixParams"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int pseudoLayer = in->Get<int>();
        cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        void *w = in->Assign<void>();
        int linLayerID = in->Get<int>();
        cudnnFilterDescriptor_t linLayerMatDesc;
        void *linLayerMat = in->Assign<void>();

        cudnnStatus_t cs = cudnnGetRNNLinLayerMatrixParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, &linLayerMat);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnFilterDescriptor_t>(linLayerMatDesc);
            out->Add<void>(linLayerMat);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNLinLayerMatrixParams Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNLinLayerBiasParams) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNLinLayerBiasParams")); 
        
        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int pseudoLayer = in->Get<int>();
        cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        void *w = in->Assign<void>();
        int linLayerID = in->Get<int>();
        cudnnFilterDescriptor_t linLayerBiasDesc;
        void *linLayerBias;

        cudnnStatus_t cs = cudnnGetRNNLinLayerBiasParams(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, &linLayerBias);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnFilterDescriptor_t>(linLayerBiasDesc);
            out->Add<void>(linLayerBias);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNLinLayerBiasParams Executed");
        
        return std::make_shared<Result>(cs, out);  
    }

    CUDNN_ROUTINE_HANDLER(RNNForwardInference) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNForwardInference"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int seqLength = in->Get<int>();
        cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        void *x = in->Assign<void>();
        cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hx = in->Assign<void>();
        cudnnTensorDescriptor_t cxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *cx = in->Assign<void>();
        cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        void* w = in->Assign<void>();
        cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
        void *y = in->Assign<void>();
        cudnnTensorDescriptor_t hyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hy = in->Assign<void>();
        cudnnTensorDescriptor_t cyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *cy = in->Assign<void>();
        void *workspace = in->Assign<void>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnRNNForwardInference(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, &yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes);
        
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<void>(y);
            out->Add<void>(hy);
            out->Add<void>(cy);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, " cudnnRNNForwardInference Executed");
        
        return std::make_shared<Result>(cs, out);   
    }

    CUDNN_ROUTINE_HANDLER(RNNForwardTraining) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNForwardTraining"));
        
        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int seqLength = in->Get<int>();
        cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        void *x = in->Assign<void>();
        cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hx = in->Assign<void>();
        cudnnTensorDescriptor_t cxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *cx = in->Assign<void>();
        cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        void *w = in->Assign<void>();
        cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
        void *y = in->Assign<void>();
        cudnnTensorDescriptor_t hyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hy = in->Assign<void>();
        cudnnTensorDescriptor_t cyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *cy = in->Assign<void>();
        void *workspace = in->Assign<void>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();
        void *reserveSpace = in->Assign<void>();
        size_t reserveSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnRNNForwardTraining(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, &yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<void>(y);
            out->Add<void>(hy);
            out->Add<void>(cy);
            out->Add<void>(reserveSpace);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnRNNForwardTraining Executed");
        
        return std::make_shared<Result>(cs, out);         
    }

    CUDNN_ROUTINE_HANDLER(RNNBackwardData) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNBackwardData"));
        
        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int seqLength = in->Get<int>();
        cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
        void *y = in->Assign<void>();
        cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dy = in->Assign<void>();
        cudnnTensorDescriptor_t dhyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dhy = in->Assign<void>();
        cudnnTensorDescriptor_t dcyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dcy = in->Assign<void>();
        cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        void *w = in->Assign<void>();
        cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hx = in->Assign<void>();
        cudnnTensorDescriptor_t cxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *cx = in->Assign<void>();
        cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dx = in->Assign<void>();
        cudnnTensorDescriptor_t dhxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dhx = in->Assign<void>();
        cudnnTensorDescriptor_t dcxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dcx = in->Assign<void>();
        void *workspace = in->Assign<void>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();
        void *reserveSpace = in->Assign<void>();
        size_t reserveSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnRNNBackwardData(handle, rnnDesc, seqLength, &yDesc, y, &dyDesc, dy,dhyDesc, dhy,  dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, &dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<void>(dx);
            out->Add<void>(dhx);
            out->Add<void>(dcx);
            out->Add<void>(reserveSpace);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnRNNBackwardData Executed");
        
        return std::make_shared<Result>(cs, out);   
    }

    CUDNN_ROUTINE_HANDLER(RNNBackwardWeights) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNBackwardWeights"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int seqLength = in->Get<int>();
        cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        void *x = in->Assign<void>();
        cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hx = in->Assign<void>();
        cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
        void *y = in->Assign<void>();
        void *workspace = in->Assign<void>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();
        cudnnFilterDescriptor_t dwDesc = in->Get<cudnnFilterDescriptor_t>();
        void *dw = in->Assign<void>();
        void *reserveSpace = in->Assign<void>();
        size_t reserveSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnRNNBackwardWeights(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, &yDesc, y, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<void>(dw);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnRNNBackwardWeights Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(SetRNNPaddingMode) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNPaddingMode"));

        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnRNNPaddingMode_t paddingMode = in->Get<cudnnRNNPaddingMode_t>();
        
        cudnnStatus_t cs = cudnnSetRNNPaddingMode(rnnDesc, paddingMode);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnRNNDescriptor_t>(rnnDesc);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnSetRNNPaddingMode Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNPaddingMode) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNPaddingMode"));

        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnRNNPaddingMode_t paddingMode = in->Get<cudnnRNNPaddingMode_t>();

        cudnnStatus_t cs = cudnnGetRNNPaddingMode(rnnDesc, &paddingMode);
        
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnRNNDescriptor_t>(rnnDesc);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNPaddingMode Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(RNNForwardTrainingEx) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNForwardTrainingEx"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnRNNDataDescriptor_t xDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *x = in->Assign<void>();
        cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hx = in->Assign<void>();
        cudnnTensorDescriptor_t cxDesc =in->Get<cudnnTensorDescriptor_t>();
        void *cx = in->Assign<void>();
        cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        void *w = in->Assign<void>();
        cudnnRNNDataDescriptor_t yDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *y = in->Assign<void>(); //OUTPUT
        cudnnTensorDescriptor_t hyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hy = in->Assign<void>(); //OUTPUT
        cudnnTensorDescriptor_t cyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *cy = in->Assign<void>(); //OUTPUT
        cudnnRNNDataDescriptor_t kDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *keys = in->Assign<void>();
        cudnnRNNDataDescriptor_t cDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *cAttn = in->Assign<void>();
        cudnnRNNDataDescriptor_t iDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *iAttn = in->Assign<void>();
        cudnnRNNDataDescriptor_t qDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *queries = in->Assign<void>();
        void *workSpace = in->Assign<void>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();
        void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT
        size_t reserveSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnRNNForwardTrainingEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<void>(y);
            out->Add<void>(hy);
            out->Add<void>(cy);
            out->Add<void>(reserveSpace);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, " cudnnRNNForwardTrainingEx Executed");
        
        return std::make_shared<Result>(cs, out);   
    }

    CUDNN_ROUTINE_HANDLER(RNNForwardInferenceEx) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNForwardInferenceEx"));
        
        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnRNNDataDescriptor_t xDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *x = in->Assign<void>();
        cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hx = in->Assign<void>();
        cudnnTensorDescriptor_t cxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *cx = in->Assign<void>();
        cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        void *w = in->Assign<void>();
        cudnnRNNDataDescriptor_t yDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *y = in->Assign<void>(); //OUTPUT
        cudnnTensorDescriptor_t hyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hy = in->Assign<void>(); //OUTPUT
        cudnnTensorDescriptor_t cyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *cy = in->Assign<void>(); //OUTPUT
        cudnnRNNDataDescriptor_t kDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *keys = in->Assign<void>();
        cudnnRNNDataDescriptor_t cDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *cAttn = in->Assign<void>();
        cudnnRNNDataDescriptor_t iDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *iAttn = in->Assign<void>();
        cudnnRNNDataDescriptor_t qDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *queries = in->Assign<void>();
        void *workSpace = in->Assign<void>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnRNNForwardInferenceEx(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<void>(y);
            out->Add<void>(hy);
            out->Add<void>(cy);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnRNNForwardInferenceEx Executed");
        
        return std::make_shared<Result>(cs, out);   
    }

    CUDNN_ROUTINE_HANDLER(RNNBackwardDataEx) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNBackwardDataEx"));
    
        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnRNNDataDescriptor_t yDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *y = in->Assign<void>();
        cudnnRNNDataDescriptor_t dyDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *dy = in->Assign<void>();
        cudnnRNNDataDescriptor_t dcDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *dcAttn = in->Assign<void>();
        cudnnTensorDescriptor_t dhyDesc = in->Get<cudnnTensorDescriptor_t>();
        void * dhy = in->Assign<void>();
        cudnnTensorDescriptor_t dcyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dcy = in->Assign<void>();
        cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        void *w = in->Assign<void>();
        cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hx = in->Assign<void>();
        cudnnTensorDescriptor_t cxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *cx = in->Assign<void>();
        cudnnRNNDataDescriptor_t dxDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *dx = in->Assign<void>(); //OUTPUT
        cudnnTensorDescriptor_t dhxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dhx = in->Assign<void>(); //OUTPUT
        cudnnTensorDescriptor_t dcxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dcx = in->Assign<void>(); //OUTPUT
        cudnnRNNDataDescriptor_t dkDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *dkeys = in->Assign<void>();
        void *workSpace = in->Assign<void>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();
        void *reserveSpace = in->Assign<void>(); //INPUT/OUTPUT
        size_t reserveSpaceSizeInBytes = in->Get<size_t>();
    
        cudnnStatus_t cs = cudnnRNNBackwardDataEx(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<void>(dx);
            out->Add<void>(dhx);
            out->Add<void>(dcx);
            out->Add<void>(reserveSpace);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnRNNBackwardDataEx Executed");
        
        return std::make_shared<Result>(cs, out);   
    }
    
    CUDNN_ROUTINE_HANDLER(RNNBackwardWeightsEx) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RNNBackwardWeightsEx"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnRNNDataDescriptor_t xDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *x = in->Assign<void>();
        cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hx = in->Assign<void>();
        cudnnRNNDataDescriptor_t yDesc = in->Get<cudnnRNNDataDescriptor_t>();
        void *y = in->Assign<void>();
        void *workSpace = in->Assign<void>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();
        cudnnFilterDescriptor_t dwDesc = in->Get<cudnnFilterDescriptor_t>();
        void *dw = in->Assign<void>(); //INPUT/OUTPUT
        void *reserveSpace = in->Assign<void>();
        size_t reserveSpaceSizeInBytes = in->Get<size_t>();
        
        cudnnStatus_t cs = cudnnRNNBackwardWeightsEx(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<void>(dw);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnRNNBackwardWeightsEx Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(SetRNNAlgorithmDescriptor) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNAlgorithmDescriptor"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        cudnnAlgorithmDescriptor_t algoDesc = in->Get<cudnnAlgorithmDescriptor_t>();

        cudnnStatus_t cs = cudnnSetRNNAlgorithmDescriptor(handle, rnnDesc, algoDesc);
        LOG4CPLUS_DEBUG(logger, "cudnnSetRNNAlgorithmDescriptor Executed");

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnAlgorithmDescriptor_t>(algoDesc);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNForwardInferenceAlgorithmMaxCount) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNForwardInferenceAlgorithmMaxCount"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int count;

        cudnnStatus_t cs = cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle, rnnDesc, &count);
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNForwardInferenceAlgorithmMaxCount Executed");

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<int>(count);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(FindRNNForwardInferenceAlgorithmEx) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindRNNForwardInferenceAlgorithmEx"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        const cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        const int seqLength = in->Get<int>();
        const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *x = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *hx = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t cxDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *cx = in->GetFromMarshal<void*>();
        const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        const void *w = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
        void *y = in->GetFromMarshal<void*>(); //OUTPUT
        const cudnnTensorDescriptor_t hyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hy = in->GetFromMarshal<void*>(); //OUTPUT
        const cudnnTensorDescriptor_t cyDesc = in->Get<cudnnTensorDescriptor_t>();
        void* cy = in->GetFromMarshal<void*>(); //OUTPUT
        const float findIntensity =  in->Get<float>();
        const int requestedAlgoCount = in->Get<int>();
        cudnnAlgorithmPerformance_t* perfResults = in->Get<cudnnAlgorithmPerformance_t>(requestedAlgoCount); //OUTPUT
        int returnedAlgoCount; //OUTPUT
        void *workspace = in->GetFromMarshal<void*>();
        size_t workSpaceSizeInBytes =in->Get<size_t>();

        cudnnStatus_t cs = cudnnFindRNNForwardInferenceAlgorithmEx(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, &yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, &returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes);
        LOG4CPLUS_DEBUG(logger, "cudnnFindRNNForwardInferenceAlgorithmEx Executed");

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->AddMarshal<void*>(y);
            out->AddMarshal<void*>(hy);
            out->AddMarshal<void*>(cy);
            out->Add<int>(returnedAlgoCount);
            out->Add<cudnnAlgorithmPerformance_t>(perfResults, returnedAlgoCount);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
            
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNForwardTrainingAlgorithmMaxCount) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNForwardTrainingAlgorithmMaxCount"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int count;

        cudnnStatus_t cs = cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle, rnnDesc, &count);
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNForwardTrainingAlgorithmMaxCount Executed");

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<int>(count);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }

        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(FindRNNForwardTrainingAlgorithmEx) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindRNNForwardTrainingAlgorithmEx"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        const cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        const int seqLength = in->Get<int>();
        const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *x = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *hx = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t cxDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *cx = in->GetFromMarshal<void*>();
        const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        const void *w = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
        void *y = in->GetFromMarshal<void*>(); //OUTPUT
        const cudnnTensorDescriptor_t hyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *hy = in->GetFromMarshal<void*>(); //OUTPUT
        const cudnnTensorDescriptor_t cyDesc = in->Get<cudnnTensorDescriptor_t>();
        void *cy = in->GetFromMarshal<void*>(); //OUTPUT
        const float findIntensity = in->Get<float>();
        const int requestedAlgoCount = in->Get<int>();
        int returnedAlgoCount; //OUTPUT
        cudnnAlgorithmPerformance_t* perfResults = in->Get<cudnnAlgorithmPerformance_t>(requestedAlgoCount); //OUTPUT
        void *workspace = in->GetFromMarshal<void*>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();
        void *reserveSpace = in->GetFromMarshal<void*>(); //INPUT/OUTPUT
        size_t reserveSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnFindRNNForwardTrainingAlgorithmEx(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, &yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount, &returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
        LOG4CPLUS_DEBUG(logger, "cudnnFindRNNForwardTrainingAlgorithmEx Executed");

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
            try {
                out->AddMarshal<void*>(y);
                out->AddMarshal<void*>(hy);
                out->AddMarshal<void*>(cy);
                out->AddMarshal<int>(returnedAlgoCount);
                out->Add<cudnnAlgorithmPerformance_t>(perfResults, returnedAlgoCount);
                out->AddMarshal<void*>(reserveSpace);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }

        return std::make_shared<Result>(cs, out);    
    }

    CUDNN_ROUTINE_HANDLER(GetRNNBackwardDataAlgorithmMaxCount) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNBackwardDataAlgorithmMaxCount"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int count;

        cudnnStatus_t cs = cudnnGetRNNBackwardDataAlgorithmMaxCount(handle, rnnDesc, &count);
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNBackwardDataAlgorithmMaxCount Executed");

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<int>(count);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }

        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(FindRNNBackwardDataAlgorithmEx) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindRNNBackwardDataAlgorithmEx"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cont cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        const int seqLength = in->Get<int>();
        const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *y = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t dyDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *dy = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t dhyDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *dhy = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t dcyDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *dcy = in->GetFromMarshal<void*>();
        const cudnnFilterDescriptor_t wDesc = in->Get<cudnnFilterDescriptor_t>();
        const void *w = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *hx = in->Assign<void>();
        const cudnnTensorDescriptor_t cxDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *cx = in->Assign<void>();
        const cudnnTensorDescriptor_t dxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dx = in->Assign<void>(); //OUTPUT
        const cudnnTensorDescriptor_t dhxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dhx = in->Assign<void>(); //OUTPUT
        const cudnnTensorDescriptor_t dcxDesc = in->Get<cudnnTensorDescriptor_t>();
        void *dcx = in->Assign<void>(); //OUTPUT
        const float findIntensity = in->Get<float>();
        const int requestedAlgoCount = in->Get<int>();
        int returnedAlgoCount; //OUTPUT
        cudnnAlgorithmPerformance_t* perfResults = in->Get<cudnnAlgorithmPerformance_t>(requestedAlgoCount);; //OUTPUT
        void *workspace = in->GetFromMarshal<void*>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();
        void *reserveSpace = in->GetFromMarshal<void*>(); //INPUT/OUTPUT
        size_t reserveSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnFindRNNBackwardDataAlgorithmEx(handle, rnnDesc, seqLength, &yDesc, y, &dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, &dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, findIntensity, requestedAlgoCount, &returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
        LOG4CPLUS_DEBUG(logger, "cudnnFindRNNBackwardDataAlgorithmEx Executed");

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->AddMarshal<void*>(dx);
            out->AddMarshal<void*>(dhx);
            out->AddMarshal<void*>(dcx);
            out->Add<int>(returnedAlgoCount);
            out->AddMarshal<cudnnAlgorithmPerformance_t>(perfResults, returnedAlgoCount);
            out->AddMarshal<void*>(reserveSpace);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }

        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(GetRNNBackwardWeightsAlgorithmMaxCount) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNBackwardWeightsAlgorithmMaxCount"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int count;

        cudnnStatus_t cs = cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle, rnnDesc, &count);
        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNBackwardWeightsAlgorithmMaxCount Executed");

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<int>(count);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(FindRNNBackwardWeightsAlgorithmEx) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("FindRNNBackwardWeightsAlgorithmEx"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        const cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        const int seqLength = in->Get<int>();
        const cudnnTensorDescriptor_t xDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *x = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t hxDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *hx = in->GetFromMarshal<void*>();
        const cudnnTensorDescriptor_t yDesc = in->Get<cudnnTensorDescriptor_t>();
        const void *y = in->GetFromMarshal<void*>();
        const float findIntensity = in->Get<float>();
        const int requestedAlgoCount = in->Get<int>();
        int returnedAlgoCount; //OUTPUT
        cudnnAlgorithmPerformance_t* perfResults = in->Get<cudnnAlgorithmPerformance_t>(requestedAlgoCount); //OUTPUT
        const void *workspace = in->GetFromMarshal<void*>();
        size_t workSpaceSizeInBytes = in->Get<size_t>();
        cudnnFilterDescriptor_t dwDesc = in->Get<cudnnFilterDescriptor_t>();
        void *dw = in->GetFromMarshal<void*>(); //INPUT/OUTPUT
        const void *reserveSpace = in->GetFromMarshal<void*>();
        size_t reserveSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnFindRNNBackwardWeightsAlgorithmEx(handle, rnnDesc, seqLength, &xDesc, x, hxDesc, hx, &yDesc, y, findIntensity, requestedAlgoCount, &returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
        LOG4CPLUS_DEBUG(logger, "cudnnFindRNNBackwardWeightsAlgorithmEx Executed");

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<int>(returnedAlgoCount);
            out->Add<cudnnAlgorithmPerformance_t>(perfResults, returnedAlgoCount);
            out->AddMarshal<void*>(dw);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }

        return std::make_shared<Result>(cs, out);  
    }

    CUDNN_ROUTINE_HANDLER(CreateAlgorithmDescriptor) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateAlgorithmDescriptor"));

        cudnnAlgorithmDescriptor_t algoDesc;

        cudnnStatus_t cs = cudnnCreateAlgorithmDescriptor(&algoDesc);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnAlgorithmDescriptor_t>(algoDesc);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnCreateAlgorithmDescriptor Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(SetAlgorithmDescriptor) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetAlgorithmDescriptor"));

        cudnnAlgorithmDescriptor_t algoDesc = in->Get<cudnnAlgorithmDescriptor_t>();
        cudnnAlgorithm_t algorithm = in->Get<cudnnAlgorithm_t>();

        cudnnStatus_t cs = cudnnSetAlgorithmDescriptor(algoDesc, algorithm);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnAlgorithmDescriptor_t>(algoDesc);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnSetAlgorithmDescriptor Executed"); 
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(GetAlgorithmDescriptor) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetAlgorithmDescriptor"));

        cudnnAlgorithmDescriptor_t algoDesc = in->Get<cudnnAlgorithmDescriptor_t>();
        cudnnAlgorithm_t algorithm = in->Get<cudnnAlgorithm_t>();
        
        cudnnStatus_t cs = cudnnGetAlgorithmDescriptor(algoDesc, &algorithm);

        LOG4CPLUS_DEBUG(logger, "cudnnGetAlgorithmDescriptor Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(CopyAlgorithmDescriptor) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CopyAlgorithmDescriptor"));

        cudnnAlgorithmDescriptor_t src = in->Get<cudnnAlgorithmDescriptor_t>();
        cudnnAlgorithmDescriptor_t dest = in->Get<cudnnAlgorithmDescriptor_t>();

        cudnnStatus_t cs = cudnnCopyAlgorithmDescriptor(src, dest);

        
        LOG4CPLUS_DEBUG(logger, "cudnnCopyAlgorithmDescriptor Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(DestroyAlgorithmDescriptor) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyAlgorithmDescriptor"));

        cudnnAlgorithmDescriptor_t algoDesc = in->Get<cudnnAlgorithmDescriptor_t>();
        
        cudnnStatus_t cs = cudnnDestroyAlgorithmDescriptor(algoDesc);

        
        LOG4CPLUS_DEBUG(logger, "cudnnDestroyAlgorithmDescriptor Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(CreateAlgorithmPerformance) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateAlgorithmPerformance"));

        cudnnAlgorithmPerformance_t algoPerf;
        int numberToCreate = in->Get<int>();

        cudnnStatus_t cs = cudnnCreateAlgorithmPerformance(&algoPerf, numberToCreate);
        
        
        LOG4CPLUS_DEBUG(logger, "cudnnCreateAlgorithmPerformance Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(SetAlgorithmPerformance) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetAlgorithmPerformance"));

        cudnnAlgorithmPerformance_t algoPerf = in->Get<cudnnAlgorithmPerformance_t>();
        cudnnAlgorithmDescriptor_t algoDesc = in->Get<cudnnAlgorithmDescriptor_t>();
        cudnnStatus_t status = in->Get<cudnnStatus_t>();
        float time = in->Get<float>();
        size_t memory = in->Get<size_t>();

        cudnnStatus_t cs = cudnnSetAlgorithmPerformance(algoPerf, algoDesc, status, time, memory);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnAlgorithmPerformance_t>(algoPerf);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnSetAlgorithmPerformance Executed");
        
        return std::make_shared<Result>(cs, out);    
    }

    CUDNN_ROUTINE_HANDLER(GetAlgorithmPerformance) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetAlgorithmPerformance"));

        cudnnAlgorithmPerformance_t algoPerf = in->Get<cudnnAlgorithmPerformance_t>();
        cudnnAlgorithmDescriptor_t algoDesc;
        cudnnStatus_t status;
        float time;
        size_t memory;

        cudnnStatus_t cs = cudnnGetAlgorithmPerformance(algoPerf, &algoDesc, &status, &time, &memory);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnAlgorithmPerformance_t>(algoPerf);
            out->Add<cudnnAlgorithmDescriptor_t>(algoDesc);
            out->Add<cudnnStatus_t>(status);
            out->Add<float>(time);
            out->Add<size_t>(memory);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetAlgorithmPerformance Executed"); 
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(DestroyAlgorithmPerformance) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyAlgorithmPerformance"));

        cudnnAlgorithmPerformance_t algoPerf = in->Get<cudnnAlgorithmPerformance_t>();
        
        cudnnStatus_t cs = cudnnDestroyAlgorithmPerformance(&algoPerf);
        LOG4CPLUS_DEBUG(logger, "cudnnDestroyAlgorithmPerformance Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(GetAlgorithmSpaceSize) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetAlgorithmSpaceSize"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnAlgorithmDescriptor_t algoDesc = in->Get<cudnnAlgorithmDescriptor_t>();
        size_t algoSpaceSizeInBytes;

        cudnnStatus_t cs = cudnnGetAlgorithmSpaceSize(handle, algoDesc, &algoSpaceSizeInBytes);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<size_t>(algoSpaceSizeInBytes);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnGetAlgorithmSpaceSize Executed");
        
        return std::make_shared<Result>(cs, out);
    }

    CUDNN_ROUTINE_HANDLER(SaveAlgorithm) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SaveAlgorithm"));  

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnAlgorithmDescriptor_t algoDesc = in->Get<cudnnAlgorithmDescriptor_t>();
        void *algoSpace = in->Assign<void>();
        size_t algoSpaceSizeInBytes = in->Get<size_t>();

        cudnnStatus_t cs = cudnnSaveAlgorithm(handle, algoDesc, algoSpace, algoSpaceSizeInBytes);

        LOG4CPLUS_DEBUG(logger, "cudnnSaveAlgorithm Executed");
        
        return std::make_shared<Result>(cs);
    }

    CUDNN_ROUTINE_HANDLER(RestoreAlgorithm) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("RestoreAlgorithm"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        void *algoSpace = in->Assign<void>();
        size_t algoSpaceSizeInBytes = in->Get<size_t>();
        cudnnAlgorithmDescriptor_t algoDesc = in->Get<cudnnAlgorithmDescriptor_t>();

        cudnnStatus_t cs = cudnnRestoreAlgorithm(handle, algoSpace, algoSpaceSizeInBytes, algoDesc);

        
        LOG4CPLUS_DEBUG(logger, "cudnnRestoreAlgorithm Executed");
        
        return std::make_shared<Result>(cs);
    }
#endif

#if CUDNN_VERSION >= 6000 && CUDNN_VERSION < 9000
    CUDNN_ROUTINE_HANDLER(SetRNNDescriptor_v6) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetRNNDescriptor_v6"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int hiddenSize = in->Get<int>();
        int numLayers  = in->Get<int>();
        cudnnDropoutDescriptor_t dropoutDesc = in->Get<cudnnDropoutDescriptor_t>();
        cudnnRNNInputMode_t inputMode = in->Get<cudnnRNNInputMode_t>();
        cudnnDirectionMode_t direction = in->Get<cudnnDirectionMode_t>();
        cudnnRNNMode_t mode = in->Get<cudnnRNNMode_t>();
        cudnnRNNAlgo_t algo = in->Get<cudnnRNNAlgo_t>();
        cudnnDataType_t mathPrec = in->Get<cudnnDataType_t>();

        cudnnStatus_t cs = cudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, mathPrec);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<cudnnRNNDescriptor_t>(rnnDesc);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }
        
        LOG4CPLUS_DEBUG(logger, "cudnnSetRNNDescriptor_v6 Executed");
        
        return std::make_shared<Result>(cs, out);         
    }

    CUDNN_ROUTINE_HANDLER(GetRNNDescriptor_v6) {
        Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetRNNDescriptor_v6"));

        cudnnHandle_t handle = in->Get<cudnnHandle_t>();
        cudnnRNNDescriptor_t rnnDesc = in->Get<cudnnRNNDescriptor_t>();
        int hiddenSize;
        int numLayers;
        cudnnDropoutDescriptor_t dropoutDesc;
        cudnnRNNInputMode_t inputMode;
        cudnnDirectionMode_t direction;
        cudnnRNNMode_t mode;
        cudnnRNNAlgo_t algo;
        cudnnDataType_t mathPrec;

        cudnnStatus_t cs = cudnnGetRNNDescriptor_v6(handle, rnnDesc, &hiddenSize, &numLayers, &dropoutDesc, &inputMode, &direction, &mode, &algo, &mathPrec);

        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        try {
            out->Add<int>(hiddenSize);
            out->Add<int>(numLayers);
            out->Add<cudnnDropoutDescriptor_t>(dropoutDesc);
            out->Add<cudnnRNNInputMode_t>(inputMode);
            out->Add<cudnnDirectionMode_t>(direction);
            out->Add<cudnnRNNMode_t>(mode);
            out->Add<cudnnRNNAlgo_t>(algo);
            out->Add<cudnnDataType_t>(mathPrec);
        } catch (const std::exception& e) {
            LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
            return std::make_shared<Result>(cs);
        }

        LOG4CPLUS_DEBUG(logger, "cudnnGetRNNDescriptor_v6 Executed");
        
        return std::make_shared<Result>(cs, out);
    }
#endif