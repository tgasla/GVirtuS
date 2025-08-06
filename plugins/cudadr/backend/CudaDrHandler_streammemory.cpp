/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications 
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDrHandler.h"

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

using namespace log4cplus;

CUDA_DRIVER_HANDLER(StreamWriteValue32) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("StreamWriteValue32"));
    
    CUstream stream = input_buffer->Get<CUstream>();
    CUdeviceptr addr = input_buffer->Get<CUdeviceptr>();
    cuuint32_t value = input_buffer->Get<cuuint32_t>();
    unsigned int flags = input_buffer->Get<unsigned int>();

    CUresult exit_code = cuStreamWriteValue32(stream, addr, value, flags);
    
    LOG4CPLUS_DEBUG(logger, "cuStreamWriteValue32 executed for stream: " << stream << ", addr: " << addr << ", value: " << value);
    return std::make_shared<Result>((cudaError_t) exit_code);
}