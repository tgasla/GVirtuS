/*
 * GVirtuS - A Virtualization Framework for GPU-Accelerated Applications
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDrHandler.h"

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

using namespace log4cplus;

CUDA_DRIVER_HANDLER(PointerGetAttribute) {
    CUpointer_attribute attribute = input_buffer->Get<CUpointer_attribute>();
    CUdeviceptr ptr = input_buffer->Get<CUdeviceptr>();

    char data[sizeof(void*)];

    CUresult exit_code = cuPointerGetAttribute(data, attribute, ptr);

    if (exit_code == CUDA_SUCCESS) {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add(data, sizeof(void*));
        return std::make_shared<Result>((cudaError_t)exit_code, out);
    }

    return std::make_shared<Result>((cudaError_t)exit_code);
}