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
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(StreamCreate) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        cudaStream_t pStream;
        cudaError_t exit_code = cudaStreamCreate(&pStream);
        out->Add<cudaStream_t>(pStream);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamCreateWithPriority) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        cudaStream_t pStream;
        unsigned int flags = input_buffer->Get<unsigned int>();
        int priority = input_buffer->Get<int>();
        cudaError_t exit_code = cudaStreamCreateWithPriority(&pStream, flags, priority);
        out->Add((pointer_t)pStream);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamCreateWithFlags) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        cudaStream_t pStream;
        unsigned int flags = input_buffer->Get<unsigned int>();
        cudaError_t exit_code = cudaStreamCreateWithFlags(&pStream, flags);
        out->Add<cudaStream_t>(pStream);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamDestroy) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return std::make_shared<Result>(cudaStreamDestroy(stream));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamWaitEvent) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
        unsigned int flags = input_buffer->Get<unsigned int>();
        return std::make_shared<Result>(cudaStreamWaitEvent(stream, event, flags));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamQuery) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return std::make_shared<Result>(cudaStreamQuery(stream));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamSynchronize) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return std::make_shared<Result>(cudaStreamSynchronize(stream));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(ThreadExchangeStreamCaptureMode) {
    try {
        cudaStreamCaptureMode mode = input_buffer->Get<cudaStreamCaptureMode>();
        cudaError_t exit_code = cudaThreadExchangeStreamCaptureMode(&mode);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<cudaStreamCaptureMode>(mode);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamIsCapturing) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaStreamCaptureStatus captureStatus;
        cudaError_t exit_code = cudaStreamIsCapturing(stream, &captureStatus);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<cudaStreamCaptureStatus>(captureStatus);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamBeginCapture) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaStreamCaptureMode mode = input_buffer->Get<cudaStreamCaptureMode>();
        return std::make_shared<Result>(cudaStreamBeginCapture(stream, mode));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(StreamEndCapture) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaGraph_t pGraph;
        cudaError_t exit_code = cudaStreamEndCapture(stream, &pGraph);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        out->Add<cudaGraph_t>(pGraph);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

/*
CUDA_ROUTINE_HANDLER(StreamAddCallback) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaStreamCallback_t callback = input_buffer->Get<cudaStreamCallback_t>();
        void* userData = input_buffer->GetFromMarshal<void*>();
        unsigned int flags = input_buffer->Get<unsigned int>();
        return std::make_shared<Result>(cudaStreamAddCallback(stream, callback, userData, flags));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}
*/
