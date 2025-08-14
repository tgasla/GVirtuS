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

CUDA_ROUTINE_HANDLER(EventCreate) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        cudaEvent_t event;
        cudaError_t exit_code = cudaEventCreate(&event);
        out->Add((pointer_t)event);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(EventCreateWithFlags) {
    try {
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
        cudaEvent_t event;
        int flags = input_buffer->Get<int>();
        cudaError_t exit_code = cudaEventCreateWithFlags(&event, flags);
        out->Add((pointer_t)event);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(EventDestroy) {
    try {
        cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
        return std::make_shared<Result>(cudaEventDestroy(event));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(EventElapsedTime) {
    try {
        float* ms = input_buffer->Assign<float>();
        cudaEvent_t start = input_buffer->Get<cudaEvent_t>();
        cudaEvent_t end = input_buffer->Get<cudaEvent_t>();
        cudaError_t exit_code = cudaEventElapsedTime(ms, start, end);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

        out->Add(ms);
        return std::make_shared<Result>(exit_code, out);
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(EventQuery) {
    try {
        cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
        return std::make_shared<Result>(cudaEventQuery(event));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(EventRecord) {
    try {
        cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return std::make_shared<Result>(cudaEventRecord(event, stream));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(EventSynchronize) {
    try {
        cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
        return std::make_shared<Result>(cudaEventSynchronize(event));
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}
