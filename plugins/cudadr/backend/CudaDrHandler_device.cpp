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
 * Written by: Flora Giannone <flora.giannone@studenti.uniparthenope.it>,
 *             Department of Applied Science
 */

#include "CudaDrHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

/*Returns the compute capability of the device*/
CUDA_DRIVER_HANDLER(DeviceComputeCapability) {
    int *major = input_buffer->Assign<int>();
    int *minor = input_buffer->Assign<int>();
    CUdevice dev = input_buffer->Get<CUdevice > ();
    CUresult exit_code = cuDeviceComputeCapability(major, minor, dev);
    std::shared_ptr<Buffer> output_buffer = std::make_shared<Buffer>();
    output_buffer->Add(major);
    output_buffer->Add(minor);
    return std::make_shared<Result>((cudaError_t) exit_code, output_buffer);
}

/*Returns a handle to a compute device*/
CUDA_DRIVER_HANDLER(DeviceGet) {
    CUdevice *device = input_buffer->Assign<CUdevice > ();
    int ordinal = input_buffer->Get<int>();
    CUresult exit_code = cuDeviceGet(device, ordinal);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(device);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Returns information about the device*/
CUDA_DRIVER_HANDLER(DeviceGetAttribute) {
    int *pi = input_buffer->Assign<int>();
    CUdevice_attribute attrib = input_buffer->Get<CUdevice_attribute > ();
    CUdevice dev = input_buffer->Get<CUdevice > ();
    CUresult exit_code = cuDeviceGetAttribute(pi, attrib, dev);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(pi);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Returns the number of compute-capable devices. */
CUDA_DRIVER_HANDLER(DeviceGetCount) {
    int *count = input_buffer->Assign<int>();
    CUresult exit_code = cuDeviceGetCount(count);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(count);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Returns an identifer string for the device.*/
CUDA_DRIVER_HANDLER(DeviceGetName) {
    char *name = input_buffer->AssignString();
    int len = input_buffer->Get<int>();
    CUdevice dev = input_buffer->Get<CUdevice > ();
    CUresult exit_code = cuDeviceGetName(name, len, dev);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddString(name);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Returns properties for a selected device. */
CUDA_DRIVER_HANDLER(DeviceGetProperties) {
    CUdevprop *prop = input_buffer->Assign<CUdevprop > ();
    CUdevice dev = input_buffer->Get<CUdevice > ();
    CUresult exit_code = cuDeviceGetProperties(prop, dev);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(prop);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Returns the total amount of memory on the device. */
CUDA_DRIVER_HANDLER(DeviceTotalMem) {
    size_t *bytes = input_buffer->Assign<size_t > ();
    CUdevice dev = input_buffer->Get<CUdevice > ();
    CUresult exit_code = cuDeviceTotalMem(bytes, dev);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(bytes);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}
