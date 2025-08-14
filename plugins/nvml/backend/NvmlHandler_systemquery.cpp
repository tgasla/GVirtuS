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
 * Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 *
 *             angzam78 <angzam78@gmail.com>
 */

#include "NvmlHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

NVML_ROUTINE_HANDLER(SystemGetCudaDriverVersion_v2) {
    int cudaDriverVersion;

    nvmlReturn_t result = nvmlSystemGetCudaDriverVersion_v2(&cudaDriverVersion);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(&cudaDriverVersion);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "nvmlSystemGetCudaDriverVersion_v2 executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(SystemGetDriverVersion) {
    unsigned int length = in->Get<unsigned int>();
    std::unique_ptr<char[]> version(new char[length]);

    nvmlReturn_t result = nvmlSystemGetDriverVersion(version.get(), length);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(version.get(), length);

    LOG4CPLUS_DEBUG(pThis->GetLogger(), "nvmlSystemGetDriverVersion executed");
    return std::make_shared<Result>(result, out);
}