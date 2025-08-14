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
 * Written by: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *             Department of Computer Science, University College Dublin
 *
 */

#include "NvrtcFrontend.h"

using namespace std;

#ifndef NVRTC_VERSION
#error NVRTC_VERSION not defined
#endif
#if NVRTC_VERSION >= 12800
export "C" nvrtcResult nvrtcGetPCHCreateStatus(nvrtcProgram prog) {
    // cerr << "nvrtcGetPCHCreateStatus called with prog: " << prog << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments((void *)prog);
    NvrtcFrontend::Execute("nvrtcGetPCHCreateStatus");
    return NvrtcFrontend::GetExitCode();
}

export "C" nvrtcResult nvrtcGetPCHHeapSize(size_t *ret) {
    // cerr << "nvrtcGetPCHHeapSize called" << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddHostPointerForArguments(ret);
    NvrtcFrontend::Execute("nvrtcGetPCHHeapSize");
    return NvrtcFrontend::GetExitCode();
}

export "C" nvrtcResult nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t *size) {
    // cerr << "nvrtcGetPCHHeapSizeRequired called with prog: " << prog << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments((void *)prog);
    NvrtcFrontend::AddHostPointerForArguments(size);
    NvrtcFrontend::Execute("nvrtcGetPCHHeapSizeRequired");
    return NvrtcFrontend::GetExitCode();
}

export "C" nvrtcResult nvrtcSetPCHHeapSize(size_t size) {
    // cerr << "nvrtcSetPCHHeapSize called with size: " << size << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddVariableForArguments(size);
    NvrtcFrontend::Execute("nvrtcSetPCHHeapSize");
    return NvrtcFrontend::GetExitCode();
}
#endif