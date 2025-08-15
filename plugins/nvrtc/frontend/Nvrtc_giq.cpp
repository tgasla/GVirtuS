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

extern "C" nvrtcResult nvrtcGetNumSupportedArchs(int *numArchs) {
    // cerr << "nvrtcGetNumSupportedArchs called with numArchs: " << (numArchs ? *numArchs : 0) <<
    // endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddHostPointerForArguments(numArchs);
    NvrtcFrontend::Execute("nvrtcGetNumSupportedArchs");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetSupportedArchs(int *supportedArchs) {
    // cerr << "nvrtcGetSupportedArchs called with supportedArchs: "
    //  << (supportedArchs ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddHostPointerForArguments(supportedArchs);
    NvrtcFrontend::Execute("nvrtcGetSupportedArchs");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcVersion(int *major, int *minor) {
    // cerr << "nvrtcVersion called with major: " << (major ? *major : 0)
    //      << ", minor: " << (minor ? *minor : 0) << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddHostPointerForArguments(major);
    NvrtcFrontend::AddHostPointerForArguments(minor);
    NvrtcFrontend::Execute("nvrtcVersion");
    return NvrtcFrontend::GetExitCode();
}
