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

extern nvrtcResult nvrtcGetTypeName(const type_info &tinfo, string *result) {
    string type_name = tinfo.name();
    // cerr << "nvrtcGetTypeName called with type_info: " << type_name
    //           << ", result pointer: " << (result ? "not null" : "null") <<  endl;

    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddVariableForArguments(type_name);
    NvrtcFrontend::Execute("nvrtcGetTypeName");

    if (NvrtcFrontend::Success() && result) {
        *result = *(NvrtcFrontend::GetOutputHostPointer<string>());
    }

    return NvrtcFrontend::GetExitCode();
}

extern nvrtcResult nvrtcGetTypeName(string *result) {
    // cerr << "nvrtcGetTypeName called with result pointer: " << (result ? "not null" : "null") <<
    // endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddHostPointerForArguments(result);
    NvrtcFrontend::Execute("nvrtcGetTypeName");
    return NvrtcFrontend::GetExitCode();
}