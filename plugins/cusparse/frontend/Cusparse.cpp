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
 * Written By: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>,
 *             Department of Science and Technologies
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            School of Computer Science, University College Dublin
 */

#include "CusparseFrontend.h"

using namespace std;

extern "C" cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddDevicePointerForArguments(handle);
    CusparseFrontend::Execute("cusparseGetVersion");
    if (CusparseFrontend::Success()) *version = CusparseFrontend::GetOutputVariable<int>();
    return CusparseFrontend::GetExitCode();
}

extern "C" const char* cusparseGetErrorString(cusparseStatus_t status) {
    CusparseFrontend::Prepare();
    CusparseFrontend::AddVariableForArguments<cusparseStatus_t>(status);
    CusparseFrontend::Execute("cusparseGetErrorString");
    return CusparseFrontend::GetOutputString();
}