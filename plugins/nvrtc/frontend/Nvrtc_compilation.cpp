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

#include <unordered_map>

#include "NvrtcFrontend.h"

using namespace std;
using CallbackType = int (*)(void *, void *);

// Struct to hold both callback and payload
typedef struct CallbackEntry {
    CallbackType callback;
    void *payload;
} CallbackEntry;

// The map: prog -> callback + payload
static std::unordered_map<nvrtcProgram, CallbackEntry> callbackRegistry;

extern "C" nvrtcResult nvrtcResultnvrtcAddNameExpression(nvrtcProgram prog,
                                                         const char *const name_expression) {
    cout << "nvrtcAddNameExpression called with prog: " << prog
         << ", name_expression: " << (name_expression ? name_expression : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddStringForArguments(name_expression);
    NvrtcFrontend::Execute("nvrtcAddNameExpression");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions,
                                           const char *const *options) {
    cout << "nvrtcCompileProgram called with prog: " << prog << ", numOptions: " << numOptions
         << ", options: " << (options ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddVariableForArguments(numOptions);
    for (int i = 0; i < numOptions; ++i) {
        NvrtcFrontend::AddStringForArguments(options[i]);
    }
    NvrtcFrontend::Execute("nvrtcCompileProgram");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char *src, const char *name,
                                          int numHeaders, const char *const *headers,
                                          const char *const *includeNames) {
    cout << "nvrtcCreateProgram called with prog: " << (prog ? *prog : nullptr)
         << ", src: " << (src ? src : "NULL") << ", name: " << (name ? name : "NULL")
         << ", numHeaders: " << numHeaders << ", headers: " << (headers ? "provided" : "NULL")
         << ", includeNames: " << (includeNames ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddStringForArguments(src);
    NvrtcFrontend::AddStringForArguments(name);
    NvrtcFrontend::AddVariableForArguments(numHeaders);
    for (int i = 0; i < numHeaders; ++i) {
        NvrtcFrontend::AddStringForArguments(headers[i]);
        NvrtcFrontend::AddStringForArguments(includeNames[i]);
    }
    NvrtcFrontend::Execute("nvrtcCreateProgram");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog) {
    cout << "nvrtcDestroyProgram called with prog: " << (prog ? *prog : nullptr) << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::Execute("nvrtcDestroyProgram");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char *cubin) {
    cout << "nvrtcGetCUBIN called with prog: " << prog
         << ", cubin: " << (cubin ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(cubin);
    NvrtcFrontend::Execute("nvrtcGetCUBIN");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t *cubinSizeRet) {
    cout << "nvrtcGetCUBINSize called with prog: " << prog
         << ", cubinSizeRet: " << (cubinSizeRet ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(cubinSizeRet);
    NvrtcFrontend::Execute("nvrtcGetCUBINSize");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char *LTOIR) {
    cout << "nvrtcGetLTOIR called with prog: " << prog
         << ", LTOIR: " << (LTOIR ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(LTOIR);
    NvrtcFrontend::Execute("nvrtcGetLTOIR");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t *LTOIRSizeRet) {
    cout << "nvrtcGetLTOIRSize called with prog: " << prog
         << ", LTOIRSizeRet: " << (LTOIRSizeRet ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(LTOIRSizeRet);
    NvrtcFrontend::Execute("nvrtcGetLTOIRSize");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char *const name_expression,
                                           const char **lowered_name) {
    cout << "nvrtcGetLoweredName called with prog: " << prog
         << ", name_expression: " << (name_expression ? name_expression : "NULL")
         << ", lowered_name: " << (lowered_name ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddStringForArguments(name_expression);
    NvrtcFrontend::AddHostPointerForArguments(lowered_name);
    NvrtcFrontend::Execute("nvrtcGetLoweredName");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetNVVM(nvrtcProgram prog, char *nvvm) {
    cout << "nvrtcGetNVVM called with prog: " << prog << ", nvvm: " << (nvvm ? "provided" : "NULL")
         << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(nvvm);
    NvrtcFrontend::Execute("nvrtcGetNVVM");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetNVVMSize(nvrtcProgram prog, size_t *nvvmSizeRet) {
    cout << "nvrtcGetNVVMSize called with prog: " << prog
         << ", nvvmSizeRet: " << (nvvmSizeRet ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(nvvmSizeRet);
    NvrtcFrontend::Execute("nvrtcGetNVVMSize");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetOptiXIR(nvrtcProgram prog, char *optixir) {
    cout << "nvrtcGetOptiXIR called with prog: " << prog
         << ", optixir: " << (optixir ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(optixir);
    NvrtcFrontend::Execute("nvrtcGetOptiXIR");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t *optixirSizeRet) {
    cout << "nvrtcGetOptiXIRSize called with prog: " << prog
         << ", optixirSizeRet: " << (optixirSizeRet ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(optixirSizeRet);
    NvrtcFrontend::Execute("nvrtcGetOptiXIRSize");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx) {
    cout << "nvrtcGetPTX called with prog: " << prog << ", ptx: " << (ptx ? "provided" : "NULL")
         << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(ptx);
    NvrtcFrontend::Execute("nvrtcGetPTX");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet) {
    cout << "nvrtcGetPTXSize called with prog: " << prog
         << ", ptxSizeRet: " << (ptxSizeRet ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(ptxSizeRet);
    NvrtcFrontend::Execute("nvrtcGetPTXSize");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log) {
    cout << "nvrtcGetProgramLog called with prog: " << prog
         << ", log: " << (log ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(log);
    NvrtcFrontend::Execute("nvrtcGetProgramLog");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet) {
    cout << "nvrtcGetProgramLogSize called with prog: " << prog
         << ", logSizeRet: " << (logSizeRet ? "provided" : "NULL") << endl;
    NvrtcFrontend::Prepare();
    NvrtcFrontend::AddDevicePointerForArguments(prog);
    NvrtcFrontend::AddHostPointerForArguments(logSizeRet);
    NvrtcFrontend::Execute("nvrtcGetProgramLogSize");
    return NvrtcFrontend::GetExitCode();
}

extern "C" nvrtcResult nvrtcSetFlowCallback(nvrtcProgram prog, int (*callback)(void *, void *),
                                            void *payload) {
    cout << "nvrtcSetFlowCallback called with prog: " << prog
         << ", callback: " << (callback ? "provided" : "NULL")
         << ", payload: " << (payload ? "provided" : "NULL") << endl;

    callbackRegistry[prog] = {callback, payload};
    return NVRTC_SUCCESS;
}