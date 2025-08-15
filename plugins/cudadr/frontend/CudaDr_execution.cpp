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
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include <dirent.h>
#include <errno.h>

#include "CudaDr.h"

using namespace std;

void listFiles(const char* path) {
    DIR* dirFile = opendir(path);
    if (dirFile) {
        struct dirent* hFile;
        errno = 0;
        while ((hFile = readdir(dirFile)) != NULL) {
            if (!strcmp(hFile->d_name, ".")) continue;
            if (!strcmp(hFile->d_name, "..")) continue;
            if (strstr(hFile->d_name, ".ptx")) printf("found a .ptx file: %s", hFile->d_name);
        }
        closedir(dirFile);
    }
}

/*Sets the parameter size for the function.*/
extern "C" CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(numbytes);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hfunc);
    CudaDrFrontend::Execute("cuParamSetSize");
    return CudaDrFrontend::GetExitCode();
}

/*Sets the block-dimensions for the function.*/
extern "C" CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(x);
    CudaDrFrontend::AddVariableForArguments(y);
    CudaDrFrontend::AddVariableForArguments(z);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hfunc);
    CudaDrFrontend::Execute("cuFuncSetBlockShape");
    return CudaDrFrontend::GetExitCode();
}

/*Launches a CUDA function.*/
extern "C" CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(grid_width);
    CudaDrFrontend::AddVariableForArguments(grid_height);
    CudaDrFrontend::AddDevicePointerForArguments((void*)f);
    CudaDrFrontend::Execute("cuLaunchGrid");
    return CudaDrFrontend::GetExitCode();
}

/*Returns information about a function.*/
extern "C" CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(pi);
    CudaDrFrontend::AddVariableForArguments(attrib);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hfunc);
    CudaDrFrontend::Execute("cuFuncGetAttribute");
    if (CudaDrFrontend::Success()) *pi = *(CudaDrFrontend::GetOutputHostPointer<int>());
    return (CUresult)(CudaDrFrontend::GetExitCode());
}

/*Sets the dynamic shared-memory size for the function.*/
extern "C" CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(bytes);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hfunc);
    CudaDrFrontend::Execute("cuFuncSetSharedSize");
    return (CUresult)(CudaDrFrontend::GetExitCode());
}

/*Launches a CUDA function.*/
extern "C" CUresult cuLaunch(CUfunction f) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*)f);
    CudaDrFrontend::Execute("cuLaunch");
    return CudaDrFrontend::GetExitCode();
}

/*Launches a CUDA function.*/
extern "C" CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                                      CUstream hStream) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(grid_width);
    CudaDrFrontend::AddVariableForArguments(grid_height);
    CudaDrFrontend::AddDevicePointerForArguments((void*)f);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hStream);
    CudaDrFrontend::Execute("cuLaunchGridAsync");
    return CudaDrFrontend::GetExitCode();
}

/*Adds a floating-point parameter to the function's argument list.*/
extern "C" CUresult cuParamSetf(CUfunction hfunc, int offset, float value) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(offset);
    CudaDrFrontend::AddVariableForArguments(value);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hfunc);
    CudaDrFrontend::Execute("cuParamSetf");
    return CudaDrFrontend::GetExitCode();
}

/*Adds an integer parameter to the function's argument list.*/
extern "C" CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(offset);
    CudaDrFrontend::AddVariableForArguments(value);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hfunc);
    CudaDrFrontend::Execute("cuParamSeti");
    return CudaDrFrontend::GetExitCode();
}

/*Adds a texture-reference to the function's argument list.*/
extern "C" CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*)hfunc);
    CudaDrFrontend::AddVariableForArguments(texunit);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hTexRef);
    CudaDrFrontend::Execute("cuParamSetTexRef");
    return CudaDrFrontend::GetExitCode();
}

/*Adds arbitrary data to the function's argument list.*/
extern "C" CUresult cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(offset);
    CudaDrFrontend::AddVariableForArguments(numbytes);
    CudaDrFrontend::AddHostPointerForArguments((char*)ptr, numbytes);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hfunc);
    CudaDrFrontend::Execute("cuParamSetv");
    return CudaDrFrontend::GetExitCode();
}

/*Sets the preferred cache configuration for a device function. */
extern "C" CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(config);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hfunc);
    CudaDrFrontend::Execute("cuFuncSetCacheConfig");
    return CudaDrFrontend::GetExitCode();
}

// new Cuda 4.0 functions
extern "C" CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                   unsigned int gridDimZ, unsigned int blockDimX,
                                   unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hstream,
                                   void** kernelParams, void** extra) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*)f);
    CudaDrFrontend::AddVariableForArguments(gridDimX);
    CudaDrFrontend::AddVariableForArguments(gridDimY);
    CudaDrFrontend::AddVariableForArguments(gridDimZ);
    CudaDrFrontend::AddVariableForArguments(blockDimX);
    CudaDrFrontend::AddVariableForArguments(blockDimY);
    CudaDrFrontend::AddVariableForArguments(blockDimZ);
    CudaDrFrontend::AddVariableForArguments(sharedMemBytes);
    CudaDrFrontend::AddDevicePointerForArguments((void*)hstream);
    CudaDrFrontend::AddDevicePointerForArguments(kernelParams);
    CudaDrFrontend::AddHostPointerForArguments(&extra);
    CudaDrFrontend::Execute("cuLaunchKernel");
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f,
                                     void** kernelParams, void** extra) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*)f);
    CudaDrFrontend::AddHostPointerForArguments<const CUlaunchConfig>(config);
    CudaDrFrontend::AddDevicePointerForArguments(kernelParams);
    CudaDrFrontend::AddHostPointerForArguments(&extra);
    CudaDrFrontend::Execute("cuLaunchKernelEx");
    return CudaDrFrontend::GetExitCode();
}