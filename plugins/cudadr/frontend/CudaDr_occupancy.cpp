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
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#include "CudaDr.h"

using namespace std;

extern "C" CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSMemSize,
                                                            CUfunction func, int numBlocks,
                                                            int blockSize) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(dynamicSMemSize);
    CudaDrFrontend::AddDevicePointerForArguments((void*)func);
    CudaDrFrontend::AddVariableForArguments(numBlocks);
    CudaDrFrontend::AddVariableForArguments(blockSize);
    CudaDrFrontend::Execute("cuOccupancyAvailableDynamicSMemPerBlock");
    if (CudaDrFrontend::Success()) {
        *dynamicSMemSize = *(CudaDrFrontend::GetOutputHostPointer<size_t>());
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func,
                                                                int blockSize,
                                                                size_t dynamicSMemSize) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(numBlocks);
    CudaDrFrontend::AddDevicePointerForArguments((void*)func);
    CudaDrFrontend::AddVariableForArguments(blockSize);
    CudaDrFrontend::AddVariableForArguments(dynamicSMemSize);
    CudaDrFrontend::Execute("cuOccupancyMaxActiveBlocksPerMultiprocessor");
    if (CudaDrFrontend::Success()) {
        *numBlocks = *(CudaDrFrontend::GetOutputHostPointer<int>());
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(dynamicSMemSize);
    CudaDrFrontend::AddDevicePointerForArguments((void*)func);
    CudaDrFrontend::AddVariableForArguments(numBlocks);
    CudaDrFrontend::AddVariableForArguments(blockSize);
    CudaDrFrontend::Execute("cuOccupancyAvailableDynamicSMemPerBlock");
    if (CudaDrFrontend::Success()) {
        numBlocks = CudaDrFrontend::GetOutputHostPointer<int>();
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func,
                                                 const CUlaunchConfig* config) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(numClusters);
    CudaDrFrontend::AddDevicePointerForArguments((void*)func);
    CudaDrFrontend::AddHostPointerForArguments<const CUlaunchConfig>(config);
    CudaDrFrontend::Execute("cuOccupancyMaxActiveClusters");
    if (CudaDrFrontend::Success()) {
        *numClusters = *(CudaDrFrontend::GetOutputHostPointer<int>());
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                                     CUfunction func,
                                                     CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                                     size_t dynamicSMemSize, int blockSizeLimit) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(minGridSize);
    CudaDrFrontend::AddHostPointerForArguments(blockSize);
    CudaDrFrontend::AddDevicePointerForArguments((void*)func);
    CudaDrFrontend::AddVariableForArguments(blockSizeToDynamicSMemSize);
    CudaDrFrontend::AddVariableForArguments(dynamicSMemSize);
    CudaDrFrontend::AddVariableForArguments(blockSizeLimit);
    CudaDrFrontend::Execute("cuOccupancyMaxPotentialBlockSize");
    if (CudaDrFrontend::Success()) {
        *minGridSize = *(CudaDrFrontend::GetOutputHostPointer<int>());
        *blockSize = *(CudaDrFrontend::GetOutputHostPointer<int>(1));
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(
    int* minGridSize, int* blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit,
    unsigned int flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(minGridSize);
    CudaDrFrontend::AddHostPointerForArguments(blockSize);
    CudaDrFrontend::AddDevicePointerForArguments((void*)func);
    CudaDrFrontend::AddVariableForArguments(blockSizeToDynamicSMemSize);
    CudaDrFrontend::AddVariableForArguments(dynamicSMemSize);
    CudaDrFrontend::AddVariableForArguments(blockSizeLimit);
    CudaDrFrontend::Execute("cuOccupancyMaxPotentialBlockSizeWithFlags");
    if (CudaDrFrontend::Success()) {
        *minGridSize = *(CudaDrFrontend::GetOutputHostPointer<int>());
        *blockSize = *(CudaDrFrontend::GetOutputHostPointer<int>(1));
    }
    return CudaDrFrontend::GetExitCode();
}

extern "C" CUresult cuOccupancyMaxPotentialClusterSize(int* clusterSize, CUfunction func,
                                                       const CUlaunchConfig* config) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(clusterSize);
    CudaDrFrontend::AddDevicePointerForArguments((void*)func);
    CudaDrFrontend::AddHostPointerForArguments<const CUlaunchConfig>(config);
    CudaDrFrontend::Execute("cuOccupancyMaxPotentialClusterSize");
    if (CudaDrFrontend::Success()) {
        *clusterSize = *(CudaDrFrontend::GetOutputHostPointer<int>());
    }
    return CudaDrFrontend::GetExitCode();
}
