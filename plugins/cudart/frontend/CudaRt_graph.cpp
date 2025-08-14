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
 *             Ting-Hui Cheng <tinghc@es.aau.dk>
 *             Department of Electronic Systems, Aalborg University
 */

#include "CudaRt.h"

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph,
                                                            cudaGraphNode_t* nodes,
                                                            size_t* numNodes) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graph);
    CudaRtFrontend::AddHostPointerForArguments(nodes);
    CudaRtFrontend::Execute("cudaGraphGetNodes");

    if (CudaRtFrontend::Success()) {
        *numNodes = CudaRtFrontend::GetOutputVariable<size_t>();
    }
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graphExec);
    CudaRtFrontend::Execute("cudaGraphExecDestroy");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t* pGraphExec,
                                                               cudaGraph_t graph,
                                                               unsigned long long flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graph);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaGraphInstantiate");

    if (CudaRtFrontend::Success()) {
        *pGraphExec = CudaRtFrontend::GetOutputVariable<cudaGraphExec_t>();
    }
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec,
                                                                        cudaGraph_t graph,
                                                                        unsigned long long flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(pGraphExec);
    CudaRtFrontend::AddDevicePointerForArguments(graph);
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaGraphInstantiateWithFlags");

    if (CudaRtFrontend::Success()) {
        *pGraphExec = (cudaGraphExec_t)CudaRtFrontend::GetOutputDevicePointer();
    }
    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaGraphDebugDotPrint(cudaGraph_t graph,
                                                                 const char* path,
                                                                 unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graph);
    CudaRtFrontend::AddStringForArguments(path);
    CudaRtFrontend::AddVariableForArguments<unsigned int>(flags);
    CudaRtFrontend::Execute("cudaGraphDebugDotPrint");

    return CudaRtFrontend::GetExitCode();
}

// TODO: needs testing
extern "C" __host__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec,
                                                          cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graphExec);
    CudaRtFrontend::AddDevicePointerForArguments(stream);
    CudaRtFrontend::Execute("cudaGraphLaunch");

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaGraphCreate");
    if (CudaRtFrontend::Success()) *pGraph = CudaRtFrontend::GetOutputVariable<cudaGraph_t>();
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(graph);
    CudaRtFrontend::Execute("cudaGraphDestroy");

    return CudaRtFrontend::GetExitCode();
}