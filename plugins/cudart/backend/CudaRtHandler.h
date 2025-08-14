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
 * Written By: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 *
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             Department of Computer Science, University College Dublin
 */

#ifndef _CUDARTHANDLER_H
#define _CUDARTHANDLER_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <CudaRt_internal.h>
#include <CudaUtil.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <gvirtus/backend/Handler.h>
#include <gvirtus/communicators/Result.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <map>
#include <string>

#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

/**
 * CudaRtHandler is used by Backend's Process(es) for storing and retrieving
 * device related data and functions.
 * CudaRtHandler has also the method Execute() that is responsible to execute a
 * named CUDA Runtime routine unmarshalling the input parameters from the
 * provided Buffer.
 */
using namespace std;
using namespace log4cplus;

using gvirtus::common::pointer_t;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

class CudaRtHandler : public gvirtus::backend::Handler {
   public:
    CudaRtHandler();
    virtual ~CudaRtHandler();
    bool CanExecute(std::string routine);
    std::shared_ptr<Result> Execute(std::string routine, std::shared_ptr<Buffer> input_buffer);

    void RegisterFatBinary(std::string &handler, void **fatCubinHandle);
    void RegisterFatBinary(const char *handler, void **fatCubinHandle);
    void RegisterFatBinaryEnd(void **fatCubinHandle);
    void **GetFatBinary(std::string &handler);
    void **GetFatBinary(const char *handler);
    void UnregisterFatBinary(std::string &handler);
    void UnregisterFatBinary(const char *handler);

    void RegisterDeviceFunction(std::string &handler, std::string &function);
    void RegisterDeviceFunction(const char *handler, const char *function);
    const char *GetDeviceFunction(std::string &handler);
    const char *GetDeviceFunction(const char *handler);

    void RegisterVar(std::string &handler, std::string &deviceName);
    void RegisterVar(const char *handler, const char *deviceName);
    const char *GetVar(std::string &handler);
    const char *GetVar(const char *handler);

    const char *GetSymbol(std::shared_ptr<Buffer> in);

    Logger &GetLogger() { return logger; }

    inline void addDeviceFunc2InfoFunc(std::string deviceFunc, NvInfoFunction infoFunction) {
        mapDeviceFunc2InfoFunc->insert(make_pair(deviceFunc, infoFunction));
    }

    inline NvInfoFunction getInfoFunc(std::string deviceFunc) {
        if (mapDeviceFunc2InfoFunc->find(deviceFunc) == mapDeviceFunc2InfoFunc->end()) {
            LOG4CPLUS_ERROR(logger,
                            "getInfoFunc: device function '" << deviceFunc << "' not found");
            throw std::runtime_error("getInfoFunc: device function not found");
        }
        return mapDeviceFunc2InfoFunc->find(deviceFunc)->second;
    };

    inline void addHost2DeviceFunc(const void *hostFunc, std::string deviceFunc) {
        mapHost2DeviceFunc->insert(make_pair(hostFunc, deviceFunc));
    }

    inline std::string getDeviceFunc(const void *hostFunc) {
        if (mapHost2DeviceFunc->find(hostFunc) == mapHost2DeviceFunc->end()) {
            LOG4CPLUS_ERROR(logger, "getDeviceFunc: host function '" << hostFunc << "' not found");
            throw std::runtime_error("getDeviceFunc: host function not found");
        }
        return mapHost2DeviceFunc->find(hostFunc)->second;
    };

    static void hexdump(void *ptr, int buflen) {
        unsigned char *buf = (unsigned char *)ptr;
        int i, j;
        for (i = 0; i < buflen; i += 16) {
            printf("%06x: ", i);
            for (j = 0; j < 16; j++)
                if (i + j < buflen)
                    printf("%02x ", buf[i + j]);
                else
                    printf("   ");
            printf(" ");
            for (j = 0; j < 16; j++)
                if (i + j < buflen) printf("%c", isprint(buf[i + j]) ? buf[i + j] : '.');
            printf("\n");
        }
    }

   private:
    log4cplus::Logger logger;
    void Initialize();
    typedef std::shared_ptr<Result> (*CudaRoutineHandler)(CudaRtHandler *, std::shared_ptr<Buffer>);
    static std::map<std::string, CudaRoutineHandler> *mspHandlers;
    std::map<std::string, void **> *mpFatBinary;
    std::map<std::string, std::string> *mpDeviceFunction;
    std::map<std::string, std::string> *mpVar;
    std::map<std::string, cudaTextureObject_t *> *mpTexture;
    std::map<std::string, cudaSurfaceObject_t *> *mpSurface;
    map<std::string, NvInfoFunction> *mapDeviceFunc2InfoFunc;
    map<const void *, std::string> *mapHost2DeviceFunc;
    void *mpShm;
    int mShmFd;
};

#define CUDA_ROUTINE_HANDLER(name) \
    std::shared_ptr<Result> handle##name(CudaRtHandler *pThis, std::shared_ptr<Buffer> input_buffer)
#define CUDA_ROUTINE_HANDLER_PAIR(name) make_pair("cuda" #name, handle##name)

/* CudaRtHandler_device */
CUDA_ROUTINE_HANDLER(ChooseDevice);
CUDA_ROUTINE_HANDLER(GetDevice);
CUDA_ROUTINE_HANDLER(GetDeviceCount);
CUDA_ROUTINE_HANDLER(GetDeviceProperties);
CUDA_ROUTINE_HANDLER(SetDevice);
CUDA_ROUTINE_HANDLER(SetDeviceFlags);
CUDA_ROUTINE_HANDLER(SetValidDevices);
CUDA_ROUTINE_HANDLER(DeviceReset);
CUDA_ROUTINE_HANDLER(DeviceSynchronize);
CUDA_ROUTINE_HANDLER(DeviceSetCacheConfig);
CUDA_ROUTINE_HANDLER(DeviceSetLimit);
CUDA_ROUTINE_HANDLER(DeviceCanAccessPeer);
CUDA_ROUTINE_HANDLER(DeviceEnablePeerAccess);
CUDA_ROUTINE_HANDLER(DeviceDisablePeerAccess);
CUDA_ROUTINE_HANDLER(IpcGetMemHandle);
CUDA_ROUTINE_HANDLER(IpcGetEventHandle);
CUDA_ROUTINE_HANDLER(IpcOpenEventHandle);
CUDA_ROUTINE_HANDLER(IpcOpenMemHandle);
CUDA_ROUTINE_HANDLER(DeviceGetDefaultMemPool);
// Testing
CUDA_ROUTINE_HANDLER(OccupancyMaxActiveBlocksPerMultiprocessor);
CUDA_ROUTINE_HANDLER(DeviceGetAttribute);
CUDA_ROUTINE_HANDLER(DeviceGetStreamPriorityRange);

/* CudaRtHandler_error */
CUDA_ROUTINE_HANDLER(GetErrorString);
CUDA_ROUTINE_HANDLER(GetLastError);
CUDA_ROUTINE_HANDLER(PeekAtLastError);

/* CudaRtHandler_event */
CUDA_ROUTINE_HANDLER(EventCreate);
CUDA_ROUTINE_HANDLER(EventCreateWithFlags);
CUDA_ROUTINE_HANDLER(EventDestroy);
CUDA_ROUTINE_HANDLER(EventElapsedTime);
CUDA_ROUTINE_HANDLER(EventQuery);
CUDA_ROUTINE_HANDLER(EventRecord);
CUDA_ROUTINE_HANDLER(EventSynchronize);

/* CudaRtHandler_execution */
CUDA_ROUTINE_HANDLER(ConfigureCall);
CUDA_ROUTINE_HANDLER(FuncGetAttributes);
CUDA_ROUTINE_HANDLER(FuncSetCacheConfig);
CUDA_ROUTINE_HANDLER(Launch);
CUDA_ROUTINE_HANDLER(LaunchKernel);
CUDA_ROUTINE_HANDLER(SetDoubleForDevice);
CUDA_ROUTINE_HANDLER(SetDoubleForHost);
CUDA_ROUTINE_HANDLER(SetupArgument);
CUDA_ROUTINE_HANDLER(PushCallConfiguration);
CUDA_ROUTINE_HANDLER(PopCallConfiguration);

/* CudaRtHandler_internal */
CUDA_ROUTINE_HANDLER(RegisterFatBinary);
CUDA_ROUTINE_HANDLER(RegisterFatBinaryEnd);
CUDA_ROUTINE_HANDLER(UnregisterFatBinary);
CUDA_ROUTINE_HANDLER(RegisterFunction);
CUDA_ROUTINE_HANDLER(RegisterVar);
CUDA_ROUTINE_HANDLER(RegisterSharedVar);
CUDA_ROUTINE_HANDLER(RegisterShared);
CUDA_ROUTINE_HANDLER(RegisterSharedMemory);
CUDA_ROUTINE_HANDLER(RequestSharedMemory);

/* CudaRtHandler_memory */
CUDA_ROUTINE_HANDLER(MemGetInfo);
CUDA_ROUTINE_HANDLER(Free);
CUDA_ROUTINE_HANDLER(FreeArray);
CUDA_ROUTINE_HANDLER(GetSymbolAddress);
CUDA_ROUTINE_HANDLER(GetSymbolSize);
CUDA_ROUTINE_HANDLER(Malloc);
CUDA_ROUTINE_HANDLER(MallocArray);
CUDA_ROUTINE_HANDLER(MallocPitch);
CUDA_ROUTINE_HANDLER(MallocManaged);
CUDA_ROUTINE_HANDLER(Memcpy);
CUDA_ROUTINE_HANDLER(Memcpy2D);
CUDA_ROUTINE_HANDLER(Memcpy3D);
CUDA_ROUTINE_HANDLER(MemcpyAsync);
CUDA_ROUTINE_HANDLER(MemcpyFromSymbol);
CUDA_ROUTINE_HANDLER(MemcpyToArray);
CUDA_ROUTINE_HANDLER(MemcpyToSymbol);
CUDA_ROUTINE_HANDLER(Memset);
CUDA_ROUTINE_HANDLER(Memset2D);
CUDA_ROUTINE_HANDLER(MemcpyFromArray);
CUDA_ROUTINE_HANDLER(MemcpyArrayToArray);
CUDA_ROUTINE_HANDLER(Memcpy2DFromArray);
CUDA_ROUTINE_HANDLER(Memcpy2DToArray);
CUDA_ROUTINE_HANDLER(Malloc3DArray);
CUDA_ROUTINE_HANDLER(MemcpyPeerAsync);
CUDA_ROUTINE_HANDLER(HostRegister);
CUDA_ROUTINE_HANDLER(HostUnregister);

/* CudaRtHandler_opengl */
CUDA_ROUTINE_HANDLER(GLSetGLDevice);
CUDA_ROUTINE_HANDLER(GraphicsGLRegisterBuffer);
CUDA_ROUTINE_HANDLER(GraphicsMapResources);
CUDA_ROUTINE_HANDLER(GraphicsResourceGetMappedPointer);
CUDA_ROUTINE_HANDLER(GraphicsUnmapResources);
CUDA_ROUTINE_HANDLER(GraphicsUnregisterResource);
CUDA_ROUTINE_HANDLER(GraphicsResourceSetMapFlags);

/* CudaRtHandler_stream_memory */
CUDA_ROUTINE_HANDLER(MemPoolCreate);
CUDA_ROUTINE_HANDLER(MemPoolGetAttribute);
CUDA_ROUTINE_HANDLER(MemPoolSetAttribute);
CUDA_ROUTINE_HANDLER(MemPoolDestroy);

/* CudaRtHandler_stream */
CUDA_ROUTINE_HANDLER(StreamCreate);
CUDA_ROUTINE_HANDLER(StreamDestroy);
CUDA_ROUTINE_HANDLER(StreamQuery);
CUDA_ROUTINE_HANDLER(StreamSynchronize);
CUDA_ROUTINE_HANDLER(StreamCreateWithFlags);
CUDA_ROUTINE_HANDLER(StreamWaitEvent);
CUDA_ROUTINE_HANDLER(StreamCreateWithPriority);
CUDA_ROUTINE_HANDLER(ThreadExchangeStreamCaptureMode);
CUDA_ROUTINE_HANDLER(StreamIsCapturing);
CUDA_ROUTINE_HANDLER(StreamBeginCapture);
CUDA_ROUTINE_HANDLER(StreamEndCapture);

/* CudaRtHandler_graph */
CUDA_ROUTINE_HANDLER(GraphCreate);
CUDA_ROUTINE_HANDLER(GraphDestroy);
CUDA_ROUTINE_HANDLER(GraphLaunch);
CUDA_ROUTINE_HANDLER(GraphGetNodes);
CUDA_ROUTINE_HANDLER(GraphInstantiate);

/* CudaRtHandler_version */
CUDA_ROUTINE_HANDLER(DriverGetVersion);
CUDA_ROUTINE_HANDLER(RuntimeGetVersion);

/* Occupancy */
CUDA_ROUTINE_HANDLER(OccupancyMaxActiveBlocksPerMultiprocessor);
CUDA_ROUTINE_HANDLER(OccupancyMaxActiveBlocksPerMultiprocessorWithFlags);

/* CudaRtHandler_api */
CUDA_ROUTINE_HANDLER(FuncSetAttribute);

#endif /* _CUDARTHANDLER_H */
