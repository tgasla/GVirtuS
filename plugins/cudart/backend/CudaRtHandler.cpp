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

#include "CudaRtHandler.h"

using namespace std;
using namespace log4cplus;

map<string, CudaRtHandler::CudaRoutineHandler> *CudaRtHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CudaRtHandler> create_t() { return std::make_shared<CudaRtHandler>(); }

CudaRtHandler::CudaRtHandler() {
    logger = Logger::getInstance(LOG4CPLUS_TEXT("CudaRtHandler"));
    mpFatBinary = new map<string, void **>();
    mpDeviceFunction = new map<string, string>();
    mpVar = new map<string, string>();
    mpTexture = new map<string, cudaTextureObject_t *>();
    mpSurface = new map<string, cudaSurfaceObject_t *>();

    mapHost2DeviceFunc = new map<const void *, std::string>();
    mapDeviceFunc2InfoFunc = new map<std::string, NvInfoFunction>();
    Initialize();
}

CudaRtHandler::~CudaRtHandler() {}

bool CudaRtHandler::CanExecute(std::string routine) {
    map<string, CudaRtHandler::CudaRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end()) return false;
    return true;
}

std::shared_ptr<Result> CudaRtHandler::Execute(std::string routine,
                                               std::shared_ptr<Buffer> input_buffer) {
    map<string, CudaRtHandler::CudaRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    LOG4CPLUS_DEBUG(logger, "Called: " << routine);
    if (it == mspHandlers->end()) throw runtime_error("No handler for '" + routine + "' found!");
    return it->second(this, input_buffer);
}

void CudaRtHandler::RegisterFatBinary(std::string &handler, void **fatCubinHandle) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it != mpFatBinary->end()) {
        mpFatBinary->erase(it);
    }
    mpFatBinary->insert(make_pair(handler, fatCubinHandle));
    LOG4CPLUS_DEBUG(logger,
                    "Registered FatBinary " << fatCubinHandle << " with handler " << handler);
}

void CudaRtHandler::RegisterFatBinary(const char *handler, void **fatCubinHandle) {
    string tmp(handler);
    RegisterFatBinary(tmp, fatCubinHandle);
}

void **CudaRtHandler::GetFatBinary(string &handler) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it == mpFatBinary->end()) throw runtime_error("Fat Binary '" + handler + "' not found");
    return it->second;
}

void **CudaRtHandler::GetFatBinary(const char *handler) {
    string tmp(handler);
    return GetFatBinary(tmp);
}

void CudaRtHandler::UnregisterFatBinary(std::string &handler) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it == mpFatBinary->end()) return;
    /* FIXME: think about freeing memory */
    LOG4CPLUS_DEBUG(logger, "Unregistered FatBinary " << it->second << " with handler " << handler);
    mpFatBinary->erase(it);
}

void CudaRtHandler::UnregisterFatBinary(const char *handler) {
    string tmp(handler);
    UnregisterFatBinary(tmp);
}

void CudaRtHandler::RegisterDeviceFunction(std::string &handler, std::string &function) {
    map<string, string>::iterator it = mpDeviceFunction->find(handler);
    if (it != mpDeviceFunction->end()) mpDeviceFunction->erase(it);
    mpDeviceFunction->insert(make_pair(handler, function));
    LOG4CPLUS_DEBUG(logger,
                    "Registered DeviceFunction " << function << " with handler " << handler);
}

void CudaRtHandler::RegisterDeviceFunction(const char *handler, const char *function) {
    string tmp1(handler);
    string tmp2(function);
    RegisterDeviceFunction(tmp1, tmp2);
}

const char *CudaRtHandler::GetDeviceFunction(std::string &handler) {
    map<string, string>::iterator it = mpDeviceFunction->find(handler);
    if (it == mpDeviceFunction->end())
        throw runtime_error("Device Function '" + handler + "' not found");
    return it->second.c_str();
}

const char *CudaRtHandler::GetDeviceFunction(const char *handler) {
    string tmp(handler);
    return GetDeviceFunction(tmp);
}

void CudaRtHandler::RegisterVar(string &handler, string &symbol) {
    logger = Logger::getInstance(LOG4CPLUS_TEXT("RegisterVar"));
    LOG4CPLUS_DEBUG(logger, "Registering Var " << symbol << " with handler " << handler);
    mpVar->insert(make_pair(handler, symbol));
    LOG4CPLUS_DEBUG(logger, "Registered Var " << symbol << " with handler " << handler);
}

void CudaRtHandler::RegisterVar(const char *handler, const char *symbol) {
    logger = Logger::getInstance(LOG4CPLUS_TEXT("RegisterVar"));
    LOG4CPLUS_DEBUG(logger, "Registering Var " << symbol << " with handler " << handler);
    string tmp1(handler);
    string tmp2(symbol);
    RegisterVar(tmp1, tmp2);
}

const char *CudaRtHandler::GetVar(string &handler) {
    map<string, string>::iterator it = mpVar->find(handler);
    if (it == mpVar->end()) return NULL;
    return it->second.c_str();
}

const char *CudaRtHandler::GetVar(const char *handler) {
    string tmp(handler);
    return GetVar(tmp);
}

// void CudaRtHandler::RegisterTexture(string &handler, cudaTextureObject_t*
// texref) {
//   mpTexture->insert(make_pair(handler, texref));
//   LOG4CPLUS_DEBUG(
//       logger, "Registered Texture " << texref << " with handler " <<
//       handler);
// }

// void CudaRtHandler::RegisterTexture(const char *handler,
//                                     cudaTextureObject_t* texref) {
//   string tmp(handler);
//   RegisterTexture(tmp, texref);
// }

// void CudaRtHandler::RegisterSurface(string &handler,
//                                     cudaSurfaceObject_t* surfref) {
//   mpSurface->insert(make_pair(handler, surfref));
//   LOG4CPLUS_DEBUG(
//       logger, "Registered Surface " << surfref << " with handler " <<
//       handler);
// }

// void CudaRtHandler::RegisterSurface(const char *handler,
//                                     cudaSurfaceObject_t*surfref) {
//   string tmp(handler);
//   RegisterSurface(tmp, surfref);
// }

// cudaTextureObject_t*CudaRtHandler::GetTexture(string &handler) {
//   map<string, cudaTextureObject_t*>::iterator it = mpTexture->find(handler);
//   if (it == mpTexture->end()) return NULL;
//   return it->second;
// }

// cudaTextureObject_t*CudaRtHandler::GetTexture(const char *handler) {
//   string tmp(handler);
//   return GetTexture(tmp);
// }

// const char *CudaRtHandler::GetTextureHandler(cudaTextureObject_t* texref) {
//   for (map<string, cudaTextureObject_t*>::iterator it = mpTexture->begin();
//        it != mpTexture->end(); it++)
//     if (it->second == texref) return it->first.c_str();
//   return NULL;
// }

// cudaSurfaceObject_t*CudaRtHandler::GetSurface(string &handler) {
//   map<string, cudaSurfaceObject_t*>::iterator it = mpSurface->find(handler);
//   if (it == mpSurface->end()) return NULL;
//   return it->second;
// }

// cudaSurfaceObject_t*CudaRtHandler::GetSurface(const char *handler) {
//   string tmp(handler);
//   return GetSurface(tmp);
// }

// const char *CudaRtHandler::GetSurfaceHandler(cudaSurfaceObject_t*surfref) {
//   for (map<string, cudaSurfaceObject_t*>::iterator it = mpSurface->begin();
//        it != mpSurface->end(); it++)
//     if (it->second == surfref) return it->first.c_str();
//   return NULL;
// }

const char *CudaRtHandler::GetSymbol(std::shared_ptr<Buffer> in) {
    char *symbol_handler = in->AssignString();
    char *symbol = in->AssignString();
    char *our_symbol = const_cast<char *>(GetVar(symbol_handler));
    if (our_symbol != NULL) symbol = const_cast<char *>(our_symbol);
    return symbol;
}

void CudaRtHandler::Initialize() {
    if (mspHandlers != NULL) return;
    mspHandlers = new map<string, CudaRtHandler::CudaRoutineHandler>();

    /* CudaRtHandler_device */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ChooseDevice));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDevice));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDeviceCount));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDeviceProperties));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDevice));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceReset));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceSynchronize));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceSetCacheConfig));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceSetLimit));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceCanAccessPeer));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceDisablePeerAccess));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceEnablePeerAccess));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(IpcGetMemHandle));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(IpcGetEventHandle));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(IpcOpenEventHandle));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(IpcOpenMemHandle));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(OccupancyMaxActiveBlocksPerMultiprocessor));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceGetDefaultMemPool));
    mspHandlers->insert(
        CUDA_ROUTINE_HANDLER_PAIR(OccupancyMaxActiveBlocksPerMultiprocessorWithFlags));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceGetAttribute));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DeviceGetStreamPriorityRange));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDeviceFlags));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetValidDevices));
    /* CudaRtHandler_error */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetErrorString));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetLastError));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(PeekAtLastError));

    /* CudaRtHandler_event */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventCreate));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventCreateWithFlags));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventDestroy));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventElapsedTime));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventQuery));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventRecord));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventSynchronize));

    /* CudaRtHandler_execution */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ConfigureCall));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(FuncGetAttributes));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Launch));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDoubleForDevice));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDoubleForHost));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetupArgument));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(PushCallConfiguration));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(PopCallConfiguration));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(LaunchKernel));
    /* CudaRtHandler_internal */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterFatBinary));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterFatBinaryEnd));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(UnregisterFatBinary));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterFunction));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterVar));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterSharedVar));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterShared));

    /* CudaRtHandler_memory */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemGetInfo));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Free));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(FreeArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetSymbolAddress));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetSymbolSize));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Malloc));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MallocArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MallocManaged));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MallocPitch));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy2D));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy3D));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyAsync));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyFromSymbol));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyToArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyToSymbol));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memset));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memset2D));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyFromArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyArrayToArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy2DFromArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy2DToArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Malloc3DArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyPeerAsync));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(HostRegister));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(HostUnregister));

    /* CudaRtHandler_opengl */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GLSetGLDevice));  // deprecated
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsGLRegisterBuffer));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsMapResources));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsResourceGetMappedPointer));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsUnmapResources));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsUnregisterResource));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphicsResourceSetMapFlags));

    /* CudaRtHandler_stream_memory */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemPoolCreate));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemPoolGetAttribute));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemPoolSetAttribute));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemPoolDestroy));

    /* CudaRtHandler_stream */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamCreate));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamDestroy));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamQuery));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamSynchronize));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamCreateWithFlags));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamWaitEvent));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamCreateWithPriority));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ThreadExchangeStreamCaptureMode));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamIsCapturing));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamBeginCapture));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamEndCapture));

    /* CudaRtHandler_graph */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphCreate));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphDestroy));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphLaunch));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphGetNodes));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GraphInstantiate));

    /* CudaRtHandler_version */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DriverGetVersion));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RuntimeGetVersion));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(FuncSetCacheConfig));
    /* CudaRtHandler_api*/
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(FuncSetAttribute));
}