/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 *  This file is part of gVirtuS.
 *
 *  gVirtuS is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  gVirtuS is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with gVirtuS; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *              Department of Computer Science, University College Dublin
 *
 *              angzam78 <angzam78@gmail.com>
 */

#ifndef NVMLHANDLER_H
#define NVMLHANDLER_H

#include <gvirtus/backend/Handler.h>
#include <gvirtus/communicators/Result.h>
#include <limits.h>
#include <nvml.h>

#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

#if (__WORDSIZE == 64)
#define BUILD_64 1
#endif
using namespace std;
using namespace log4cplus;

class NvmlHandler : public gvirtus::backend::Handler {
   public:
    NvmlHandler();
    virtual ~NvmlHandler();
    bool CanExecute(std::string routine);
    std::shared_ptr<gvirtus::communicators::Result> Execute(
        std::string routine, std::shared_ptr<gvirtus::communicators::Buffer> input_buffer);
    log4cplus::Logger &GetLogger() { return logger; }

   private:
    log4cplus::Logger logger;
    void Initialize();
    typedef std::shared_ptr<gvirtus::communicators::Result> (*NvmlRoutineHandler)(
        NvmlHandler *, std::shared_ptr<gvirtus::communicators::Buffer>);
    static std::map<std::string, NvmlRoutineHandler> *mspHandlers;
};

#define NVML_ROUTINE_HANDLER(name)                                \
    std::shared_ptr<gvirtus::communicators::Result> handle##name( \
        NvmlHandler *pThis, std::shared_ptr<gvirtus::communicators::Buffer> in)
#define NVML_ROUTINE_HANDLER_PAIR(name) make_pair("nvml" #name, handle##name)

// Error reporting
NVML_ROUTINE_HANDLER(ErrorString);
// Initialization and cleanup
NVML_ROUTINE_HANDLER(InitWithFlags);
NVML_ROUTINE_HANDLER(Init);
NVML_ROUTINE_HANDLER(Shutdown);
// System queries
NVML_ROUTINE_HANDLER(SystemGetCudaDriverVersion_v2);
NVML_ROUTINE_HANDLER(SystemGetDriverVersion);
// Device queries
NVML_ROUTINE_HANDLER(DeviceGetCount);
NVML_ROUTINE_HANDLER(DeviceGetHandleByIndex);
NVML_ROUTINE_HANDLER(DeviceGetTemperature);
NVML_ROUTINE_HANDLER(DeviceGetIndex);
NVML_ROUTINE_HANDLER(DeviceGetName);
NVML_ROUTINE_HANDLER(DeviceGetPciInfo_v3);
NVML_ROUTINE_HANDLER(DeviceGetPersistenceMode);
NVML_ROUTINE_HANDLER(DeviceGetDisplayActive);
NVML_ROUTINE_HANDLER(DeviceGetEccMode);
NVML_ROUTINE_HANDLER(DeviceGetFanSpeed);
//////////// VIBED ///////////////////
NVML_ROUTINE_HANDLER(DeviceGetPerformanceState);
NVML_ROUTINE_HANDLER(DeviceGetPowerUsage);
NVML_ROUTINE_HANDLER(DeviceGetEnforcedPowerLimit);
NVML_ROUTINE_HANDLER(DeviceGetMemoryInfo);
NVML_ROUTINE_HANDLER(DeviceGetUtilizationRates);
NVML_ROUTINE_HANDLER(DeviceGetComputeMode);
NVML_ROUTINE_HANDLER(DeviceGetMigMode);
// NVML_ROUTINE_HANDLER(DeviceGetVirtualizationMode);
NVML_ROUTINE_HANDLER(DeviceGetMaxMigDeviceCount);
NVML_ROUTINE_HANDLER(DeviceIsMigDeviceHandle);
NVML_ROUTINE_HANDLER(DeviceValidateInforom);

// Event handling
NVML_ROUTINE_HANDLER(EventSetCreate);
NVML_ROUTINE_HANDLER(EventSetFree);
// Internal
// NVML_ROUTINE_HANDLER(InternalGetExportTable);

#endif /* NVMLHANDLER_H */
