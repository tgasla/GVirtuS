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
 *             angzam78 <angzam78@gmail.com>
 */

#include "NvmlHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

std::map<string, NvmlHandler::NvmlRoutineHandler>* NvmlHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<NvmlHandler> create_t() { return std::make_shared<NvmlHandler>(); }

extern "C" int HandlerInit() { return 0; }

NvmlHandler::NvmlHandler() {
    logger = Logger::getInstance(LOG4CPLUS_TEXT("NvmlHandler"));
    Initialize();
}

NvmlHandler::~NvmlHandler() {}

bool NvmlHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

std::shared_ptr<Result> NvmlHandler::Execute(std::string routine,
                                             std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger, "Called " << routine);
    map<string, NvmlHandler::NvmlRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end()) throw runtime_error("No handler for '" + routine + "' found!");
    try {
        return it->second(this, input_buffer);
    } catch (const std::exception& e) {
        LOG4CPLUS_DEBUG(logger, LOG4CPLUS_TEXT("Exception: ") << e.what());
    }
    return NULL;
}

void NvmlHandler::Initialize() {
    if (mspHandlers != NULL) return;
    mspHandlers = new map<string, NvmlHandler::NvmlRoutineHandler>();

    // Error reporting
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(ErrorString));
    // Initialization and Cleanup
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(InitWithFlags));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(Init));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(Shutdown));
    // System queries
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(SystemGetCudaDriverVersion_v2));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(SystemGetDriverVersion));
    // Device queries
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetCount));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetHandleByIndex));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetTemperature));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetIndex));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetName));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetPciInfo_v3));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetPersistenceMode));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetDisplayActive));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetEccMode));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetFanSpeed));
    //////////// VIBED ///////////////////
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetPerformanceState));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetPowerUsage));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetEnforcedPowerLimit));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetMemoryInfo));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetUtilizationRates));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetComputeMode));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetMigMode));
    // mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetVirtualizationMode));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceGetMaxMigDeviceCount));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceIsMigDeviceHandle));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(DeviceValidateInforom));

    // Event handling
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(EventSetCreate));
    mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(EventSetFree));
    // Internal
    // mspHandlers->insert(NVML_ROUTINE_HANDLER_PAIR(InternalGetExportTable));
}