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

#include "NvmlHandler.h"

using namespace std;
using namespace log4cplus;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

NVML_ROUTINE_HANDLER(DeviceGetCount) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetCount"));

    unsigned int deviceCount;
    
    nvmlReturn_t result = nvmlDeviceGetCount(&deviceCount);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(deviceCount);

    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetCount executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetHandleByIndex) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetHandleByIndex"));
    unsigned int index = in->Get<unsigned int>();
    nvmlDevice_t deviceHandle;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(index, &deviceHandle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add<nvmlDevice_t>(deviceHandle);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetHandleByIndex executed" );
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetTemperature) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetTemperature"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    nvmlTemperatureSensors_t sensorType = in->Get<nvmlTemperatureSensors_t>();
    unsigned int temp;
    nvmlReturn_t result = nvmlDeviceGetTemperature(*device, sensorType, &temp);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(temp);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetTemperature executed");
    return std::make_shared<Result>(result, out);
}
 
NVML_ROUTINE_HANDLER(DeviceGetIndex) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetIndex"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    unsigned int *index = in->Assign<unsigned int>();
    nvmlReturn_t result = nvmlDeviceGetIndex(*device, index);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(index);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetIndex executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetName) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetName"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    unsigned int length = in->Get<unsigned int>();
    char *name = new char[length];
    nvmlReturn_t result = nvmlDeviceGetName(*device, name, length);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(name, length);
    delete[] name;
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetName executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetPciInfo_v3) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetPciInfo_v3"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    nvmlPciInfo_t pci;
    nvmlReturn_t result = nvmlDeviceGetPciInfo_v3(*device, &pci);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(&pci);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetPciInfo_v3 executed");
    return std::make_shared<Result>(result, out);
}
    
NVML_ROUTINE_HANDLER(DeviceGetPersistenceMode) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetPersistenceMode"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    nvmlEnableState_t mode;
    nvmlReturn_t result = nvmlDeviceGetPersistenceMode(*device, &mode);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(&mode);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetPersistenceMode executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetDisplayActive) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetDisplayActive"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    nvmlEnableState_t mode;
    nvmlReturn_t result = nvmlDeviceGetDisplayActive(*device, &mode);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(&mode);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetDisplayActive executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetEccMode) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetEccMode"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    nvmlEnableState_t current, pending;
    nvmlReturn_t result = nvmlDeviceGetEccMode(*device, &current, &pending);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(&current);
    out->Add(&pending);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetEccMode executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetFanSpeed) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetFanSpeed"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    unsigned int speed;
    nvmlReturn_t result = nvmlDeviceGetFanSpeed(*device, &speed);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(speed);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetFanSpeed executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetPerformanceState) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetPerformanceState"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    nvmlPstates_t pState;
    nvmlReturn_t result = nvmlDeviceGetPerformanceState(*device, &pState);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(&pState);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetPerformanceState executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetPowerUsage) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetPowerUsage"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    unsigned int power;
    nvmlReturn_t result = nvmlDeviceGetPowerUsage(*device, &power);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(power);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetPowerUsage executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetEnforcedPowerLimit) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetEnforcedPowerLimit"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    unsigned int limit;
    nvmlReturn_t result = nvmlDeviceGetEnforcedPowerLimit(*device, &limit);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(limit);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetEnforcedPowerLimit executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetMemoryInfo) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetMemoryInfo"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    nvmlMemory_t memoryinfo;
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(*device, &memoryinfo);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(&memoryinfo);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetMemoryInfo executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetUtilizationRates) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetUtilizationRates"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    nvmlUtilization_t utilization;
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(*device, &utilization);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(&utilization);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetUtilizationRates executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetComputeMode) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetComputeMode"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    nvmlComputeMode_t mode;
    nvmlReturn_t result = nvmlDeviceGetComputeMode(*device, &mode);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(&mode);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetComputeMode executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetMigMode) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetMigMode"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    unsigned int current, pending;
    nvmlReturn_t result = nvmlDeviceGetMigMode(*device, &current, &pending);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(current);
    out->Add(pending);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetMigMode executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceGetMaxMigDeviceCount) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceGetMaxMigDeviceCount"));

    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    unsigned int count;
    nvmlReturn_t result = nvmlDeviceGetMaxMigDeviceCount(*device, &count);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(count);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceGetMaxMigDeviceCount executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceIsMigDeviceHandle) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceIsMigDeviceHandle"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    unsigned int isMig;
    nvmlReturn_t result = nvmlDeviceIsMigDeviceHandle(*device, &isMig);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(isMig);
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceIsMigDeviceHandle executed");
    return std::make_shared<Result>(result, out);
}

NVML_ROUTINE_HANDLER(DeviceValidateInforom) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DeviceValidateInforom"));
    nvmlDevice_t *device = in->Assign<nvmlDevice_t>();
    nvmlReturn_t result = nvmlDeviceValidateInforom(*device);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    LOG4CPLUS_DEBUG(logger, "nvmlDeviceValidateInforom executed");
    return std::make_shared<Result>(result, out);
}