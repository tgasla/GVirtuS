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

#include "NvmlFrontend.h"

using namespace std;

extern "C" nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
    NvmlFrontend::Prepare();
    NvmlFrontend::Execute("nvmlDeviceGetCount");
    if (NvmlFrontend::Success()) *deviceCount = NvmlFrontend::GetOutputVariable<unsigned int>();
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddVariableForArguments(index);
    NvmlFrontend::AddDevicePointerForArguments(device);
    NvmlFrontend::Execute("nvmlDeviceGetHandleByIndex");
    if (NvmlFrontend::Success()) {
        *device = NvmlFrontend::GetOutputVariable<nvmlDevice_t>();
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device,
                                                 nvmlTemperatureSensors_t sensorType,
                                                 unsigned int *temp) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::AddVariableForArguments(sensorType);
    NvmlFrontend::Execute("nvmlDeviceGetTemperature");
    if (NvmlFrontend::Success()) *temp = NvmlFrontend::GetOutputVariable<unsigned int>();
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::AddHostPointerForArguments(index);
    NvmlFrontend::Execute("nvmlDeviceGetIndex");
    if (NvmlFrontend::Success()) {
        unsigned int *index_backend = NvmlFrontend::GetOutputHostPointer<unsigned int>();
        std::memcpy(index, index_backend, sizeof(unsigned int));
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::AddVariableForArguments(length);
    NvmlFrontend::Execute("nvmlDeviceGetName");
    if (NvmlFrontend::Success()) {
        char *name_backend = NvmlFrontend::GetOutputHostPointer<char>();
        std::memcpy(name, name_backend, length);
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetPciInfo_v3");
    if (NvmlFrontend::Success()) {
        nvmlPciInfo_t *pci_backend = NvmlFrontend::GetOutputHostPointer<nvmlPciInfo_t>();
        std::memcpy(pci, pci_backend, sizeof(nvmlPciInfo_t));
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetPersistenceMode");
    if (NvmlFrontend::Success()) {
        nvmlEnableState_t *mode_backend = NvmlFrontend::GetOutputHostPointer<nvmlEnableState_t>();
        std::memcpy(mode, mode_backend, sizeof(nvmlEnableState_t));
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t *mode) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetDisplayActive");
    if (NvmlFrontend::Success()) {
        nvmlEnableState_t *mode_backend = NvmlFrontend::GetOutputHostPointer<nvmlEnableState_t>();
        std::memcpy(mode, mode_backend, sizeof(nvmlEnableState_t));
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current,
                                             nvmlEnableState_t *pending) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetEccMode");
    if (NvmlFrontend::Success()) {
        nvmlEnableState_t *current_backend =
            NvmlFrontend::GetOutputHostPointer<nvmlEnableState_t>();
        std::memcpy(current, current_backend, sizeof(nvmlEnableState_t));
        nvmlEnableState_t *pending_backend =
            NvmlFrontend::GetOutputHostPointer<nvmlEnableState_t>();
        std::memcpy(pending, pending_backend, sizeof(nvmlEnableState_t));
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetFanSpeed");
    if (NvmlFrontend::Success()) *speed = NvmlFrontend::GetOutputVariable<unsigned int>();
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t *pState) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetPerformanceState");
    if (NvmlFrontend::Success()) *pState = NvmlFrontend::GetOutputVariable<nvmlPstates_t>();
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetPowerUsage");
    if (NvmlFrontend::Success()) *power = NvmlFrontend::GetOutputVariable<unsigned int>();
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetEnforcedPowerLimit");
    if (NvmlFrontend::Success()) *limit = NvmlFrontend::GetOutputVariable<unsigned int>();
    return NvmlFrontend::GetExitCode();
}

// TODO: check issues with nvmlDeviceGetMemoryInfo
extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetMemoryInfo");
    if (NvmlFrontend::Success()) {
        nvmlMemory_t *memory_backend = NvmlFrontend::GetOutputHostPointer<nvmlMemory_t>();
        std::memcpy(memory, memory_backend, sizeof(nvmlMemory_t));
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device,
                                                      nvmlUtilization_t *utilization) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetUtilizationRates");
    if (NvmlFrontend::Success()) {
        nvmlUtilization_t *utilization_backend =
            NvmlFrontend::GetOutputHostPointer<nvmlUtilization_t>();
        std::memcpy(utilization, utilization_backend, sizeof(nvmlUtilization_t));
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t *mode) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetComputeMode");
    if (NvmlFrontend::Success()) {
        nvmlComputeMode_t *mode_backend = NvmlFrontend::GetOutputHostPointer<nvmlComputeMode_t>();
        std::memcpy(mode, mode_backend, sizeof(nvmlComputeMode_t));
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *current,
                                             unsigned int *pending) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetMigMode");
    if (NvmlFrontend::Success()) {
        *current = NvmlFrontend::GetOutputVariable<unsigned int>();
        *pending = NvmlFrontend::GetOutputVariable<unsigned int>();
    }
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceGetMaxMigDeviceCount");
    if (NvmlFrontend::Success()) *count = NvmlFrontend::GetOutputVariable<unsigned int>();
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device,
                                                    unsigned int *isMigDevice) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceIsMigDeviceHandle");
    if (NvmlFrontend::Success()) *isMigDevice = NvmlFrontend::GetOutputVariable<unsigned int>();
    return NvmlFrontend::GetExitCode();
}

extern "C" nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) {
    NvmlFrontend::Prepare();
    NvmlFrontend::AddHostPointerForArguments(&device);
    NvmlFrontend::Execute("nvmlDeviceValidateInforom");
    return NvmlFrontend::GetExitCode();
}

//// API Compatibility //////////////////////////////////////////////////

#if CUDA_VERSION <= 12060

typedef struct {
    unsigned int version;
    nvmlTemperatureSensors_t sensorType;
    int temperature;
} nvmlTemperature_v1_t;

typedef nvmlTemperature_v1_t nvmlTemperature_t;

extern "C" nvmlReturn_t nvmlDeviceGetTemperatureV(nvmlDevice_t device, nvmlTemperature_t *temp) {
    unsigned int temperature;
    nvmlReturn_t ret = nvmlDeviceGetTemperature(device, temp->sensorType, &temperature);
    temp->temperature = temperature;
    return ret;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory) {
    nvmlMemory_t memory_v1;
    nvmlReturn_t ret = nvmlDeviceGetMemoryInfo(device, &memory_v1);
    if (ret == NVML_SUCCESS) {
        memory->version = 2;
        memory->total = memory_v1.total;
        memory->free = memory_v1.free;
        memory->used = memory_v1.used;
    }
    return ret;
}

#endif