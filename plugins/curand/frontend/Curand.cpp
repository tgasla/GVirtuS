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
 * Edited By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>,
 *            School of Computer Science, University College Dublin
 */

#include <mutex>
#include <unordered_map>

#include "CurandFrontend.h"

using namespace std;

static std::mutex generator_type_mutex;  // Mutex to protect access to generator_is_host_map
static std::unordered_map<curandGenerator_t, bool>
    generator_is_host_map;  // true if host, false if device

/* Helper Functions */

bool isHostGenerator(curandGenerator_t generator) {
    std::lock_guard<std::mutex> lock(generator_type_mutex);
    auto it = generator_is_host_map.find(generator);
    if (it != generator_is_host_map.end()) {
        return it->second;
    }
    // Default if unknown, assume device generator
    return false;
}

/* HOST API */

extern "C" curandStatus_t CURANDAPI curandCreateGenerator(curandGenerator_t* generator,
                                                          curandRngType_t rng_type) {
    CurandFrontend::Prepare();
    CurandFrontend::AddVariableForArguments<curandRngType_t>(rng_type);
    CurandFrontend::Execute("curandCreateGenerator");
    if (CurandFrontend::Success()) {
        *generator = CurandFrontend::GetOutputVariable<curandGenerator_t>();
        std::lock_guard<std::mutex> lock(generator_type_mutex);
        generator_is_host_map[*generator] = false;  // device generator
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t CURANDAPI curandCreateGeneratorHost(curandGenerator_t* generator,
                                                              curandRngType_t rng_type) {
    CurandFrontend::Prepare();
    CurandFrontend::AddVariableForArguments<curandRngType_t>(rng_type);
    CurandFrontend::Execute("curandCreateGeneratorHost");
    if (CurandFrontend::Success()) {
        *generator = CurandFrontend::GetOutputVariable<curandGenerator_t>();
        std::lock_guard<std::mutex> lock(generator_type_mutex);
        generator_is_host_map[*generator] = true;  // host generator
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                             unsigned long long seed) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<unsigned long long>(seed);
    CurandFrontend::Execute("curandSetPseudoRandomGeneratorSeed");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator,
                                                   unsigned long long offset) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<unsigned long long>(offset);
    CurandFrontend::Execute("curandSetGeneratorOffset");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator,
                                                                  unsigned int num_dimensions) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<unsigned int>(num_dimensions);
    CurandFrontend::Execute("curandSetQuasiRandomGeneratorDimensions");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int* outputPtr,
                                         size_t num) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::Execute("curandGenerate");

    if (isHostGenerator(generator) && CurandFrontend::Success()) {
        unsigned int* backend_output = CurandFrontend::GetOutputHostPointer<unsigned int>(num);
        std::memcpy(outputPtr, backend_output, sizeof(unsigned int) * num);
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateLongLong(curandGenerator_t generator,
                                                 unsigned long long* outputPtr, size_t num) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::Execute("curandGenerateLongLong");

    if (isHostGenerator(generator) && CurandFrontend::Success()) {
        unsigned long long* backend_output =
            CurandFrontend::GetOutputHostPointer<unsigned long long>(num);
        std::memcpy(outputPtr, backend_output, sizeof(unsigned long long) * num);
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* outputPtr,
                                                size_t num) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::Execute("curandGenerateUniform");

    if (isHostGenerator(generator) && CurandFrontend::Success()) {
        float* backend_output = CurandFrontend::GetOutputHostPointer<float>(num);
        std::memcpy(outputPtr, backend_output, sizeof(float) * num);
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateNormal(curandGenerator_t generator, float* outputPtr,
                                               size_t num, float mean, float stddev) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    CurandFrontend::AddVariableForArguments<float>(mean);
    CurandFrontend::AddVariableForArguments<float>(stddev);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::Execute("curandGenerateNormal");

    if (isHostGenerator(generator) && CurandFrontend::Success()) {
        float* backend_output = CurandFrontend::GetOutputHostPointer<float>(num);
        std::memcpy(outputPtr, backend_output, sizeof(float) * num);
    }
    return CurandFrontend::GetExitCode();
}

// Alternative implementation for curandGenerateNormal. Use only with the alternative implementation
// of backend also. The only difference is that here,
// in case of a host generator, we pass the whole output array to the backend.
// This is more clear, but we do not really need to pass the whole array, as the array before the
// execution of the function does not contain any meaningful data and also the backend will either
// way need to allocate a new array locally extern "C" curandStatus_t
// curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t num, float mean, float
// stddev) {
//     CurandFrontend::Prepare();
//     CurandFrontend::AddDevicePointerForArguments(generator);
//     CurandFrontend::AddVariableForArguments<size_t>(num);
//     CurandFrontend::AddVariableForArguments<float>(mean);
//     CurandFrontend::AddVariableForArguments<float>(stddev);
//     isHostGenerator(generator)
//     ? CurandFrontend::AddHostPointerForArguments<float>(outputPtr, num)
//     : CurandFrontend::AddDevicePointerForArguments(outputPtr);
//     CurandFrontend::Execute("curandGenerateNormal");

//     if (isHostGenerator(generator) && CurandFrontend::Success()) {
//         float* backend_output = CurandFrontend::GetOutputHostPointer<float>(num);
//         std::memcpy(outputPtr, backend_output, sizeof(float) * num);
//     }
//     return CurandFrontend::GetExitCode();
// }

extern "C" curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float* outputPtr,
                                                  size_t num, float mean, float stddev) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    CurandFrontend::AddVariableForArguments<float>(mean);
    CurandFrontend::AddVariableForArguments<float>(stddev);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::Execute("curandGenerateLogNormal");

    if (isHostGenerator(generator) && CurandFrontend::Success()) {
        float* backend_output = CurandFrontend::GetOutputHostPointer<float>(num);
        std::memcpy(outputPtr, backend_output, sizeof(float) * num);
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGeneratePoisson(curandGenerator_t generator,
                                                unsigned int* outputPtr, size_t num,
                                                double lambda) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    CurandFrontend::AddVariableForArguments<double>(lambda);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::Execute("curandGeneratePoisson");

    if (isHostGenerator(generator) && CurandFrontend::Success()) {
        unsigned int* backend_output = CurandFrontend::GetOutputHostPointer<unsigned int>(num);
        std::memcpy(outputPtr, backend_output, sizeof(unsigned int) * num);
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator,
                                                      double* outputPtr, size_t num) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::Execute("curandGenerateUniformDouble");

    if (isHostGenerator(generator) && CurandFrontend::Success()) {
        double* backend_output = CurandFrontend::GetOutputHostPointer<double>(num);
        std::memcpy(outputPtr, backend_output, sizeof(double) * num);
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double* outputPtr,
                                                     size_t n, double mean, double stddev) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<size_t>(n);
    CurandFrontend::AddVariableForArguments<double>(mean);
    CurandFrontend::AddVariableForArguments<double>(stddev);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::Execute("curandGenerateNormalDouble");

    if (isHostGenerator(generator) && CurandFrontend::Success()) {
        double* backend_output = CurandFrontend::GetOutputHostPointer<double>(n);
        std::memcpy(outputPtr, backend_output, sizeof(double) * n);
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator,
                                                        double* outputPtr, size_t n, double mean,
                                                        double stddev) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::AddVariableForArguments<size_t>(n);
    CurandFrontend::AddVariableForArguments<double>(mean);
    CurandFrontend::AddVariableForArguments<double>(stddev);
    CurandFrontend::AddDevicePointerForArguments(outputPtr);
    CurandFrontend::Execute("curandGenerateLogNormalDouble");

    if (isHostGenerator(generator) && CurandFrontend::Success()) {
        double* backend_output = CurandFrontend::GetOutputHostPointer<double>(n);
        std::memcpy(outputPtr, backend_output, sizeof(double) * n);
    }
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t CURANDAPI curandDestroyGenerator(curandGenerator_t generator) {
    CurandFrontend::Prepare();
    CurandFrontend::AddDevicePointerForArguments(generator);
    CurandFrontend::Execute("curandDestroyGenerator");

    if (CurandFrontend::Success()) {
        std::lock_guard<std::mutex> lock(generator_type_mutex);
        generator_is_host_map.erase(generator);
    }

    return CurandFrontend::GetExitCode();
}

// /* --- HOST API END --- */