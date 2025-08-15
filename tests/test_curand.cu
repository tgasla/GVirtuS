/*
 * Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             School of Computer Science, University College Dublin
 */

#include <curand.h>
#include <gtest/gtest.h>

#include <iostream>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess) << "CUDA error: " << cudaGetErrorString(err)
#define CURAND_CHECK(err) ASSERT_EQ((err), CURAND_STATUS_SUCCESS)

TEST(cuRAND, CreateDestroyGenerator) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandDestroyGenerator(generator));
}

TEST(cuRAND, CreateDestroyGeneratorHost) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandDestroyGenerator(generator));
}

TEST(cuRAND, SetSeed) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));
    CURAND_CHECK(curandDestroyGenerator(generator));
}

TEST(cuRAND, GenerateDevice) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));

    const size_t n = 10;
    unsigned int* output;
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(unsigned int)));

    CURAND_CHECK(curandGenerate(generator, output, n));

    unsigned int host_output[n];
    CUDA_CHECK(cudaMemcpy(host_output, output, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    bool all_zero = true;
    for (size_t i = 0; i < n; ++i) {
        if (host_output[i] != 0) {
            all_zero = false;
            break;
        }
    }
    ASSERT_FALSE(all_zero);  // Generated numbers should not all be zero

    CURAND_CHECK(curandDestroyGenerator(generator));
    CUDA_CHECK(cudaFree(output));
}

TEST(cuRAND, GenerateHost) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));

    const size_t n = 10;
    unsigned int* output = (unsigned int*)malloc(n * sizeof(unsigned int));
    ASSERT_NE(output, nullptr);

    CURAND_CHECK(curandGenerate(generator, output, n));

    bool all_zero = true;
    for (size_t i = 0; i < n; ++i) {
        if (output[i] != 0) {
            all_zero = false;
            break;
        }
    }
    ASSERT_FALSE(all_zero);  // Generated numbers should not all be zero

    CURAND_CHECK(curandDestroyGenerator(generator));
    free(output);
}

TEST(cuRAND, GenerateLongLongDevice) {
    curandGenerator_t generator;
    const size_t num = 10;

    // Create a quasi-random number generator
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_QUASI_SOBOL64));

    // Set dimensions (required for quasi generators)
    CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(generator, 1));

    // Allocate device memory
    unsigned long long* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, num * sizeof(unsigned long long)));

    // Generate quasi-random numbers
    CURAND_CHECK(curandGenerateLongLong(generator, d_output, num));

    // Copy results back to host for checking
    unsigned long long h_output[num];
    CUDA_CHECK(
        cudaMemcpy(h_output, d_output, num * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // Clean up
    CUDA_CHECK(cudaFree(d_output));
    CURAND_CHECK(curandDestroyGenerator(generator));
}

TEST(cuRAND, GenerateLongLongHost) {
    curandGenerator_t generator;
    const size_t num = 10;

    // Create a QUASI-random number generator (host generator)
    CURAND_CHECK(curandCreateGeneratorHost(&generator, CURAND_RNG_QUASI_SOBOL64));

    // Set dimensions (required for quasi generators)
    CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(generator, 1));

    // Allocate host memory for output
    unsigned long long* h_output = new unsigned long long[num];

    // Generate random numbers on host
    CURAND_CHECK(curandGenerateLongLong(generator, h_output, num));

    delete[] h_output;
    CURAND_CHECK(curandDestroyGenerator(generator));
}

TEST(cuRAND, GenerateUniformDevice) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));

    const size_t n = 10;
    float* output;
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(float)));

    CURAND_CHECK(curandGenerateUniform(generator, output, n));

    float host_output[n];
    CUDA_CHECK(cudaMemcpy(host_output, output, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        ASSERT_GE(host_output[i], 0.0f);
        ASSERT_LT(host_output[i], 1.0f);
    }

    CURAND_CHECK(curandDestroyGenerator(generator));
    CUDA_CHECK(cudaFree(output));
}

TEST(cuRAND, GenerateUniformHost) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL));

    const size_t n = 10;
    float* output = (float*)malloc(n * sizeof(float));
    ASSERT_NE(output, nullptr);

    CURAND_CHECK(curandGenerateUniform(generator, output, n));

    for (size_t i = 0; i < n; ++i) {
        ASSERT_GE(output[i], 0.0f);
        ASSERT_LT(output[i], 1.0f);
    }

    CURAND_CHECK(curandDestroyGenerator(generator));
    free(output);
}

TEST(cuRAND, GenerateNormalDevice) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 9012ULL));

    const size_t n = 1000;  // Larger sample for stats
    float* output;
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(float)));

    const float mean = 5.0f;
    const float stddev = 2.0f;
    CURAND_CHECK(curandGenerateNormal(generator, output, n, mean, stddev));

    float host_output[n];
    CUDA_CHECK(cudaMemcpy(host_output, output, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Basic sanity: check mean and stddev roughly
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += host_output[i];
    }
    float sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, mean, 0.2f);

    float variance_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = host_output[i] - mean;
        variance_sum += diff * diff;
    }
    float sample_stddev = sqrt(variance_sum / n);
    ASSERT_NEAR(sample_stddev, stddev, 0.3f);

    CURAND_CHECK(curandDestroyGenerator(generator));
    CUDA_CHECK(cudaFree(output));
}

TEST(cuRAND, GenerateNormalHost) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 9012ULL));

    const size_t n = 1000;
    float* output = (float*)malloc(n * sizeof(float));
    ASSERT_NE(output, nullptr);

    const float mean = 5.0f;
    const float stddev = 2.0f;
    CURAND_CHECK(curandGenerateNormal(generator, output, n, mean, stddev));

    // Basic sanity: check mean and stddev roughly
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += output[i];
    }
    float sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, mean, 0.2f);

    float variance_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = output[i] - mean;
        variance_sum += diff * diff;
    }
    float sample_stddev = sqrt(variance_sum / n);
    ASSERT_NEAR(sample_stddev, stddev, 0.3f);

    CURAND_CHECK(curandDestroyGenerator(generator));
    free(output);
}

TEST(cuRAND, GenerateLogNormalDevice) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 8642ULL));

    const size_t n = 1000;
    float* output;
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(float)));

    const float mean = 0.0f;
    const float stddev = 0.5f;
    CURAND_CHECK(curandGenerateLogNormal(generator, output, n, mean, stddev));

    float host_output[n];
    CUDA_CHECK(cudaMemcpy(host_output, output, n * sizeof(float), cudaMemcpyDeviceToHost));

    // All values should be positive
    for (size_t i = 0; i < n; ++i) {
        ASSERT_GT(host_output[i], 0.0f);
    }

    CURAND_CHECK(curandDestroyGenerator(generator));
    CUDA_CHECK(cudaFree(output));
}

TEST(cuRAND, GenerateLogNormalHost) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 8642ULL));

    const size_t n = 1000;
    float* output = (float*)malloc(n * sizeof(float));
    ASSERT_NE(output, nullptr);

    const float mean = 0.0f;
    const float stddev = 0.5f;
    CURAND_CHECK(curandGenerateLogNormal(generator, output, n, mean, stddev));

    // All values should be positive
    for (size_t i = 0; i < n; ++i) {
        ASSERT_GT(output[i], 0.0f);
    }

    CURAND_CHECK(curandDestroyGenerator(generator));
    free(output);
}

TEST(cuRAND, GeneratePoissonDevice) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 3456ULL));

    const size_t n = 1000;
    unsigned int* output;
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(unsigned int)));

    double lambda = 4.5;
    CURAND_CHECK(curandGeneratePoisson(generator, output, n, lambda));

    unsigned int host_output[n];
    CUDA_CHECK(cudaMemcpy(host_output, output, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Basic sanity checks: values >= 0 and mean close to lambda
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ASSERT_GE(host_output[i], 0u);
        sum += host_output[i];
    }
    double sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, lambda, 0.2);

    CURAND_CHECK(curandDestroyGenerator(generator));
    CUDA_CHECK(cudaFree(output));
}

TEST(cuRAND, GeneratePoissonHost) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 3456ULL));

    const size_t n = 1000;
    unsigned int* output = (unsigned int*)malloc(n * sizeof(unsigned int));
    ASSERT_NE(output, nullptr);

    double lambda = 4.5;
    CURAND_CHECK(curandGeneratePoisson(generator, output, n, lambda));

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        ASSERT_GE(output[i], 0u);
        sum += output[i];
    }
    double sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, lambda, 0.2);

    CURAND_CHECK(curandDestroyGenerator(generator));
    free(output);
}

TEST(cuRAND, GenerateUniformDoubleDevice) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 7890ULL));

    const size_t n = 10;
    double* output;
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(double)));

    CURAND_CHECK(curandGenerateUniformDouble(generator, output, n));

    double host_output[n];
    CUDA_CHECK(cudaMemcpy(host_output, output, n * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        ASSERT_GE(host_output[i], 0.0);
        ASSERT_LT(host_output[i], 1.0);
    }

    CURAND_CHECK(curandDestroyGenerator(generator));
    CUDA_CHECK(cudaFree(output));
}

TEST(cuRAND, GenerateUniformDoubleHost) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 7890ULL));

    const size_t n = 10;
    double* output = (double*)malloc(n * sizeof(double));
    ASSERT_NE(output, nullptr);

    CURAND_CHECK(curandGenerateUniformDouble(generator, output, n));

    for (size_t i = 0; i < n; ++i) {
        ASSERT_GE(output[i], 0.0);
        ASSERT_LT(output[i], 1.0);
    }

    CURAND_CHECK(curandDestroyGenerator(generator));
    free(output);
}

TEST(cuRAND, GenerateNormalDoubleDevice) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 2468ULL));

    const size_t n = 1000;
    double* output;
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(double)));

    const double mean = 10.0;
    const double stddev = 3.0;
    CURAND_CHECK(curandGenerateNormalDouble(generator, output, n, mean, stddev));

    double host_output[n];
    CUDA_CHECK(cudaMemcpy(host_output, output, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Basic sanity: check mean and stddev roughly
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += host_output[i];
    }
    double sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, mean, 0.2);

    double variance_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = host_output[i] - mean;
        variance_sum += diff * diff;
    }
    double sample_stddev = sqrt(variance_sum / n);
    ASSERT_NEAR(sample_stddev, stddev, 0.3);

    CURAND_CHECK(curandDestroyGenerator(generator));
    CUDA_CHECK(cudaFree(output));
}

TEST(cuRAND, GenerateNormalDoubleHost) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 2468ULL));

    const size_t n = 1000;
    double* output = (double*)malloc(n * sizeof(double));
    ASSERT_NE(output, nullptr);

    const double mean = 10.0;
    const double stddev = 3.0;
    CURAND_CHECK(curandGenerateNormalDouble(generator, output, n, mean, stddev));

    // Basic sanity: check mean and stddev roughly
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += output[i];
    }
    double sample_mean = sum / n;
    ASSERT_NEAR(sample_mean, mean, 0.2);

    double variance_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = output[i] - mean;
        variance_sum += diff * diff;
    }
    double sample_stddev = sqrt(variance_sum / n);
    ASSERT_NEAR(sample_stddev, stddev, 0.3);

    CURAND_CHECK(curandDestroyGenerator(generator));
    free(output);
}

TEST(cuRAND, GenerateLogNormalDoubleDevice) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1357ULL));

    const size_t n = 1000;
    double* output;
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(double)));

    const double mean = 0.0;    // mean of underlying normal
    const double stddev = 0.5;  // stddev of underlying normal
    CURAND_CHECK(curandGenerateLogNormalDouble(generator, output, n, mean, stddev));

    double host_output[n];
    CUDA_CHECK(cudaMemcpy(host_output, output, n * sizeof(double), cudaMemcpyDeviceToHost));

    // All outputs should be positive
    for (size_t i = 0; i < n; ++i) {
        ASSERT_GT(host_output[i], 0.0);
    }

    CURAND_CHECK(curandDestroyGenerator(generator));
    CUDA_CHECK(cudaFree(output));
}

TEST(cuRAND, GenerateLogNormalDoubleHost) {
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, 1357ULL));

    const size_t n = 1000;
    double* output = (double*)malloc(n * sizeof(double));
    ASSERT_NE(output, nullptr);

    const double mean = 0.0;
    const double stddev = 0.5;
    CURAND_CHECK(curandGenerateLogNormalDouble(generator, output, n, mean, stddev));

    // All outputs should be positive
    for (size_t i = 0; i < n; ++i) {
        ASSERT_GT(output[i], 0.0);
    }

    CURAND_CHECK(curandDestroyGenerator(generator));
    free(output);
}