/*
 * Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             School of Computer Science, University College Dublin
 */

#include <cufft.h>
#include <cufftXt.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess) << "CUDA error: " << cudaGetErrorString(err)
#define CUFFT_CHECK(err) ASSERT_EQ((err), CUFFT_SUCCESS)

TEST(cuFFT, cufftCreateDestroy) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));

    // Check if the plan was created successfully
    ASSERT_NE(plan, 0);

    // Clean up
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, Plan1D) {
    const int N = 8;
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    // Check if the plan was created successfully
    ASSERT_NE(plan, 0);

    // Clean up
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, Plan2D) {
    const int NX = 4, NY = 4;
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, NX, NY, CUFFT_C2C));

    // Check if the plan was created successfully
    ASSERT_NE(plan, 0);

    // Clean up
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, Plan3D) {
    const int NX = 4, NY = 4, NZ = 4;
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan3d(&plan, NX, NY, NZ, CUFFT_C2C));
    ASSERT_NE(plan, 0);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, PlanMany) {
    const int rank = 1;
    int n[] = {8};
    const int batch = 2;
    cufftHandle plan;
    CUFFT_CHECK(cufftPlanMany(&plan, rank, n, nullptr, 1, 0,  // inembed, istride, idist
                              nullptr, 1, 0,                  // onembed, ostride, odist
                              CUFFT_C2C, batch));
    ASSERT_NE(plan, 0);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, Estimate1D) {
    size_t workSize = 0;
    CUFFT_CHECK(cufftEstimate1d(8, CUFFT_C2C, 1, &workSize));
    ASSERT_GT(workSize, 0u);
}

TEST(cuFFT, Estimate2D) {
    size_t workSize = 0;
    CUFFT_CHECK(cufftEstimate2d(4, 4, CUFFT_C2C, &workSize));
    ASSERT_GT(workSize, 0u);
}

TEST(cuFFT, Estimate3D) {
    size_t workSize = 0;
    CUFFT_CHECK(cufftEstimate3d(4, 4, 4, CUFFT_C2C, &workSize));
    ASSERT_GT(workSize, 0u);
}

TEST(cuFFT, EstimateMany) {
    int n[] = {8};
    size_t workSize = 0;
    CUFFT_CHECK(cufftEstimateMany(1, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 2, &workSize));
    ASSERT_GT(workSize, 0u);
}

TEST(cuFFT, Plan1DAndExecC2C) {
    const int N = 8;
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    std::vector<cufftComplex> input(N), output(N);

    // Fill input with known values
    for (int i = 0; i < N; ++i) {
        input[i].x = static_cast<float>(i);
        input[i].y = 0.0f;
    }

    cufftComplex *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(cufftComplex) * N));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), sizeof(cufftComplex) * N, cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftExecC2C(plan, d_input, d_input, CUFFT_FORWARD));
    CUDA_CHECK(
        cudaMemcpy(output.data(), d_input, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost));

    // Optional: Check at least that output is not all zero
    float totalEnergy = 0.0f;
    for (auto &v : output) {
        totalEnergy += v.x * v.x + v.y * v.y;
    }
    ASSERT_GT(totalEnergy, 0.0f);

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_input));
}

TEST(cuFFT, CreateMakePlan1dAndExecC2C) {
    const int N = 8;  // FFT size

    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));

    size_t workSize = 0;
    CUFFT_CHECK(cufftMakePlan1d(plan, N, CUFFT_C2C, 1, &workSize));

    cufftComplex *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(cufftComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(cufftComplex) * N));

    // Initialize input on host
    cufftComplex h_in[N];
    for (int i = 0; i < N; ++i) {
        h_in[i].x = static_cast<float>(i);
        h_in[i].y = 0.0f;
    }

    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));

    // Execute FFT
    CUFFT_CHECK(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));

    // Copy result back
    cufftComplex h_out[N];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

    // Basic check: output should not all be zero
    bool all_zero = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i].x != 0 || h_out[i].y != 0) {
            all_zero = false;
            break;
        }
    }
    ASSERT_FALSE(all_zero) << "All FFT output values are zero — transform likely failed.";

    // Cleanup
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

TEST(cuFFT, MakePlan2D) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    size_t workSize = 0;
    CUFFT_CHECK(cufftMakePlan2d(plan, 4, 4, CUFFT_C2C, &workSize));
    ASSERT_GT(workSize, 0u);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, MakePlan3D) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    size_t workSize = 0;
    CUFFT_CHECK(cufftMakePlan3d(plan, 4, 4, 4, CUFFT_C2C, &workSize));
    ASSERT_GT(workSize, 0u);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, MakePlanMany) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));

    int n[] = {8};
    size_t workSize = 0;

    CUFFT_CHECK(
        cufftMakePlanMany(plan, 1, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 2, &workSize));

    ASSERT_GT(workSize, 0u);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, MakePlanMany64) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));

    long long int n[] = {8};
    long long int inembed[] = {8};
    long long int onembed[] = {8};
    long long int istride = 1, ostride = 1;
    long long int idist = 8, odist = 8;
    long long int howmany = 2;
    size_t workSize = 0;

    CUFFT_CHECK(cufftMakePlanMany64(plan, 1, n, inembed, istride, idist, onembed, ostride, odist,
                                    CUFFT_C2C, howmany, &workSize));

    ASSERT_GT(workSize, 0u);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, Plan2DAndExecC2C) {
    const int NX = 4, NY = 4;
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, NX, NY, CUFFT_C2C));

    std::vector<cufftComplex> input(NX * NY), output(NX * NY);

    for (int i = 0; i < NX * NY; ++i) {
        input[i].x = static_cast<float>(i);
        input[i].y = 0.0f;
    }

    cufftComplex *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(cufftComplex) * NX * NY));
    CUDA_CHECK(
        cudaMemcpy(d_input, input.data(), sizeof(cufftComplex) * NX * NY, cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftExecC2C(plan, d_input, d_input, CUFFT_FORWARD));
    CUDA_CHECK(
        cudaMemcpy(output.data(), d_input, sizeof(cufftComplex) * NX * NY, cudaMemcpyDeviceToHost));

    // Check for non-zero output
    float energy = 0;
    for (auto &v : output) {
        energy += v.x * v.x + v.y * v.y;
    }
    ASSERT_GT(energy, 0.0f);

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_input));
}

TEST(cuFFT, ExecR2CAndC2R) {
    const int N = 16;
    cufftHandle plan_fwd, plan_inv;

    CUFFT_CHECK(cufftPlan1d(&plan_fwd, N, CUFFT_R2C, 1));
    CUFFT_CHECK(cufftPlan1d(&plan_inv, N, CUFFT_C2R, 1));

    std::vector<float> input(N);
    std::vector<float> output(N);
    std::vector<cufftComplex> spectrum(N / 2 + 1);

    for (int i = 0; i < N; ++i) input[i] = sin(2 * M_PI * i / N);  // A sine wave

    float *d_input;
    cufftComplex *d_spectrum;
    float *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_spectrum, sizeof(cufftComplex) * (N / 2 + 1)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * N));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftExecR2C(plan_fwd, d_input, d_spectrum));
    CUFFT_CHECK(cufftExecC2R(plan_inv, d_spectrum, d_output));

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, sizeof(float) * N, cudaMemcpyDeviceToHost));

    // Normalize inverse FFT output
    for (int i = 0; i < N; ++i) output[i] /= N;

    for (int i = 0; i < N; ++i) ASSERT_NEAR(input[i], output[i], 1e-3f);

    CUFFT_CHECK(cufftDestroy(plan_fwd));
    CUFFT_CHECK(cufftDestroy(plan_inv));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_spectrum));
    CUDA_CHECK(cudaFree(d_output));
}

TEST(cuFFT, GetSize1D) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    size_t workSize = 0;
    CUFFT_CHECK(cufftGetSize1d(plan, 8, CUFFT_C2C, 1, &workSize));
    ASSERT_GT(workSize, 0u);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, GetSize2D) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    size_t workSize = 0;
    CUFFT_CHECK(cufftGetSize2d(plan, 4, 4, CUFFT_C2C, &workSize));
    ASSERT_GT(workSize, 0u);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, GetSize3D) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    size_t workSize = 0;
    CUFFT_CHECK(cufftGetSize3d(plan, 4, 4, 4, CUFFT_C2C, &workSize));
    ASSERT_GT(workSize, 0u);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, GetSizeMany) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    int n[] = {8};
    size_t workSize = 0;
    CUFFT_CHECK(
        cufftGetSizeMany(plan, 1, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 2, &workSize));
    ASSERT_GT(workSize, 0u);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, GetSizeMany64) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    long long int n[] = {8};
    long long int inembed[] = {8};
    long long int onembed[] = {8};
    size_t workSize = 0;
    CUFFT_CHECK(
        cufftGetSizeMany64(plan, 1, n, inembed, 1, 8, onembed, 1, 8, CUFFT_C2C, 2, &workSize));
    ASSERT_GT(workSize, 0u);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, GetSize) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    int n[] = {8};
    size_t workSize = 0;
    CUFFT_CHECK(
        cufftMakePlanMany(plan, 1, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 1, &workSize));
    size_t queriedSize = 0;
    CUFFT_CHECK(cufftGetSize(plan, &queriedSize));
    ASSERT_EQ(workSize, queriedSize);
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, SetWorkArea) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    void *workArea;
    CUDA_CHECK(cudaMalloc(&workArea, 1024));
    CUFFT_CHECK(cufftSetWorkArea(plan, workArea));
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(workArea));
}

TEST(cuFFT, SetAutoAllocation) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftSetAutoAllocation(plan, 0));  // Manual mode
    CUFFT_CHECK(cufftSetAutoAllocation(plan, 1));  // Back to default
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, GetVersion) {
    int version = 0;
    CUFFT_CHECK(cufftGetVersion(&version));
    ASSERT_GT(version, 0);
}

TEST(cuFFT, SetStream) {
    cufftHandle plan;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUFFT_CHECK(cufftPlan1d(&plan, 16, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftSetStream(plan, stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, GetProperty) {
    int property = 0;
    CUFFT_CHECK(cufftGetProperty(MAJOR_VERSION, &property));
    ASSERT_GT(property, 0);
}

TEST(cuFFT, ExecZ2Z) {
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, 8, CUFFT_Z2Z, 1));

    cufftDoubleComplex *data;
    CUDA_CHECK(cudaMalloc(&data, sizeof(cufftDoubleComplex) * 8));

    cufftDoubleComplex h_data[8];
    for (int i = 0; i < 8; i++) {
        h_data[i].x = i;
        h_data[i].y = 0;
    }
    CUDA_CHECK(cudaMemcpy(data, h_data, sizeof(h_data), cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftExecZ2Z(plan, data, data, CUFFT_FORWARD));

    CUDA_CHECK(cudaMemcpy(h_data, data, sizeof(h_data), cudaMemcpyDeviceToHost));

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(data));
}

TEST(cuFFT, ExecD2Z) {
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, 8, CUFFT_D2Z, 1));

    double *idata;
    cufftDoubleComplex *odata;
    CUDA_CHECK(cudaMalloc(&idata, sizeof(double) * 8));
    CUDA_CHECK(cudaMalloc(&odata, sizeof(cufftDoubleComplex) * 5));

    double h_idata[8];
    for (int i = 0; i < 8; i++) h_idata[i] = i;
    CUDA_CHECK(cudaMemcpy(idata, h_idata, sizeof(h_idata), cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftExecD2Z(plan, idata, odata));

    // Optionally copy output back to host for inspection
    cufftDoubleComplex h_odata[5];
    CUDA_CHECK(cudaMemcpy(h_odata, odata, sizeof(h_odata), cudaMemcpyDeviceToHost));

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(idata));
    CUDA_CHECK(cudaFree(odata));
}

TEST(cuFFT, ExecZ2D) {
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, 8, CUFFT_Z2D, 1));

    cufftDoubleComplex *idata;
    double *odata;
    CUDA_CHECK(cudaMalloc(&idata, sizeof(cufftDoubleComplex) * 5));
    CUDA_CHECK(cudaMalloc(&odata, sizeof(double) * 8));

    cufftDoubleComplex h_idata[5];
    for (int i = 0; i < 5; i++) {
        h_idata[i].x = i;
        h_idata[i].y = 0;
    }
    CUDA_CHECK(cudaMemcpy(idata, h_idata, sizeof(h_idata), cudaMemcpyHostToDevice));

    CUFFT_CHECK(cufftExecZ2D(plan, idata, odata));

    double h_odata[8];
    CUDA_CHECK(cudaMemcpy(h_odata, odata, sizeof(h_odata), cudaMemcpyDeviceToHost));

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(idata));
    CUDA_CHECK(cudaFree(odata));
}

TEST(cuFFT, XtSetGPUs) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2 || deviceCount > 8) GTEST_SKIP() << "Test requires 2-8 GPUs";

    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));  // Step 1: Create

    int gpus[1] = {0};
    CUFFT_CHECK(cufftXtSetGPUs(plan, 1, gpus));  // Step 2: Set GPUs

    int n[1] = {8};  // 1D FFT of length 8
    size_t workSize = 0;
    CUFFT_CHECK(cufftMakePlanMany(plan, 1, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 1,
                                  &workSize));  // Step 3: Make Plan

    CUFFT_CHECK(cufftDestroy(plan));
}

// Not supported by GVirtuS
TEST(cuFFT, XtMakePlanMany) {
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    long long int n[] = {8};
    long long int inembed[] = {8};
    long long int onembed[] = {8};
    long long int howmany = 1;
    size_t workSize = 0;
    CUFFT_CHECK(cufftXtMakePlanMany(plan, 1, n, inembed, 1, 8, CUDA_C_32F, onembed, 1, 8,
                                    CUDA_C_32F, howmany, &workSize, CUDA_C_32F));
    CUFFT_CHECK(cufftDestroy(plan));
}

TEST(cuFFT, XtMallocFree) {
    constexpr size_t N = 8;
    cufftHandle plan;
    cudaLibXtDesc *deviceDesc = nullptr;

    // Create 1D plan
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    // Allocate device memory using XtMalloc
    CUFFT_CHECK(cufftXtMalloc(plan, &deviceDesc, CUFFT_XT_FORMAT_INPLACE));
    ASSERT_NE(deviceDesc, nullptr);

    // Free the memory using XtFree
    CUFFT_CHECK(cufftXtFree(deviceDesc));

    // Destroy plan
    CUFFT_CHECK(cufftDestroy(plan));
}

// TEST(cuFFT, CufftXtMallocMemcpyFree) {
//     constexpr size_t N = 8;
//     cufftHandle plan;
//     cudaLibXtDesc* deviceDesc = nullptr;
//     float* hostData = new float[N];

//     for (size_t i = 0; i < N; ++i)
//         hostData[i] = static_cast<float>(i);

//     // Create cuFFT plan
//     CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, 1));

//     // Allocate memory using Xt API
//     CUFFT_CHECK(cufftXtMalloc(plan, &deviceDesc, CUFFT_XT_FORMAT_INPLACE));

//     // Use the opaque descriptor directly in XtMemcpy
//     CUFFT_CHECK(cufftXtMemcpy(plan,
//                               reinterpret_cast<void*>(deviceDesc),
//                               reinterpret_cast<void*>(hostData),
//                               CUFFT_COPY_HOST_TO_DEVICE));

//     // Clear hostData before copying back
//     for (size_t i = 0; i < N; ++i)
//         hostData[i] = 0.0f;

//     // Copy back from device to host
//     CUFFT_CHECK(cufftXtMemcpy(plan,
//                               reinterpret_cast<void*>(hostData),
//                               reinterpret_cast<void*>(deviceDesc),
//                               CUFFT_COPY_DEVICE_TO_HOST));

//     // Check round-trip correctness
//     for (size_t i = 0; i < N; ++i)
//         ASSERT_FLOAT_EQ(hostData[i], static_cast<float>(i));

//     // Cleanup
//     // CUFFT_CHECK(cufftXtFree(deviceDesc));
//     CUFFT_CHECK(cufftDestroy(plan));
//     delete[] hostData;
// }

// TEST(cuFFT, CufftXtExecDescriptorC2C) {
//     // int deviceCount = 0;
//     // CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
//     // if (deviceCount < 2)
//     //     GTEST_SKIP() << "Test requires multiple GPUs";

//     constexpr size_t N = 8;
//     cufftHandle planFwd, planInv;
//     cudaLibXtDesc* deviceDescFwd = nullptr;
//     cudaLibXtDesc* deviceDescInv = nullptr;
//     cufftComplex* hostData = new cufftComplex[N];

//     // Initialize host input with complex numbers
//     for (size_t i = 0; i < N; ++i) {
//         hostData[i].x = static_cast<float>(i);
//         hostData[i].y = 0.0f;
//     }

//     // Create forward and inverse cuFFT plans
//     CUFFT_CHECK(cufftPlan1d(&planFwd, N, CUFFT_C2C, 1));
//     CUFFT_CHECK(cufftPlan1d(&planInv, N, CUFFT_C2C, 1));

//     // Allocate Xt memory
//     CUFFT_CHECK(cufftXtMalloc(planFwd, &deviceDescFwd, CUFFT_XT_FORMAT_INPLACE));
//     CUFFT_CHECK(cufftXtMalloc(planInv, &deviceDescInv, CUFFT_XT_FORMAT_INPLACE));

//     // Host to device
//     CUFFT_CHECK(cufftXtMemcpy(planFwd,
//                               reinterpret_cast<void*>(deviceDescFwd),
//                               reinterpret_cast<void*>(hostData),
//                               CUFFT_COPY_HOST_TO_DEVICE));

//     // Execute forward FFT
//     CUFFT_CHECK(cufftXtExecDescriptorC2C(planFwd, deviceDescFwd, deviceDescInv, CUFFT_FORWARD));

//     // Execute inverse FFT (in-place)
//     CUFFT_CHECK(cufftXtExecDescriptorC2C(planInv, deviceDescInv, deviceDescFwd, CUFFT_INVERSE));

//     // Copy result back to host
//     CUFFT_CHECK(cufftXtMemcpy(planInv,
//                               reinterpret_cast<void*>(hostData),
//                               reinterpret_cast<void*>(deviceDescFwd),
//                               CUFFT_COPY_DEVICE_TO_HOST));

//     // Check that output matches original input (within tolerance)
//     for (size_t i = 0; i < N; ++i) {
//         ASSERT_NEAR(hostData[i].x / N, static_cast<float>(i), 1e-3);
//         ASSERT_NEAR(hostData[i].y / N, 0.0f, 1e-3);
//     }

//     // Cleanup
//     CUFFT_CHECK(cufftXtFree(deviceDescFwd));
//     CUFFT_CHECK(cufftXtFree(deviceDescInv));
//     CUFFT_CHECK(cufftDestroy(planFwd));
//     CUFFT_CHECK(cufftDestroy(planInv));
//     delete[] hostData;
// }
