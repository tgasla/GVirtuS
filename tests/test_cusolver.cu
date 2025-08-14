/*
 * Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             School of Computer Science, University College Dublin
 */

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <gtest/gtest.h>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess)
#define CUSOLVER_CHECK(err) ASSERT_EQ((err), CUSOLVER_STATUS_SUCCESS)

TEST(CuSolver, CreateDestroy) {
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));
}

TEST(CuSolver, GetSetStream) {
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

    cudaStream_t returned_stream;
    CUSOLVER_CHECK(cusolverDnGetStream(handle, &returned_stream));
    ASSERT_EQ(stream, returned_stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));
}