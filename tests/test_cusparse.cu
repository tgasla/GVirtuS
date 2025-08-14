/*
 * Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             School of Computer Science, University College Dublin
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <gtest/gtest.h>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess) << "CUDA error: " << cudaGetErrorString(err)
#define CUSPARSE_CHECK(err) \
    ASSERT_EQ((err), CUSPARSE_STATUS_SUCCESS) << "CuSPARSE error: " << cusparseGetErrorString(err)

TEST(CuSPARSE, CreateDestroy) {
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUSPARSE_CHECK(cusparseDestroy(handle));
}

TEST(CuSPARSE, GetErrorString) {
    const char* msg = cusparseGetErrorString(CUSPARSE_STATUS_SUCCESS);
    ASSERT_NE(msg, nullptr);
}

TEST(CuSPARSE, GetVersion) {
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    int version;
    CUSPARSE_CHECK(cusparseGetVersion(handle, &version));
    ASSERT_GT(version, 999);  // Version should be greater than 1.0
    CUSPARSE_CHECK(cusparseDestroy(handle));
}

TEST(CuSPARSE, SetGetStream) {
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUSPARSE_CHECK(cusparseSetStream(handle, stream));

    cudaStream_t returned_stream;
    CUSPARSE_CHECK(cusparseGetStream(handle, &returned_stream));
    ASSERT_EQ(stream, returned_stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUSPARSE_CHECK(cusparseDestroy(handle));
}