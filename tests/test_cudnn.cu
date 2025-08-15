/*
 * Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             School of Computer Science, University College Dublin
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <gtest/gtest.h>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess) << "CUDA Error: " << cudaGetErrorString(err)
#define CUDNN_CHECK(err) \
    ASSERT_EQ((err), CUDNN_STATUS_SUCCESS) << "CUDNN Error: " << cudnnGetErrorString(err)

TEST(cuDNN, CreateDestroy) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, GetVersion) {
    size_t version = cudnnGetVersion();
    ASSERT_GT(version, 0);
    // std::cout << "cuDNN version: " << version << std::endl;
}

TEST(cuDNN, SetGetStream) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDNN_CHECK(cudnnSetStream(handle, stream));

    cudaStream_t got_stream;
    CUDNN_CHECK(cudnnGetStream(handle, &got_stream));
    ASSERT_EQ(stream, got_stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, TensorDescriptorCreateDestroy) {
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
}

TEST(cuDNN, TensorDescriptorSetGet) {
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 3, 4));

    cudnnDataType_t dataType;
    int n, c, h, w, nStride, cStride, hStride, wStride;

    CUDNN_CHECK(cudnnGetTensor4dDescriptor(desc, &dataType, &n, &c, &h, &w, &nStride, &cStride,
                                           &hStride, &wStride));
    ASSERT_EQ(n, 1);
    ASSERT_EQ(c, 2);
    ASSERT_EQ(h, 3);
    ASSERT_EQ(w, 4);

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
}

TEST(cuDNN, TensorNdDescriptorSetGet) {
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

    const int nbDims = 4;
    int dimA[nbDims] = {1, 2, 3, 4};  // NCHW
    int strideA[nbDims] = {
        2 * 3 * 4,  // N stride = C * H * W
        3 * 4,      // C stride = H * W
        4,          // H stride = W
        1           // W stride = 1
    };

    CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, nbDims, dimA, strideA));

    // Prepare to retrieve
    cudnnDataType_t dataTypeOut;
    int nbDimsOut;
    int dimOut[nbDims];
    int strideOut[nbDims];

    CUDNN_CHECK(
        cudnnGetTensorNdDescriptor(desc, nbDims, &dataTypeOut, &nbDimsOut, dimOut, strideOut));
    ASSERT_EQ(dataTypeOut, CUDNN_DATA_FLOAT);
    ASSERT_EQ(nbDimsOut, nbDims);
    for (int i = 0; i < nbDims; ++i) {
        ASSERT_EQ(dimOut[i], dimA[i]);
        ASSERT_EQ(strideOut[i], strideA[i]);
    }

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
}

TEST(cuDNN, ActivationForwardReLU) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 3));

    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(
        cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    float h_input[] = {-1.0f, 0.0f, 2.0f};
    float h_output[3] = {};

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(h_input)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(h_output)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(
        cudnnActivationForward(handle, actDesc, &alpha, desc, d_input, &beta, desc, d_output));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    ASSERT_FLOAT_EQ(h_output[0], 0.0f);
    ASSERT_FLOAT_EQ(h_output[1], 0.0f);
    ASSERT_FLOAT_EQ(h_output[2], 2.0f);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, PoolingForwardMax) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnPoolingDescriptor_t pool_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));

    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 2, 2));
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2, 2,
                                            0, 0, 1, 1));

    float h_input[] = {1, 2, 3, 4};  // max is 4
    float h_output[1] = {};

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(h_input)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(h_output)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(handle, pool_desc, &alpha, in_desc, d_input, &beta, out_desc,
                                    d_output));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    ASSERT_FLOAT_EQ(h_output[0], 4.0f);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, FilterDescriptorCreateDestroy) {
    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
}

TEST(cuDNN, ErrorString) {
    const char* msg = cudnnGetErrorString(CUDNN_STATUS_ALLOC_FAILED);
    ASSERT_TRUE(msg != nullptr);
}

TEST(cuDNN, ConvolutionForward) {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Input dimensions: NCHW
    int n = 1, c = 1, h = 5, w = 5;
    int kernel_size = 3;
    int pad = 0, stride = 1, dilation = 1;

    // Output dimensions
    int out_n = 1, out_c = 1;
    int out_h = (h + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (w + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Allocate and initialize host memory
    std::vector<float> h_input(n * c * h * w, 1.0f);  // Input filled with ones
    std::vector<float> h_filter(out_c * c * kernel_size * kernel_size,
                                1.0f);  // Filter filled with ones
    std::vector<float> h_output(out_n * out_c * out_h * out_w,
                                0.0f);  // Output initialized to zeros

    // Allocate device memory
    float *d_input, *d_filter, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter, h_filter.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter.data(), h_filter.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_c,
                                           c, kernel_size, kernel_size));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, dilation,
                                                dilation, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Set output descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n,
                                           out_c, out_h, out_w));

    // Select convolution algorithm
    cudnnConvolutionFwdAlgoPerf_t perf_results;
    int returned_algo_count = 0;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc, conv_desc,
                                                     output_desc, 1, &returned_algo_count,
                                                     &perf_results));
    cudnnConvolutionFwdAlgo_t algo = perf_results.algo;

    // Allocate workspace
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc,
                                                        output_desc, algo, &workspace_bytes));
    void* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
    }

    // Perform the convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, filter_desc, d_filter,
                                        conv_desc, algo, d_workspace, workspace_bytes, &beta,
                                        output_desc, d_output));

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Verify the result
    for (size_t i = 0; i < h_output.size(); ++i) {
        ASSERT_NEAR(h_output[i], 9.0f, 1e-5f) << "Mismatch at index " << i;
    }

    // Cleanup
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, ConvolutionBackwardData) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Dimensions
    int n = 1, c = 1, h = 5, w = 5;
    int kernel_size = 3;
    int pad = 0, stride = 1, dilation = 1;
    int out_h = (h + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (w + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Host-side buffers
    std::vector<float> h_input(n * c * h * w, 1.0f);                   // input filled with ones
    std::vector<float> h_filter(c * kernel_size * kernel_size, 1.0f);  // filter filled with ones
    std::vector<float> h_output_grad(n * c * out_h * out_w,
                                     1.0f);                // output gradient (dy) filled with ones
    std::vector<float> h_input_grad(n * c * h * w, 0.0f);  // dx to be filled

    // Device-side buffers
    float *d_input, *d_filter, *d_output_grad, *d_input_grad;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter, h_filter.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_grad, h_output_grad.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_grad, h_input_grad.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_filter, h_filter.data(), h_filter.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_grad, h_output_grad.data(), h_output_grad.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c,
                                           out_h, out_w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, c, c,
                                           kernel_size, kernel_size));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, dilation,
                                                dilation, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Workspace
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn, filter_desc, output_desc, conv_desc, input_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        &workspace_bytes));

    void* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
    }

    // Compute backward data (dx)
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(cudnn, &alpha, filter_desc, d_filter, output_desc,
                                             d_output_grad, conv_desc,
                                             CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, d_workspace,
                                             workspace_bytes, &beta, input_desc, d_input_grad));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_input_grad.data(), d_input_grad, h_input_grad.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Verify (each element in dx should be equal to number of times the kernel overlaps it)
    // With 3x3 kernel and 5x5 input, interior pixels get full 3x3 overlap = 9
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            int count = 0;
            for (int m = 0; m < kernel_size; ++m) {
                for (int n = 0; n < kernel_size; ++n) {
                    int y = i - m + pad;
                    int x = j - n + pad;
                    if (y >= 0 && y < out_h && x >= 0 && x < out_w) count++;
                }
            }
            float expected =
                static_cast<float>(count);  // number of times the kernel covered this pixel
            int idx = i * w + j;
            ASSERT_NEAR(h_input_grad[idx], expected, 1e-5f)
                << "Mismatch at (" << i << "," << j << ")";
        }
    }

    // Cleanup
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output_grad));
    CUDA_CHECK(cudaFree(d_input_grad));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, ConvolutionBackwardFilter) {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    int n = 1, c = 1, h = 5, w = 5;
    int kernel_size = 3;
    int pad = 0, stride = 1, dilation = 1;

    int out_h = (h + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = out_h;
    int out_n = n, out_c = c;

    std::vector<float> h_input(n * c * h * w, 1.0f);                // Input filled with 1.0
    std::vector<float> h_output_grad(n * c * out_h * out_w, 1.0f);  // dOutput filled with 1.0
    std::vector<float> h_filter_grad(c * c * kernel_size * kernel_size,
                                     0.0f);  // dFilter initialized to 0.0

    float *d_input, *d_output_grad, *d_filter_grad;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_grad, h_output_grad.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter_grad, h_filter_grad.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_grad, h_output_grad.data(), h_output_grad.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n,
                                           out_c, out_h, out_w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_c,
                                           c, kernel_size, kernel_size));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, dilation,
                                                dilation, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Get algorithm using the v7 API
    int returned_algo_count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t perf_results;

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn, input_desc, output_desc,
                                                              conv_desc, filter_desc, 1,
                                                              &returned_algo_count, &perf_results));

    ASSERT_GT(returned_algo_count, 0);
    cudnnConvolutionBwdFilterAlgo_t algo = perf_results.algo;

    // Allocate workspace
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn, input_desc, output_desc, conv_desc, filter_desc, algo, &workspace_bytes));

    void* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
    }

    // Run convolution backward filter
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnn, &alpha, input_desc, d_input, output_desc,
                                               d_output_grad, conv_desc, algo, d_workspace,
                                               workspace_bytes, &beta, filter_desc, d_filter_grad));

    CUDA_CHECK(cudaMemcpy(h_filter_grad.data(), d_filter_grad, h_filter_grad.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Verify: each value in the filter gradient should be equal to 9.0 (from 3x3 patch of 1s)
    for (size_t i = 0; i < h_filter_grad.size(); ++i) {
        ASSERT_NEAR(h_filter_grad[i], 9.0f, 1e-5f) << "Mismatch at index " << i;
    }

    // Cleanup
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_grad));
    CUDA_CHECK(cudaFree(d_filter_grad));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, ActivationBackwardReLU) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    const int n = 1, c = 1, h = 2, w = 3;
    const int total_size = n * c * h * w;

    std::vector<float> h_input = {-1.0f, 0.0f, 1.0f, 2.0f, -3.0f, 4.0f};  // Input before ReLU
    std::vector<float> h_output = {0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 4.0f};   // Output after ReLU
    std::vector<float> h_grad_output(total_size, 1.0f);  // dY (incoming gradient): all ones
    std::vector<float> h_grad_input(total_size, 0.0f);   // dX: what we compute

    float *d_input, *d_output, *d_grad_output, *d_grad_input;
    CUDA_CHECK(cudaMalloc(&d_input, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, total_size * sizeof(float)));

    CUDA_CHECK(
        cudaMemcpy(d_input, h_input.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_output, h_output.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output.data(), total_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    cudnnActivationDescriptor_t act_desc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
    CUDNN_CHECK(
        cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;

    CUDNN_CHECK(cudnnActivationBackward(cudnn, act_desc, &alpha, desc, d_output,  // y
                                        desc, d_grad_output,                      // dy
                                        desc, d_input,                            // x
                                        &beta, desc, d_grad_input                 // dx
                                        ));

    CUDA_CHECK(cudaMemcpy(h_grad_input.data(), d_grad_input, total_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::vector<float> expected_grad_input = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

    for (int i = 0; i < total_size; ++i) {
        ASSERT_NEAR(h_grad_input[i], expected_grad_input[i], 1e-5f) << "Mismatch at index " << i;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_grad_input));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(act_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, BatchNormForwardTraining) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    const int n = 1, c = 2, h = 1, w = 3;  // Small 1x2x1x3 tensor
    const int total_size = n * c * h * w;
    const int param_size = c;

    std::vector<float> h_input = {
        1.0f, 2.0f, 3.0f,  // Channel 0
        4.0f, 5.0f, 6.0f   // Channel 1
    };
    std::vector<float> h_grad_output(total_size, 1.0f);  // dy
    std::vector<float> h_output(total_size, 0.0f);
    std::vector<float> h_grad_input(total_size, 0.0f);

    std::vector<float> h_scale(param_size, 1.0f);  // gamma
    std::vector<float> h_bias(param_size, 0.0f);   // beta
    std::vector<float> h_running_mean(param_size, 0.0f);
    std::vector<float> h_running_var(param_size, 1.0f);

    std::vector<float> h_saved_mean(param_size);
    std::vector<float> h_saved_var(param_size);

    std::vector<float> h_dscale(param_size, 0.0f);
    std::vector<float> h_dbias(param_size, 0.0f);

    float *d_input, *d_output, *d_grad_output, *d_grad_input;
    float *d_scale, *d_bias, *d_running_mean, *d_running_var;
    float *d_saved_mean, *d_saved_var;
    float *d_dscale, *d_dbias;

    CUDA_CHECK(cudaMalloc(&d_input, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, total_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_scale, param_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, param_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_running_mean, param_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_running_var, param_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_saved_mean, param_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_saved_var, param_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dscale, param_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dbias, param_size * sizeof(float)));

    CUDA_CHECK(
        cudaMemcpy(d_input, h_input.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output.data(), total_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_scale, h_scale.data(), param_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_bias, h_bias.data(), param_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_running_mean, h_running_mean.data(), param_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_running_var, h_running_var.data(), param_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudnnTensorDescriptor_t x_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    cudnnTensorDescriptor_t bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(bn_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1, 1));

    float alpha = 1.0f, beta = 0.0f;
    double epsilon = 1e-5;
    double exponential_average_factor = 1.0;

    // Forward training pass
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        cudnn, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, x_desc, d_input, x_desc, d_output, bn_desc,
        d_scale, d_bias, exponential_average_factor, d_running_mean, d_running_var, epsilon,
        d_saved_mean, d_saved_var));

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_grad_input));
    CUDA_CHECK(cudaFree(d_scale));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_running_mean));
    CUDA_CHECK(cudaFree(d_running_var));
    CUDA_CHECK(cudaFree(d_saved_mean));
    CUDA_CHECK(cudaFree(d_saved_var));
    CUDA_CHECK(cudaFree(d_dscale));
    CUDA_CHECK(cudaFree(d_dbias));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

// TEST(cuDNN, BatchNormForwardTrainingAndBackward) {
//     cudnnHandle_t cudnn;
//     CUDNN_CHECK(cudnnCreate(&cudnn));

//     const int n = 1, c = 2, h = 1, w = 3; // Small 1x2x1x3 tensor
//     const int total_size = n * c * h * w;
//     const int param_size = c;

//     std::vector<float> h_input = {
//         1.0f, 2.0f, 3.0f,   // Channel 0
//         4.0f, 5.0f, 6.0f    // Channel 1
//     };
//     std::vector<float> h_grad_output(total_size, 1.0f);  // dy
//     std::vector<float> h_output(total_size, 0.0f);
//     std::vector<float> h_grad_input(total_size, 0.0f);

//     std::vector<float> h_scale(param_size, 1.0f);  // gamma
//     std::vector<float> h_bias(param_size, 0.0f);   // beta
//     std::vector<float> h_running_mean(param_size, 0.0f);
//     std::vector<float> h_running_var(param_size, 1.0f);

//     std::vector<float> h_saved_mean(param_size);
//     std::vector<float> h_saved_var(param_size);

//     std::vector<float> h_dscale(param_size, 0.0f);
//     std::vector<float> h_dbias(param_size, 0.0f);

//     float *d_input, *d_output, *d_grad_output, *d_grad_input;
//     float *d_scale, *d_bias, *d_running_mean, *d_running_var;
//     float *d_saved_mean, *d_saved_var;
//     float *d_dscale, *d_dbias;

//     CUDA_CHECK(cudaMalloc(&d_input, total_size * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_output, total_size * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_grad_output, total_size * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_grad_input, total_size * sizeof(float)));

//     CUDA_CHECK(cudaMalloc(&d_scale, param_size * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_bias, param_size * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_running_mean, param_size * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_running_var, param_size * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_saved_mean, param_size * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_saved_var, param_size * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_dscale, param_size * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_dbias, param_size * sizeof(float)));

//     CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), total_size * sizeof(float),
//     cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output.data(),
//     total_size * sizeof(float), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_scale,
//     h_scale.data(), param_size * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), param_size * sizeof(float),
//     cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_running_mean, h_running_mean.data(),
//     param_size * sizeof(float), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_running_var,
//     h_running_var.data(), param_size * sizeof(float), cudaMemcpyHostToDevice));

//     cudnnTensorDescriptor_t x_desc;
//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
//     CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h,
//     w));

//     cudnnTensorDescriptor_t bn_desc;
//     CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
//     CUDNN_CHECK(cudnnSetTensor4dDescriptor(bn_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c, 1,
//     1));

//     float alpha = 1.0f, beta = 0.0f;
//     double epsilon = 1e-5;
//     double exponential_average_factor = 1.0;

//     // Forward training pass
//     CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
//         cudnn,
//         CUDNN_BATCHNORM_SPATIAL,
//         &alpha, &beta,
//         x_desc, d_input,
//         x_desc, d_output,
//         bn_desc, d_scale, d_bias,
//         exponential_average_factor,
//         d_running_mean, d_running_var,
//         epsilon,
//         d_saved_mean, d_saved_var
//     ));

//     // Backward pass
//     CUDNN_CHECK(cudnnBatchNormalizationBackward(
//         cudnn,
//         CUDNN_BATCHNORM_SPATIAL,
//         &alpha, &beta,
//         &alpha, &beta,
//         x_desc, d_input,
//         x_desc, d_grad_output,
//         x_desc, d_grad_input,
//         bn_desc, d_scale,
//         d_dscale, d_dbias,
//         epsilon,
//         d_saved_mean, d_saved_var
//     ));

//     // Copy results back
//     CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, total_size * sizeof(float),
//     cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h_grad_input.data(), d_grad_input, total_size
//     * sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h_dscale.data(), d_dscale,
//     param_size * sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h_dbias.data(),
//     d_dbias, param_size * sizeof(float), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(h_saved_mean.data(), d_saved_mean, param_size * sizeof(float),
//     cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h_saved_var.data(), d_saved_var, param_size *
//     sizeof(float), cudaMemcpyDeviceToHost));

//     // Expect saved mean = mean of each channel
//     ASSERT_NEAR(h_saved_mean[0], 2.0f, 1e-3);
//     ASSERT_NEAR(h_saved_mean[1], 5.0f, 1e-3);
//     ASSERT_NEAR(h_saved_var[0], 1.0f, 1e-2);  // (1+0+1)/3 = 0.6667 approx
//     ASSERT_NEAR(h_saved_var[1], 1.0f, 1e-2);

//     // Expect dBias to be 3.0 (sum of dy per channel)
//     ASSERT_NEAR(h_dbias[0], 3.0f, 1e-3);
//     ASSERT_NEAR(h_dbias[1], 3.0f, 1e-3);

//     // Expect dScale to be near 0 (since input normalized mean is 0)
//     ASSERT_NEAR(h_dscale[0], 0.0f, 1e-2);
//     ASSERT_NEAR(h_dscale[1], 0.0f, 1e-2);

//     // Cleanup
//     CUDA_CHECK(cudaFree(d_input));
//     CUDA_CHECK(cudaFree(d_output));
//     CUDA_CHECK(cudaFree(d_grad_output));
//     CUDA_CHECK(cudaFree(d_grad_input));
//     CUDA_CHECK(cudaFree(d_scale));
//     CUDA_CHECK(cudaFree(d_bias));
//     CUDA_CHECK(cudaFree(d_running_mean));
//     CUDA_CHECK(cudaFree(d_running_var));
//     CUDA_CHECK(cudaFree(d_saved_mean));
//     CUDA_CHECK(cudaFree(d_saved_var));
//     CUDA_CHECK(cudaFree(d_dscale));
//     CUDA_CHECK(cudaFree(d_dbias));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
//     CUDNN_CHECK(cudnnDestroyTensorDescriptor(bn_desc));
//     CUDNN_CHECK(cudnnDestroy(cudnn));
// }

TEST(cuDNN, GetConvolution2dForwardOutputDim) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    const int n = 1, c = 1, h = 5, w = 5;
    const int k = 1, kh = 3, kw = 3;
    const int pad = 1, stride = 1, dilation = 1;

    cudnnTensorDescriptor_t input_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CUDNN_CHECK(
        cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, kh, kw));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, dilation,
                                                dilation, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &out_n,
                                                      &out_c, &out_h, &out_w));

    ASSERT_EQ(out_n, n);
    ASSERT_EQ(out_c, k);
    ASSERT_EQ(out_h, h);  // With padding=1, stride=1, output height = input height
    ASSERT_EQ(out_w, w);  // Same for width

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, GetConvolutionForwardAlgorithm_v7) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    const int n = 1, c = 1, h = 5, w = 5;
    const int k = 1, kh = 3, kw = 3;
    const int pad = 1, stride = 1, dilation = 1;

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CUDNN_CHECK(
        cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, kh, kw));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, dilation,
                                                dilation, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    int out_n, out_c, out_h, out_w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &out_n,
                                                      &out_c, &out_h, &out_w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n,
                                           out_c, out_h, out_w));

    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results[1];

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_desc, filter_desc, conv_desc,
                                                       output_desc, 1, &returned_algo_count,
                                                       perf_results));

    ASSERT_GE(returned_algo_count, 1);
    ASSERT_EQ(perf_results[0].status, CUDNN_STATUS_SUCCESS);
    ASSERT_NE(perf_results[0].algo,
              CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);  // or just check algo is valid

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, AddTensorFloat) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    const int n = 1, c = 1, h = 2, w = 2;
    float alpha = 2.0f, beta = 1.0f;

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    std::vector<float> h_A = {1, 2, 3, 4};
    std::vector<float> h_B = {10, 20, 30, 40};
    std::vector<float> h_result(4);

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, desc, d_A, &beta, desc, d_B));

    CUDA_CHECK(
        cudaMemcpy(h_result.data(), d_B, h_result.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(h_result[i], beta * h_B[i] + alpha * h_A[i]);  // Expected: 1*B + 2*A
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, AddTensorDouble) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    const int n = 1, c = 1, h = 2, w = 2;
    double alpha = 2.0, beta = 1.0;

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n, c, h, w));

    std::vector<double> h_A = {1, 2, 3, 4};
    std::vector<double> h_B = {10, 20, 30, 40};
    std::vector<double> h_result(4);

    double *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(double), cudaMemcpyHostToDevice));

    CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha, desc, d_A, &beta, desc, d_B));

    CUDA_CHECK(
        cudaMemcpy(h_result.data(), d_B, h_result.size() * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; ++i) {
        ASSERT_DOUBLE_EQ(h_result[i], beta * h_B[i] + alpha * h_A[i]);  // Expected: 1*B + 2*A
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, TransformTensorFloat) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    const int n = 1, c = 1, h = 2, w = 2;
    float alpha = 3.0f, beta = 0.0f;

    cudnnTensorDescriptor_t src_desc, dest_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&src_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dest_desc));

    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(src_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(dest_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    std::vector<float> h_src = {1, 2, 3, 4};
    std::vector<float> h_dest(4, 0);

    float *d_src, *d_dest;
    CUDA_CHECK(cudaMalloc(&d_src, h_src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dest, h_dest.size() * sizeof(float)));

    CUDA_CHECK(
        cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_dest, h_dest.data(), h_dest.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDNN_CHECK(cudnnTransformTensor(cudnn, &alpha, src_desc, d_src, &beta, dest_desc, d_dest));

    CUDA_CHECK(
        cudaMemcpy(h_dest.data(), d_dest, h_dest.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(h_dest[i], alpha * h_src[i]);  // Expected: 3 * source
    }

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dest));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(src_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dest_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, TransformTensorDouble) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    const int n = 1, c = 1, h = 2, w = 2;
    double alpha = 3.0, beta = 0.0;

    cudnnTensorDescriptor_t src_desc, dest_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&src_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dest_desc));

    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(src_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n, c, h, w));
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(dest_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n, c, h, w));

    std::vector<double> h_src = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> h_dest(4, 0.0);

    double* d_src;
    double* d_dest;
    CUDA_CHECK(cudaMalloc(&d_src, h_src.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dest, h_dest.size() * sizeof(double)));

    CUDA_CHECK(
        cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_dest, h_dest.data(), h_dest.size() * sizeof(double), cudaMemcpyHostToDevice));

    CUDNN_CHECK(cudnnTransformTensor(cudnn, &alpha, src_desc, d_src, &beta, dest_desc, d_dest));

    CUDA_CHECK(
        cudaMemcpy(h_dest.data(), d_dest, h_dest.size() * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; ++i) {
        ASSERT_DOUBLE_EQ(h_dest[i], alpha * h_src[i]);  // Expected: 3.0 * source
    }

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dest));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(src_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dest_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, PoolingBackward) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    const int n = 1, c = 1, h = 4, w = 4;
    const int window = 2, padding = 0, stride = 2;

    cudnnTensorDescriptor_t input_desc, output_desc, dInput_desc, dOutput_desc;
    cudnnPoolingDescriptor_t pooling_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dInput_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dOutput_desc));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc));

    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                            window, window, padding, padding, stride, stride));

    int out_n, out_c, out_h, out_w;
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(pooling_desc, input_desc, &out_n, &out_c, &out_h,
                                                  &out_w));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n,
                                           out_c, out_h, out_w));
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(dInput_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dOutput_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n,
                                           out_c, out_h, out_w));

    std::vector<float> h_input(n * c * h * w, 1.0f);
    std::vector<float> h_output(out_n * out_c * out_h * out_w);
    std::vector<float> h_dOutput(out_n * out_c * out_h * out_w, 1.0f);  // Grad from next layer
    std::vector<float> h_dInput(n * c * h * w, 0.0f);

    float *d_input, *d_output, *d_dInput, *d_dOutput;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dInput, h_dInput.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dOutput, h_dOutput.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dOutput, h_dOutput.data(), h_dOutput.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Forward pooling
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(cudnn, pooling_desc, &alpha, input_desc, d_input, &beta,
                                    output_desc, d_output));

    // Backward pooling
    CUDNN_CHECK(cudnnPoolingBackward(cudnn, pooling_desc, &alpha, output_desc, d_output,
                                     dOutput_desc, d_dOutput, input_desc, d_input, &beta,
                                     dInput_desc, d_dInput));

    CUDA_CHECK(cudaMemcpy(h_dInput.data(), d_dInput, h_dInput.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Basic check: gradients should be positive and distributed in max pooling positions
    for (auto val : h_dInput) {
        ASSERT_GE(val, 0);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_dInput));
    CUDA_CHECK(cudaFree(d_dOutput));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dInput_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dOutput_desc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, FindConvolutionForwardAlgorithmEx) {
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CUDNN_CHECK(cudnnCreate(&cudnn));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

    // Define tensor dimensions (NCHW)
    const int N = 1, C = 3, H = 32, W = 32;
    const int K = 16, R = 3,
              S = 3;  // filters: output channels, input channels, filter height, filter width

    // Setup input tensor descriptor (NCHW, float)
    CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    // Setup filter descriptor
    CUDNN_CHECK(
        cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));

    // Setup convolution descriptor
    const int pad_h = 1, pad_w = 1;
    const int stride_h = 1, stride_w = 1;
    const int dilation_h = 1, dilation_w = 1;
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_h, stride_w,
                                                dilation_h, dilation_w, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Get output dimensions
    int outN, outC, outH, outW;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &outN, &outC,
                                                      &outH, &outW));

    // Setup output tensor descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outN,
                                           outC, outH, outW));

    // Allocate device memory for input, filter, output
    size_t inputBytes = N * C * H * W * sizeof(float);
    size_t filterBytes = K * C * R * S * sizeof(float);
    size_t outputBytes = outN * outC * outH * outW * sizeof(float);

    float* d_input = nullptr;
    float* d_filter = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, inputBytes));
    CUDA_CHECK(cudaMemset(d_input, 0, inputBytes));

    CUDA_CHECK(cudaMalloc(&d_filter, filterBytes));
    CUDA_CHECK(cudaMemset(d_filter, 0, filterBytes));

    CUDA_CHECK(cudaMalloc(&d_output, outputBytes));
    CUDA_CHECK(cudaMemset(d_output, 0, outputBytes));

    // Allocate workspace
    size_t workspaceSize = 0;
    // Just get a reasonable workspace size for one algorithm to be safe
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, inputDesc, filterDesc, convDesc, outputDesc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &workspaceSize));

    // Make workspace at least 1MB for safety (optional)
    workspaceSize = (workspaceSize > (1 << 20)) ? workspaceSize : (1 << 20);

    void* d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspaceSize));

    // Prepare perfResults array
    const int requestedAlgoCount = 5;
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults[requestedAlgoCount];

    // Run the algorithm finder
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        cudnn, inputDesc, d_input, filterDesc, d_filter, convDesc, outputDesc, d_output,
        requestedAlgoCount, &returnedAlgoCount, perfResults, d_workspace, workspaceSize));

    // Checks
    ASSERT_GT(returnedAlgoCount, 0);
    for (int i = 0; i < returnedAlgoCount; i++) {
        ASSERT_GE(perfResults[i].time, 0.0f);
        ASSERT_GE(perfResults[i].memory, 0);
        ASSERT_EQ(perfResults[i].status, CUDNN_STATUS_SUCCESS);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_workspace));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

TEST(cuDNN, RNNCreateDestroyDescriptor) {
    cudnnRNNDescriptor_t rnnDesc;
    CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnnDesc));
    CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnnDesc));
}

TEST(cuDNN, RNNCreateDestroyDataDescriptor) {
    cudnnRNNDataDescriptor_t rnnDataDesc;
    CUDNN_CHECK(cudnnCreateRNNDataDescriptor(&rnnDataDesc));
    CUDNN_CHECK(cudnnDestroyRNNDataDescriptor(rnnDataDesc));
}

TEST(cuDNN, RNNSetGetDataDescriptor) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnRNNDataDescriptor_t rnnDataDesc;
    CUDNN_CHECK(cudnnCreateRNNDataDescriptor(&rnnDataDesc));

    // Descriptor setup
    cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    int maxSeqLength = 5;
    int batchSize = 3;
    int vectorSize = 8;
    std::vector<int> seqLengths = {5, 4, 3};
    float paddingFillValue = 0.0f;

    CUDNN_CHECK(cudnnSetRNNDataDescriptor(rnnDataDesc, dataType, layout, maxSeqLength, batchSize,
                                          vectorSize, seqLengths.data(),
                                          &paddingFillValue  // must match `dataType` in type
                                          ));

    // Prepare output variables
    cudnnDataType_t outDataType;
    cudnnRNNDataLayout_t outLayout;
    int outMaxSeqLength = 0, outBatchSize = 0, outVectorSize = 0;
    std::vector<int> outSeqLengths(batchSize);
    float outPaddingFill = -1.0f;  // initial garbage value

    CUDNN_CHECK(cudnnGetRNNDataDescriptor(rnnDataDesc, &outDataType, &outLayout, &outMaxSeqLength,
                                          &outBatchSize, &outVectorSize, batchSize,
                                          outSeqLengths.data(), &outPaddingFill));

    // Validate roundtrip
    ASSERT_EQ(outDataType, dataType);
    ASSERT_EQ(outLayout, layout);
    ASSERT_EQ(outMaxSeqLength, maxSeqLength);
    ASSERT_EQ(outBatchSize, batchSize);
    ASSERT_EQ(outVectorSize, vectorSize);
    ASSERT_FLOAT_EQ(outPaddingFill, paddingFillValue);
    for (int i = 0; i < batchSize; ++i) {
        ASSERT_EQ(outSeqLengths[i], seqLengths[i]);
    }

    CUDNN_CHECK(cudnnDestroyRNNDataDescriptor(rnnDataDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, DropoutDescriptorCreateDestroy) {
    cudnnDropoutDescriptor_t dropoutDesc;
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropoutDesc));
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropoutDesc));
}

TEST(cuDNN, GetDropoutDescriptor) {
    cudnnHandle_t handle;
    cudnnDropoutDescriptor_t dropoutDesc;

    // Create cuDNN handle and dropout descriptor
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropoutDesc));

    // These will hold output from cudnnGetDropoutDescriptor
    float dropout;
    void* states;
    unsigned long long seed;

    // This will most likely return default/uninitialized values,
    // but the goal is to ensure it doesn't crash or return an error
    ASSERT_EQ(cudnnGetDropoutDescriptor(dropoutDesc, handle, &dropout, &states, &seed),
              CUDNN_STATUS_SUCCESS);

    // Clean up
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropoutDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, SetGetDropoutDescriptor) {
    cudnnHandle_t handle;
    cudnnDropoutDescriptor_t dropoutDesc;

    // Create cuDNN handle and dropout descriptor
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropoutDesc));

    // Allocate memory for states
    size_t stateSize;
    float dropout = 0.25f;
    unsigned long long seed = 123456789ULL;

    // Query size required for states
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &stateSize));
    void* states;
    cudaMalloc(&states, stateSize);

    // Set the dropout descriptor
    CUDNN_CHECK(cudnnSetDropoutDescriptor(dropoutDesc, handle, dropout, states, stateSize, seed));

    // Get the descriptor values
    float returned_dropout;
    void* returned_states;
    unsigned long long returned_seed;

    CUDNN_CHECK(cudnnGetDropoutDescriptor(dropoutDesc, handle, &returned_dropout, &returned_states,
                                          &returned_seed));

    // Assert values are correct
    ASSERT_FLOAT_EQ(returned_dropout, dropout);
    ASSERT_EQ(returned_seed, seed);
    ASSERT_EQ(returned_states, states);

    // Cleanup
    cudaFree(states);
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropoutDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
}

TEST(cuDNN, RNNForward) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // RNN config
    const int inputSize = 8;
    const int hiddenSize = 16;
    const int numLayers = 1;
    const int batchSize = 2;
    const int seqLength = 4;
    const int projSize = hiddenSize;

    cudnnRNNDescriptor_t rnnDesc;
    CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnnDesc));

    cudnnDropoutDescriptor_t dropoutDesc;
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropoutDesc));

    // Dummy dropout config (no dropout)
    void* states;
    size_t stateSize;
    CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &stateSize));
    CUDA_CHECK(cudaMalloc(&states, stateSize));
    CUDNN_CHECK(cudnnSetDropoutDescriptor(dropoutDesc, handle, 0.0f, states, stateSize, 0));

    CUDNN_CHECK(cudnnSetRNNDescriptor_v8(
        rnnDesc, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
        CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, inputSize,
        hiddenSize, projSize, numLayers, dropoutDesc, 0));

    // Input and output RNNDataDescriptors
    cudnnRNNDataDescriptor_t xDesc, yDesc;
    CUDNN_CHECK(cudnnCreateRNNDataDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateRNNDataDescriptor(&yDesc));

    int seqLengthArray[batchSize];
    for (int i = 0; i < batchSize; ++i) seqLengthArray[i] = seqLength;

    CUDNN_CHECK(cudnnSetRNNDataDescriptor(xDesc, CUDNN_DATA_FLOAT,
                                          CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, seqLength,
                                          batchSize, inputSize, seqLengthArray, nullptr));

    CUDNN_CHECK(cudnnSetRNNDataDescriptor(yDesc, CUDNN_DATA_FLOAT,
                                          CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, seqLength,
                                          batchSize, hiddenSize, seqLengthArray, nullptr));

    // Allocate dummy input/output memory
    size_t inputBytes = seqLength * batchSize * inputSize * sizeof(float);
    size_t outputBytes = seqLength * batchSize * hiddenSize * sizeof(float);
    float* x;
    float* y;
    CUDA_CHECK(cudaMalloc(&x, inputBytes));
    CUDA_CHECK(cudaMalloc(&y, outputBytes));

    // Initial/Final hidden/cell states
    cudnnTensorDescriptor_t hDesc, cDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cDesc));

    int dims[3] = {numLayers, batchSize, hiddenSize};
    int strides[3] = {batchSize * hiddenSize, hiddenSize, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(hDesc, CUDNN_DATA_FLOAT, 3, dims, strides));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(cDesc, CUDNN_DATA_FLOAT, 3, dims, strides));

    float *hx, *cx, *hy, *cy;
    CUDA_CHECK(cudaMalloc(&hx, numLayers * batchSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cx, numLayers * batchSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hy, numLayers * batchSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cy, numLayers * batchSize * hiddenSize * sizeof(float)));

    // Allocate weights
    size_t weightSpaceSize;
    CUDNN_CHECK(cudnnGetRNNWeightSpaceSize(handle, rnnDesc, &weightSpaceSize));
    void* weightSpace;
    CUDA_CHECK(cudaMalloc(&weightSpace, weightSpaceSize));

    // Workspace and reserve space
    size_t workspaceSize, reserveSize;
    CUDNN_CHECK(cudnnGetRNNTempSpaceSizes(handle, rnnDesc, CUDNN_FWD_MODE_TRAINING, xDesc,
                                          &workspaceSize, &reserveSize));
    void* workspace;
    void* reserveSpace;
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));
    CUDA_CHECK(cudaMalloc(&reserveSpace, reserveSize));

    // Allocate devSeqLengths
    int32_t* devSeqLengths;
    CUDA_CHECK(cudaMalloc(&devSeqLengths, batchSize * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(devSeqLengths, seqLengthArray, batchSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // Run forward
    CUDNN_CHECK(cudnnRNNForward(handle, rnnDesc, CUDNN_FWD_MODE_TRAINING, devSeqLengths, xDesc, x,
                                yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize,
                                weightSpace, workspaceSize, workspace, reserveSize, reserveSpace));

    // Cleanup
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(hx));
    CUDA_CHECK(cudaFree(hy));
    CUDA_CHECK(cudaFree(cx));
    CUDA_CHECK(cudaFree(cy));
    CUDA_CHECK(cudaFree(weightSpace));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(reserveSpace));
    CUDA_CHECK(cudaFree(states));
    CUDA_CHECK(cudaFree(devSeqLengths));

    CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnnDesc));
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropoutDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(hDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cDesc));
    CUDNN_CHECK(cudnnDestroyRNNDataDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyRNNDataDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
}

// Not implemented yet
// TEST(cuDNN, FusedOpsPlanCreateDestroy) {
//     cudnnFusedOpsPlan_t plan;

//     // Choose a valid fused op. For example: convolution + bias + activation
//     cudnnFusedOps_t ops = CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS;

//     // Create the fused ops plan
//     CUDNN_CHECK(cudnnCreateFusedOpsPlan(&plan, ops));

//     // Now destroy the plan if it was created
//     CUDNN_CHECK(cudnnDestroyFusedOpsPlan(plan));
// }

TEST(cuDNN, createDestroyLRNDescriptor) {
    cudnnLRNDescriptor_t lrnDesc;
    CUDNN_CHECK(cudnnCreateLRNDescriptor(&lrnDesc));
    CUDNN_CHECK(cudnnDestroyLRNDescriptor(lrnDesc));
}

TEST(cuDNN, SetGetLRNDescriptor) {
    cudnnLRNDescriptor_t lrnDesc;
    CUDNN_CHECK(cudnnCreateLRNDescriptor(&lrnDesc));

    // Set LRN descriptor
    unsigned n = 5;  // Local size
    double alpha = 1.0f, beta = 0.75f, k = 1e-4f;
    CUDNN_CHECK(cudnnSetLRNDescriptor(lrnDesc, n, alpha, beta, k));

    // Get LRN descriptor
    unsigned out_n;
    double out_alpha, out_beta, out_k;
    CUDNN_CHECK(cudnnGetLRNDescriptor(lrnDesc, &out_n, &out_alpha, &out_beta, &out_k));

    ASSERT_EQ(out_n, n);
    ASSERT_FLOAT_EQ(out_alpha, alpha);
    ASSERT_FLOAT_EQ(out_beta, beta);
    ASSERT_FLOAT_EQ(out_k, k);

    CUDNN_CHECK(cudnnDestroyLRNDescriptor(lrnDesc));
}

TEST(cuDNN, LRNCrossChannelForwardFloat) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    const int N = 1, C = 3, H = 5, W = 5;
    const size_t tensorSize = N * C * H * W;

    // Allocate host and device memory
    std::vector<float> h_input(tensorSize, 1.0f);  // simple input
    std::vector<float> h_output(tensorSize, 0.0f);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, tensorSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, tensorSize * sizeof(float)));

    CUDA_CHECK(
        cudaMemcpy(d_input, h_input.data(), tensorSize * sizeof(float), cudaMemcpyHostToDevice));

    // Create and set tensor descriptors
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    // Create and set LRN descriptor
    cudnnLRNDescriptor_t lrnDesc;
    CUDNN_CHECK(cudnnCreateLRNDescriptor(&lrnDesc));
    unsigned lrnN = 5;
    double lrnAlpha = 1e-4;
    double lrnBeta = 0.75;
    double lrnK = 2.0;
    CUDNN_CHECK(cudnnSetLRNDescriptor(lrnDesc, lrnN, lrnAlpha, lrnBeta, lrnK));

    // Perform forward pass
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnLRNCrossChannelForward(handle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha,
                                            desc, d_input, &beta, desc, d_output));

    // Copy result back to host
    CUDA_CHECK(
        cudaMemcpy(h_output.data(), d_output, tensorSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Optional: Validate results are different from input
    for (size_t i = 0; i < tensorSize; ++i) {
        ASSERT_NE(h_output[i],
                  0.0f);  // since input was all 1s, LRN will produce something non-zero
    }

    // Cleanup
    CUDNN_CHECK(cudnnDestroyLRNDescriptor(lrnDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

TEST(cuDNN, LRNCrossChannelForwardDouble) {
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    const int N = 1, C = 3, H = 5, W = 5;
    const size_t tensorSize = N * C * H * W;

    // Allocate host and device memory
    std::vector<double> h_input(tensorSize, 1.0);  // simple input
    std::vector<double> h_output(tensorSize, 0.0);

    double* d_input = nullptr;
    double* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, tensorSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, tensorSize * sizeof(double)));

    CUDA_CHECK(
        cudaMemcpy(d_input, h_input.data(), tensorSize * sizeof(double), cudaMemcpyHostToDevice));

    // Create and set tensor descriptors
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W));

    // Create and set LRN descriptor
    cudnnLRNDescriptor_t lrnDesc;
    CUDNN_CHECK(cudnnCreateLRNDescriptor(&lrnDesc));
    unsigned lrnN = 5;
    double lrnAlpha = 1e-4;
    double lrnBeta = 0.75;
    double lrnK = 2.0;
    CUDNN_CHECK(cudnnSetLRNDescriptor(lrnDesc, lrnN, lrnAlpha, lrnBeta, lrnK));

    // Perform forward pass
    const double alpha = 1.0;
    const double beta = 0.0;
    CUDNN_CHECK(cudnnLRNCrossChannelForward(handle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha,
                                            desc, d_input, &beta, desc, d_output));

    // Copy result back to host
    CUDA_CHECK(
        cudaMemcpy(h_output.data(), d_output, tensorSize * sizeof(double), cudaMemcpyDeviceToHost));

    // Optional: Validate results are different from input
    for (size_t i = 0; i < tensorSize; ++i) {
        ASSERT_NE(h_output[i], 0.0);  // since input was all 1s, LRN will produce something non-zero
    }

    // Cleanup
    CUDNN_CHECK(cudnnDestroyLRNDescriptor(lrnDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

TEST(cuDNN, BackendCreateAndDestroyDescriptor) {
    cudnnBackendDescriptor_t desc;

    CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                             &desc));
    ASSERT_NE(desc, nullptr);

    CUDNN_CHECK(cudnnBackendDestroyDescriptor(desc));
}

// TEST(CudnnBackendEngineHeurTest, SetGetAttributes) {
//     cudnnHandle_t handle;
//     CUDNN_CHECK(cudnnCreate(&handle));

//     // Step 1: Create a simple operation graph descriptor
//     cudnnBackendDescriptor_t opGraph;
//     CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraph));

//     CUDNN_CHECK(cudnnBackendSetAttribute(
//         opGraph,
//         CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
//         CUDNN_TYPE_HANDLE,
//         1,
//         &handle));

//     CUDNN_CHECK(cudnnBackendSetAttribute(
//         opGraph,
//         CUDNN_ATTR_OPERATIONGRAPH_OPS,
//         CUDNN_TYPE_BACKEND_DESCRIPTOR,
//         1,
//         &CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR));

//     // For test, just finalize empty graph
//     CUDNN_CHECK(cudnnBackendFinalize(opGraph));

//     // Step 2: Create heuristics descriptor
//     cudnnBackendDescriptor_t heurDesc;
//     CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heurDesc));

//     // Step 3: Set CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH
//     CUDNN_CHECK(cudnnBackendSetAttribute(
//         heurDesc,
//         CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
//         CUDNN_TYPE_BACKEND_DESCRIPTOR,
//         1,
//         &opGraph));

//     // Step 4: Set CUDNN_ATTR_ENGINEHEUR_MODE = CUDNN_HEUR_MODE_INSTANT
//     cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_INSTANT;
//     CUDNN_CHECK(cudnnBackendSetAttribute(
//         heurDesc,
//         CUDNN_ATTR_ENGINEHEUR_MODE,
//         CUDNN_TYPE_HEUR_MODE,
//         1,
//         &mode));

//     // Step 5: Finalize heuristics descriptor
//     CUDNN_CHECK(cudnnBackendFinalize(heurDesc));

//     // Step 6: Get attribute CUDNN_ATTR_ENGINEHEUR_RESULTS with element count 0 to get the real
//     count int64_t returnedCount = 0; CUDNN_CHECK(cudnnBackendGetAttribute(
//         heurDesc,
//         CUDNN_ATTR_ENGINEHEUR_RESULTS,
//         CUDNN_TYPE_BACKEND_DESCRIPTOR,
//         0,
//         &returnedCount,
//         nullptr));

//     std::cout << "Returned element count for ENGINEHEUR_RESULTS: " << returnedCount << std::endl;
//     ASSERT_GT(returnedCount, 0);

//     // Step 7: Try to get attribute with element count = returnedCount
//     cudnnBackendDescriptor_t results[returnedCount]; // Max 24 elements as example
//     int64_t actualReturnedCount;
//     CUDNN_CHECK(cudnnBackendGetAttribute(
//         heurDesc,
//         CUDNN_ATTR_ENGINEHEUR_RESULTS,
//         CUDNN_TYPE_BACKEND_DESCRIPTOR,
//         returnedCount,
//         &actualReturnedCount,
//         results));

//     // Print status, expect success or failure (your reported error is BAD_PARAM)
//     std::cout << "GetAttribute with element count " << actualReturnedCount << std::endl;
// }