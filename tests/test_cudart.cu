/*
 * Written By: Theodoros Aslanidis <theodoros.aslanidis@ucdconnect.ie>
 *             School of Computer Science, University College Dublin
 */

#include <cuda.h> /* cuuint64_t */
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#define CUDA_CHECK(err) ASSERT_EQ((err), cudaSuccess)

__device__ int intDeviceVariable = 0;

TEST(cudaRT, getDeviceCount) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    ASSERT_GT(count, 0);
}

TEST(cudaRT, ThreadExchangeStreamCaptureMode) {
    cudaStreamCaptureMode original_mode = cudaStreamCaptureModeThreadLocal;
    cudaStreamCaptureMode previous_mode;

    // Exchange thread-local with current mode, original_mode gets overwritten with previous
    previous_mode = original_mode;
    CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&original_mode));

    // Ensure that exchange actually happened: value at `original_mode` now holds the previous
    ASSERT_NE(previous_mode, original_mode);

    // Now push the original mode back to restore thread state
    CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&original_mode));

    // Ensure that the original mode is restored
    ASSERT_EQ(original_mode, previous_mode);
}

TEST(cudaRT, MemPoolGetAttribute) {
    cudaMemPool_t memPool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, 0));

    cuuint64_t threshold = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold));
}

TEST(cudaRT, MallocFree) {
    void* devPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devPtr, 1024));
    CUDA_CHECK(cudaFree(devPtr));
}

TEST(cudaRT, MemcpySync) {
    int h_src = 42;
    int h_dst = 0;
    int* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_ptr, &h_src, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&h_dst, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(h_dst, 42);

    CUDA_CHECK(cudaFree(d_ptr));
}

TEST(cudaRT, MemcpyAsync) {
    int h_src = 24;
    int h_dst = 0;
    int* d_ptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(int)));
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_ptr, &h_src, sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_dst, d_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(h_dst, 24);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_ptr));
}

TEST(cudaRT, Memset) {
    int* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ptr, 0, sizeof(int)));

    int h_val = 1;
    CUDA_CHECK(cudaMemcpy(&h_val, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(h_val, 0);

    CUDA_CHECK(cudaFree(d_ptr));
}

TEST(cudaRT, StreamCreateDestroySynchronize) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(cudaRT, StreamCaptureBeginEnd) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal;
    CUDA_CHECK(cudaStreamBeginCapture(stream, mode));
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(cudaRT, GraphCreateDestroy) {
    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

__global__ void dummyKernel() {
}

TEST(cudaRT, GraphInstantiateDestroy) {
    cudaGraph_t graph;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal;
    cudaGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    CUDA_CHECK(cudaStreamBeginCapture(stream, mode));
    dummyKernel<<<1, 1, 0, stream>>>(); 
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes, &numNodes));
    ASSERT_EQ(numNodes, 1);
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(cudaRT, GraphLaunch) {
    cudaGraph_t graph;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal;
    cudaGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    CUDA_CHECK(cudaStreamBeginCapture(stream, mode));
    dummyKernel<<<1, 1, 0, stream>>>(); 
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphGetNodes(graph, nodes, &numNodes));
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TEST(cudaRT, GetDevice) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
}

TEST(cudaRT, SetDevice) {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
}

TEST(cudaRT, DeviceSynchronize) { CUDA_CHECK(cudaDeviceSynchronize()); }

__global__ void simpleKernel(int* output) { *output = 123; }

TEST(cudaRT, LaunchKernel) {
    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(int)));

    void* args[] = {&d_output};

    dim3 grid(1), block(1);
    CUDA_CHECK(cudaLaunchKernel((const void*)simpleKernel, grid, block, args, 0, nullptr));

    int h_output = 0;
    CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(h_output, 123);

    CUDA_CHECK(cudaFree(d_output));
}

TEST(cudaRT, PushCallConfiguration) {
    dim3 grid(1), block(1);
    size_t shared = 0;
    cudaStream_t stream = 0;
    CUDA_CHECK(__cudaPushCallConfiguration(grid, block, shared, stream));
}

TEST(CudaRT, KernelLaunchWithTripletSyntax) {
    int* d_out = nullptr;
    int h_out = 0;

    // Allocate memory on device
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));

    // Launch kernel with <<<>>> syntax
    simpleKernel<<<1, 1>>>(d_out);

    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify kernel result
    ASSERT_EQ(h_out, 123);

    CUDA_CHECK(cudaFree(d_out));
}

TEST(cudaRT, EventCreateRecordSynchronizeElapsedTimeDestroy) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    ASSERT_GT(elapsed_ms, 0.0f);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
