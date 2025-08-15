#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <fstream>

#define N 512  // Matrix size: N x N

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Host function to initialize matrices
void initializeMatrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = rand() % 10;  // Random float between 0-9
    }
}

// Host function to verify the result (optional)
void verifyResult(float *A, float *B, float *C, int width) {
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < width; ++k) {
                expected += A[row * width + k] * B[k * width + col];
            }
            float diff = abs(C[row * width + col] - expected);
            if (diff > 1e-4) {
                printf("Mismatch at (%d, %d): GPU = %f, CPU = %f\n",
                       row, col, C[row * width + col], expected);
                return;
            }
        }
    }
    printf("Result verified successfully!\n");
}

int main() {
    std::ofstream csvFile("timing_output.csv");
    csvFile << "RunType,Iteration,ExecutionTime(ms)\n";

    int size = N * N;
    size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize input matrices
    initializeMatrix(h_A, size);
    initializeMatrix(h_B, size);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Test with different iteration counts
    int testIterations[] = {1, 100, 1000, 10000};

    for (int iter : testIterations) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        // ==================== Without Graph (StreamSync) ====================
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iter; ++i) {
            matrixMul<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, N);
            cudaStreamSynchronize(stream);
            //matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
            //cudaDeviceSynchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
        double averageTimeMs = totalMs / iter;
        csvFile << "WithoutGraph(NotSysc)," << iter << "," << averageTimeMs << "\n";

        // ==================== Without Graph (NotSysc) ====================
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iter; ++i) {
            matrixMul<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, N);
            //matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        }
        end = std::chrono::high_resolution_clock::now();
        totalMs = std::chrono::duration<double, std::milli>(end - start).count();
        averageTimeMs = totalMs / iter;
        csvFile << "WithoutGraph(Sysc)," << iter << "," << averageTimeMs << "\n";

        // ==================== With Graph ====================
        bool graphCreated = false;
        cudaGraph_t graph;
        cudaGraphExec_t instance;

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iter; ++i) {
            if (!graphCreated) {
                cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
                matrixMul<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, N);
                cudaStreamEndCapture(stream, &graph);
                cudaGraphInstantiate(&instance, graph, 0);
                cudaGraphDestroy(graph);
                graphCreated = true;
                // end = std::chrono::high_resolution_clock::now();
                // totalMs = std::chrono::duration<double, std::milli>(end - start).count();
                //std::cout << "graph, iterations: " << iter << " average: " << averageTimeMs << " ms\n";
                //csvFile << "Graph," << 0 << "," << totalMs << "\n";
            }
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);
        }
        end = std::chrono::high_resolution_clock::now();
        totalMs = std::chrono::duration<double, std::milli>(end - start).count();
        averageTimeMs = totalMs / iter;

        std::cout << "With graph, iterations: " << iter
                  << " average: " << averageTimeMs << " ms\n";
        csvFile << "WithGraph," << iter << "," << averageTimeMs << "\n";

        cudaStreamDestroy(stream);
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify result
    verifyResult(h_A, h_B, h_C, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    csvFile.close();
    return 0;
}
