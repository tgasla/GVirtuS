#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <cublas_v2.h>

// extern "C" {
//     __global__ void kernel_conv(float* input, 
//                 float* output, 
//                 float* weights, 
//                 float* bias, 
//                 int input_height, 
//                 int input_width, 
//                 int input_channels,
//                 int num_filters,
//                 int kernel_size,
//             int padding) {
//         int x = blockIdx.x * blockDim.x + threadIdx.x;  
//         int y = blockIdx.y * blockDim.y + threadIdx.y;  
//         int z = blockIdx.z;  
//         int output_height=input_height+2*padding-kernel_size+1;
//         int output_width=input_width+2*padding-kernel_size+1;

//         if (x < output_width && y < output_height && z < num_filters) {
//             float sum = 0.0f;

//             for (int c = 0; c < input_channels; ++c) {
//                 for (int i = 0; i < kernel_size; ++i) {
//                     for (int j = 0; j < kernel_size; ++j) {
//                         int input_x = x + i - padding; 
//                         int input_y = y + j - padding;
//                         if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height) {
//                             int input_idx = c * (input_height * input_width) + input_x * input_height + input_y;
//                             int weight_idx = z * (input_channels * kernel_size * kernel_size) 
//                                         + c * (kernel_size * kernel_size) 
//                                         + i * kernel_size + j;
//                             sum += input[input_idx] * weights[weight_idx];
//                         }
//                     }
//                 }
//             }

//             int output_idx = z * (output_height * output_width) + x * output_height + y;
//             output[output_idx] = sum + bias[z];
//             }
//         }

//     __global__ void kernel_maxpool(float* input, 
//                 float* output, 
//                 int input_channels, 
//                 int input_height, 
//                 int input_width, 
//                 int pool_size) {
//         int x = blockIdx.x * blockDim.x + threadIdx.x; 
//         int y = blockIdx.y * blockDim.y + threadIdx.y; 
//         int z = blockIdx.z;  

//         int output_height = input_height / pool_size;
//         int output_width = input_width / pool_size;

//         if (x < output_width && y < output_height && z < input_channels) {
//             float max_val = -1e38; 
//             for (int i = 0; i < pool_size; ++i) {
//                 for (int j = 0; j < pool_size; ++j) {
//                     int input_x = x * pool_size + i;
//                     int input_y = y * pool_size + j;
//                     int input_idx = z * (input_height * input_width) + input_x * input_height + input_y;
//                     max_val = fmaxf(max_val, input[input_idx]);
//                 }
//             }

//             int output_idx = z * (output_height * output_width) + x * output_height + y;
//             output[output_idx] = max_val;
//         }
//         }

//         __global__ void kernel_fc(float* input, 
//                     float* output, 
//                     float* weights, 
//                     float* bias, 
//                     int input_size, 
//                     int output_size) {
//         int x = blockIdx.x * blockDim.x + threadIdx.x;

//         if (x < output_size) {
//             float sum = 0.0f;
//             for (int i = 0; i < input_size; ++i) {
//                 sum += input[i] * weights[x * input_size + i];
//                 }
//             output[x] = sum + bias[x];
//         }
//         }

//     __global__ void kernel_flatten(float* input, float* output, 
//                                 int channels, int height, int width) {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x;
//         int input_size = channels * height * width;

//         if (idx < input_size) {
//             int c = idx / (height * width);
//             int h = (idx % (height * width)) / width;
//             int w = (idx % width);
//             output[idx] = input[c * (height * width) + h * width + w];
//         }
//     }

//     ReLU 
//     __global__ void kernel_relu(float* input, float* output, int size) {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x;
//         if (idx < size) {
//             output[idx] = fmaxf(0.0f, input[idx]);
//         }
//     }

//     void fully_connected_layer(const float* input, 
//                            const float* weights, 
//                            const float* bias, 
//                            float* output, 
//                            int M, 
//                            int N, 
//                            int K) {
//                             const float alpha = 1.0f;
//                             const float beta = 0.0f;
//                             cublasHandle_t handle;
//                             cublasCreate(&handle);
                         
//                             cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, weights, K, input, K, &beta, output, M);
//                             cublasSgemv(handle, CUBLAS_OP_T,K, M, &alpha, weights, K, input, 1, &beta, output, 1);
//                             cublasSaxpy(handle, M, &alpha, bias, 1, output, 1);
                            
//                             cublasDestroy(handle);
// }

//     void forward_pass(float (*data)[64], float *output, float (*weights_conv1)[3][3], float *bias_conv1,float (*weights_conv2)[32][3][3], float *bias_conv2, float (*weights_fc1)[16384], float *bias_fc1, float (*weights_fc2)[128], float *bias_fc2) {
//         Memory allocations and copying
//         cudaEvent_t start, stop;
//         cudaEventCreate(&start);
//         cudaEventCreate(&stop);
//         float time_taken = 0.0;
//         cudaEventRecord(start, NULL);

//         float (*input)[64][64];  

//         float (*d_weights_conv1)[32][3][3];  
//         float (*d_bias_conv1)[32];         
//         float (*d_output_conv1)[64][64][32];     

//         float (*d_output_pool1)[32][32][32];  

//         float (*d_weights_conv2)[64][32][3][3]; 
//         float (*d_bias_conv2)[64];         
//         float (*d_output_conv2)[32][32][64];    

//         float (*d_output_pool2)[16][16][64];  

//         float (*d_weights_fc1)[128][16384];  
//         float (*d_bias_fc1)[128];       
//         float (*d_output_fc1)[128];         

//         float (*d_weights_fc2)[40][128];    
//         float (*d_bias_fc2)[40];         
//         float (*d_output_fc2)[40];       
//         float (*d_flattened)[16384];

// 		cudaError_t err =cudaMalloc((void**)&input, 64 * 64 * sizeof(float));
//         if (err != cudaSuccess) {
//             printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
//         }
// 		cudaMemcpy(input, data, 64 * 64 * sizeof(float), cudaMemcpyHostToDevice);

// 		cudaMalloc((void**)&d_output_conv1, 64 * 64 * 32 * sizeof(float));
// 		cudaMalloc((void**)&d_weights_conv1, 32 * 3 * 3 * sizeof(float));
// 		cudaMalloc((void**)&d_bias_conv1, 32 * sizeof(float));

//         cudaMemcpy(d_weights_conv1, weights_conv1, 32 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
// 		cudaMemcpy(d_bias_conv1, bias_conv1, 32 * sizeof(float), cudaMemcpyHostToDevice);

//         cudaMalloc((void**)&d_output_pool1, 32 * 32 * 32 * sizeof(float));
        
// 		cudaMalloc((void**)&d_output_conv2, 32 * 32 * 64 * sizeof(float));
// 		cudaMalloc((void**)&d_weights_conv2, 64 *32* 3 * 3 * sizeof(float));
// 		cudaMalloc((void**)&d_bias_conv2, 64 * sizeof(float));

//         cudaMemcpy(d_weights_conv2, weights_conv2, 64 *32* 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
// 		cudaMemcpy(d_bias_conv2, bias_conv2, 64 * sizeof(float), cudaMemcpyHostToDevice);

//         cudaMalloc((void**)&d_output_pool2, 16 * 16 * 64 * sizeof(float));

//         cudaMalloc((void**)&d_weights_fc1, 128 * 16384 * sizeof(float));

//         cudaMemcpy(d_weights_fc1, weights_fc1, 128 * 16384 * sizeof(float), cudaMemcpyHostToDevice);

// 		cudaMalloc((void**)&d_bias_fc1, 128 * sizeof(float));

// 		cudaMemcpy(d_bias_fc1, bias_fc1, 128 * sizeof(float), cudaMemcpyHostToDevice);

// 		cudaError_t err1 =cudaMalloc((void**)&d_output_fc1, 128 * sizeof(float));
//         if (err1 != cudaSuccess) {
//             printf("CUDA malloc failed: %s\n", cudaGetErrorString(err1));
//         }

//         cudaMalloc((void**)&d_weights_fc2, 40 * 128 * sizeof(float));

//         cudaMemcpy(d_weights_fc2, weights_fc2, 40 * 128 * sizeof(float), cudaMemcpyHostToDevice);

// 		cudaMalloc((void**)&d_bias_fc2, 40 * sizeof(float));

// 		cudaMemcpy(d_bias_fc2, bias_fc2, 40 * sizeof(float), cudaMemcpyHostToDevice);

// 		cudaMalloc((void**)&d_output_fc2, 40 * sizeof(float));

//         cudaMalloc((void**)&d_flattened, 16384 * sizeof(float));
       
//         int input_channel=1;
//         int input_height = 64;
//         int input_width = 64;
//         int output_channels_conv1 = 32;
//         int output_channels_conv2 = 64;
//         int output_size_fc1 = 128;
        
//         dim3 threadsPerBlockConv(16, 16);  
//         dim3 numBlocksConv((input_width + threadsPerBlockConv.x - 1) / threadsPerBlockConv.x, 
//                         (input_height + threadsPerBlockConv.y - 1) / threadsPerBlockConv.y, 
//                         output_channels_conv1); 
//         dim3 numBlocksConv2((32 + threadsPerBlockConv.x - 1) / threadsPerBlockConv.x, 
//                         (32 + threadsPerBlockConv.y - 1) / threadsPerBlockConv.y, 
//                         output_channels_conv2);  

//         dim3 threadsPerBlockPool(2, 2);  
//         dim3 numBlocksPool((input_width / 2 + threadsPerBlockPool.x - 1) / threadsPerBlockPool.x, 
//                         (input_height / 2 + threadsPerBlockPool.y - 1) / threadsPerBlockPool.y, 
//                         output_channels_conv1); 
//         dim3 numBlocksPool2((32 / 2 + threadsPerBlockPool.x - 1) / threadsPerBlockPool.x, 
//                         (32 / 2 + threadsPerBlockPool.y - 1) / threadsPerBlockPool.y, 
//                         output_channels_conv2);  

//         dim3 threadsPerBlockFC(128);  
//         dim3 numBlocksFC(128); 
//         int blockSize = 512;  
//         int gridSize = (16384 + blockSize - 1) / blockSize; 
//         dim3 grid(gridSize);  

//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		float milliseconds = 0.00f;
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//         cudaEventRecord(start, NULL);
    
//         kernel_conv<<<numBlocksConv, threadsPerBlockConv>>>(reinterpret_cast<float*>(input), reinterpret_cast<float*>(d_output_conv1), reinterpret_cast<float*>(d_weights_conv1), reinterpret_cast<float*>(d_bias_conv1),  input_height, input_width,input_channel,output_channels_conv1, 3,1);
//         cudaDeviceSynchronize();

//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//         cudaEventRecord(start, NULL);

//         kernel_maxpool<<<numBlocksPool, threadsPerBlockPool>>>(reinterpret_cast<float*>(d_output_conv1), reinterpret_cast<float*>(d_output_pool1), 
//                                                             output_channels_conv1, input_height, 
//                                                             input_width, 2);
//         cudaDeviceSynchronize();

//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//         cudaEventRecord(start, NULL);

//         kernel_conv<<<numBlocksConv2, threadsPerBlockConv>>>(reinterpret_cast<float*>(d_output_pool1), reinterpret_cast<float*>(d_output_conv2), reinterpret_cast<float*>(d_weights_conv2), 
//                                                             reinterpret_cast<float*>(d_bias_conv2),32, 
//                                                             32, output_channels_conv1, 
//                                                             output_channels_conv2, 3,1);
//         cudaDeviceSynchronize();

//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//         cudaEventRecord(start, NULL);

//         kernel_maxpool<<<numBlocksPool2, threadsPerBlockPool>>>(reinterpret_cast<float*>(d_output_conv2), reinterpret_cast<float*>(d_output_pool2), 
//                                                             output_channels_conv2, 32, 
//                                                             32 , 2);
//         cudaDeviceSynchronize();

//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//         cudaEventRecord(start, NULL);

//         kernel_flatten<<<numBlocksFC, threadsPerBlockFC>>>(reinterpret_cast<float*>(d_output_pool2), reinterpret_cast<float*>(d_flattened), 
//         output_channels_conv2,16, 16);
//         cudaDeviceSynchronize();

//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//         cudaEventRecord(start, NULL);

//         fully_connected_layer(reinterpret_cast<float*>(d_flattened), reinterpret_cast<float*>(d_weights_fc1), reinterpret_cast<float*>(d_bias_fc1), reinterpret_cast<float*>(d_output_fc1), 128,1,16384);
//         kernel_fc<<<numBlocksFC, threadsPerBlockFC>>>(reinterpret_cast<float*>(d_flattened), reinterpret_cast<float*>(d_output_fc1), reinterpret_cast<float*>(d_weights_fc1), 
//                                                             reinterpret_cast<float*>(d_bias_fc1), 16384, 128); 
//         cudaDeviceSynchronize();

//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//         cudaEventRecord(start, NULL);

//         kernel_relu<<<numBlocksFC, threadsPerBlockFC>>>(reinterpret_cast<float*>(d_output_fc1), reinterpret_cast<float*>(d_output_fc1), output_size_fc1);
//         cudaDeviceSynchronize();

//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//         cudaEventRecord(start, NULL);

//         fully_connected_layer(reinterpret_cast<float*>(d_output_fc1), reinterpret_cast<float*>(d_weights_fc2), reinterpret_cast<float*>(d_bias_fc2), reinterpret_cast<float*>(d_output_fc2), 40,1,128);
//                 kernel_fc<<<numBlocksFC, threadsPerBlockFC>>>(reinterpret_cast<float*>(d_output_fc1), reinterpret_cast<float*>(d_output_fc2), reinterpret_cast<float*>(d_weights_fc2), 
//                                                     reinterpret_cast<float*>(d_bias_fc2), output_size_fc1, output_size_fc2);
//         cudaDeviceSynchronize();

//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//         cudaEventRecord(start, NULL);

// 		cudaMemcpy(output, d_output_fc2, 40 * sizeof(float), cudaMemcpyDeviceToHost);

//         for (int i = 0; i < 40; i++) {
//             printf("%f ", output[i]);
//         }
//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//         cudaEventRecord(start, NULL);

//         cudaFree(input);
// 		cudaFree(d_output_conv1);
// 		cudaFree(d_weights_conv1);
// 		cudaFree(d_bias_conv1);
// 		cudaFree(d_output_conv2);
// 		cudaFree(d_weights_conv2);
// 		cudaFree(d_bias_conv2);
// 		cudaFree(d_weights_fc1);
// 		cudaFree(d_bias_fc1);
//         cudaFree(d_output_fc1);
// 		cudaFree(d_weights_fc2);
// 		cudaFree(d_bias_fc2);
//         cudaFree(d_output_fc2);
//         cudaFree(d_flattened);

//         cudaEventRecord(stop, NULL);
// 		cudaEventSynchronize(stop);
// 		cudaEventElapsedTime(&milliseconds, start, stop);
// 		time_taken = milliseconds;
//         printf("Ex time = %f (ms) \n", time_taken);
//     }
// } 
 
extern "C" {
    __global__ void kernel_conv(float* input, 
                float* output, 
                float* weights, 
                float* bias, 
                int input_height, 
                int input_width, 
                int input_channels,
                int num_filters,
                int kernel_size,
            int padding) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;  
        int y = blockIdx.y * blockDim.y + threadIdx.y;  
        int z = blockIdx.z;  
        int output_height=input_height+2*padding-kernel_size+1;
        int output_width=input_width+2*padding-kernel_size+1;

        if (x < output_width && y < output_height && z < num_filters) {
            float sum = 0.0f;

            for (int c = 0; c < input_channels; ++c) {
                for (int i = 0; i < kernel_size; ++i) {
                    for (int j = 0; j < kernel_size; ++j) {
                        int input_x = x + i - padding; 
                        int input_y = y + j - padding;
                        if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height) {
                            int input_idx = c * (input_height * input_width) + input_x * input_height + input_y;
                            int weight_idx = z * (input_channels * kernel_size * kernel_size) 
                                        + c * (kernel_size * kernel_size) 
                                        + i * kernel_size + j;
                            sum += input[input_idx] * weights[weight_idx];
                        }
                    }
                }
            }

            int output_idx = z * (output_height * output_width) + x * output_height + y;
            output[output_idx] = sum + bias[z];
            }
        }

    __global__ void kernel_maxpool(float* input, 
                float* output, 
                int input_channels, 
                int input_height, 
                int input_width, 
                int pool_size) {
        int x = blockIdx.x * blockDim.x + threadIdx.x; 
        int y = blockIdx.y * blockDim.y + threadIdx.y; 
        int z = blockIdx.z;  

        int output_height = input_height / pool_size;
        int output_width = input_width / pool_size;

        if (x < output_width && y < output_height && z < input_channels) {
            float max_val = -1e38; 
            for (int i = 0; i < pool_size; ++i) {
                for (int j = 0; j < pool_size; ++j) {
                    int input_x = x * pool_size + i;
                    int input_y = y * pool_size + j;
                    int input_idx = z * (input_height * input_width) + input_x * input_height + input_y;
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }

            int output_idx = z * (output_height * output_width) + x * output_height + y;
            output[output_idx] = max_val;
        }
        }

        __global__ void kernel_fc(float* input, 
                    float* output, 
                    float* weights, 
                    float* bias, 
                    int input_size, 
                    int output_size) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;

        if (x < output_size) {
            float sum = 0.0f;
            for (int i = 0; i < input_size; ++i) {
                sum += input[i] * weights[x * input_size + i];
                }
            output[x] = sum + bias[x];
        }
        }

    __global__ void kernel_flatten(float* input, float* output, 
                                int channels, int height, int width) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int input_size = channels * height * width;

        if (idx < input_size) {
            int c = idx / (height * width);
            int h = (idx % (height * width)) / width;
            int w = (idx % width);
            output[idx] = input[c * (height * width) + h * width + w];
        }
    }

    // ReLU 
    __global__ void kernel_relu(float* input, float* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = fmaxf(0.0f, input[idx]);
        }
    }

    void fully_connected_layer(const float* input, 
                           const float* weights, 
                           const float* bias, 
                           float* output, 
                           int M, 
                           int N, 
                           int K) {
                            const float alpha = 1.0f;
                            const float beta = 0.0f;
                            cublasHandle_t handle;
                            cublasCreate(&handle);
                         
                            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, weights, K, input, K, &beta, output, M);
                            // cublasSgemv(handle, CUBLAS_OP_T,K, M, &alpha, weights, K, input, 1, &beta, output, 1);
                            cublasSaxpy(handle, M, &alpha, bias, 1, output, 1);
                            
                            cublasDestroy(handle);
}


class Model {
    public:
    // int M, N, O;
    // float (*weights)[5][5];
    // float (*bias);
    // float (*output)[24][24];
    // float (*pre_output)[24][24];
	// float (*output)[24][24];
    
    // Function to build the model (load parameters into GPU memory)
    Model(float (*weights_conv1)[3][3], float *bias_conv1,float (*weights_conv2)[32][3][3], float *bias_conv2, float (*weights_fc1)[16384], float *bias_fc1, float (*weights_fc2)[128], float *bias_fc2)
        // M(M), N(N), O(O),
		// weights(new float[6][5][5]),
		// bias(new float[6]),
		// output(new float[6][24][24]),
		// pre_output(new float[6][24][24])
    {
        cudaMalloc((void**)&d_output_conv1, 64 * 64 * 32 * sizeof(float));
        cudaMalloc((void**)&d_weights_conv1, 32 * 3 * 3 * sizeof(float));
        cudaMalloc((void**)&d_bias_conv1, 32 * sizeof(float));
        cudaMalloc((void**)&d_output_pool1, 32 * 32 * 32 * sizeof(float));
        cudaMalloc((void**)&d_output_conv2, 32 * 32 * 64 * sizeof(float));
        cudaMalloc((void**)&d_weights_conv2, 64 *32* 3 * 3 * sizeof(float));
        cudaMalloc((void**)&d_bias_conv2, 64 * sizeof(float));
        cudaMalloc((void**)&d_output_pool2, 16 * 16 * 64 * sizeof(float));
        cudaMalloc((void**)&d_weights_fc1, 128 * 16384 * sizeof(float));
        cudaMalloc((void**)&d_bias_fc1, 128 * sizeof(float));
        cudaMalloc((void**)&d_output_fc1, 128 * sizeof(float));
        cudaMalloc((void**)&d_weights_fc2, 40 * 128 * sizeof(float));
        cudaMalloc((void**)&d_bias_fc2, 40 * sizeof(float));
        cudaMalloc((void**)&d_output_fc2, 40 * sizeof(float));
        cudaMalloc((void**)&d_flattened, 16384 * sizeof(float));

        cudaMemcpy(d_weights_conv1, weights_conv1, 32 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bias_conv1, bias_conv1, 32 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights_conv2, weights_conv2, 64 *32* 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bias_conv2, bias_conv2, 64 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights_fc1, weights_fc1, 128 * 16384 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bias_fc1, bias_fc1, 128 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights_fc2, weights_fc2, 40 * 128 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bias_fc2, bias_fc2, 40 * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // Forward pass to calculate the output
    void forward_pass(float (*data)[64], float *output) {
        float (*input)[64][64];  
        cudaMalloc((void**)&input, 64 * 64 * sizeof(float));
		cudaMemcpy(input, data, 64 * 64 * sizeof(float), cudaMemcpyHostToDevice);
        int input_height = 64;
        int input_width = 64;
        int input_channel=1;
        int output_channels_conv1 = 32;
        int output_channels_conv2 = 64;
        int output_size_fc1 = 128;
        // cudaMemcpy(output, input, 40 * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < 40; i++) {
        //     printf("%f ", output[i]);
        // }
        
        dim3 threadsPerBlockConv(16, 16);  
        dim3 numBlocksConv((input_width + threadsPerBlockConv.x - 1) / threadsPerBlockConv.x, 
                        (input_height + threadsPerBlockConv.y - 1) / threadsPerBlockConv.y, 
                        output_channels_conv1); 
        dim3 numBlocksConv2((32 + threadsPerBlockConv.x - 1) / threadsPerBlockConv.x, 
                        (32 + threadsPerBlockConv.y - 1) / threadsPerBlockConv.y, 
                        output_channels_conv2);  

        dim3 threadsPerBlockPool(2, 2);  
        dim3 numBlocksPool((input_width / 2 + threadsPerBlockPool.x - 1) / threadsPerBlockPool.x, 
                        (input_height / 2 + threadsPerBlockPool.y - 1) / threadsPerBlockPool.y, 
                        output_channels_conv1); 
        dim3 numBlocksPool2((32 / 2 + threadsPerBlockPool.x - 1) / threadsPerBlockPool.x, 
                        (32 / 2 + threadsPerBlockPool.y - 1) / threadsPerBlockPool.y, 
                        output_channels_conv2);  

        dim3 threadsPerBlockFC(128);  
        dim3 numBlocksFC(128); 
        int blockSize = 512;  
        int gridSize = (16384 + blockSize - 1) / blockSize; 
        dim3 grid(gridSize);  
    
        kernel_conv<<<numBlocksConv, threadsPerBlockConv>>>(reinterpret_cast<float*>(input), reinterpret_cast<float*>(d_output_conv1), reinterpret_cast<float*>(d_weights_conv1), reinterpret_cast<float*>(d_bias_conv1),  input_height, input_width,input_channel,output_channels_conv1, 3,1);
        cudaDeviceSynchronize();

        kernel_maxpool<<<numBlocksPool, threadsPerBlockPool>>>(reinterpret_cast<float*>(d_output_conv1), reinterpret_cast<float*>(d_output_pool1), 
                                                            output_channels_conv1, input_height, 
                                                            input_width, 2);
        cudaDeviceSynchronize();

        kernel_conv<<<numBlocksConv2, threadsPerBlockConv>>>(reinterpret_cast<float*>(d_output_pool1), reinterpret_cast<float*>(d_output_conv2), reinterpret_cast<float*>(d_weights_conv2), 
                                                            reinterpret_cast<float*>(d_bias_conv2),32, 
                                                            32, output_channels_conv1, 
                                                            output_channels_conv2, 3,1);
        cudaDeviceSynchronize();

        kernel_maxpool<<<numBlocksPool2, threadsPerBlockPool>>>(reinterpret_cast<float*>(d_output_conv2), reinterpret_cast<float*>(d_output_pool2), 
                                                            output_channels_conv2, 32, 
                                                            32 , 2);
        cudaDeviceSynchronize();

        kernel_flatten<<<numBlocksFC, threadsPerBlockFC>>>(reinterpret_cast<float*>(d_output_pool2), reinterpret_cast<float*>(d_flattened), 
        output_channels_conv2,16, 16);
        cudaDeviceSynchronize();

        fully_connected_layer(reinterpret_cast<float*>(d_flattened), reinterpret_cast<float*>(d_weights_fc1), reinterpret_cast<float*>(d_bias_fc1), reinterpret_cast<float*>(d_output_fc1), 128,1,16384);
        // kernel_fc<<<numBlocksFC, threadsPerBlockFC>>>(reinterpret_cast<float*>(d_flattened), reinterpret_cast<float*>(d_output_fc1), reinterpret_cast<float*>(d_weights_fc1), 
        //                                                     reinterpret_cast<float*>(d_bias_fc1), 16384, 128); 
        cudaDeviceSynchronize();

        kernel_relu<<<numBlocksFC, threadsPerBlockFC>>>(reinterpret_cast<float*>(d_output_fc1), reinterpret_cast<float*>(d_output_fc1), output_size_fc1);
        cudaDeviceSynchronize();

        fully_connected_layer(reinterpret_cast<float*>(d_output_fc1), reinterpret_cast<float*>(d_weights_fc2), reinterpret_cast<float*>(d_bias_fc2), reinterpret_cast<float*>(d_output_fc2), 40,1,128);
                // kernel_fc<<<numBlocksFC, threadsPerBlockFC>>>(reinterpret_cast<float*>(d_output_fc1), reinterpret_cast<float*>(d_output_fc2), reinterpret_cast<float*>(d_weights_fc2), 
        //                                             reinterpret_cast<float*>(d_bias_fc2), output_size_fc1, output_size_fc2);
        cudaDeviceSynchronize();

		cudaMemcpy(output, d_output_fc2, 40 * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(input);
    }

    ~Model() {
        // Free device memory
		cudaFree(d_output_conv1);
		cudaFree(d_weights_conv1);
		cudaFree(d_bias_conv1);
		cudaFree(d_output_conv2);
		cudaFree(d_weights_conv2);
		cudaFree(d_bias_conv2);
		cudaFree(d_weights_fc1);
		cudaFree(d_bias_fc1);
        cudaFree(d_output_fc1);
		cudaFree(d_weights_fc2);
		cudaFree(d_bias_fc2);
        cudaFree(d_output_fc2);
        cudaFree(d_flattened);
    }

private:
    float (*d_weights_conv1)[32][3][3];  
    float (*d_bias_conv1)[32];         
    float (*d_output_conv1)[64][64][32];     

    float (*d_output_pool1)[32][32][32];  

    float (*d_weights_conv2)[64][32][3][3]; 
    float (*d_bias_conv2)[64];         
    float (*d_output_conv2)[32][32][64];    

    float (*d_output_pool2)[16][16][64];  

    float (*d_weights_fc1)[128][16384];  
    float (*d_bias_fc1)[128];       
    float (*d_output_fc1)[128];         

    float (*d_weights_fc2)[40][128];    
    float (*d_bias_fc2)[40];         
    float (*d_output_fc2)[40];       
    float (*d_flattened)[16384];
    };

    Model* create_model(float (*weights_conv1)[3][3], float *bias_conv1, 
                        float (*weights_conv2)[32][3][3], float *bias_conv2, 
                        float (*weights_fc1)[16384], float *bias_fc1, 
                        float (*weights_fc2)[128], float *bias_fc2) {
        return new Model(weights_conv1, bias_conv1, weights_conv2, bias_conv2, weights_fc1, bias_fc1, weights_fc2, bias_fc2);
    }
    
    // Function to delete a Layer instance
    void delete_model(Model* model) {
        delete model;
    }

    void forward_pass(Model* model, float (*data)[64], float *output) {
        model->forward_pass(data, output);
    }
} 