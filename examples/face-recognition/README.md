## Human face recognition

### dataset

Olivetti faces dataset

- [5.6.1. The Olivetti faces dataset &#8212; scikit-learn 0.19.2 documentation](https://scikit-learn.org/0.19/datasets/olivetti_faces.html)

- 400 human face images of 40 people (10 for each)

- size of each image is 64x64

- each image is quantized to 256 grey levels and stored as unsigned 8-bit integers

- randomly select 320 images for training set and 80 images for test set

- image sample

- ![](C:\Users\UE18WC\OneDrive%20-%20Aalborg%20Universitet\Desktop\samples\human_face\images\test_image_0.png)

### model

- recognition (classification task)

- model structure (CNN)
  
  - convolutional layer with 32 filters (1 * 64 * 64 -> 32 * 64 * 64)
  
  - maxpooling (32 * 64 * 64 -> 32 * 32 * 32)
  
  - convolutional layer with 64 filters (32 * 32 * 32 -> 64 * 32 * 32)
  
  - maxpooling (64 * 32 * 32 ->  64 * 16 * 16)
  
  - flatten (64 * 16 * 16 -> 16384)
  
  - fully connected layer (16384 -> 128)
  
  - relu activation (128)
  
  - fully connected layer (128 -> 40)

## experiments

test 10 images, the prediction accuracy is 90%

Repeat 10 times consecutively

| （second） | Run on backend | Run on frontend with TCP/IP <br/>(both on laptop) |
| -------- | -------------- | ------------------------------------------------- |
| 1        | 2.824892       | 2.181764                                          |
| 2        | 0.048304       | 0.232937                                          |
| 3        | 0.048830       | 0.233665                                          |
| 4        | 0.046227       | 0.236016                                          |
| 5        | 0.046448       | 0.230157                                          |
| 6        | 0.048053       | 0.221213                                          |
| 7        | 0.046144       | 0.218673                                          |
| 8        | 0.047910       | 0.220521                                          |
| 9        | 0.046386       | 0.210538                                          |
| 10       | 0.046363       | 0.218717                                          |
| Average  | 0.324956       | 0.420420                                          |

readed size on backend

| Function name             |         |
| ------------------------- | ------- |
| cudaRegisterFatBinary     | 18711   |
| cudaRegisterFunction      | 139     |
| cudaRegisterFunction      | 145     |
| cudaRegisterFunction      | 135     |
| cudaRegisterFunction      | 145     |
| cudaRegisterFunction      | 139     |
| cudaRegisterFatBinaryEnd  | 18711   |
| cudaMalloc                | 8       |
| cudaMemcpy                | 16142   |
| cudaMalloc                | 8       |
| cudaMalloc                | 8       |
| cudaMalloc                | 8       |
| cudaMemcpy                | 1180    |
| cudaMemcpy                | 156     |
| cudaMalloc                | 8       |
| cudaMalloc                | 8       |
| cudaMalloc                | 8       |
| cudaMalloc                | 8       |
| cudaMemcpy                | 73756   |
| cudaMemcpy                | 284     |
| cudaMalloc                | 8       |
| cudaMalloc                | 8       |
| cudaMemcpy                | 8388636 |
| cudaMalloc                | 8       |
| cudaMemcpy                | 540     |
| cudaMalloc                | 8       |
| cudaMalloc                | 8       |
| cudaMemcpy                | 20508   |
| cudaMalloc                | 8       |
| cudaMemcpy                | 188     |
| cudaMalloc                | 8       |
| cudaMalloc                | 8       |
| cudaPushCallConfiguration | 40      |
| cudaPopCallConfiguration  | 0       |
| cudaLaunchKernel          | 112     |
| cudaDeviceSynchronize     | 0       |
| cudaPushCallConfiguration | 40      |
| cudaPopCallConfiguration  | 0       |
| cudaLaunchKernel          | 88      |
| cudaDeviceSynchronize     | 0       |
| cudaPushCallConfiguration | 40      |
| cudaPopCallConfiguration  | 0       |
| cudaLaunchKernel          | 112     |
| cudaDeviceSynchronize     | 0       |
| cudaPushCallConfiguration | 40      |
| cudaPopCallConfiguration  | 0       |
| cudaLaunchKernel          | 88      |
| cudaDeviceSynchronize     | 0       |
| cudaPushCallConfiguration | 40      |
| cudaPopCallConfiguration  | 0       |
| cudaLaunchKernel          | 84      |
| cudaDeviceSynchronize     | 0       |
| cublasCreate_v2           | 0       |
| cublasSgemm_v2            | 88      |
| cublasSaxpy_v2            | 48      |
| cublasDestroy_v2          | 8       |
| cudaDeviceSynchronize     | 0       |
| cudaPushCallConfiguration | 40      |
| cudaPopCallConfiguration  | 0       |
| cudaLaunchKernel          | 76      |
| cudaDeviceSynchronize     | 0       |
| cublasCreate_v2           | 0       |
| cublasSgemm_v2            | 88      |
| cublasSaxpy_v2            | 48      |
| cublasDestroy_v2          | 8       |
| cudaDeviceSynchronize     | 0       |
| cudaMemcpy                | 29      |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaFree                  | 8       |
| cudaUnregisterFatBinary   | 31      |

excution time

- disable logs 
- fully-connected layer spends lots of time for the first time running, probably caused by the initializtion of cublas library

| function            | test1 (ms)  | test2 (ms) |
| ------------------- | ----------- | ---------- |
| copy data to device | 16.683935   | 16.427872  |
| conv1               | 1.870848    | 2.422784   |
| maxpool1            | 1.743872    | 2.451456   |
| conv2               | 5.326848    | 5.277696   |
| maxpool2            | 4.952064    | 4.368384   |
| flatten             | 2.208768    | 1.755136   |
| fc1                 | 2148.074463 | 5.448704   |
| relu                | 1.957888    | 2.029568   |
| fc2                 | 6.960128    | 2.493440   |
| copy data to host   | 0.345408    | 0.196800   |
| cudafree            | 2.199904    | 2.646112   |

## todolist

- optimize the code of loading model

- try to output the size of buffer on FE

- function for pytorch/tensorflow
