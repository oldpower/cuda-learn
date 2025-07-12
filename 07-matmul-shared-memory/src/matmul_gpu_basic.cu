#include "cuda_runtime_api.h"

#include "utils.hpp"

__global__ void MatmulKernel(float *M_device,float *N_device,float *P_device,int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0.0;

    for (int k=0;k<width;k++)
    {
        float M_element = M_device[width * y + k];
        float N_element = N_device[width * k + x];
        P_element += M_element*N_element;
    }
    P_device[width * y + x] = P_element;
}

void MatmulOnDevice(float *M_host,float *N_host, float* P_host, int width,int blockSize)
{
    //设置矩阵大小
    int size = width * width * sizeof(float);
    
    //分配GPU空间
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc(&M_device,size));
    CUDA_CHECK(cudaMalloc(&N_device,size));
    float *P_device;
    CUDA_CHECK(cudaMalloc(&P_device,size));

    //将host数据copy至GPU
    CUDA_CHECK(cudaMemcpy(M_device,M_host,size,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device,N_host,size,cudaMemcpyHostToDevice));

    //核函数matmul
    dim3 dimBlock(blockSize,blockSize);
    dim3 dimGrid(width/blockSize,width/blockSize);
    MatmulKernel<<<dimGrid,dimBlock>>>(M_device,N_device,P_device,width);

    // 将结果copy到host
    CUDA_CHECK(cudaMemcpy(P_host,P_device,size,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 排查kernal错误
    LAST_KERNEL_CHECK();

    //Free
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}
