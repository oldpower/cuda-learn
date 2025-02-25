#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.hpp"

__global__ void print_idx_kernal()
{
    printf("block idx: (%3d,%3d,%3d), thread inx: (%3d,%3d,%3d) \n",
            blockIdx.z,blockIdx.y,blockIdx.x,
            threadIdx.z,threadIdx.y,threadIdx.x);
}

__global__ void print_dim_kernal()
{
    printf("grid dim: (%3d,%3d,%3d), block dim(%3d,%3d,%3d)\n",
            gridDim.z,gridDim.y,gridDim.x,
            blockDim.z,blockDim.y,blockDim.x);
}
__global__ void print_thread_idx_per_block_kernal()
{
    int index = threadIdx.z + blockDim.x * blockDim.y +
                threadIdx.y + blockDim.x +
                threadIdx.y;
    printf("block idx:(%3d,%3d,%3d),thread idx:%3d\n",
            blockIdx.z,blockIdx.y,blockIdx.x,
            index);
}

__global__ void print_thread_idx_per_grid_kernal()
{
    int bSize = blockDim.z * blockDim.y * blockDim.x;

    int bIndex = blockIdx.z * gridDim.x * gridDim.y +
                blockIdx.y * gridDim.x +
                blockIdx.x;
    int tIndex = threadIdx.z * blockDim.x * blockDim.y +
                threadIdx.y * blockDim.x + 
                threadIdx.x;

    int index = bIndex * bSize + tIndex;

    printf("block idx: %3d, thread idx in block: %3d, thread idx: %3d\n",
            bIndex,tIndex,index);
}

void print_idx_device(dim3 grid,dim3 block)
{
    print_idx_kernal<<<grid,block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void print_dim_device(dim3 gird,dim3 block)
{
    print_dim_kernal<<<gird,block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}
void print_thread_idx_per_block_device(dim3 grid,dim3 block)
{

    print_thread_idx_per_block_kernal<<<grid,block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}
void print_thread_idx_device(dim3 grid,dim3 block)
{
    print_thread_idx_per_grid_kernal<<<grid,block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
} 


