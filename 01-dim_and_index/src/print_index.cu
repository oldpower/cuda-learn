#include <stdio.h>
#include <cuda_runtime.h>

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

void print_one_dim()
{
    int inputSize = 8;
    int blockDim = 4;
    int gridDim = inputSize/blockDim;

    dim3 block(blockDim);
    dim3 grid(gridDim);

    // 打印核的索引
    print_idx_kernal<<<grid,block>>>();
    // 打印核的维度
    print_dim_kernal<<<grid,block>>>();
    // 打印核的每个block的thread索引
    print_thread_idx_per_block_kernal<<<grid,block>>>();
    // 打印核的每个grid的thread索引
    print_thread_idx_per_grid_kernal<<<grid,block>>>();

    cudaDeviceSynchronize();

}
int main()
{
    print_one_dim(); 

    return 0;
}
