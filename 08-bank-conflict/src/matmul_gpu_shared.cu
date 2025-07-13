#include "cuda_runtime_api.h"

#include "utils.hpp"


__global__ void MatmulSharedStaticKernel (float *M_device, float *N_device, float *P_device, int width){
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    for(int m = 0 ; m < width/BLOCKSIZE ; m++){
    	M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
	N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty) * width + x];

	__syncthreads();

	for(int k = 0 ; k < BLOCKSIZE; k++){
	    P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
	}

	__syncthreads();
    }

    P_device[y * width + x] = P_element;

}

__global__ void MatmulSharedDynamicKernel (float *M_device, float *N_device, float *P_device, int width, int blockSize){
    //动态共享内存是一维的，使用时需要extern声明
    extern __shared__ float deviceShared[];

    int stride = blockSize * blockSize;

    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    for(int m = 0; m < width / blockSize; m++){
    	deviceShared[ty * blockSize + tx] = M_device[y * width + (m * blockSize + tx)];
	deviceShared[stride + (ty * blockSize + tx)] = N_device[(m * blockSize + ty) * width + x];

	__syncthreads();

	for(int k = 0; k < blockSize; k++){
	    P_element += deviceShared[ty * blockSize + k] * deviceShared[stride + (k * blockSize + tx)];
	}

	__syncthreads();
    }

    if (y < width && x < width){
    	P_device[y * width + x] = P_element;
    }
}


void MatmulSharedOnDevice(float *M_host,float *N_host, float* P_host, int width,int blockSize, bool staticMem)
{
    //设置矩阵大小
    int size = width * width * sizeof(float);
    //设置共享内存块大小
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    
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

    if (staticMem){
    	MatmulSharedStaticKernel <<<dimGrid,dimBlock>>>(M_device,N_device,P_device,width);
    }else{
	// nullptr 是默认 stream
	MatmulSharedDynamicKernel <<<dimGrid, dimBlock, sMemSize, nullptr>>>(M_device, N_device, P_device, width, blockSize);
    }

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
