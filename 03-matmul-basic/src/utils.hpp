#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <cuda_runtime.h>

# define CUDACHECK(call){                                               \
    cudaError_t error = call;                                           \
    if(error != cudaSuccess){                                           \
        print("ERROR: %s:%d, ",__FILE__,__LINE__);                      \
        print("CODE:%d, DETAIL:%s \n",error,cudaGetErrorString(error)); \
        exit(1);                                                        \
    }                                                                   \
}

#define BLOCKSIZE 16

void initMatrix(float* data,int size,int low,int high,int seed);
void printMat(float* data,int size);
void compareMat(float* h_data,float* d_data,int size);


#endif //__UTILS__HPP__
