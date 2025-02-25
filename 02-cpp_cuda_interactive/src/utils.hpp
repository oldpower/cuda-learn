#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda_runtime.h>

// 一般cuda的check都是这样写宏
#define CUDA_CHECK(call){} {                                                    \
    cudaError_t error = call;                                                   \
    if(error != cudaSuccess){                                                   \
        printf("ERROR:%s:%d",__FILE__,__LINE__);                                \
        printf("CODE:%d, DETAIL:%s\n",error,cudaGetErrorString(error));         \
        exit(1);                                                                \
    }                                                                           \
                                                                                \
}                                                                               

#endif // __UTILS_HPP__
