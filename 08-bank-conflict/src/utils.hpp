#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <system_error>
#include <stdarg.h>

# define CUDACHECK(call){                                               \
    cudaError_t error = call;                                           \
    if(error != cudaSuccess){                                           \
        print("ERROR: %s:%d, ",__FILE__,__LINE__);                      \
        print("CODE:%d, DETAIL:%s \n",error,cudaGetErrorString(error)); \
        exit(1);                                                        \
    }                                                                   \
}

# define CUDA_CHECK(call)       __cudaCheck(call,__FILE__,__LINE__)
# define LAST_KERNEL_CHECK()	__kernalCheck(__FILE__,__LINE__)
# define LOG(...)		__log_info(__VA_ARGS__)

#define BLOCKSIZE 16


inline static void __cudaCheck(cudaError_t err, const char* file, const int line){
    if(err != cudaSuccess){                                           
        printf("ERROR: %s:%d, ",__FILE__,__LINE__);                      
        printf("CODE:%d, DETAIL:%s \n",err, cudaGetErrorString(err)); 
        exit(1);                                                        
    }                                                                   
}

inline static void __kernalCheck(const char* file, const int line) {
	cudaError_t err = cudaPeekAtLastError();
	if(err != cudaSuccess){
		printf("ERROR: %s:%d, ", file, line);
		printf("CODE: %s, DETAL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
		exit(1);
	}
}

static void __log_info(const char* format, ...){
    char msg[1000];
    va_list args;
    va_start(args,format);

    vsnprintf(msg, sizeof(msg), format, args);
    
    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

void initMatrix(float* data,int size,int low,int high,int seed);
void printMat(float* data,int size);
void compareMat(float* h_data,float* d_data,int size);


#endif //__UTILS__HPP__
