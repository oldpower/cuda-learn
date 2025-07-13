#include <stdio.h>
#include <cuda_runtime.h>
#include "timer.hpp"
#include "utils.hpp"
#include "matmul.hpp"


int seed;
int main()
{
    Timer timer;
    int width = 1<<8;
    int min = 0;
    int max = 1;
    int size = width * width;
    int blockSize = 16;
    bool statMem = true;
    char str[100];
    
    float *h_matM = (float*)malloc(size * sizeof(float));
    float *h_matN = (float*)malloc(size * sizeof(float));
    float *h_matP = (float*)malloc(size * sizeof(float));
    float *d_matP = (float*)malloc(size * sizeof(float));

    seed = 1;
    initMatrix(h_matM,size,min,max,seed);
    seed += 1;
    initMatrix(h_matN,size,min,max,seed);

    // cpu
    timer.start_cpu();
    MatmulOnHost(h_matM,h_matN,h_matP,width);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("matmul in cpu");

    LOG("Input size is %d x %d", width, width);
    /* GPU warmup */
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu(warmup)");

    /* GPU general implementation <<<256, 16>>>*/
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(without shared memory)<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    // /* GPU general implementation <<<256, 16>>>*/
    timer.start_gpu();
    MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(static))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    timer.start_gpu();
    MatmulSharedConflictOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(static, bank conflict))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    timer.start_gpu();
    MatmulSharedConflictPadOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(static, conflict pad))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    /* GPU general implementation <<<256, 16>>>*/
    statMem = false;
    timer.start_gpu();
    MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(dynamic))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    timer.start_gpu();
    MatmulSharedConflictOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(dynamic, bank conflict))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    timer.start_gpu();
    MatmulSharedConflictPadOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(shared memory(dynamic, conflict pad))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    return 0;
}
