#include "timer.hpp"
#include "utils.hpp"
#include "matmul.hpp"


int seed;
int main()
{
    Timer timer;
    int width = 1<<10;
    int min = 0;
    int max = 1;
    int size = width * width;
    int blockSize = 1;
    
    float *h_matM = (float*)malloc(size * sizeof(float));
    float *h_matN = (float*)malloc(size * sizeof(float));
    float *h_matP = (float*)malloc(size * sizeof(float));
    float *d_matP = (float*)malloc(size * sizeof(float));

    seed = 1;
    initMatrix(h_matM,size,min,max,seed);
    seed += 1;
    initMatrix(h_matN,size,min,max,seed);

    // cpu
    timer.start();
    MatmulOnHost(h_matM,h_matN,h_matP,width);
    timer.stop();
    timer.duration<Timer::ms>("matmul in cpu");

    // gpu warmup
    timer.start();
    MatmulOnDevice(h_matM, h_matN, h_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in gpu(warmup)");
    
    // gpu blockSize 16;
    blockSize = 16;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, h_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in gpu(blockSize = 16)");
    compareMat(h_matP, d_matP, size);

    
    // gpu blockSize 1;
    blockSize = 1;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, h_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in gpu(blockSize = 1)");
    compareMat(h_matP, d_matP, size);

    // gpu blockSize 32;
    blockSize = 32;
    timer.start();
    MatmulOnDevice(h_matM, h_matN, h_matP, width, blockSize);
    timer.stop();
    timer.duration<Timer::ms>("matmul in gpu(blockSize = 32)");
    compareMat(h_matP, d_matP, size);

    return 0;
}
