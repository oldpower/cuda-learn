#include <cuda_runtime.h>
#include "print_index.hpp"

void print_one_dim(int inputSize,int blockSize)
{
    int gridSize = inputSize/blockSize;

    dim3 block(blockSize);
    dim3 grid(gridSize);

    print_thread_idx_device(block,grid);
}

int main()
{
    int inputSize;
    int blockSize;
    
    inputSize = 8;
    blockSize = 4;

    print_one_dim(inputSize,blockSize);

    return 0;
}
