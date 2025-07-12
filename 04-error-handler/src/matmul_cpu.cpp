#include "matmul.hpp"

void MatmulOnHost(float *M_host, float *N_host, float *P_host, int width)
{
    for(int i=0;i<width;i++)
    {
        for(int j=0;j<width;j++)
        {
            float sum = 0;
            for(int k=0;k<width;k++)
            {
                float a = M_host[i*width + k];
                float b = N_host[k*width + j];
                sum +=a*b;
            }
            P_host[i*width + j]=sum;
        }
    }
}
