#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

__global__ void cudaFill(void* output)
{
    const uint64_t idx = ((uint64_t)blockIdx.x << 10) + threadIdx.x;
    *((uint64_t*)output + idx) = 0x0807060504030201ULL;
}

void Fill(void* output, uint64_t samples)
{
    cudaFill << < samples >> 12, 0x400 >> > (output);
}

int main()
{
    const uint64_t samples = (uint64_t)1 << 13;
    void* cpuArr = malloc(samples << 2);
    void* gpuArr;
    cudaMalloc(&gpuArr, samples << 2);

    Fill(gpuArr, samples);

    cudaMemcpy(cpuArr, gpuArr, samples << 2, cudaMemcpyDeviceToHost);

    // 16-bit version
    for (uint64_t i = 0; i < samples; i++)
    {
        uint16_t* ptr = (uint16_t*)cpuArr + i;
        for (int j = 16; j--;)
            printf("%u", (bool)((*ptr >> j) & 1));
        printf("\n");
    }

    // 64-bit version
    for (uint64_t i = 0; i < samples >> 2; i++)
    {
        uint64_t* ptr = (uint64_t*)cpuArr + i;
        for (int j = 64; j--;)
            printf("%u", (bool)((*ptr >> j) & 1));
        printf("\n");
    }

    return 0;
}
