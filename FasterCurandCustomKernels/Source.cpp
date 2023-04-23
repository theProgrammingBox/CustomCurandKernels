#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

__global__ void cudaFill(void* output, uint64_t seed, uint64_t offset)
{
    const uint64_t idx = ((uint64_t)blockIdx.x << 10) + threadIdx.x;
    seed ^= (uint64_t)__brev((uint32_t)seed) << 32;
    offset += idx;
    offset = ((offset >> 32) + (offset << 32)) ^ (0x0E4125884092CA03ULL - seed);
    offset ^= ((offset << 49) | (offset >> 15)) ^ ((offset << 24) | (offset >> 40));
    offset *= 0x9FB21C651E98DF25ULL;
    offset ^= (offset >> 35) + 8;
    offset *= 0x9FB21C651E98DF25ULL;
    offset ^= offset >> 28;
    *((uint64_t*)output + idx) = offset;
}

void Fill(void* output, uint64_t f16s, uint64_t& seed, uint64_t& offset)
{
    // cuda assert multiple of 1024
    assert((f16s & 0x3ff) == 0);
    cudaFill<<<f16s >> 12, 0x400>>>(output, seed, offset);
    seed++;
    offset += f16s;
}

void TestStats(void* gpuArr, void* cpuArr, uint64_t f16s)
{
    cudaMemcpy(cpuArr, gpuArr, f16s << 2, cudaMemcpyDeviceToHost);

    uint64_t sum = 0;
    for (uint64_t i = 0; i < f16s; i++)
        sum += ((uint16_t*)cpuArr)[i];
    double mean = (double)sum / f16s;
    printf("mean: %f\n", mean);

    double variance = 0;
    for (uint64_t i = 0; i < f16s; i++)
    {
        double diff = ((uint16_t*)cpuArr)[i] - mean;
        variance += diff * diff;
    }
    variance /= f16s;
    printf("variance: %f\n", variance);

    double stdDev = sqrt(variance);
    printf("standard deviation: %f\n\n", stdDev);
}

int main()
{
    const uint64_t f16s = (uint64_t)1 << 28;

    void* cpuArr = malloc(f16s << 2);
    void* gpuArr;
    cudaMalloc(&gpuArr, f16s << 2);

    uint64_t seed = 0;
    uint64_t offset = 0;
    float milliseconds = 0;
    cudaEvent_t start, stop;

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    //warmup
    Fill(gpuArr, f16s, seed, offset);
    curandGenerate(gen, (uint32_t*)gpuArr, f16s >> 1);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    Fill(gpuArr, f16s, seed, offset);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %fms\n", milliseconds);
    TestStats(gpuArr, cpuArr, f16s);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    curandGenerate(gen, (uint32_t*)gpuArr, f16s >> 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %fms\n", milliseconds);
    TestStats(gpuArr, cpuArr, f16s);


    /*for (uint64_t i = 0; i < f16s; i++)
    {
        uint16_t* ptr = (uint16_t*)cpuArr + i;
        for (uint8_t j = 16; j--;)
            printf("%u", (bool)((*ptr >> j) & 1));
        printf("\n");
    }*/

    free(cpuArr);
    cudaFree(gpuArr);
    curandDestroyGenerator(gen);

    return 0;
}
