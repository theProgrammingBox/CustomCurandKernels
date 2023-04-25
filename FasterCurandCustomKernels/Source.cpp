#include <iostream>
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

__global__ void cudaFill(void* output, uint64_t seed, uint64_t offset)
{
    const __half scale = __float2half(0.000030517578125f);
    const uint64_t idx = ((uint64_t)blockIdx.x << 10) + threadIdx.x;
    seed ^= (uint64_t)__brev((uint32_t)seed) << 32;
    offset += idx;
    offset = ((offset >> 32) + (offset << 32)) ^ (0x0E4125884092CA03ULL - seed);
    offset ^= ((offset << 49) | (offset >> 15)) ^ ((offset << 24) | (offset >> 40));
    offset *= 0x9FB21C651E98DF25ULL;
    offset ^= (offset >> 35) + 8;
    offset *= 0x9FB21C651E98DF25ULL;
    offset ^= offset >> 28;
    __half arr[4];
    arr[0] = __hmul(__float2half(offset & 0x7fff), scale);
    arr[1] = __hmul(__float2half((offset >> 16) & 0x7fff), scale);
    arr[2] = __hmul(__float2half((offset >> 32) & 0x7fff), scale);
    arr[3] = __hmul(__float2half((offset >> 48) & 0x7fff), scale);
    *((uint64_t*)arr) ^= offset & 0x8000800080008000;
    *((uint64_t*)output + idx) = *((uint64_t*)arr);
    // __half* location = (__half*)output + (idx << 2);
    // location[0] = __hmul(__float2half(offset & 0x7fff), scale);
    // location[1] = __hmul(__float2half((offset >> 16) & 0x7fff), scale);
    // location[2] = __hmul(__float2half((offset >> 32) & 0x7fff), scale);
    // location[3] = __hmul(__float2half((offset >> 48) & 0x7fff), scale);
    // *((uint64_t*)output + idx) ^= offset & 0x8000800080008000;
}

void Fill(void* output, uint64_t f16s, uint64_t& seed, uint64_t& offset)
{
    // assert multiple of 4096
    assert((f16s & 0xfff) == 0);
    cudaFill << < f16s >> 12, 0x400 >> > (output, seed, offset);
    seed++;
    offset += 3;
}

void TestStats(void* gpuArr, void* cpuArr, uint64_t f16s)
{
    cudaMemcpy(cpuArr, gpuArr, f16s << 2, cudaMemcpyDeviceToHost);

    double sum = 0;
    for (uint64_t i = 0; i < f16s; i++)
        sum += __half2float(((__half*)cpuArr)[i]);
    double mean = sum / f16s;
    printf("mean: %f\n", mean);

    double variance = 0;
    for (uint64_t i = 0; i < f16s; i++)
    {
        double diff = __half2float(((__half*)cpuArr)[i]) - mean;
        variance += diff * diff;
    }
    variance /= f16s;
    printf("variance: %f\n", variance);

    double stdDev = sqrt(variance);
    printf("standard deviation: %f\n\n", stdDev);
}

int main()
{
    const uint64_t f16s = (uint64_t)1 << 24;
    const uint16_t iterations = 0xfff;

    void* cpuArr = malloc(f16s << 2);
    void* gpuArr;
    cudaMalloc(&gpuArr, f16s << 2);

    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    uint64_t offset = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    float milliseconds = 0;
    float averageMilliseconds = 0;
    cudaEvent_t start, stop;

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    //warmup
    for (uint16_t i = iterations; i--;)
    {
        Fill(gpuArr, f16s, seed, offset);
    }

    printf("Testing cuRAND...\n");
    averageMilliseconds = 0;
    for (uint16_t i = iterations; i--;)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        curandGenerate(gen, (uint32_t*)gpuArr, f16s >> 1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        averageMilliseconds += milliseconds;
    }
    printf("average time: %fms\n", averageMilliseconds / iterations);
    // TestStats(gpuArr, cpuArr, f16s);
    printf("\n");
    double curandTime = averageMilliseconds / iterations;

    printf("Testing custom kernel...\n");
    averageMilliseconds = 0;
    for (uint16_t i = iterations; i--;)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        Fill(gpuArr, f16s, seed, offset);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        averageMilliseconds += milliseconds;
    }
    printf("average time: %fms\n", averageMilliseconds / iterations);
    TestStats(gpuArr, cpuArr, f16s);
    double customTime = averageMilliseconds / iterations;

    printf("A %f%% difference in performance.\n\n", (curandTime - customTime) / curandTime * 100);

    for (uint64_t i = 0; i < 0xf; i++)
    {
        // uint16_t* ptr = (uint16_t*)cpuArr + i;
        // for (uint8_t j = 16; j--;)
        //     printf("%u", (bool)((*ptr >> j) & 1));
        // printf("\n");
        printf("%f\n", __half2float(((__half*)cpuArr)[i]));
    }
    printf("\n");

    free(cpuArr);
    cudaFree(gpuArr);
    curandDestroyGenerator(gen);

    return 0;
}
