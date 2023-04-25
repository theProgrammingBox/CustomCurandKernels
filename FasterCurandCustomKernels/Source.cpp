#include <iostream>
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>



__global__ void cudaFill(const void* output, uint64_t seed, uint64_t offset)
{
    const float scale = 2.3283064365386962890625e-10f;
    const float nTwo = -2.0f;
    const float twoPi = 6.283185307179586476925286766559f;

    const uint64_t idx = ((uint64_t)blockIdx.x << 10) + threadIdx.x;
    seed ^= (uint64_t)__brev((uint32_t)seed) << 32;
    offset += idx;
    offset = ((offset >> 32) + (offset << 32)) ^ (0x0E4125884092CA03ULL - seed);
    offset ^= ((offset << 49) | (offset >> 15)) ^ ((offset << 24) | (offset >> 40));
    offset *= 0x9FB21C651E98DF25ULL;
    offset ^= (offset >> 35) + 8;
    offset *= 0x9FB21C651E98DF25ULL;
    offset ^= offset >> 28;

    float u1 = (float)(offset & 0xffffffff | 0x1) * scale;
    float u2 = (float)(offset >> 32) * scale;

    float r1 = sqrt(nTwo * log(u1));
    float r2 = sqrt(nTwo * log(u2));
    float theta1 = twoPi * u1;
    float theta2 = twoPi * u2;
    float sin1, cos1, sin2, cos2;
    sincos(theta1, &sin1, &cos1);
    sincos(theta2, &sin2, &cos2);

    __half arr[4];
    arr[0] = __float2half(r1 * cos2);
    arr[1] = __float2half(r1 * sin2);
    arr[2] = __float2half(r2 * cos1);
    arr[3] = __float2half(r2 * sin1);
    *((uint64_t*)output + idx) = *(uint64_t*)arr;
 }

void Fill(const void* output, uint64_t f16s, uint64_t& seed, uint64_t& offset)
{
    // assert multiple of 4096
    assert((f16s & 0xfff) == 0);
    cudaFill <<<f16s >> 12, 0x400>>> (output, seed, offset);
    seed++;
    offset += 3;
}



__global__ void cudaFloat2Half(const void* floatInput, const void* halfOutput)
{
    const uint64_t idx = ((uint64_t)blockIdx.x << 10) + threadIdx.x;
    ((__half*)halfOutput)[idx] = __float2half(((float*)floatInput)[idx]);
}

void cuRANDFill(curandGenerator_t gen, const void* floatOutput, const void* halfOutput, uint64_t f16s)
{
    // assert multiple of 1024
    assert((f16s & 0x3ff) == 0);
    curandGenerateUniform(gen, (float*)floatOutput, f16s);
    cudaFloat2Half <<<f16s >> 10, 0x400>>> (floatOutput, halfOutput);
}



void TestStats(void* gpuHalfArr, void* cpuArr, uint64_t f16s)
{
    cudaMemcpy(cpuArr, gpuHalfArr, f16s << 2, cudaMemcpyDeviceToHost);

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

    for (uint64_t i = 0x400; i < 0x40f; i++)
        printf("%f\n", __half2float(((__half*)cpuArr)[i]));
    printf("\n\n");
}



int main()
{
    const uint64_t f16s = (uint64_t)1 << 24;
    const uint16_t iterations = 0xfff;

    void* cpuArr = malloc(f16s << 2);
    void* gpuHalfArr;
    void* gpuFloatArr;
    cudaMalloc(&gpuHalfArr, f16s << 2);
    cudaMalloc(&gpuFloatArr, f16s << 4);

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
        cuRANDFill(gen, gpuFloatArr, gpuHalfArr, f16s);
        Fill(gpuHalfArr, f16s, seed, offset);
    }

    printf("Testing cuRAND...\n");
    averageMilliseconds = 0;
    for (uint16_t i = iterations; i--;)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cuRANDFill(gen, gpuFloatArr, gpuHalfArr, f16s);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        averageMilliseconds += milliseconds;
    }
    printf("average time: %fms\n", averageMilliseconds / iterations);
    TestStats(gpuHalfArr, cpuArr, f16s);
    double curandTime = averageMilliseconds / iterations;

    printf("Testing custom kernel...\n");
    averageMilliseconds = 0;
    for (uint16_t i = iterations; i--;)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        Fill(gpuHalfArr, f16s, seed, offset);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        averageMilliseconds += milliseconds;
    }
    printf("average time: %fms\n", averageMilliseconds / iterations);
    TestStats(gpuHalfArr, cpuArr, f16s);
    double customTime = averageMilliseconds / iterations;

    printf("A %f%% difference in performance.\n\n", curandTime / customTime * 100);

    free(cpuArr);
    cudaFree(gpuHalfArr);
    curandDestroyGenerator(gen);

    return 0;
}
