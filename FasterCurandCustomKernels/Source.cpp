#include <iostream>
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

__global__ void CudaTest(const void* output, uint32_t seed1, uint32_t seed2)
{
    const float scale = 0.0000152587890625f;
    uint32_t idx = threadIdx.x + (blockIdx.x << 10);

    seed1 = (0xE558D374 ^ idx + seed1 ^ seed2) * 0xAA69E974;
    seed1 = (seed1 >> 13 ^ seed1) * 0x8B7A1B65;

    ((__half*)&seed1)[0] = __float2half((((uint16_t*)&seed1)[0] | 1) * scale);
    ((__half*)&seed1)[1] = __float2half((((uint16_t*)&seed1)[1] | 1) * scale);

    *((uint32_t*)output + idx) = *(uint32_t*)&seed1;
}

void Test(const void* output, uint64_t f16s, uint32_t seed1 = 0, uint32_t seed2 = 0)
{
    assert((f16s & 0x7ff) == 0);
    CudaTest<<<f16s >> 11, 1024>>>(output, seed1, seed2);
}

int main()
{
    const uint64_t f16s = 1 << 24;
    const uint32_t iterations = 1 << 16;
    const uint32_t iterationStart = iterations - (1 << 12);
    const uint32_t runs = iterations - iterationStart;

    void* cpuArr = malloc(f16s << 2);
    void* gpuHalfArr;
    cudaMalloc(&gpuHalfArr, f16s << 2);

    float milliseconds[iterations];
    cudaEvent_t start, stop;

    for (uint32_t i = 0; i < iterations; ++i)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        Test(gpuHalfArr, f16s);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(milliseconds + i, start, stop);
    }

    double mean = 0;
    for (uint32_t i = iterationStart; i < iterations; ++i)
		mean += milliseconds[i];
    mean /= runs;
    printf("mean: %fms\n", mean);

    double variance = 0;
    for (uint32_t i = iterationStart; i < iterations; ++i)
    {
        double diff = milliseconds[i] - mean;
		variance += diff * diff;
    }
    variance /= runs;
    printf("variance: %fms\n", variance);

    double stdDev = sqrt(variance);
    printf("stdDev: %fms\n\n", stdDev);

    cudaMemcpy(cpuArr, gpuHalfArr, f16s << 2, cudaMemcpyDeviceToHost);
    mean = 0;
    for (uint32_t i = 0; i < f16s; ++i)
        mean += __half2float(*((__half*)cpuArr + i));
    mean /= f16s;
    printf("mean: %f\n", mean);

    variance = 0;
    for (uint32_t i = 0; i < f16s; ++i)
    {
		double diff = __half2float(*((__half*)cpuArr + i)) - mean;
		variance += diff * diff;
	}
    variance /= f16s;
    printf("variance: %f\n", variance);

    stdDev = sqrt(variance);
    printf("stdDev: %f\n\n", stdDev);

    return 0;
}