#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

__global__ void cudaGenerate(__half* output, uint32_t seed, uint32_t samples)
{
	const __half ntwo = __float2half(-2.0f);
	const __half twoPi = __float2half(6.28318530717958647692f);
	uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) << 1;

	samples = idx * 0xcc9e2d51;
	samples = (samples << 15) | (samples >> 17);
	samples *= 0x1b873593;
	seed ^= samples;
	seed = (seed << 13) | (seed >> 19);
	seed = seed * 5 + 0xe6546b64;
	seed ^= 4;
	seed ^= seed >> 16;
	seed *= 0x85ebca6b;
	seed ^= seed >> 13;
	seed *= 0xc2b2ae35;
	seed ^= seed >> 16;

	__half u1f16 = __float2half(uint16_t(seed) * 0.0000152587890625f);
	__half u2f16 = __float2half(uint16_t(seed >> 16) * 0.0000152587890625f);

	__half r = hsqrt(__hmul(ntwo, hlog(u1f16)));
	__half cosu2 = hcos(__hmul(twoPi, u2f16));
	__half sinu2 = hsin(__hmul(twoPi, u2f16));
	output[idx] = __hmul(r, cosu2);
	output[idx + 1] = __hmul(r, sinu2);
}

int main()
{
	const uint32_t iterations = 20000;
	const uint32_t samples = 1000000;
	__half* output = new __half[samples];
	__half* d_output;
	cudaMalloc(&d_output, samples * sizeof(__half));
	float time;
	uint32_t seed = 0;
	cudaEvent_t start, stop;

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, seed);


	for (uint32_t i = 0; i < iterations; i++)
		cudaGenerate << <std::ceil(samples * 0.5f * 0.0009765625f), 1024 >> > (d_output, seed++, samples);


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (uint32_t i = 0; i < iterations; i++)
		cudaGenerate << <std::ceil(samples * 0.5f * 0.0009765625f), 1024 >> > (d_output, seed++, samples);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time: %f ms\n", time / iterations);


	cudaMemcpy(output, d_output, samples * sizeof(__half), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i++)
		printf("%f\n", __half2float(output[i]));

	// historgam
	// const uint32_t bins = 100;
	// const float scale = float(bins) / samples;
	// float hist[bins];
	// memset(hist, 0, bins * sizeof(float));
	// for (int i = 0; i < samples; i++)
	// {
	// 	int bin = (output[i] * 0.3 + 1) * bins * 0.5f;
	// 	if (bin >= 0 && bin < bins)
	// 		hist[bin]++;
	// }
	// for (int i = 0; i < bins; i++)
	// {
	// 	for (int j = 0; j < hist[i] * scale * 40; j++)
	// 		printf("*");
	// 	printf("\n");
	// }
	// printf("\n");

	// cleanup
	cudaFree(d_output);
	delete[] output;
	return 0;
}