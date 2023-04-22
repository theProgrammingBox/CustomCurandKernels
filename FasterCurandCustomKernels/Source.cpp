#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

__global__ void cudaGenerate(uint32_t* output, uint64_t seed, uint32_t samples)
{
	uint64_t idx = ((uint64_t)blockIdx.x << 10) + (threadIdx.x << 1);
	seed ^= (uint64_t)__brev((uint32_t)seed) << 32;
	uint64_t const bitflip = 0x0E4125884092CA03ULL - seed;
	uint64_t const input64 = (idx >> 32) + (idx << 32);
	uint64_t keyed = input64 ^ bitflip;
	keyed ^= ((keyed << 49) | (keyed >> 15)) ^ ((keyed << 24) | (keyed >> 40));
	keyed *= 0x9FB21C651E98DF25ULL;
	keyed ^= (keyed >> 35) + 8;
	keyed *= 0x9FB21C651E98DF25ULL;
	keyed ^= keyed >> 28;
	output[idx] = keyed;
	output[idx + 1] = keyed >> 32;
}

int main()
{
	const uint32_t iterations = 100000;
	const uint32_t samples = 1000000;


	float time;
	uint64_t seed = 0;
	cudaEvent_t start, stop;

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, seed);


	uint32_t* output = new uint32_t[samples];
	uint32_t* d_output;
	cudaMalloc(&d_output, samples * sizeof(uint32_t));


	for (uint32_t i = 0; i < iterations; i++)
		cudaGenerate << <(samples >> 11) + bool(samples & 0x7ff), 0x400 >> > (d_output, seed++, samples);


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (uint32_t i = 0; i < iterations; i++)
		curandGenerate(gen, d_output, samples);
	// cudaGenerate <<<(samples >> 11) + bool(samples & 0x7ff), 0x400>>> (d_output, seed++, samples);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time: %f ms\n", time / iterations);


	cudaMemcpy(output, d_output, samples * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	for (int i = 1024; i < 1030; i++)
	{
		// printf("%f\n", __half2float(output[i]));
		for (int j = 0; j < 16; j++)
		{
			printf("%u", *(uint16_t*)(output + i) >> (15 - j) & 1);
		}
		printf("\n");
	}


	//historgam
	/*const uint32_t bins = 100;
	const float scale = float(bins) / samples;
	float hist[bins];
	memset(hist, 0, bins * sizeof(float));
	for (int i = 0; i < samples; i++)
	{
		int bin = floor(__half2float(output[i]) * bins);
		if (bin >= 0 && bin < bins)
			hist[bin]++;
	}
	for (int i = 0; i < bins; i++)
	{
		printf("%u: %f\n", i, hist[i] * scale);
	}
	printf("\n");*/

	cudaFree(d_output);
	delete[] output;
	return 0;
}