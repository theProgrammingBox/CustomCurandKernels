#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

__global__ void devCudaGenerate(__half* output, uint64_t seed)
{
	const uint64_t idx = ((uint64_t)blockIdx.x << 12) + (threadIdx.x << 2);
	const __half scale = __float2half(0.0000152587890625f);
	seed ^= (uint64_t)__brev((uint32_t)seed) << 32;
	uint64_t keyed = ((idx >> 32) + (idx << 32)) ^ (0x0E4125884092CA03ULL - seed);
	keyed ^= ((keyed << 49) | (keyed >> 15)) ^ ((keyed << 24) | (keyed >> 40));
	keyed *= 0x9FB21C651E98DF25ULL;
	keyed ^= (keyed >> 35) + 8;
	keyed *= 0x9FB21C651E98DF25ULL;
	keyed ^= keyed >> 28;
	__half uniform1 = __float2half(uint16_t(keyed)) * scale;
	__half uniform2 = __float2half(uint16_t(keyed >> 16)) * scale;
	__half uniform3 = __float2half(uint16_t(keyed >> 32)) * scale;
	__half uniform4 = __float2half(uint16_t(keyed >> 48)) * scale;
	output[idx] = uniform1;
	output[idx + 1] = uniform2;
	output[idx + 2] = uniform3;
	output[idx + 3] = uniform4;
}

void cudaGenerate(__half* output, uint64_t& seed, uint64_t samples)
{
	devCudaGenerate << <(samples >> 12) + bool(samples & 0x1fff), 0x400 >> > (output, seed++);
}

int main()
{
	const uint64_t iterations = 10000;
	const uint64_t samples = 10000000;


	float time;
	uint64_t seed = 0;
	cudaEvent_t start, stop;

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, seed);


	__half* output = new __half[samples];
	__half* d_output;
	cudaMalloc(&d_output, samples * sizeof(__half));


	for (uint32_t i = 0; i < iterations; i++)
		cudaGenerate(d_output, seed, samples);


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (uint32_t i = 0; i < iterations; i++)
		// curandGenerate(gen, d_output, samples);
		cudaGenerate(d_output, seed, samples);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time: %f ms\n", time / iterations);


	cudaMemcpy(output, d_output, samples * sizeof(__half), cudaMemcpyDeviceToHost);
	for (int i = 1024; i < 1050; i++)
	{
		printf("%f\n", __half2float(output[i]));
		// for (int j = 8; j--;)
		//   printf("%u", *(uint8_t*)(output + i) >> j & 1);
		// printf("\n");
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