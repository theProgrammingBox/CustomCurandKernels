#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

__global__ void cudaGenerate(float* output, uint32_t seed, uint32_t samples)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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
	float u1 = seed * 2.3283064365386963e-10f;
	seed ^= 4;
	seed ^= seed >> 16;
	seed *= 0x85ebca6b;
	seed ^= seed >> 13;
	seed *= 0xc2b2ae35;
	seed ^= seed >> 16;
	float u2 = seed * 2.3283064365386963e-10f;

	output[idx] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

int main()
{
	const uint32_t iterations = 10;
	const uint32_t samples = 1000000;
	float* output = new float[samples];
	float* d_output;
	cudaMalloc(&d_output, samples * sizeof(float));
	float time;
	uint32_t seed = 0;
	cudaEvent_t start, stop;

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, seed);


	for (uint32_t i = 0; i < iterations; i++)
		curandGenerateUniform(gen, d_output, samples);


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (uint32_t i = 0; i < iterations; i++)
		curandGenerateUniform(gen, d_output, samples);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time: %f ms\n", time / iterations);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (uint32_t i = 0; i < iterations; i++)
		cudaGenerate <<<std::ceil(0.0009765625f * samples), 1024 >>> (d_output, seed++, samples);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time: %f ms\n", time / iterations);


	cudaMemcpy(output, d_output, samples * sizeof(float), cudaMemcpyDeviceToHost);

	// historgam
	const uint32_t bins = 50;
	const float scale = float(bins) / samples;
	float hist[bins];
	memset(hist, 0, bins * sizeof(float));
	for (int i = 0; i < samples; i++)
	{
		int bin = (output[i] * 0.4 + 1) * bins * 0.5f;
		if (bin >= 0 && bin < bins)
			hist[bin]++;
	}
	for (int i = 0; i < bins; i++)
	{
		for (int j = 0; j < hist[i] * scale * 20; j++)
			printf("*");
		printf("\n");
	}
	printf("\n");

	// cleanup
	cudaFree(d_output);
	delete[] output;
	return 0;
}