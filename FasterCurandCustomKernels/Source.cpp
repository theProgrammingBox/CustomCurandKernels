#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

////////////////////////////////////////////////////////////////////////////////////

alignas(64) static const uint8_t XXH3_kSecret[192] = {
		0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c,
		0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
		0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21,
		0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c,
		0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3,
		0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8,
		0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d,
		0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64,
		0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
		0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
		0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce,
		0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
};

static uint32_t XXH_swap32(uint32_t x)
{
	return  ((x << 24) & 0xff000000) |
		((x << 8) & 0x00ff0000) |
		((x >> 8) & 0x0000ff00) |
		((x >> 24) & 0x000000ff);
}

#  define XXH_rotl64(x,r) (((x) << (r)) | ((x) >> (64 - (r))))

static uint64_t XXH_read64(const void* memPtr)
{
	uint64_t val;
	XXH_memcpy(&val, memPtr, sizeof(val));
	return val;
}

uint64_t XXH_readLE64(const void* ptr)
{
	return XXH_CPU_LITTLE_ENDIAN ? XXH_read64(ptr) : XXH_swap64(XXH_read64(ptr));
}

static void* XXH_memcpy(void* dest, const void* src, size_t size)
{
	return memcpy(dest, src, size);
}

static uint32_t read32(const void* memPtr)
{
	uint32_t val;
	XXH_memcpy(&val, memPtr, sizeof(val));
	return val;
}

uint32_t XXH_readLE32(const void* ptr)
{
	return read32(ptr)
}

uint64_t XXH_xorshift64(uint64_t v64, int shift)
{
	return v64 ^ (v64 >> shift);
}

static uint64_t XXH3_rrmxmx(uint64_t h64, uint64_t len)
{
	h64 ^= XXH_rotl64(h64, 49) ^ XXH_rotl64(h64, 24);
	h64 *= 0x9FB21C651E98DF25ULL;
	h64 ^= (h64 >> 35) + len;
	h64 *= 0x9FB21C651E98DF25ULL;
	return XXH_xorshift64(h64, 28);
}

uint64_t XXH3_len_4to8_64b(const uint8_t* input, size_t len, const uint8_t* secret, uint64_t seed)
{
	seed ^= (uint64_t)XXH_swap32((uint32_t)seed) << 32;
	uint32_t const input1 = XXH_readLE32(input);
	uint32_t const input2 = XXH_readLE32(input + len - 4);
	uint64_t const bitflip = (XXH_readLE64(secret + 8) ^ XXH_readLE64(secret + 16)) - seed;
	uint64_t const input64 = input2 + (((uint64_t)input1) << 32);
	uint64_t const keyed = input64 ^ bitflip;
	return XXH3_rrmxmx(keyed, len);
}

uint64_t XXH3_len_0to16_64b(const uint8_t* input, size_t len, const uint8_t* secret, uint64_t seed)
{
	return XXH3_len_4to8_64b(input, len, secret, seed);
}

uint64_t XXH3_64bits_internal(const void* __restrict input, size_t len, uint64_t seed64, const void* __restrict secret, size_t secretLen, XXH3_hashLong64_f f_hashLong)
{
	return XXH3_len_0to16_64b((const uint8_t*)input, len, (const uint8_t*)secret, seed64);
}

uint64_t XXH3_64bits(const void* input, size_t length)
{
	return XXH3_64bits_internal(input, length, 0, XXH3_kSecret, sizeof(XXH3_kSecret), XXH3_hashLong_64b_default);
}

uint64_t XXH3_64bits(const uint8_t* input, size_t len, const uint8_t* secret, uint64_t seed)
{
	seed ^= (uint64_t)XXH_swap32((uint32_t)seed) << 32;
}

////////////////////////////////////////////////////////////////////////////////////

__global__ void cudaGenerate(uint64_t* output, uint64_t seed, uint32_t samples)
{
	uint64_t idx = ((uint64_t)blockIdx.x << 10) + threadIdx.x;
	seed ^= (uint64_t)__brev((uint32_t)seed) << 32;
	uint64_t const bitflip = 0E4125884092CA03ULL - seed;
	uint64_t const input64 = (idx >> 32) + (idx << 32);
	uint64_t keyed = input64 ^ bitflip;
	//keyed ^= XXH_rotl64(keyed, 49) ^ XXH_rotl64(keyed, 24);
	keyed ^= ((keyed << 49) | (keyed >> 15)) ^ ((keyed << 24) | (keyed >> 40));
	output[idx] = bitflip;
}

int main()
{
	const uint32_t iterations = 100000;
	const uint32_t samples = 1000000;


	__half* output = new __half[samples];
	__half* d_output;
	cudaMalloc(&d_output, samples * sizeof(__half));


	float time;
	uint32_t seed = 0;
	cudaEvent_t start, stop;


	for (uint32_t i = 0; i < iterations; i++)
		cudaGenerate << <std::ceil(samples * 0.0009765625f), 1024 >> > (d_output, seed++, samples);


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (uint32_t i = 0; i < iterations; i++)
		cudaGenerate << <std::ceil(samples * 0.0009765625f), 1024 >> > (d_output, seed++, samples);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time: %f ms\n", time / iterations);


	cudaMemcpy(output, d_output, samples * sizeof(__half), cudaMemcpyDeviceToHost);
	for (int i = 1024; i < 1030; i++)
	{
		printf("%f\n", __half2float(output[i]));
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