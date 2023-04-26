#include <iostream>
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

__global__ void CudaTest(const void* output, uint32_t seed1, uint32_t seed2)
{
    const int32_t kn[128] = {
        0x76ad2212, 0x00000000, 0x600f1b53, 0x6ce447a6, 0x725b46a2, 0x7560051d, 0x774921eb, 0x789a25bd,
        0x799045c3, 0x7a4bce5d, 0x7adf629f, 0x7b5682a6, 0x7bb8a8c6, 0x7c0ae722, 0x7c50cce7, 0x7c8cec5b,
        0x7cc12cd6, 0x7ceefed2, 0x7d177e0b, 0x7d3b8883, 0x7d5bce6c, 0x7d78dd64, 0x7d932886, 0x7dab0e57,
        0x7dc0dd30, 0x7dd4d688, 0x7de73185, 0x7df81cea, 0x7e07c0a3, 0x7e163efa, 0x7e23b587, 0x7e303dfd,
        0x7e3beec2, 0x7e46db77, 0x7e51155d, 0x7e5aabb3, 0x7e63abf7, 0x7e6c222c, 0x7e741906, 0x7e7b9a18,
        0x7e82adfa, 0x7e895c63, 0x7e8fac4b, 0x7e95a3fb, 0x7e9b4924, 0x7ea0a0ef, 0x7ea5b00d, 0x7eaa7ac3,
        0x7eaf04f3, 0x7eb3522a, 0x7eb765a5, 0x7ebb4259, 0x7ebeeafd, 0x7ec2620a, 0x7ec5a9c4, 0x7ec8c441,
        0x7ecbb365, 0x7ece78ed, 0x7ed11671, 0x7ed38d62, 0x7ed5df12, 0x7ed80cb4, 0x7eda175c, 0x7edc0005,
        0x7eddc78e, 0x7edf6ebf, 0x7ee0f647, 0x7ee25ebe, 0x7ee3a8a9, 0x7ee4d473, 0x7ee5e276, 0x7ee6d2f5,
        0x7ee7a620, 0x7ee85c10, 0x7ee8f4cd, 0x7ee97047, 0x7ee9ce59, 0x7eea0eca, 0x7eea3147, 0x7eea3568,
        0x7eea1aab, 0x7ee9e071, 0x7ee98602, 0x7ee90a88, 0x7ee86d08, 0x7ee7ac6a, 0x7ee6c769, 0x7ee5bc9c,
        0x7ee48a67, 0x7ee32efc, 0x7ee1a857, 0x7edff42f, 0x7ede0ffa, 0x7edbf8d9, 0x7ed9ab94, 0x7ed7248d,
        0x7ed45fae, 0x7ed1585c, 0x7ece095f, 0x7eca6ccb, 0x7ec67be2, 0x7ec22eee, 0x7ebd7d1a, 0x7eb85c35,
        0x7eb2c075, 0x7eac9c20, 0x7ea5df27, 0x7e9e769f, 0x7e964c16, 0x7e8d44ba, 0x7e834033, 0x7e781728,
        0x7e6b9933, 0x7e5d8a1a, 0x7e4d9ded, 0x7e3b737a, 0x7e268c2f, 0x7e0e3ff5, 0x7df1aa5d, 0x7dcf8c72,
        0x7da61a1e, 0x7d72a0fb, 0x7d30e097, 0x7cd9b4ab, 0x7c600f1a, 0x7ba90bdc, 0x7a722176, 0x77d664e5,
    };

    const uint32_t fn2[128] = {
            0x3f800000, 0x3f76ae78, 0x3f6fb039, 0x3f69bd3a, 0x3f646c92, 0x3f5f8cdb, 0x3f5b0216, 0x3f56ba86,
            0x3f52aa0c, 0x3f4ec7f0, 0x3f4b0da6, 0x3f47761e, 0x3f43fd53, 0x3f40a003, 0x3f3d5b7d, 0x3f3a2d82,
            0x3f37142b, 0x3f340dd6, 0x3f311918, 0x3f2e34b7, 0x3f2b5f9d, 0x3f2898d4, 0x3f25df84, 0x3f2332e7,
            0x3f20924f, 0x3f1dfd1c, 0x3f1b72bd, 0x3f18f2b1, 0x3f167c7d, 0x3f140fb5, 0x3f11abf3, 0x3f0f50d9,
            0x3f0cfe12, 0x3f0ab34c, 0x3f08703e, 0x3f0634a1, 0x3f040036, 0x3f01d2c0, 0x3eff580d, 0x3efb17a7,
            0x3ef6e3ec, 0x3ef2bc7d, 0x3eeea102, 0x3eea9128, 0x3ee68ca0, 0x3ee29320, 0x3edea460, 0x3edac01e,
            0x3ed6e61c, 0x3ed3161b, 0x3ecf4fe5, 0x3ecb9342, 0x3ec7dffe, 0x3ec435e9, 0x3ec094d4, 0x3ebcfc92,
            0x3eb96cf8, 0x3eb5e5de, 0x3eb2671e, 0x3eaef092, 0x3eab8216, 0x3ea81b8a, 0x3ea4bccc, 0x3ea165bf,
            0x3e9e1644, 0x3e9ace41, 0x3e978d98, 0x3e945432, 0x3e9121f5, 0x3e8df6cb, 0x3e8ad29c, 0x3e87b554,
            0x3e849ede, 0x3e818f27, 0x3e7d0c3a, 0x3e77075b, 0x3e710f91, 0x3e6b24bc, 0x3e6546c0, 0x3e5f757f,
            0x3e59b0df, 0x3e53f8c8, 0x3e4e4d22, 0x3e48add7, 0x3e431ad5, 0x3e3d9407, 0x3e38195e, 0x3e32aaca,
            0x3e2d483e, 0x3e27f1ad, 0x3e22a70d, 0x3e1d6855, 0x3e18357f, 0x3e130e85, 0x3e0df364, 0x3e08e41b,
            0x3e03e0aa, 0x3dfdd226, 0x3df3fab7, 0x3dea3b16, 0x3de09356, 0x3dd7038f, 0x3dcd8bdf, 0x3dc42c6b,
            0x3dbae55f, 0x3db1b6eb, 0x3da8a14c, 0x3d9fa4c4, 0x3d96c1a1, 0x3d8df83b, 0x3d8548f8, 0x3d79689a,
            0x3d68757b, 0x3d57b9c6, 0x3d4736da, 0x3d36ee4f, 0x3d26e1fe, 0x3d171416, 0x3d07872a, 0x3cf07ca9,
            0x3cd27abd, 0x3cb511fd, 0x3c984e77, 0x3c788042, 0x3c41fa5f, 0x3c0d4db6, 0x3bb5d458, 0x3b2ef4f2,
    };

    const uint32_t wn2[128] = {
            0x30eda334, 0x2f0b6da4, 0x2f39ca49, 0x2f5a647f, 0x2f7472bb, 0x2f8549b7, 0x2f8f0673, 0x2f97cc6b,
            0x2f9fd5f7, 0x2fa74a60, 0x2fae457f, 0x2fb4dbd9, 0x2fbb1d0b, 0x2fc11543, 0x2fc6ce35, 0x2fcc4fc3,
            0x2fd1a069, 0x2fd6c593, 0x2fdbc3d4, 0x2fe09f16, 0x2fe55ab5, 0x2fe9f9a0, 0x2fee7e66, 0x2ff2eb48,
            0x2ff74247, 0x2ffb852c, 0x2fffb591, 0x3001ea74, 0x3003f23f, 0x3005f2c2, 0x3007ec89, 0x3009e012,
            0x300bcdd4, 0x300db63a, 0x300f99aa, 0x30117880, 0x30135314, 0x301529b8, 0x3016fcb8, 0x3018cc5d,
            0x301a98ec, 0x301c62a4, 0x301e29c3, 0x301fee85, 0x3021b120, 0x302371cc, 0x302530bb, 0x3026ee1f,
            0x3028aa28, 0x302a6506, 0x302c1ee4, 0x302dd7f1, 0x302f9055, 0x3031483d, 0x3032ffd1, 0x3034b73b,
            0x30366ea2, 0x3038262f, 0x3039de0a, 0x303b965b, 0x303d4f48, 0x303f08fa, 0x3040c398, 0x30427f4a,
            0x30443c39, 0x3045fa8e, 0x3047ba71, 0x30497c0d, 0x304b3f8d, 0x304d051c, 0x304ecce7, 0x3050971c,
            0x305263ea, 0x30543382, 0x30560616, 0x3057dbda, 0x3059b504, 0x305b91cb, 0x305d726a, 0x305f571e,
            0x30614027, 0x30632dc6, 0x30652042, 0x306717e6, 0x306914fd, 0x306b17db, 0x306d20d5, 0x306f304a,
            0x3071469a, 0x3073642d, 0x30758975, 0x3077b6e7, 0x3079ed05, 0x307c2c58, 0x307e7576, 0x3080647f,
            0x308193d1, 0x3082c90f, 0x308404a0, 0x308546f7, 0x3086908f, 0x3087e1f2, 0x30893bb9, 0x308a9e8b,
            0x308c0b26, 0x308d825c, 0x308f051a, 0x3090946f, 0x3092318c, 0x3093ddd1, 0x30959ad5, 0x30976a70,
            0x30994ecc, 0x309b4a7a, 0x309d608e, 0x309f94c2, 0x30a1ebae, 0x30a46b0c, 0x30a71a2b, 0x30aa0290,
            0x30ad30f7, 0x30b0b700, 0x30b4ae16, 0x30b93cef, 0x30bea2f6, 0x30c5539f, 0x30ce4706, 0x30dc53e2,
    };

    const float* fn = (float*)fn2;
    const float* wn = (float*)wn2;

    const float scale = 0.0000152587890625f;
    const __half twoPi = __float2half(6.283185307179586476925286766559f);
    const __half negTwo = __float2half(-2.0f);
    const uint32_t idx = threadIdx.x + (blockIdx.x << 10);

    seed1 = (0xE558D374 ^ idx + seed1 ^ seed2) * 0xAA69E974;
    seed1 = (seed1 >> 13 ^ seed1) * 0x8B7A1B65;

    *(__half*)output = __float2half(wn[seed1 & 0x3f]);
    /*((__half*)&seed1)[0] = __float2half(((uint16_t*)&seed1)[0] * scale);
    ((__half*)&seed1)[1] = __float2half(((uint16_t*)&seed1)[1] * scale);

    *((uint32_t*)output + idx) = *(uint32_t*)&seed1;*/
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
    {
        float h = __half2float(*((__half*)cpuArr + i));
        mean += h;
        /*if (h == 0)
            printf("0 at %d\n", i);*/
    }
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