// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define HIP_CHECK(expression)                      \
    {                                              \
        const hipError_t status = expression;      \
        if (status != hipSuccess)                  \
        {                                          \
            std::cerr << "HIP error "              \
                      << status << ": "            \
                      << hipGetErrorString(status) \
                      << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;    \
        }                                          \
    }

constexpr int block_size  = 256;
constexpr int input_size  = 1 << 24; // 16 M elements

// [Sphinx msad start]
// msad_u8: SAD of four packed bytes, skipping positions where the src1 byte is zero.
// Useful for block matching over padded reference windows where padding is coded as zero.
__global__ __launch_bounds__(block_size)
void msad_u8(const unsigned int *src0,
             const unsigned int *src1,
             unsigned int       *output,
             int                 n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n)
    {
        return;
    }
    output[gid] = __builtin_amdgcn_msad_u8(src0[gid], src1[gid], 0u);
}
// [Sphinx msad end]

// [Sphinx sad variants start]
// sad_u8 accumulates into bits [15:0]; sad_hi_u8 shifts each byte difference
// left 16 before accumulating, placing the result in bits [31:16]. One call
// to each packs two independent 4-byte SADs into one register.
__global__ __launch_bounds__(block_size)
void sad_variants(const unsigned int *src0_lo,
                  const unsigned int *src0_hi,
                  const unsigned int *src1_lo,
                  const unsigned int *src1_hi,
                  unsigned int       *output,
                  int                 n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n)
    {
        return;
    }
    unsigned int packed = __builtin_amdgcn_sad_u8(src0_lo[gid], src1_lo[gid], 0u);
    packed              = __builtin_amdgcn_sad_hi_u8(src0_hi[gid], src1_hi[gid], packed);
    output[gid]         = packed;
}
// [Sphinx sad variants end]

// [Sphinx qsad start]
// qsad_pk_u16_u8: four overlapping 8-byte SADs in one instruction.
// src0 is a 64-bit sliding reference window; src1 is the 4-byte query.
// The four results are packed as uint16 values in the returned uint64.
__global__ __launch_bounds__(block_size)
void qsad_pk_u16_u8(const unsigned long long *src0,
                    const unsigned int        *src1,
                    unsigned long long        *output,
                    int                        n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n)
    {
        return;
    }
    output[gid] = __builtin_amdgcn_qsad_pk_u16_u8(src0[gid], src1[gid], 0ull);
}
// [Sphinx qsad end]

// [Sphinx udot4 start]
// udot4: dot product of four unsigned byte pairs accumulated into uint32 (dot7-insts).
// Bytes are packed inline per thread before calling the intrinsic.
__global__ __launch_bounds__(block_size)
void udot4(const unsigned char *a, const unsigned char *b, unsigned int *output, int n)
{
    int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = gid * 4;
    if (base + 3 >= n)
    {
        return;
    }
    unsigned int a_word = (unsigned int)a[base + 0]
                        | ((unsigned int)a[base + 1] << 8)
                        | ((unsigned int)a[base + 2] << 16)
                        | ((unsigned int)a[base + 3] << 24);
    unsigned int b_word = (unsigned int)b[base + 0]
                        | ((unsigned int)b[base + 1] << 8)
                        | ((unsigned int)b[base + 2] << 16)
                        | ((unsigned int)b[base + 3] << 24);
    output[gid] = __builtin_amdgcn_udot4(a_word, b_word, 0u, false);
}
// [Sphinx udot4 end]

// [Sphinx fdot2 start]
// fdot2: dot product of two FP16 pairs accumulated into FP32 (dot10-insts).
// Each thread loads one pair of half2 vectors from consecutive elements.
__global__ __launch_bounds__(block_size)
void fdot2(const __fp16 *a, const __fp16 *b, float *output, int n)
{
    int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = gid * 2;
    if (base + 1 >= n)
    {
        return;
    }
    using half2 = __attribute__((vector_size(4))) __fp16;
    half2 va    = *reinterpret_cast<const half2 *>(&a[base]);
    half2 vb    = *reinterpret_cast<const half2 *>(&b[base]);
    output[gid] = __builtin_amdgcn_fdot2(va, vb, 0.0f, false);
}
// [Sphinx fdot2 end]

// [Sphinx cvt start]
// cvt_pk_u8_f32: clamp-and-convert one FP32 to uint8 and insert it at byte position i of dst.
// Four calls pack a full word suitable for use with sad_u8, udot4, or similar.
__global__ __launch_bounds__(block_size)
void cvt_pk_u8_f32(const float *input, unsigned int *output, int n)
{
    int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    int base = gid * 4;
    if (base + 3 >= n)
    {
        return;
    }
    unsigned int word = 0u;
    word = __builtin_amdgcn_cvt_pk_u8_f32(input[base + 0], 0, word);
    word = __builtin_amdgcn_cvt_pk_u8_f32(input[base + 1], 1, word);
    word = __builtin_amdgcn_cvt_pk_u8_f32(input[base + 2], 2, word);
    word = __builtin_amdgcn_cvt_pk_u8_f32(input[base + 3], 3, word);
    output[gid] = word;
}
// [Sphinx cvt end]

// CPU references

unsigned int cpu_msad_u8(unsigned int src0, unsigned int src1)
{
    unsigned int result = 0;
    for (int k = 0; k < 4; ++k)
    {
        unsigned int a = (src0 >> (k * 8)) & 0xff;
        unsigned int b = (src1 >> (k * 8)) & 0xff;
        if (b != 0)
        {
            result += (a > b) ? (a - b) : (b - a);
        }
    }
    return result;
}

unsigned int cpu_sad_u8(unsigned int src0, unsigned int src1)
{
    unsigned int result = 0;
    for (int k = 0; k < 4; ++k)
    {
        unsigned int a = (src0 >> (k * 8)) & 0xff;
        unsigned int b = (src1 >> (k * 8)) & 0xff;
        result        += (a > b) ? (a - b) : (b - a);
    }
    return result;
}

// sad_hi_u8 shifts each per-byte difference left 16 before accumulating.
unsigned int cpu_sad_hi_u8(unsigned int src0, unsigned int src1)
{
    unsigned int result = 0;
    for (int k = 0; k < 4; ++k)
    {
        unsigned int a = (src0 >> (k * 8)) & 0xff;
        unsigned int b = (src1 >> (k * 8)) & 0xff;
        result        += ((a > b) ? (a - b) : (b - a)) << 16;
    }
    return result;
}

unsigned int cpu_udot4(const unsigned char *a, const unsigned char *b, int base)
{
    unsigned int result = 0;
    for (int k = 0; k < 4; ++k)
    {
        result += (unsigned int)a[base + k] * (unsigned int)b[base + k];
    }
    return result;
}

float cpu_fdot2(const __fp16 *a, const __fp16 *b, int base)
{
    return (float)a[base] * (float)b[base]
         + (float)a[base + 1] * (float)b[base + 1];
}

int main()
{
    constexpr int n  = input_size;
    int           nb = (n + block_size - 1) / block_size;

    std::mt19937                          gen(42);
    std::uniform_int_distribution<int>    dist_byte(0, 255);
    std::uniform_real_distribution<float> dist_f16(-4.0f, 4.0f);

    std::cout << "Running math intrinsic kernels..." << std::endl;

    // msad_u8
    {
        // ~25% of src1 bytes are zero to exercise the masking behaviour.
        std::vector<unsigned int> h_src0(n), h_src1(n), h_out(n);
        for (int i = 0; i < n; ++i)
        {
            h_src0[i] = (unsigned int)dist_byte(gen)
                      | ((unsigned int)dist_byte(gen) << 8)
                      | ((unsigned int)dist_byte(gen) << 16)
                      | ((unsigned int)dist_byte(gen) << 24);
            h_src1[i] = ((i % 4 == 0) ? 0u : (unsigned int)dist_byte(gen))
                      | ((i % 4 == 1) ? 0u : (unsigned int)dist_byte(gen) << 8)
                      | ((unsigned int)dist_byte(gen) << 16)
                      | ((unsigned int)dist_byte(gen) << 24);
        }

        unsigned int *d_src0, *d_src1, *d_out;
        HIP_CHECK(hipMalloc(&d_src0, sizeof(unsigned int) * n));
        HIP_CHECK(hipMalloc(&d_src1, sizeof(unsigned int) * n));
        HIP_CHECK(hipMalloc(&d_out,  sizeof(unsigned int) * n));
        HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), sizeof(unsigned int) * n, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), sizeof(unsigned int) * n, hipMemcpyHostToDevice));

        msad_u8<<<nb, block_size>>>(d_src0, d_src1, d_out, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, sizeof(unsigned int) * n, hipMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < n && passed; ++i)
        {
            if (h_out[i] != cpu_msad_u8(h_src0[i], h_src1[i]))
            {
                passed = false;
            }
        }
        std::cout << "  msad_u8:              " << (passed ? "PASSED" : "FAILED") << std::endl;

        HIP_CHECK(hipFree(d_src0));
        HIP_CHECK(hipFree(d_src1));
        HIP_CHECK(hipFree(d_out));
    }

    // sad_u8 + sad_hi_u8
    {
        std::vector<unsigned int> h_lo0(n), h_hi0(n), h_lo1(n), h_hi1(n), h_out(n);
        for (int i = 0; i < n; ++i)
        {
            h_lo0[i] = (unsigned int)dist_byte(gen) | ((unsigned int)dist_byte(gen) << 8) | ((unsigned int)dist_byte(gen) << 16) | ((unsigned int)dist_byte(gen) << 24);
            h_hi0[i] = (unsigned int)dist_byte(gen) | ((unsigned int)dist_byte(gen) << 8) | ((unsigned int)dist_byte(gen) << 16) | ((unsigned int)dist_byte(gen) << 24);
            h_lo1[i] = (unsigned int)dist_byte(gen) | ((unsigned int)dist_byte(gen) << 8) | ((unsigned int)dist_byte(gen) << 16) | ((unsigned int)dist_byte(gen) << 24);
            h_hi1[i] = (unsigned int)dist_byte(gen) | ((unsigned int)dist_byte(gen) << 8) | ((unsigned int)dist_byte(gen) << 16) | ((unsigned int)dist_byte(gen) << 24);
        }

        unsigned int *d_lo0, *d_hi0, *d_lo1, *d_hi1, *d_out;
        HIP_CHECK(hipMalloc(&d_lo0, sizeof(unsigned int) * n));
        HIP_CHECK(hipMalloc(&d_hi0, sizeof(unsigned int) * n));
        HIP_CHECK(hipMalloc(&d_lo1, sizeof(unsigned int) * n));
        HIP_CHECK(hipMalloc(&d_hi1, sizeof(unsigned int) * n));
        HIP_CHECK(hipMalloc(&d_out, sizeof(unsigned int) * n));
        HIP_CHECK(hipMemcpy(d_lo0, h_lo0.data(), sizeof(unsigned int) * n, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_hi0, h_hi0.data(), sizeof(unsigned int) * n, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_lo1, h_lo1.data(), sizeof(unsigned int) * n, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_hi1, h_hi1.data(), sizeof(unsigned int) * n, hipMemcpyHostToDevice));

        sad_variants<<<nb, block_size>>>(d_lo0, d_hi0, d_lo1, d_hi1, d_out, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, sizeof(unsigned int) * n, hipMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < n && passed; ++i)
        {
            unsigned int ref = cpu_sad_u8(h_lo0[i], h_lo1[i])
                             + cpu_sad_hi_u8(h_hi0[i], h_hi1[i]);
            if (h_out[i] != ref)
            {
                passed = false;
            }
        }
        std::cout << "  sad_u8 + sad_hi_u8:   " << (passed ? "PASSED" : "FAILED") << std::endl;

        HIP_CHECK(hipFree(d_lo0));
        HIP_CHECK(hipFree(d_hi0));
        HIP_CHECK(hipFree(d_lo1));
        HIP_CHECK(hipFree(d_hi1));
        HIP_CHECK(hipFree(d_out));
    }

    // qsad_pk_u16_u8
    {
        std::vector<unsigned long long> h_src0(n);
        std::vector<unsigned int>       h_src1(n);
        std::vector<unsigned long long> h_out(n);
        for (int i = 0; i < n; ++i)
        {
            h_src0[i] = (unsigned long long)dist_byte(gen)
                      | ((unsigned long long)dist_byte(gen) << 8)
                      | ((unsigned long long)dist_byte(gen) << 16)
                      | ((unsigned long long)dist_byte(gen) << 24)
                      | ((unsigned long long)dist_byte(gen) << 32)
                      | ((unsigned long long)dist_byte(gen) << 40)
                      | ((unsigned long long)dist_byte(gen) << 48)
                      | ((unsigned long long)dist_byte(gen) << 56);
            h_src1[i] = (unsigned int)dist_byte(gen) | ((unsigned int)dist_byte(gen) << 8) | ((unsigned int)dist_byte(gen) << 16) | ((unsigned int)dist_byte(gen) << 24);
        }

        unsigned long long *d_src0, *d_out;
        unsigned int       *d_src1;
        HIP_CHECK(hipMalloc(&d_src0, sizeof(unsigned long long) * n));
        HIP_CHECK(hipMalloc(&d_src1, sizeof(unsigned int)       * n));
        HIP_CHECK(hipMalloc(&d_out,  sizeof(unsigned long long) * n));
        HIP_CHECK(hipMemcpy(d_src0, h_src0.data(), sizeof(unsigned long long) * n, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_src1, h_src1.data(), sizeof(unsigned int)       * n, hipMemcpyHostToDevice));

        qsad_pk_u16_u8<<<nb, block_size>>>(d_src0, d_src1, d_out, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, sizeof(unsigned long long) * n, hipMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < n && passed; ++i)
        {
            for (int s = 0; s < 4 && passed; ++s)
            {
                unsigned int sad = 0;
                for (int k = 0; k < 4; ++k)
                {
                    unsigned int a = (h_src0[i] >> ((s + k) * 8)) & 0xff;
                    unsigned int b = (h_src1[i] >> (k * 8)) & 0xff;
                    sad           += (a > b) ? (a - b) : (b - a);
                }
                if (((h_out[i] >> (s * 16)) & 0xffff) != (sad & 0xffff))
                {
                    passed = false;
                }
            }
        }
        std::cout << "  qsad_pk_u16_u8:       " << (passed ? "PASSED" : "FAILED") << std::endl;

        HIP_CHECK(hipFree(d_src0));
        HIP_CHECK(hipFree(d_src1));
        HIP_CHECK(hipFree(d_out));
    }

    // udot4
    {
        int nb4 = (n / 4 + block_size - 1) / block_size;

        std::vector<unsigned char> h_a(n), h_b(n);
        std::vector<unsigned int>  h_out(n / 4);
        for (int i = 0; i < n; ++i)
        {
            h_a[i] = (unsigned char)dist_byte(gen);
            h_b[i] = (unsigned char)dist_byte(gen);
        }

        unsigned char *d_a, *d_b;
        unsigned int  *d_out;
        HIP_CHECK(hipMalloc(&d_a,   sizeof(unsigned char) * n));
        HIP_CHECK(hipMalloc(&d_b,   sizeof(unsigned char) * n));
        HIP_CHECK(hipMalloc(&d_out, sizeof(unsigned int)  * (n / 4)));
        HIP_CHECK(hipMemcpy(d_a, h_a.data(), sizeof(unsigned char) * n, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, h_b.data(), sizeof(unsigned char) * n, hipMemcpyHostToDevice));

        udot4<<<nb4, block_size>>>(d_a, d_b, d_out, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, sizeof(unsigned int) * (n / 4), hipMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < n / 4 && passed; ++i)
        {
            if (h_out[i] != cpu_udot4(h_a.data(), h_b.data(), i * 4))
            {
                passed = false;
            }
        }
        std::cout << "  udot4:                " << (passed ? "PASSED" : "FAILED") << std::endl;

        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_out));
    }

    // fdot2
    {
        int nb2 = (n / 2 + block_size - 1) / block_size;

        std::vector<__fp16> h_a(n), h_b(n);
        std::vector<float>  h_out(n / 2);
        for (int i = 0; i < n; ++i)
        {
            h_a[i] = (__fp16)dist_f16(gen);
            h_b[i] = (__fp16)dist_f16(gen);
        }

        __fp16 *d_a, *d_b;
        float  *d_out;
        HIP_CHECK(hipMalloc(&d_a,   sizeof(__fp16) * n));
        HIP_CHECK(hipMalloc(&d_b,   sizeof(__fp16) * n));
        HIP_CHECK(hipMalloc(&d_out, sizeof(float)  * (n / 2)));
        HIP_CHECK(hipMemcpy(d_a, h_a.data(), sizeof(__fp16) * n, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_b, h_b.data(), sizeof(__fp16) * n, hipMemcpyHostToDevice));

        fdot2<<<nb2, block_size>>>(d_a, d_b, d_out, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, sizeof(float) * (n / 2), hipMemcpyDeviceToHost));

        bool        passed = true;
        const float tol    = 1e-3f;
        for (int i = 0; i < n / 2 && passed; ++i)
        {
            float ref = cpu_fdot2(h_a.data(), h_b.data(), i * 2);
            if (std::fabs(h_out[i] - ref) > tol * (std::fabs(ref) + 1.0f))
            {
                passed = false;
            }
        }
        std::cout << "  fdot2 (fp16->f32):    " << (passed ? "PASSED" : "FAILED") << std::endl;

        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_b));
        HIP_CHECK(hipFree(d_out));
    }

    // cvt_pk_u8_f32
    {
        int nb4 = (n / 4 + block_size - 1) / block_size;

        // Integer-valued floats avoid any rounding ambiguity at the byte boundary.
        std::uniform_int_distribution<int> dist_u8(0, 255);
        std::vector<float>        h_in(n);
        std::vector<unsigned int> h_out(n / 4);
        for (int i = 0; i < n; ++i)
        {
            h_in[i] = (float)dist_u8(gen);
        }

        float        *d_in;
        unsigned int *d_out;
        HIP_CHECK(hipMalloc(&d_in,  sizeof(float)        * n));
        HIP_CHECK(hipMalloc(&d_out, sizeof(unsigned int) * (n / 4)));
        HIP_CHECK(hipMemcpy(d_in, h_in.data(), sizeof(float) * n, hipMemcpyHostToDevice));

        cvt_pk_u8_f32<<<nb4, block_size>>>(d_in, d_out, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, sizeof(unsigned int) * (n / 4), hipMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < n / 4 && passed; ++i)
        {
            for (int k = 0; k < 4 && passed; ++k)
            {
                unsigned int expected = (unsigned int)(int)h_in[i * 4 + k];
                unsigned int got      = (h_out[i] >> (k * 8)) & 0xff;
                if (got != expected)
                {
                    passed = false;
                }
            }
        }
        std::cout << "  cvt_pk_u8_f32:        " << (passed ? "PASSED" : "FAILED") << std::endl;

        HIP_CHECK(hipFree(d_in));
        HIP_CHECK(hipFree(d_out));
    }

    std::cout << "Execution completed successfully." << std::endl;
    return EXIT_SUCCESS;
}
