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

#include <hip/hip_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <string>
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

constexpr int M_GLOBAL    = 256;
constexpr int N_GLOBAL    = 256;
constexpr int K_GLOBAL    = 256;
constexpr int warp_size   = 64;
constexpr int block_size  = 256;
constexpr int timing_runs = 200;

#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || \
    defined(__gfx942__) || defined(__gfx950__)
#define HAS_SMFMAC_MAI
#endif

#if defined(__gfx942__) || defined(__gfx950__)
#define HAS_SMFMAC_FP8
#endif

#if defined(__gfx950__)
#define HAS_SMFMAC_GFX950
#endif

// ---- FP8 E4M3 helpers (via hip/hip_fp8.h) ----
// gfx94x uses FNUZ encoding; gfx950 and later use OCP (standard) encoding.
// The interpretation is selected at runtime based on the device architecture.

static __hip_fp8_interpretation_t fp8_interpret = __HIP_E4M3_FNUZ;

static __hip_fp8_storage_t float_to_fp8(float v)
{
    return __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, fp8_interpret);
}

static float fp8_to_float(__hip_fp8_storage_t v)
{
    __half hf = __hip_cvt_fp8_to_halfraw(v, fp8_interpret);
    return __half2float(hf);
}

// ---- Host-side A compression (4:2 sparsity, keep positions {0,2}) ----

static void compress_A_f16(const std::vector<float> &A, int M, int K,
                           std::vector<_Float16> &out)
{
    out.resize(M * (K / 2));
    for (int m = 0; m < M; ++m)
    {
        int ci = 0;
        for (int k = 0; k < K; k += 4)
        {
            out[m * (K / 2) + ci++] = static_cast<_Float16>(A[m * K + k + 0]);
            out[m * (K / 2) + ci++] = static_cast<_Float16>(A[m * K + k + 2]);
        }
    }
}

static void compress_A_fp8(const std::vector<float> &A, int M, int K,
                           std::vector<uint8_t> &out)
{
    out.resize(M * (K / 2));
    for (int m = 0; m < M; ++m)
    {
        int ci = 0;
        for (int k = 0; k < K; k += 4)
        {
            out[m * (K / 2) + ci++] = float_to_fp8(A[m * K + k + 0]);
            out[m * (K / 2) + ci++] = float_to_fp8(A[m * K + k + 2]);
        }
    }
}

// ---- CPU reference (sparse GEMM) ----

static void cpu_ref_sparse(const std::vector<float> &A, const std::vector<float> &B,
                           std::vector<float> &D, int M, int N, int K, bool f16)
{
    D.assign(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                if (k % 4 == 0 || k % 4 == 2)
                {
                    float a = A[m * K + k];
                    float b = B[k * N + n];
                    if (f16)
                    {
                        a = static_cast<float>(static_cast<_Float16>(a));
                        b = static_cast<float>(static_cast<_Float16>(b));
                    }
                    acc += a * b;
                }
            }
            D[m * N + n] = acc;
        }
    }
}

// ---- Verification ----

static void verify(const std::vector<float> &ref, const float *d_output,
                   int M, int N, float rtol = 5e-2f, float atol = 5e-2f)
{
    std::vector<float> gpu(M * N);
    HIP_CHECK(hipMemcpy(gpu.data(), d_output, sizeof(float) * M * N,
                        hipMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < M * N; ++i)
    {
        float diff = std::fabs(gpu[i] - ref[i]);
        if (diff > atol + rtol * std::fabs(ref[i]))
        {
            if (mismatches < 3)
            {
                std::cerr << "    mismatch [" << i / N << "," << i % N
                          << "]: gpu=" << gpu[i] << " ref=" << ref[i] << std::endl;
            }
            ++mismatches;
        }
    }
    if (mismatches > 0)
    {
        std::cout << "FAILED (" << mismatches << " / " << M * N << ")" << std::endl;
    }
    else
    {
        std::cout << "PASSED" << std::endl;
    }
}

#ifdef ENABLE_TIMERS
static void time_kernel(const std::string &label, const std::function<void()> &launch)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    float total_ms = 0.0f;
    for (int run = 0; run < timing_runs; ++run)
    {
        HIP_CHECK(hipEventRecord(start));
        launch();
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    std::cout << "    [timer] " << label << ": avg "
              << (total_ms / timing_runs * 1e3f) << " us over "
              << timing_runs << " runs" << std::endl;
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}
#endif

// ============================================================================
// Kernels
// ============================================================================

// [Sphinx lds_baseline start]
__global__ __launch_bounds__(block_size)
void lds_matmul_f16(const _Float16 *__restrict__ A, const _Float16 *__restrict__ B,
                    float *__restrict__ D, int M, int N, int K)
{
    constexpr int CTA_M = 32, CTA_N = 32, TILE_K = 32;

    const int tid    = threadIdx.x;
    const int warp_m = (tid / warp_size) / 2;
    const int warp_n = (tid / warp_size) % 2;
    const int lane   = tid % warp_size;
    const int cta_m  = blockIdx.y * CTA_M;
    const int cta_n  = blockIdx.x * CTA_N;

    __shared__ _Float16 sA[CTA_M * TILE_K];
    __shared__ _Float16 sB[TILE_K * CTA_N];

    int out_n      = lane % 16;
    int out_m_base = (lane / 16) * 4;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kt = 0; kt < K / TILE_K; ++kt)
    {
        int k0 = kt * TILE_K;
        for (int i = tid; i < CTA_M * TILE_K; i += block_size)
        {
            int r = i / TILE_K, c = i % TILE_K;
            sA[i] = A[(cta_m + r) * K + k0 + c];
        }
        for (int i = tid; i < TILE_K * CTA_N; i += block_size)
        {
            int r = i / CTA_N, c = i % CTA_N;
            sB[i] = B[(k0 + r) * N + cta_n + c];
        }
        __syncthreads();

        for (int r = 0; r < 4; ++r)
        {
            int m_local = warp_m * 16 + out_m_base + r;
            int n_local = warp_n * 16 + out_n;
            for (int k = 0; k < TILE_K; ++k)
            {
                acc[r] += static_cast<float>(sA[m_local * TILE_K + k]) *
                          static_cast<float>(sB[k * CTA_N + n_local]);
            }
        }
        __syncthreads();
    }

    int sub_m = cta_m + warp_m * 16;
    int sub_n = cta_n + warp_n * 16;
    for (int r = 0; r < 4; ++r)
    {
        D[(sub_m + out_m_base + r) * N + sub_n + out_n] = acc[r];
    }
}
// [Sphinx lds_baseline end]

#ifdef HAS_SMFMAC_MAI
// [Sphinx smfmac_f32_16x16x32_f16 start]
__global__ __launch_bounds__(block_size)
void smfmac_f32_16x16x32_f16(const _Float16 *__restrict__ A_comp,
                              const _Float16 *__restrict__ B,
                              float *__restrict__ D, int M, int N, int K)
{
    constexpr int CTA_M = 32, CTA_N = 32, TILE_K = 32;

    const int tid    = threadIdx.x;
    const int warp_m = (tid / warp_size) / 2;
    const int warp_n = (tid / warp_size) % 2;
    const int lane   = tid % warp_size;
    const int g      = lane / 16;
    const int n      = lane % 16;
    const int cta_m  = blockIdx.y * CTA_M;
    const int cta_n  = blockIdx.x * CTA_N;
    const int K_comp = K / 2;

    constexpr int tk_comp = TILE_K / 2;
    __shared__ _Float16 sA[CTA_M * tk_comp];
    __shared__ _Float16 sB[TILE_K * CTA_N];

    using float4v = __attribute__((ext_vector_type(4))) float;
    float4v d = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kt = 0; kt < K / TILE_K; ++kt)
    {
        int k0 = kt * TILE_K;
        for (int i = tid; i < CTA_M * tk_comp; i += block_size)
        {
            int r = i / tk_comp, c = i % tk_comp;
            sA[i] = A_comp[(cta_m + r) * K_comp + k0 / 2 + c];
        }
        for (int i = tid; i < TILE_K * CTA_N; i += block_size)
        {
            int r = i / CTA_N, c = i % CTA_N;
            sB[i] = B[(k0 + r) * N + cta_n + c];
        }
        __syncthreads();

        using half4v = __attribute__((ext_vector_type(4))) _Float16;
        half4v a;
        int a_off = (warp_m * 16 + n) * tk_comp + g * 4;
        for (int i = 0; i < 4; ++i)
        {
            a[i] = sA[a_off + i];
        }

        using half8v = __attribute__((ext_vector_type(8))) _Float16;
        half8v b;
        int b_col = warp_n * 16 + n;
        for (int v = 0; v < 4; ++v)
        {
            int kk = g * 8 + v * 2;
            b[v * 2 + 0] = sB[kk * CTA_N + b_col];
            b[v * 2 + 1] = sB[(kk + 1) * CTA_N + b_col];
        }

        d = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, d, 0x88, 0, 0);
        __syncthreads();
    }

    int sub_m = cta_m + warp_m * 16;
    int sub_n = cta_n + warp_n * 16;
    for (int r = 0; r < 4; ++r)
    {
        D[(sub_m + (g * 4) + r) * N + sub_n + n] = d[r];
    }
}
// [Sphinx smfmac_f32_16x16x32_f16 end]

// [Sphinx smfmac_f32_32x32x16_f16 start]
__global__ __launch_bounds__(block_size)
void smfmac_f32_32x32x16_f16(const _Float16 *__restrict__ A_comp,
                              const _Float16 *__restrict__ B,
                              float *__restrict__ D, int M, int N, int K)
{
    constexpr int CTA_M = 64, CTA_N = 64, TILE_K = 16;

    const int tid    = threadIdx.x;
    const int warp_m = (tid / warp_size) / 2;
    const int warp_n = (tid / warp_size) % 2;
    const int lane   = tid % warp_size;
    const int g      = lane / 16;
    const int cta_m  = blockIdx.y * CTA_M;
    const int cta_n  = blockIdx.x * CTA_N;
    const int K_comp = K / 2;

    constexpr int tk_comp = TILE_K / 2;
    __shared__ _Float16 sA[CTA_M * tk_comp];
    __shared__ _Float16 sB[TILE_K * CTA_N];

    using float16v = __attribute__((ext_vector_type(16))) float;
    float16v d = {};

    for (int kt = 0; kt < K / TILE_K; ++kt)
    {
        int k0 = kt * TILE_K;
        for (int i = tid; i < CTA_M * tk_comp; i += block_size)
        {
            int r = i / tk_comp, c = i % tk_comp;
            sA[i] = A_comp[(cta_m + r) * K_comp + k0 / 2 + c];
        }
        for (int i = tid; i < TILE_K * CTA_N; i += block_size)
        {
            int r = i / CTA_N, c = i % CTA_N;
            sB[i] = B[(k0 + r) * N + cta_n + c];
        }
        __syncthreads();

        using half4v = __attribute__((ext_vector_type(4))) _Float16;
        half4v a;
        int a_row = warp_m * 32 + (g % 2) * 16 + (lane % 16);
        int a_col = (g / 2) * 4;
        for (int i = 0; i < 4; ++i)
        {
            a[i] = sA[a_row * tk_comp + a_col + i];
        }

        using half8v = __attribute__((ext_vector_type(8))) _Float16;
        half8v b;
        int b_col = warp_n * 32 + (g % 2) * 16 + (lane % 16);
        int k_off = (g / 2) * 8;
        for (int v = 0; v < 4; ++v)
        {
            int kk = k_off + v * 2;
            b[v * 2 + 0] = sB[kk * CTA_N + b_col];
            b[v * 2 + 1] = sB[(kk + 1) * CTA_N + b_col];
        }

        d = __builtin_amdgcn_smfmac_f32_32x32x16_f16(a, b, d, 0x88, 0, 0);
        __syncthreads();
    }

    int sub_m = cta_m + warp_m * 32;
    int sub_n = cta_n + warp_n * 32;
    int j_base = lane % 32;
    int grp    = lane / 32;
    for (int r = 0; r < 16; ++r)
    {
        int i_out = (r / 4) * 8 + grp * 4 + (r % 4);
        D[(sub_m + i_out) * N + sub_n + j_base] = d[r];
    }
}
// [Sphinx smfmac_f32_32x32x16_f16 end]
#endif // HAS_SMFMAC_MAI

#ifdef HAS_SMFMAC_FP8
// [Sphinx smfmac_f32_16x16x64_fp8_fp8 start]
__global__ __launch_bounds__(block_size)
void smfmac_f32_16x16x64_fp8_fp8(const uint8_t *__restrict__ A_comp,
                                  const uint8_t *__restrict__ B,
                                  float *__restrict__ D, int M, int N, int K)
{
    constexpr int CTA_M = 32, CTA_N = 32, TILE_K = 64;

    const int tid    = threadIdx.x;
    const int warp_m = (tid / warp_size) / 2;
    const int warp_n = (tid / warp_size) % 2;
    const int lane   = tid % warp_size;
    const int g      = lane / 16;
    const int n      = lane % 16;
    const int cta_m  = blockIdx.y * CTA_M;
    const int cta_n  = blockIdx.x * CTA_N;
    const int K_comp = K / 2;

    constexpr int tk_comp = TILE_K / 2;
    __shared__ uint8_t sA[CTA_M * tk_comp];
    __shared__ uint8_t sB[TILE_K * CTA_N];

    using float4v = __attribute__((ext_vector_type(4))) float;
    float4v d = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kt = 0; kt < K / TILE_K; ++kt)
    {
        int k0 = kt * TILE_K;
        for (int i = tid; i < CTA_M * tk_comp; i += block_size)
        {
            int r = i / tk_comp, c = i % tk_comp;
            sA[i] = A_comp[(cta_m + r) * K_comp + k0 / 2 + c];
        }
        for (int i = tid; i < TILE_K * CTA_N; i += block_size)
        {
            int r = i / CTA_N, c = i % CTA_N;
            sB[i] = B[(k0 + r) * N + cta_n + c];
        }
        __syncthreads();

        using int2v = __attribute__((ext_vector_type(2))) int;
        int2v a;
        int a_off = (warp_m * 16 + n) * tk_comp + g * 8;
        for (int j = 0; j < 2; ++j)
        {
            a[j] = static_cast<int>(
                uint32_t(sA[a_off + j*4 + 0])       | (uint32_t(sA[a_off + j*4 + 1]) << 8) |
                (uint32_t(sA[a_off + j*4 + 2]) << 16) | (uint32_t(sA[a_off + j*4 + 3]) << 24));
        }

        using int4v = __attribute__((ext_vector_type(4))) int;
        int4v b;
        int b_col = warp_n * 16 + n;
        for (int v = 0; v < 4; ++v)
        {
            int kk = g * 16 + v * 4;
            b[v] = static_cast<int>(
                uint32_t(sB[kk * CTA_N + b_col])       | (uint32_t(sB[(kk+1) * CTA_N + b_col]) << 8) |
                (uint32_t(sB[(kk+2) * CTA_N + b_col]) << 16) | (uint32_t(sB[(kk+3) * CTA_N + b_col]) << 24));
        }

        d = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, d, 0x8888, 0, 0);
        __syncthreads();
    }

    int sub_m = cta_m + warp_m * 16;
    int sub_n = cta_n + warp_n * 16;
    for (int r = 0; r < 4; ++r)
    {
        D[(sub_m + g * 4 + r) * N + sub_n + n] = d[r];
    }
}
// [Sphinx smfmac_f32_16x16x64_fp8_fp8 end]
#endif // HAS_SMFMAC_FP8

#ifdef HAS_SMFMAC_GFX950
// [Sphinx smfmac_f32_16x16x128_fp8_fp8 start]
__global__ __launch_bounds__(block_size)
void smfmac_f32_16x16x128_fp8_fp8(const uint8_t *__restrict__ A_comp,
                                   const uint8_t *__restrict__ B,
                                   float *__restrict__ D, int M, int N, int K)
{
    constexpr int CTA_M = 32, CTA_N = 32, TILE_K = 128;

    const int tid    = threadIdx.x;
    const int warp_m = (tid / warp_size) / 2;
    const int warp_n = (tid / warp_size) % 2;
    const int lane   = tid % warp_size;
    const int g      = lane / 16;
    const int n      = lane % 16;
    const int cta_m  = blockIdx.y * CTA_M;
    const int cta_n  = blockIdx.x * CTA_N;
    const int K_comp = K / 2;

    constexpr int tk_comp = TILE_K / 2;
    __shared__ uint8_t sA[CTA_M * tk_comp];
    __shared__ uint8_t sB[TILE_K * CTA_N];

    using float4v = __attribute__((ext_vector_type(4))) float;
    float4v d = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kt = 0; kt < K / TILE_K; ++kt)
    {
        int k0 = kt * TILE_K;
        for (int i = tid; i < CTA_M * tk_comp; i += block_size)
        {
            int r = i / tk_comp, c = i % tk_comp;
            sA[i] = A_comp[(cta_m + r) * K_comp + k0 / 2 + c];
        }
        for (int i = tid; i < TILE_K * CTA_N; i += block_size)
        {
            int r = i / CTA_N, c = i % CTA_N;
            sB[i] = B[(k0 + r) * N + cta_n + c];
        }
        __syncthreads();

        using int4v = __attribute__((ext_vector_type(4))) int;
        int4v a;
        int a_off = (warp_m * 16 + n) * tk_comp + g * 16;
        for (int j = 0; j < 4; ++j)
        {
            a[j] = static_cast<int>(
                uint32_t(sA[a_off + j*4 + 0])       | (uint32_t(sA[a_off + j*4 + 1]) << 8) |
                (uint32_t(sA[a_off + j*4 + 2]) << 16) | (uint32_t(sA[a_off + j*4 + 3]) << 24));
        }

        using int8v = __attribute__((ext_vector_type(8))) int;
        int8v b;
        int b_col = warp_n * 16 + n;
        for (int v = 0; v < 8; ++v)
        {
            int kk = (v / 4) * 64 + g * 16 + (v % 4) * 4;
            b[v] = static_cast<int>(
                uint32_t(sB[kk * CTA_N + b_col])       | (uint32_t(sB[(kk+1) * CTA_N + b_col]) << 8) |
                (uint32_t(sB[(kk+2) * CTA_N + b_col]) << 16) | (uint32_t(sB[(kk+3) * CTA_N + b_col]) << 24));
        }

        d = __builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8(
                a, b, d, static_cast<int>(0x88888888u), 0, 0);
        __syncthreads();
    }

    int sub_m = cta_m + warp_m * 16;
    int sub_n = cta_n + warp_n * 16;
    for (int r = 0; r < 4; ++r)
    {
        D[(sub_m + g * 4 + r) * N + sub_n + n] = d[r];
    }
}
// [Sphinx smfmac_f32_16x16x128_fp8_fp8 end]
#endif // HAS_SMFMAC_GFX950

// ============================================================================
// Main
// ============================================================================

int main()
{
    int deviceid = 0;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceid));
    std::cout << "Device: " << props.name << std::endl;
    std::cout << "GCN arch: " << props.gcnArchName << std::endl;
    std::cout << "Problem: " << M_GLOBAL << "x" << N_GLOBAL << "x" << K_GLOBAL << std::endl;

    // Select FP8 interpretation based on architecture.
    // gfx94x uses FNUZ encoding; gfx950 and later use standard OCP encoding.
    std::string arch(props.gcnArchName);
    if (arch.find("gfx94") != std::string::npos)
    {
        fp8_interpret = __HIP_E4M3_FNUZ;
    }
    else
    {
        fp8_interpret = __HIP_E4M3;
    }

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int>    fp8_dist(-2, 2);

    std::cout << "\nRunning sparse MFMA examples..." << std::endl;

    // LDS dense baseline
    {
        constexpr int CTA_M = 32, CTA_N = 32;
        dim3 grid(N_GLOBAL / CTA_N, M_GLOBAL / CTA_M);

        std::vector<_Float16> A_f16(M_GLOBAL * K_GLOBAL), B_f16(K_GLOBAL * N_GLOBAL);
        std::mt19937 gen_bl(99);
        for (auto &v : A_f16) { v = static_cast<_Float16>(dist(gen_bl)); }
        for (auto &v : B_f16) { v = static_cast<_Float16>(dist(gen_bl)); }

        _Float16 *dA = nullptr, *dB = nullptr;
        float    *dD = nullptr;
        HIP_CHECK(hipMalloc(&dA, sizeof(_Float16) * A_f16.size()));
        HIP_CHECK(hipMalloc(&dB, sizeof(_Float16) * B_f16.size()));
        HIP_CHECK(hipMalloc(&dD, sizeof(float) * M_GLOBAL * N_GLOBAL));
        HIP_CHECK(hipMemcpy(dA, A_f16.data(), sizeof(_Float16) * A_f16.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, B_f16.data(), sizeof(_Float16) * B_f16.size(), hipMemcpyHostToDevice));

        lds_matmul_f16<<<grid, block_size>>>(dA, dB, dD, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

#ifdef ENABLE_TIMERS
        time_kernel("lds_matmul_f16 (dense baseline)", [&]()
        {
            lds_matmul_f16<<<grid, block_size>>>(dA, dB, dD, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        });
#endif

        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dD));
    }

#ifdef HAS_SMFMAC_MAI
    // smfmac_f32_16x16x32_f16
    {
        constexpr int CTA_M = 32, CTA_N = 32;
        dim3 grid(N_GLOBAL / CTA_N, M_GLOBAL / CTA_M);

        std::vector<float> A(M_GLOBAL * K_GLOBAL), B(K_GLOBAL * N_GLOBAL);
        for (float &v : A) { v = dist(gen); }
        for (float &v : B) { v = dist(gen); }

        std::vector<float> D_ref;
        cpu_ref_sparse(A, B, D_ref, M_GLOBAL, N_GLOBAL, K_GLOBAL, true);

        std::vector<_Float16> A_comp;
        compress_A_f16(A, M_GLOBAL, K_GLOBAL, A_comp);
        std::vector<_Float16> B_f16(K_GLOBAL * N_GLOBAL);
        for (int i = 0; i < K_GLOBAL * N_GLOBAL; ++i) { B_f16[i] = static_cast<_Float16>(B[i]); }

        _Float16 *dA = nullptr, *dB = nullptr;
        float    *dD = nullptr;
        HIP_CHECK(hipMalloc(&dA, sizeof(_Float16) * A_comp.size()));
        HIP_CHECK(hipMalloc(&dB, sizeof(_Float16) * B_f16.size()));
        HIP_CHECK(hipMalloc(&dD, sizeof(float) * M_GLOBAL * N_GLOBAL));
        HIP_CHECK(hipMemcpy(dA, A_comp.data(), sizeof(_Float16) * A_comp.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, B_f16.data(), sizeof(_Float16) * B_f16.size(), hipMemcpyHostToDevice));

        smfmac_f32_16x16x32_f16<<<grid, block_size>>>(dA, dB, dD, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        std::cout << "  smfmac_f32_16x16x32_f16: ";
        verify(D_ref, dD, M_GLOBAL, N_GLOBAL);
#ifdef ENABLE_TIMERS
        time_kernel("smfmac_16x16x32_f16", [&]()
        {
            smfmac_f32_16x16x32_f16<<<grid, block_size>>>(dA, dB, dD, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        });
#endif
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dD));
    }

    // smfmac_f32_32x32x16_f16
    {
        constexpr int CTA_M = 64, CTA_N = 64;
        dim3 grid(N_GLOBAL / CTA_N, M_GLOBAL / CTA_M);

        std::vector<float> A(M_GLOBAL * K_GLOBAL), B(K_GLOBAL * N_GLOBAL);
        for (float &v : A) { v = dist(gen); }
        for (float &v : B) { v = dist(gen); }

        std::vector<float> D_ref;
        cpu_ref_sparse(A, B, D_ref, M_GLOBAL, N_GLOBAL, K_GLOBAL, true);

        std::vector<_Float16> A_comp;
        compress_A_f16(A, M_GLOBAL, K_GLOBAL, A_comp);
        std::vector<_Float16> B_f16(K_GLOBAL * N_GLOBAL);
        for (int i = 0; i < K_GLOBAL * N_GLOBAL; ++i) { B_f16[i] = static_cast<_Float16>(B[i]); }

        _Float16 *dA = nullptr, *dB = nullptr;
        float    *dD = nullptr;
        HIP_CHECK(hipMalloc(&dA, sizeof(_Float16) * A_comp.size()));
        HIP_CHECK(hipMalloc(&dB, sizeof(_Float16) * B_f16.size()));
        HIP_CHECK(hipMalloc(&dD, sizeof(float) * M_GLOBAL * N_GLOBAL));
        HIP_CHECK(hipMemcpy(dA, A_comp.data(), sizeof(_Float16) * A_comp.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, B_f16.data(), sizeof(_Float16) * B_f16.size(), hipMemcpyHostToDevice));

        smfmac_f32_32x32x16_f16<<<grid, block_size>>>(dA, dB, dD, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        std::cout << "  smfmac_f32_32x32x16_f16: ";
        verify(D_ref, dD, M_GLOBAL, N_GLOBAL);
#ifdef ENABLE_TIMERS
        time_kernel("smfmac_32x32x16_f16", [&]()
        {
            smfmac_f32_32x32x16_f16<<<grid, block_size>>>(dA, dB, dD, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        });
#endif
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dD));
    }
#endif // HAS_SMFMAC_MAI

#ifdef HAS_SMFMAC_FP8
    // smfmac_f32_16x16x64_fp8_fp8
    {
        constexpr int CTA_M = 32, CTA_N = 32;
        dim3 grid(N_GLOBAL / CTA_N, M_GLOBAL / CTA_M);

        std::vector<float> A(M_GLOBAL * K_GLOBAL), B(K_GLOBAL * N_GLOBAL);
        for (float &v : A) { v = static_cast<float>(fp8_dist(gen)); }
        for (float &v : B) { v = static_cast<float>(fp8_dist(gen)); }
        for (float &v : A) { v = fp8_to_float(float_to_fp8(v)); }
        for (float &v : B) { v = fp8_to_float(float_to_fp8(v)); }

        std::vector<float> D_ref;
        cpu_ref_sparse(A, B, D_ref, M_GLOBAL, N_GLOBAL, K_GLOBAL, false);

        std::vector<uint8_t> A_comp;
        compress_A_fp8(A, M_GLOBAL, K_GLOBAL, A_comp);
        std::vector<uint8_t> B_fp8(K_GLOBAL * N_GLOBAL);
        for (int i = 0; i < K_GLOBAL * N_GLOBAL; ++i) { B_fp8[i] = float_to_fp8(B[i]); }

        uint8_t *dA = nullptr, *dB = nullptr;
        float   *dD = nullptr;
        HIP_CHECK(hipMalloc(&dA, A_comp.size()));
        HIP_CHECK(hipMalloc(&dB, B_fp8.size()));
        HIP_CHECK(hipMalloc(&dD, sizeof(float) * M_GLOBAL * N_GLOBAL));
        HIP_CHECK(hipMemcpy(dA, A_comp.data(), A_comp.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, B_fp8.data(), B_fp8.size(), hipMemcpyHostToDevice));

        smfmac_f32_16x16x64_fp8_fp8<<<grid, block_size>>>(dA, dB, dD, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        std::cout << "  smfmac_f32_16x16x64_fp8_fp8: ";
        verify(D_ref, dD, M_GLOBAL, N_GLOBAL, 0.5f, 0.5f);
#ifdef ENABLE_TIMERS
        time_kernel("smfmac_16x16x64_fp8_fp8", [&]()
        {
            smfmac_f32_16x16x64_fp8_fp8<<<grid, block_size>>>(dA, dB, dD, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        });
#endif
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dD));
    }
#endif // HAS_SMFMAC_FP8

#ifdef HAS_SMFMAC_GFX950
    // smfmac_f32_16x16x128_fp8_fp8
    {
        constexpr int CTA_M = 32, CTA_N = 32;
        dim3 grid(N_GLOBAL / CTA_N, M_GLOBAL / CTA_M);

        std::vector<float> A(M_GLOBAL * K_GLOBAL), B(K_GLOBAL * N_GLOBAL);
        for (float &v : A) { v = static_cast<float>(fp8_dist(gen)); }
        for (float &v : B) { v = static_cast<float>(fp8_dist(gen)); }
        for (float &v : A) { v = fp8_to_float(float_to_fp8(v)); }
        for (float &v : B) { v = fp8_to_float(float_to_fp8(v)); }

        std::vector<float> D_ref;
        cpu_ref_sparse(A, B, D_ref, M_GLOBAL, N_GLOBAL, K_GLOBAL, false);

        std::vector<uint8_t> A_comp;
        compress_A_fp8(A, M_GLOBAL, K_GLOBAL, A_comp);
        std::vector<uint8_t> B_fp8(K_GLOBAL * N_GLOBAL);
        for (int i = 0; i < K_GLOBAL * N_GLOBAL; ++i) { B_fp8[i] = float_to_fp8(B[i]); }

        uint8_t *dA = nullptr, *dB = nullptr;
        float   *dD = nullptr;
        HIP_CHECK(hipMalloc(&dA, A_comp.size()));
        HIP_CHECK(hipMalloc(&dB, B_fp8.size()));
        HIP_CHECK(hipMalloc(&dD, sizeof(float) * M_GLOBAL * N_GLOBAL));
        HIP_CHECK(hipMemcpy(dA, A_comp.data(), A_comp.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, B_fp8.data(), B_fp8.size(), hipMemcpyHostToDevice));

        smfmac_f32_16x16x128_fp8_fp8<<<grid, block_size>>>(dA, dB, dD, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        std::cout << "  smfmac_f32_16x16x128_fp8_fp8: ";
        verify(D_ref, dD, M_GLOBAL, N_GLOBAL, 0.5f, 0.5f);
#ifdef ENABLE_TIMERS
        time_kernel("smfmac_16x16x128_fp8_fp8", [&]()
        {
            smfmac_f32_16x16x128_fp8_fp8<<<grid, block_size>>>(dA, dB, dD, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        });
#endif
        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dD));
    }
#endif // HAS_SMFMAC_GFX950

    std::cout << "Execution completed successfully." << std::endl;
    return EXIT_SUCCESS;
}
