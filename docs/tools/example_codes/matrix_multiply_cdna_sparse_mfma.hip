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

// ---------------------------------------------------------------------------
// matrix_multiply_cdna_sparse_mfma.hip
//
// FP16 sparse MFMA (SMFMAC) intrinsics for CDNA GPUs.
// Demonstrates smfmac_f32_16x16x32_f16 and smfmac_f32_32x32x16_f16 using
// a policy-based kernel template.
//
// Architecture: gfx908, gfx90a, gfx940, gfx942, gfx950.
// Compile with: hipcc -O3 -std=c++17 --offload-arch=gfx942 \
//                   matrix_multiply_cdna_sparse_mfma.hip -o sparse_mfma_cdna
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define HIP_CHECK(expression)                      \
    {                                              \
        const hipError_t status = (expression);     \
        if (status != hipSuccess)                  \
        {                                          \
            std::cerr << "HIP error "              \
                      << status << ": "            \
                      << hipGetErrorString(status) \
                      << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;    \
            std::exit(EXIT_FAILURE);               \
        }                                          \
    }

constexpr int M           = 256;
constexpr int N           = 256;
constexpr int K           = 256;
constexpr int warp_size   = 64;
constexpr int block_size  = 256;
constexpr int WARMUP_RUNS = 3;
constexpr int TIMING_RUNS = 10;

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

// ---- CPU reference (sparse GEMM) ----

static void cpu_ref_sparse(const std::vector<float> &A, const std::vector<float> &B,
                           std::vector<float> &D, int M, int N, int K)
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
                    float a = static_cast<float>(static_cast<_Float16>(A[m * K + k]));
                    float b = static_cast<float>(static_cast<_Float16>(B[k * N + n]));
                    acc += a * b;
                }
            }
            D[m * N + n] = acc;
        }
    }
}

// ---- Verification ----

static bool verify_result(const std::vector<float> &ref, const float *d_output,
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
    return mismatches == 0;
}

// ---- Timing ----

template<typename KernelLaunch>
float time_kernel_ms(KernelLaunch launch, int warmup_runs, int timing_runs)
{
    for (int i = 0; i < warmup_runs; ++i)
    {
        launch();
        HIP_CHECK(hipDeviceSynchronize());
    }

    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    HIP_CHECK(hipEventRecord(ev_start));
    for (int i = 0; i < timing_runs; ++i)
        launch();
    HIP_CHECK(hipEventRecord(ev_stop));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    return elapsed_ms / static_cast<float>(timing_runs);
}

static void print_metrics(const char *label, bool passed, float avg_ms,
                           long long m, long long n, long long k)
{
    const double bytes_min =
        sizeof(float) * static_cast<double>(m * k + k * n + m * n);
    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n)
                         * static_cast<double>(k);

    std::cout << label << ": " << (passed ? "PASSED" : "FAILED") << "\n"
              << "  Average kernel time  : " << avg_ms << " ms\n"
              << "  Eff. bandwidth       : "
              << bytes_min / (avg_ms * 1.0e-3) / 1.0e9 << " GB/s\n"
              << "  Arithmetic throughput: "
              << flops / (avg_ms * 1.0e-3) / 1.0e12 << " TFLOPS\n\n";
}

// ============================================================================
// SparseTilePolicy -- shared-memory layout for sparse MFMA kernels
// ============================================================================

// [Sphinx sparse_tile_policy start]
template<typename ElemA_, typename ElemB_, int CtaM_, int CtaN_, int TileK_>
struct SparseTilePolicy
{
    using ElemA = ElemA_;
    using ElemB = ElemB_;
    static constexpr int cta_m   = CtaM_;
    static constexpr int cta_n   = CtaN_;
    static constexpr int tile_k  = TileK_;
    static constexpr int tk_comp = TileK_ / 2;

    struct SharedStorage
    {
        ElemA_ sA[CtaM_ * (TileK_ / 2)];
        ElemB_ sB[TileK_ * CtaN_];
    };
};
// [Sphinx sparse_tile_policy end]

// ============================================================================
// ComputePolicy structs -- FP16 sparse MFMA variants
// ============================================================================

// [Sphinx smfmac_cdna_16x16_f16_policy start]
struct SmfmacCdna16x16F16Policy
{
    static constexpr int thread_tile_m = 16;
    static constexpr int thread_tile_n = 16;

    using v4float = float [[clang::ext_vector_type(4)]];
    using Accumulator = v4float;

    using v4half = _Float16 [[clang::ext_vector_type(4)]];
    using v8half = _Float16 [[clang::ext_vector_type(8)]];
    using AFrag = v4half;
    using BFrag = v8half;

    __device__ static void zero(Accumulator &d) { d = {0.0f, 0.0f, 0.0f, 0.0f}; }

    __device__ static void load_a(const _Float16 *sA, int warp_m, int lane,
                                  int tk_comp, AFrag &a)
    {
        const int g = lane / 16;
        const int n = lane % 16;
        int a_off = (warp_m * 16 + n) * tk_comp + g * 4;
        for (int i = 0; i < 4; ++i)
        {
            a[i] = sA[a_off + i];
        }
    }

    __device__ static void load_b(const _Float16 *sB, int warp_n, int lane,
                                  int cta_n, BFrag &b)
    {
        const int g = lane / 16;
        const int n = lane % 16;
        int b_col = warp_n * 16 + n;
        for (int v = 0; v < 4; ++v)
        {
            int kk = g * 8 + v * 2;
            b[v * 2 + 0] = sB[kk * cta_n + b_col];
            b[v * 2 + 1] = sB[(kk + 1) * cta_n + b_col];
        }
    }

    __device__ static void mma(Accumulator &d, const AFrag &a, const BFrag &b)
    {
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || \
    defined(__gfx942__) || defined(__gfx950__)
        d = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, d, 0x88, 0, 0);
#endif
    }

    __device__ static void store_c(const Accumulator &d, float *D,
                                   int sub_m, int sub_n, int N, int lane)
    {
        const int g = lane / 16;
        const int n = lane % 16;
        for (int r = 0; r < 4; ++r)
        {
            D[(sub_m + g * 4 + r) * N + sub_n + n] = d[r];
        }
    }
};
// [Sphinx smfmac_cdna_16x16_f16_policy end]

// [Sphinx smfmac_cdna_32x32_f16_policy start]
struct SmfmacCdna32x32F16Policy
{
    static constexpr int thread_tile_m = 32;
    static constexpr int thread_tile_n = 32;

    using v16float = float [[clang::ext_vector_type(16)]];
    using Accumulator = v16float;

    using v4half = _Float16 [[clang::ext_vector_type(4)]];
    using v8half = _Float16 [[clang::ext_vector_type(8)]];
    using AFrag = v4half;
    using BFrag = v8half;

    __device__ static void zero(Accumulator &d) { d = {}; }

    __device__ static void load_a(const _Float16 *sA, int warp_m, int lane,
                                  int tk_comp, AFrag &a)
    {
        const int g = lane / 16;
        int a_row = warp_m * 32 + (g % 2) * 16 + (lane % 16);
        int a_col = (g / 2) * 4;
        for (int i = 0; i < 4; ++i)
        {
            a[i] = sA[a_row * tk_comp + a_col + i];
        }
    }

    __device__ static void load_b(const _Float16 *sB, int warp_n, int lane,
                                  int cta_n, BFrag &b)
    {
        const int g = lane / 16;
        int b_col = warp_n * 32 + (g % 2) * 16 + (lane % 16);
        int k_off = (g / 2) * 8;
        for (int v = 0; v < 4; ++v)
        {
            int kk = k_off + v * 2;
            b[v * 2 + 0] = sB[kk * cta_n + b_col];
            b[v * 2 + 1] = sB[(kk + 1) * cta_n + b_col];
        }
    }

    __device__ static void mma(Accumulator &d, const AFrag &a, const BFrag &b)
    {
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || \
    defined(__gfx942__) || defined(__gfx950__)
        d = __builtin_amdgcn_smfmac_f32_32x32x16_f16(a, b, d, 0x88, 0, 0);
#endif
    }

    __device__ static void store_c(const Accumulator &d, float *D,
                                   int sub_m, int sub_n, int N, int lane)
    {
        int j_base = lane % 32;
        int grp    = lane / 32;
        for (int r = 0; r < 16; ++r)
        {
            int i_out = (r / 4) * 8 + grp * 4 + (r % 4);
            D[(sub_m + i_out) * N + sub_n + j_base] = d[r];
        }
    }
};
// [Sphinx smfmac_cdna_32x32_f16_policy end]

// ============================================================================
// Generic sparse MFMA kernel template
// ============================================================================

// [Sphinx matrix_multiply_sparse start]
template<typename TilePolicy, typename ComputePolicy>
__global__ __launch_bounds__(block_size)
void matrix_multiply_sparse(const typename TilePolicy::ElemA *__restrict__ A_comp,
                       const typename TilePolicy::ElemB *__restrict__ B,
                       float *__restrict__ D, int M, int N, int K)
{
    constexpr int CTA_M   = TilePolicy::cta_m;
    constexpr int CTA_N   = TilePolicy::cta_n;
    constexpr int TILE_K  = TilePolicy::tile_k;
    constexpr int tk_comp = TilePolicy::tk_comp;

    const int tid    = threadIdx.x;
    const int warp_m = (tid / warp_size) / 2;
    const int warp_n = (tid / warp_size) % 2;
    const int lane   = tid % warp_size;
    const int cta_m  = blockIdx.y * CTA_M;
    const int cta_n  = blockIdx.x * CTA_N;
    const int K_comp = K / 2;

    __shared__ typename TilePolicy::SharedStorage smem;

    typename ComputePolicy::Accumulator d;
    ComputePolicy::zero(d);

    for (int kt = 0; kt < K / TILE_K; ++kt)
    {
        int k0 = kt * TILE_K;
        for (int i = tid; i < CTA_M * tk_comp; i += block_size)
        {
            int r = i / tk_comp, c = i % tk_comp;
            smem.sA[i] = A_comp[(cta_m + r) * K_comp + k0 / 2 + c];
        }
        for (int i = tid; i < TILE_K * CTA_N; i += block_size)
        {
            int r = i / CTA_N, c = i % CTA_N;
            smem.sB[i] = B[(k0 + r) * N + cta_n + c];
        }
        __syncthreads();

        typename ComputePolicy::AFrag a;
        typename ComputePolicy::BFrag b;
        ComputePolicy::load_a(smem.sA, warp_m, lane, tk_comp, a);
        ComputePolicy::load_b(smem.sB, warp_n, lane, CTA_N, b);
        ComputePolicy::mma(d, a, b);
        __syncthreads();
    }

    int sub_m = cta_m + warp_m * ComputePolicy::thread_tile_m;
    int sub_n = cta_n + warp_n * ComputePolicy::thread_tile_n;
    ComputePolicy::store_c(d, D, sub_m, sub_n, N, lane);
}
// [Sphinx matrix_multiply_sparse end]

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
    std::cout << "Problem: " << M << "x" << N << "x" << K << std::endl;

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::cout << "\nRunning FP16 sparse MFMA examples..." << std::endl;

    // smfmac_f32_16x16x32_f16
    {
        constexpr int CTA_M = 32, CTA_N = 32;
        dim3 grid(N / CTA_N, M / CTA_M);

        std::vector<float> A(M * K), B(K * N);
        for (float &v : A) { v = dist(gen); }
        for (float &v : B) { v = dist(gen); }

        std::vector<float> D_ref;
        cpu_ref_sparse(A, B, D_ref, M, N, K);

        std::vector<_Float16> A_comp;
        compress_A_f16(A, M, K, A_comp);
        std::vector<_Float16> B_f16(K * N);
        for (int i = 0; i < K * N; ++i) { B_f16[i] = static_cast<_Float16>(B[i]); }

        _Float16 *dA = nullptr, *dB = nullptr;
        float    *dD = nullptr;
        HIP_CHECK(hipMalloc(&dA, sizeof(_Float16) * A_comp.size()));
        HIP_CHECK(hipMalloc(&dB, sizeof(_Float16) * B_f16.size()));
        HIP_CHECK(hipMalloc(&dD, sizeof(float) * M * N));
        HIP_CHECK(hipMemcpy(dA, A_comp.data(), sizeof(_Float16) * A_comp.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, B_f16.data(), sizeof(_Float16) * B_f16.size(), hipMemcpyHostToDevice));

        auto launch_16x16 = [&]()
        {
            matrix_multiply_sparse<SparseTilePolicy<_Float16, _Float16, 32, 32, 32>,
                             SmfmacCdna16x16F16Policy>
                <<<grid, block_size>>>(dA, dB, dD, M, N, K);
            HIP_CHECK(hipGetLastError());
        };

        float ms_16 = time_kernel_ms(launch_16x16, WARMUP_RUNS, TIMING_RUNS);
        bool ok_16 = verify_result(D_ref, dD, M, N);
        print_metrics("smfmac_f32_16x16x32_f16", ok_16, ms_16,
                      M, N, K);

        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dD));
    }

    // smfmac_f32_32x32x16_f16
    {
        constexpr int CTA_M = 64, CTA_N = 64;
        dim3 grid(N / CTA_N, M / CTA_M);

        std::vector<float> A(M * K), B(K * N);
        for (float &v : A) { v = dist(gen); }
        for (float &v : B) { v = dist(gen); }

        std::vector<float> D_ref;
        cpu_ref_sparse(A, B, D_ref, M, N, K);

        std::vector<_Float16> A_comp;
        compress_A_f16(A, M, K, A_comp);
        std::vector<_Float16> B_f16(K * N);
        for (int i = 0; i < K * N; ++i) { B_f16[i] = static_cast<_Float16>(B[i]); }

        _Float16 *dA = nullptr, *dB = nullptr;
        float    *dD = nullptr;
        HIP_CHECK(hipMalloc(&dA, sizeof(_Float16) * A_comp.size()));
        HIP_CHECK(hipMalloc(&dB, sizeof(_Float16) * B_f16.size()));
        HIP_CHECK(hipMalloc(&dD, sizeof(float) * M * N));
        HIP_CHECK(hipMemcpy(dA, A_comp.data(), sizeof(_Float16) * A_comp.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, B_f16.data(), sizeof(_Float16) * B_f16.size(), hipMemcpyHostToDevice));

        auto launch_32x32 = [&]()
        {
            matrix_multiply_sparse<SparseTilePolicy<_Float16, _Float16, 64, 64, 16>,
                             SmfmacCdna32x32F16Policy>
                <<<grid, block_size>>>(dA, dB, dD, M, N, K);
            HIP_CHECK(hipGetLastError());
        };

        float ms_32 = time_kernel_ms(launch_32x32, WARMUP_RUNS, TIMING_RUNS);
        bool ok_32 = verify_result(D_ref, dD, M, N);
        print_metrics("smfmac_f32_32x32x16_f16", ok_32, ms_32,
                      M, N, K);

        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dD));
    }

    std::cout << "Execution completed successfully." << std::endl;
    return EXIT_SUCCESS;
}
