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
// matrix_multiply_cdna3_sparse_mfma.hip
//
// FP8 sparse MFMA (SMFMAC) intrinsic for CDNA3 GPUs.
// Demonstrates smfmac_f32_16x16x64_fp8_fp8 using a policy-based kernel
// template.
//
// Architecture: gfx942, gfx950.
// Compile with: hipcc -O3 -std=c++17 --offload-arch=gfx942 \
//                   matrix_multiply_cdna3_sparse_mfma.hip -o sparse_mfma_cdna3
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>

#include <hip/hip_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
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

// ---- FP8 E4M3 helpers (via hip/hip_fp8.h) ----

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
                    acc += A[m * K + k] * B[k * N + n];
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
// SparseTilePolicy
// ============================================================================

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

// ============================================================================
// ComputePolicy -- FP8 sparse MFMA (K=64, CDNA3)
// ============================================================================

// [Sphinx smfmac_cdna3_fp8_policy start]
struct SmfmacCdna3FP8Policy
{
    static constexpr int thread_tile_m = 16;
    static constexpr int thread_tile_n = 16;

    using v4float = float [[clang::ext_vector_type(4)]];
    using Accumulator = v4float;

    using v2int = int [[clang::ext_vector_type(2)]];
    using v4int = int [[clang::ext_vector_type(4)]];
    using AFrag = v2int;
    using BFrag = v4int;

    __device__ static void zero(Accumulator &d) { d = {0.0f, 0.0f, 0.0f, 0.0f}; }

    __device__ static void load_a(const uint8_t *sA, int warp_m, int lane,
                                  int tk_comp, AFrag &a)
    {
        const int g = lane / 16;
        const int n = lane % 16;
        int a_off = (warp_m * 16 + n) * tk_comp + g * 8;
        for (int j = 0; j < 2; ++j)
        {
            a[j] = static_cast<int>(
                uint32_t(sA[a_off + j*4 + 0])       | (uint32_t(sA[a_off + j*4 + 1]) << 8) |
                (uint32_t(sA[a_off + j*4 + 2]) << 16) | (uint32_t(sA[a_off + j*4 + 3]) << 24));
        }
    }

    __device__ static void load_b(const uint8_t *sB, int warp_n, int lane,
                                  int cta_n, BFrag &b)
    {
        const int g = lane / 16;
        const int n = lane % 16;
        int b_col = warp_n * 16 + n;
        for (int v = 0; v < 4; ++v)
        {
            int kk = g * 16 + v * 4;
            b[v] = static_cast<int>(
                uint32_t(sB[kk * cta_n + b_col])       | (uint32_t(sB[(kk+1) * cta_n + b_col]) << 8) |
                (uint32_t(sB[(kk+2) * cta_n + b_col]) << 16) | (uint32_t(sB[(kk+3) * cta_n + b_col]) << 24));
        }
    }

    __device__ static void mma(Accumulator &d, const AFrag &a, const BFrag &b)
    {
#if defined(__gfx942__) || defined(__gfx950__)
        d = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, d, 0x8888, 0, 0);
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
// [Sphinx smfmac_cdna3_fp8_policy end]

// ============================================================================
// Generic sparse MFMA kernel template
// ============================================================================

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
    std::uniform_int_distribution<int> fp8_dist(-2, 2);

    std::cout << "\nRunning FP8 sparse MFMA example (K=64, CDNA3)..." << std::endl;

    // smfmac_f32_16x16x64_fp8_fp8
    {
        constexpr int CTA_M = 32, CTA_N = 32;
        dim3 grid(N / CTA_N, M / CTA_M);

        std::vector<float> A(M * K), B(K * N);
        for (float &v : A) { v = static_cast<float>(fp8_dist(gen)); }
        for (float &v : B) { v = static_cast<float>(fp8_dist(gen)); }
        for (float &v : A) { v = fp8_to_float(float_to_fp8(v)); }
        for (float &v : B) { v = fp8_to_float(float_to_fp8(v)); }

        std::vector<float> D_ref;
        cpu_ref_sparse(A, B, D_ref, M, N, K);

        std::vector<uint8_t> A_comp;
        compress_A_fp8(A, M, K, A_comp);
        std::vector<uint8_t> B_fp8(K * N);
        for (int i = 0; i < K * N; ++i) { B_fp8[i] = float_to_fp8(B[i]); }

        uint8_t *dA = nullptr, *dB = nullptr;
        float   *dD = nullptr;
        HIP_CHECK(hipMalloc(&dA, A_comp.size()));
        HIP_CHECK(hipMalloc(&dB, B_fp8.size()));
        HIP_CHECK(hipMalloc(&dD, sizeof(float) * M * N));
        HIP_CHECK(hipMemcpy(dA, A_comp.data(), A_comp.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, B_fp8.data(), B_fp8.size(), hipMemcpyHostToDevice));

        auto launch = [&]()
        {
            matrix_multiply_sparse<SparseTilePolicy<uint8_t, uint8_t, 32, 32, 64>,
                             SmfmacCdna3FP8Policy>
                <<<grid, block_size>>>(dA, dB, dD, M, N, K);
            HIP_CHECK(hipGetLastError());
        };

        float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        bool ok = verify_result(D_ref, dD, M, N, 0.5f, 0.5f);
        print_metrics("smfmac_f32_16x16x64_fp8_fp8", ok, ms,
                      M, N, K);

        HIP_CHECK(hipFree(dA));
        HIP_CHECK(hipFree(dB));
        HIP_CHECK(hipFree(dD));
    }

    std::cout << "Execution completed successfully." << std::endl;
    return EXIT_SUCCESS;
}
