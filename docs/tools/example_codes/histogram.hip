// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

constexpr int input_size  = 1 << 24; // 16 M elements
constexpr int num_bins    = 256;
constexpr int block_size  = 256;
constexpr int timing_runs = 100;

// [Sphinx histogram naive kernel start]
__global__ void histogram_naive(const unsigned int *input, unsigned int *histogram, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n)
    {
        unsigned int bin = input[gid] % num_bins;
        atomicAdd(&histogram[bin], 1u);
    }
}
// [Sphinx histogram naive kernel end]

// [Sphinx histogram local kernel start]
__global__ void histogram_local(const unsigned int *input, unsigned int *histogram, int n)
{
    __shared__ unsigned int local_hist[num_bins];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Initialize the per-block histogram to zero.
    for (int i = tid; i < num_bins; i += blockDim.x)
    {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Accumulate this block's elements into the local histogram.
    if (gid < n)
    {
        unsigned int bin = input[gid] % num_bins;
        atomicAdd(&local_hist[bin], 1u);
    }
    __syncthreads();

    // Merge the local histogram into the global histogram.
    for (int i = tid; i < num_bins; i += blockDim.x)
    {
        atomicAdd(&histogram[i], local_hist[i]);
    }
}
// [Sphinx histogram local kernel end]

// [Sphinx histogram multi kernel start]
__global__ void histogram_multi(const unsigned int *input, unsigned int *histogram, int n)
{
    __shared__ unsigned int local_hist[num_bins];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 4) + tid;

    for (int i = tid; i < num_bins; i += blockDim.x)
    {
        local_hist[i] = 0;
    }
    __syncthreads();

    if (gid + 3 * blockDim.x < n)
    {
        uint4 v = reinterpret_cast<const uint4 *>(input)[blockIdx.x * blockDim.x + tid];
        atomicAdd(&local_hist[v.x % num_bins], 1u);
        atomicAdd(&local_hist[v.y % num_bins], 1u);
        atomicAdd(&local_hist[v.z % num_bins], 1u);
        atomicAdd(&local_hist[v.w % num_bins], 1u);
    }
    else
    {
        if (gid                  < n) { atomicAdd(&local_hist[input[gid                 ] % num_bins], 1u); }
        if (gid +     blockDim.x < n) { atomicAdd(&local_hist[input[gid +     blockDim.x] % num_bins], 1u); }
        if (gid + 2 * blockDim.x < n) { atomicAdd(&local_hist[input[gid + 2 * blockDim.x] % num_bins], 1u); }
        if (gid + 3 * blockDim.x < n) { atomicAdd(&local_hist[input[gid + 3 * blockDim.x] % num_bins], 1u); }
    }
    __syncthreads();

    for (int i = tid; i < num_bins; i += blockDim.x)
    {
        atomicAdd(&histogram[i], local_hist[i]);
    }
}
// [Sphinx histogram multi kernel end]

std::vector<unsigned int> cpu_reference(const std::vector<unsigned int> &input)
{
    std::vector<unsigned int> hist(num_bins, 0u);
    for (unsigned int v : input)
    {
        hist[v % num_bins]++;
    }
    return hist;
}

void verify(const std::vector<unsigned int> &ref, const unsigned int *d_histogram)
{
    std::vector<unsigned int> h_histogram(num_bins);
    HIP_CHECK(hipMemcpy(h_histogram.data(), d_histogram, sizeof(unsigned int) * num_bins,
                        hipMemcpyDeviceToHost));

    for (int i = 0; i < num_bins; ++i)
    {
        if (h_histogram[i] != ref[i])
        {
            std::cout << "FAILED (bin " << i
                      << ": got " << h_histogram[i]
                      << ", expected " << ref[i] << ")" << std::endl;
            return;
        }
    }
    std::cout << "PASSED" << std::endl;
}

#ifdef ENABLE_TIMERS
void time_kernel(const std::string &label, const std::function<void()> &launch)
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
    std::cout << "    [timer] " << label << ": avg " << total_ms / timing_runs
              << " ms over " << timing_runs << " runs" << std::endl;
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}
#endif

int main()
{
    std::vector<unsigned int> h_input(input_size);
    std::mt19937 gen(42);
    std::uniform_int_distribution<unsigned int> dist(0, num_bins - 1);
    for (unsigned int &v : h_input)
    {
        v = dist(gen);
    }

    std::vector<unsigned int> ref = cpu_reference(h_input);

    int nb1 = (input_size + block_size - 1) / block_size;
    int nb4 = (input_size + block_size * 4 - 1) / (block_size * 4);

    unsigned int *d_input     = nullptr;
    unsigned int *d_histogram = nullptr;

    HIP_CHECK(hipMalloc(&d_input,     sizeof(unsigned int) * input_size));
    HIP_CHECK(hipMalloc(&d_histogram, sizeof(unsigned int) * num_bins));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), sizeof(unsigned int) * input_size,
                        hipMemcpyHostToDevice));

    std::cout << "Running histogram kernels..." << std::endl;

    // Naive kernel
    HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
    histogram_naive<<<nb1, block_size>>>(d_input, d_histogram, input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Naive: ";
    verify(ref, d_histogram);
#ifdef ENABLE_TIMERS
    time_kernel("naive", [&]()
    {
        HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
        histogram_naive<<<nb1, block_size>>>(d_input, d_histogram, input_size);
    });
#endif

    // Local memory kernel
    HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
    histogram_local<<<nb1, block_size>>>(d_input, d_histogram, input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Local memory: ";
    verify(ref, d_histogram);
#ifdef ENABLE_TIMERS
    time_kernel("local memory", [&]()
    {
        HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
        histogram_local<<<nb1, block_size>>>(d_input, d_histogram, input_size);
    });
#endif

    // Multi-element (uint4) kernel
    HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
    histogram_multi<<<nb4, block_size>>>(d_input, d_histogram, input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Vectorized loads: ";
    verify(ref, d_histogram);
#ifdef ENABLE_TIMERS
    time_kernel("vectorized loads", [&]()
    {
        HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
        histogram_multi<<<nb4, block_size>>>(d_input, d_histogram, input_size);
    });
#endif

    std::cout << "Execution completed successfully." << std::endl;

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_histogram));

    return EXIT_SUCCESS;
}
