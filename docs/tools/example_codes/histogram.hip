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

#ifndef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 16
#endif

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

// [Sphinx histogram shared kernel start]
__global__ void histogram_shared(const unsigned int *input,
                                 unsigned int *histogram,
                                 int n)
{
    __shared__ unsigned int shared_hist[num_bins];

    int tid = threadIdx.x;
    int base = (blockIdx.x * blockDim.x * ITEMS_PER_THREAD) + tid;

    // 1. Initialize shared histogram
    for (int i = tid; i < num_bins; i += blockDim.x)
    {
        shared_hist[i] = 0;
    }
    __syncthreads();

    // 2. Process multiple items per thread (grid-stride style)
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int idx = base + i * blockDim.x;

        if (idx < n)
        {
            unsigned int bin = input[idx] % num_bins;
            atomicAdd(&shared_hist[bin], 1u);
        }
    }

    __syncthreads();

    // 3. Merge into global histogram
    for (int i = tid; i < num_bins; i += blockDim.x)
    {
        atomicAdd(&histogram[i], shared_hist[i]);
    }
}
// [Sphinx histogram shared kernel end]

// [Sphinx histogram partial kernel start]
__global__ void histogram_partial(const unsigned int *input,
                                  unsigned int *partial_histogram,
                                  int n)
{
    __shared__ unsigned int shared_hist[num_bins];

    int tid  = threadIdx.x;
    int base = (blockIdx.x * blockDim.x * ITEMS_PER_THREAD) + tid;

    for (int i = tid; i < num_bins; i += blockDim.x)
    {
        shared_hist[i] = 0;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int idx = base + i * blockDim.x;
        if (idx < n)
        {
            unsigned int bin = input[idx] % num_bins;
            atomicAdd(&shared_hist[bin], 1u);
        }
    }
    __syncthreads();

    unsigned int *block_hist = partial_histogram + blockIdx.x * num_bins;
    for (int i = tid; i < num_bins; i += blockDim.x)
    {
        block_hist[i] = shared_hist[i];
    }
}
// [Sphinx histogram partial kernel end]

// [Sphinx histogram reduce kernel start]
__global__ void histogram_reduce(const unsigned int *partial_histogram,
                                 unsigned int *histogram,
                                 int num_partial)
{
    int bin = blockIdx.x;
    int tid = threadIdx.x;

    if (bin >= num_bins)
        return;

    unsigned int sum = 0;

    // Each thread accumulates a chunk
    for (int i = tid; i < num_partial; i += blockDim.x)
    {
        sum += partial_histogram[i * num_bins + bin];
    }

    // Shared memory reduction
    __shared__ unsigned int sdata[block_size];
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write result
    if (tid == 0)
        histogram[bin] = sdata[0];
}
// [Sphinx histogram reduce kernel end]

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
    HIP_CHECK(hipEventRecord(start));
    for (int run = 0; run < timing_runs; ++run)
    {
        launch();
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    std::cout << "    [timer] " << label << ": avg " << total_ms / timing_runs
              << " ms over " << timing_runs << " runs" << std::endl;
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}
#endif

int main()
{
    std::vector<unsigned int> h_input(input_size, 0);
    std::mt19937 gen(42);
    std::uniform_int_distribution<unsigned int> dist(0, num_bins - 1);
    for (unsigned long long i = 0; i < input_size; i++)
    {
        unsigned int &v  = h_input[i];
        if( i % 4 )
          v = dist(gen);
        else
          v = 1;
    }

    std::vector<unsigned int> ref = cpu_reference(h_input);

    int nb     = (input_size + block_size - 1) / block_size;
    int nb_ipt = (input_size + block_size * ITEMS_PER_THREAD - 1)
                 / (block_size * ITEMS_PER_THREAD);

    unsigned int *d_input     = nullptr;
    unsigned int *d_histogram = nullptr;

    HIP_CHECK(hipMalloc(&d_input,     sizeof(unsigned int) * input_size));
    HIP_CHECK(hipMalloc(&d_histogram, sizeof(unsigned int) * num_bins));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), sizeof(unsigned int) * input_size,
                        hipMemcpyHostToDevice));

    std::cout << "Running histogram kernels..." << std::endl;

    // Naive kernel
    HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
    histogram_naive<<<nb, block_size>>>(d_input, d_histogram, input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Naive: ";
    verify(ref, d_histogram);
#ifdef ENABLE_TIMERS
    time_kernel("naive", [&]()
    {
        HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
        histogram_naive<<<nb, block_size>>>(d_input, d_histogram, input_size);
    });
#endif

    // shared memory kernel
    HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
    histogram_shared<<<nb_ipt, block_size>>>(d_input, d_histogram, input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Shared memory: ";
    verify(ref, d_histogram);
#ifdef ENABLE_TIMERS
    time_kernel("shared memory", [&]()
    {
        HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
        histogram_shared<<<nb_ipt, block_size>>>(d_input, d_histogram, input_size);
    });
#endif

    // Partial histograms + reduction
    unsigned int *d_partial = nullptr;
    HIP_CHECK(hipMalloc(&d_partial, sizeof(unsigned int) * nb_ipt * num_bins));

    HIP_CHECK(hipMemset(d_histogram, 0, sizeof(unsigned int) * num_bins));
    histogram_partial<<<nb_ipt, block_size>>>(d_input, d_partial, input_size);
    HIP_CHECK(hipGetLastError());
    histogram_reduce<<<num_bins, block_size>>>(d_partial, d_histogram, nb_ipt);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Partial + reduce: ";
    verify(ref, d_histogram);
#ifdef ENABLE_TIMERS
    time_kernel("partial + reduce", [&]()
    {
        histogram_partial<<<nb_ipt, block_size>>>(d_input, d_partial, input_size);
        histogram_reduce<<<num_bins, block_size>>>(d_partial, d_histogram, nb_ipt);
    });
#endif

    std::cout << "Execution completed successfully." << std::endl;

    HIP_CHECK(hipFree(d_partial));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_histogram));

    return EXIT_SUCCESS;
}
