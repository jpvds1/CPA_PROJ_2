#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <papi.h>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include "helpers.h"
#include <utility>

#ifdef USE_SYCL
#include <CL/sycl.hpp>
using namespace cl::sycl;
#endif

void referenceMultiplication(const double *A, const double *B, double *C, int n)
{
    std::fill(C, C + n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
}

std::pair<double, double> sequentialVersion(const int n, const int blockSize, const double *A, const double *B, double *C)
{
    auto start = std::chrono::high_resolution_clock::now();
    long long energyBefore = readEnergy();

    for (int i = 0; i < n; i += blockSize)
    {
        int iMax = std::min(i + blockSize, n);
        for (int k = 0; k < n; k += blockSize)
        {
            int kMax = std::min(k + blockSize, n);
            for (int j = 0; j < n; j += blockSize)
            {
                int jMax = std::min(j + blockSize, n);

                for (int ii = i; ii < iMax; ++ii)
                {
                    for (int kk = k; kk < kMax; ++kk)
                    {
                        double a_val = A[ii * n + kk];
// Use compiler vectorization hints
#pragma omp simd aligned(C, B : 64)
                        for (int jj = j; jj < jMax; ++jj)
                        {
                            C[ii * n + jj] += a_val * B[kk * n + jj];
                        }
                    }
                }
            }
        }
    }

    long long energyAfter = readEnergy();
    auto end = std::chrono::high_resolution_clock::now();

    double timeTaken = std::chrono::duration<double, std::milli>(end - start).count();
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    return {timeTaken, energyConsumed};
}

#ifdef USE_OMP
std::pair<double, double> parallelVersion(const int n, const int cores, const int blockSize, const double *A, const double *B, double *C)
{
    omp_set_dynamic(0);
    omp_set_num_threads(cores);

    auto start = std::chrono::high_resolution_clock::now();
    long long energyBefore = readEnergy();

#pragma omp parallel for collapse(3) schedule(static)
    for (int i = 0; i < n; i += blockSize)
        for (int j = 0; j < n; j += blockSize)
            for (int k = 0; k < n; k += blockSize)
            {
                int iMax = std::min(i + blockSize, n);
                int jMax = std::min(j + blockSize, n);
                int kMax = std::min(k + blockSize, n);

                for (int ii = i; ii < iMax; ++ii)
                    for (int kk = k; kk < kMax; ++kk)
                    {
                        double a_val = A[ii * n + kk];
                        for (int jj = j; jj < jMax; ++jj)
                            C[ii * n + jj] += a_val * B[kk * n + jj];
                    }
            }

    long long energyAfter = readEnergy();
    auto end = std::chrono::high_resolution_clock::now();

    double timeTaken = std::chrono::duration<double, std::milli>(end - start).count();
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    return {timeTaken, energyConsumed};
}
#endif

#ifdef USE_SYCL
std::pair<double, double> SYCLVersion_GPU(const int n, const int blockSize, cl::sycl::queue &q, const double *A, const double *B, double *C)
{
    double *A_dev = cl::sycl::malloc_device<double>(n * n, q);
    double *B_dev = cl::sycl::malloc_device<double>(n * n, q);
    double *C_dev = cl::sycl::malloc_device<double>(n * n, q);

    q.memcpy(A_dev, A, n * n * sizeof(double)).wait();
    q.memcpy(B_dev, B, n * n * sizeof(double)).wait();

    auto start = std::chrono::high_resolution_clock::now();
    long long powerBefore = readGpuPowerMilliwatts();

    size_t globalSize = (n + blockSize - 1) / blockSize * blockSize;

    q.submit([&](handler &h)
             {
        local_accessor<double, 2> A_tile({static_cast<size_t>(blockSize), static_cast<size_t>(blockSize)}, h);
        local_accessor<double, 2> B_tile({static_cast<size_t>(blockSize), static_cast<size_t>(blockSize)}, h);

        h.parallel_for(nd_range<2>({globalSize, globalSize}, {static_cast<size_t>(blockSize), static_cast<size_t>(blockSize)}),
            [=](nd_item<2> item) {
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                int li = item.get_local_id(0);
                int lj = item.get_local_id(1);

                double sum = 0.0;

                for (int t = 0; t < n; t += blockSize) {
                    if (row < n && t + lj < n)
                        A_tile[li][lj] = A_dev[row * n + t + lj];
                    else
                        A_tile[li][lj] = 0.0;

                    if (t + li < n && col < n)
                        B_tile[li][lj] = B_dev[(t + li) * n + col];
                    else
                        B_tile[li][lj] = 0.0;

                    item.barrier(access::fence_space::local_space);

                    for (int k = 0; k < blockSize; ++k)
                        sum += A_tile[li][k] * B_tile[k][lj];

                    item.barrier(access::fence_space::local_space);
                }

                if (row < n && col < n)
                    C_dev[row * n + col] = sum;
            }); });

    q.wait();

    long long powerAfter = readGpuPowerMilliwatts();
    auto end = std::chrono::high_resolution_clock::now();

    q.memcpy(C, C_dev, n * n * sizeof(double)).wait();

    double timeTaken = std::chrono::duration<double, std::milli>(end - start).count();
    double avgPowerMw = (powerBefore + powerAfter) / 2.0;
    double energyConsumed = (avgPowerMw * timeTaken) / 1e6;

    cl::sycl::free(A_dev, q);
    cl::sycl::free(B_dev, q);
    cl::sycl::free(C_dev, q);

    return {timeTaken, energyConsumed};
}

std::pair<double, double> SYCLVersion_CPU(const int n, const int blockSize, cl::sycl::queue &q, const double *A, const double *B, double *C)
{
    // Use host-allocated memory on CPU for better cache locality
    double *A_dev = cl::sycl::malloc_host<double>(n * n, q);
    double *B_dev = cl::sycl::malloc_host<double>(n * n, q);
    double *C_dev = cl::sycl::malloc_host<double>(n * n, q);

    std::memcpy(A_dev, A, n * n * sizeof(double));
    std::memcpy(B_dev, B, n * n * sizeof(double));
    std::memset(C_dev, 0, n * n * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();
    long long energyBefore = readEnergy();

    size_t numBlocks = (n + blockSize - 1) / blockSize;

    // Parallel over output blocks
    q.submit([&](cl::sycl::handler &h)
             { h.parallel_for(cl::sycl::range<2>(numBlocks, numBlocks), [=](cl::sycl::id<2> blockIdx)
                              {
            int bi = blockIdx[0];
            int bj = blockIdx[1];

            int iStart = bi * blockSize;
            int jStart = bj * blockSize;

            for (int bk = 0; bk < numBlocks; ++bk)
            {
                int kStart = bk * blockSize;

                for (int i = iStart; i < std::min(iStart + blockSize, n); ++i)
                {
                    for (int k = kStart; k < std::min(kStart + blockSize, n); ++k)
                    {
                        double a_val = A_dev[i * n + k];

                        for (int j = jStart; j < std::min(jStart + blockSize, n); ++j)
                        {
                            C_dev[i * n + j] += a_val * B_dev[k * n + j];
                        }
                    }
                }
            } }); })
        .wait();

    long long energyAfter = readEnergy();
    auto end = std::chrono::high_resolution_clock::now();

    std::memcpy(C, C_dev, n * n * sizeof(double));

    cl::sycl::free(A_dev, q);
    cl::sycl::free(B_dev, q);
    cl::sycl::free(C_dev, q);

    double timeTaken = std::chrono::duration<double, std::milli>(end - start).count();
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    return {timeTaken, energyConsumed};
}

std::pair<double, double> SYCLVersion(const int n, const int blockSize, cl::sycl::queue &q, const double *A, const double *B, double *C)
{
    if (q.get_device().is_gpu())
        return SYCLVersion_GPU(n, blockSize, q, A, B, C);
    else
        return SYCLVersion_CPU(n, blockSize, q, A, B, C);
}
#endif

int main()
{
    std::pair<double, double> results;
    std::vector<int> sizes;
    for (int i = 1024; i <= 8192; i += 1024)
        sizes.push_back(i);

    // Build reference matrix using minimal size
    double *A_ref, *B_ref, *C_ref, *A, *B, *C_out;
    setupArrays(&A_ref, &B_ref, &C_ref, sizes[0]);
    referenceMultiplication(A_ref, B_ref, C_ref, sizes[0]);

    settings s = getSettings(); // <- interactive selection restored

    int EventSet = PAPI_NULL;
    long long values[7];

    if (setupPAPI(EventSet) != PAPI_OK)
    {
        std::cerr << "PAPI setup failed!" << std::endl;
        return 1;
    }

    switch (s.algorithmChoice)
    {
    case 0: // Sequential
        for (int size : sizes)
        {
            setupArrays(&A, &B, &C_out, size);
            std::cout << "Running Sequential for size: " << size << std::endl;

            PAPI_start(EventSet);
            results = sequentialVersion(size, s.blockSize, A, B, C_out);
            PAPI_stop(EventSet, values);

            saveResult("Sequential", size, s.cores, s.blockSize, results.first, results.second, values);

            if (size == sizes[0])
            {
                if (!matricesEqual(C_out, C_ref, size, 1e-10))
                    std::cerr << "Sequential result does not match reference!" << std::endl;
                else
                    std::cout << "Sequential result matches reference.\n";
            }

            freeArrays(A, B, C_out);
        }
        break;

    case 1: // OpenMP
#ifdef USE_OMP
        for (int size : sizes)
        {
            setupArrays(&A, &B, &C_out, size);
            std::cout << "Running Parallel (OpenMP) for size: " << size << std::endl;

            PAPI_start(EventSet);
            results = parallelVersion(size, s.cores, s.blockSize, A, B, C_out);
            PAPI_stop(EventSet, values);

            saveResult("Parallel", size, s.cores, s.blockSize, results.first, results.second, values);

            if (size == sizes[0])
            {
                if (!matricesEqual(C_out, C_ref, size))
                    std::cerr << "Parallel result does not match reference!" << std::endl;
                else
                    std::cout << "Parallel result matches reference.\n";
            }

            freeArrays(A, B, C_out);
        }
#else
        std::cerr << "OpenMP support is not enabled in this build.\n";
#endif
        break;

    case 2: // SYCL
#ifdef USE_SYCL
        try
        {
            auto platforms = cl::sycl::platform::get_platforms();
            if (platforms.empty())
            {
                std::cerr << "No SYCL platforms found.\n";
                break;
            }

            auto platform = platforms.at(s.platformIndex);
            auto device = platform.get_devices().at(s.deviceIndex);
            cl::sycl::queue q(device);

            std::string label = device.is_gpu() ? "SYCL_GPU" : "SYCL_CPU";

            for (int size : sizes)
            {
                setupArrays(&A, &B, &C_out, size);
                std::cout << "Running SYCL for size: " << size
                          << " on: " << device.get_info<cl::sycl::info::device::name>() << std::endl;

                PAPI_start(EventSet);
                results = SYCLVersion(size, s.blockSize, q, A, B, C_out);
                PAPI_stop(EventSet, values);

                saveResult(label, size, s.cores, s.blockSize, results.first, results.second, values);

                if (size == sizes[0])
                {
                    if (!matricesEqual(C_out, C_ref, size))
                        std::cerr << "SYCL result does not match reference!" << std::endl;
                    else
                        std::cout << "SYCL result matches reference.\n";
                }

                freeArrays(A, B, C_out);
            }
        }
        catch (std::exception &e)
        {
            std::cerr << "SYCL error: " << e.what() << std::endl;
        }
#else
        std::cerr << "SYCL support is not enabled in this build.\n";
#endif
        break;

    default:
        std::cerr << "Invalid algorithm selection.\n";
        break;
    }

    cleanupPAPI(EventSet);
    freeArrays(A_ref, B_ref, C_ref);

    std::cout << "All tests completed successfully.\n";

    return 0;
}
