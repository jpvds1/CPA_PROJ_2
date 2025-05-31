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

std::pair<double, double> sequentialVersion(const int n, const int blockSize, const double *A, const double *B, double* C)
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
std::pair<double, double> parallelVersion(const int n, const int cores, const int blockSize, const double *A, const double *B, double* C)
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
std::pair<double, double> SYCLVersion(const int n, const int blockSize, cl::sycl::queue &q, const double *A, const double *B, double* C)
{
    double *A_dev = cl::sycl::malloc_device<double>(n * n, q);
    double *B_dev = cl::sycl::malloc_device<double>(n * n, q);
    double *C_dev = cl::sycl::malloc_device<double>(n * n, q);

    q.memcpy(A_dev, A, n * n * sizeof(double)).wait();
    q.memcpy(B_dev, B, n * n * sizeof(double)).wait();

    auto start = std::chrono::high_resolution_clock::now();
    long long energyBefore = readEnergy();

    // Launch 2D parallel_for over output tiles
    size_t numBlocks = (n + blockSize - 1) / blockSize;

    q.submit([&](auto &h)
             { h.parallel_for(cl::sycl::range<2>(numBlocks, numBlocks), [=](cl::sycl::id<2> blockIdx)
                              {
            int bi = blockIdx[0]; // block row
            int bj = blockIdx[1]; // block col

            int rowStart = bi * blockSize;
            int colStart = bj * blockSize;

            for (int i = rowStart; i < std::min(rowStart + blockSize, n); ++i) {
                for (int j = colStart; j < std::min(colStart + blockSize, n); ++j) {
                    double sum = 0.0;
                    for (int kBlock = 0; kBlock < numBlocks; ++kBlock) {
                        int kStart = kBlock * blockSize;
                        for (int k = kStart; k < std::min(kStart + blockSize, n); ++k) {
                            sum += A_dev[i * n + k] * B_dev[k * n + j];
                        }
                    }
                    C_dev[i * n + j] = sum;
                }
            } }); });

    q.wait();

    long long energyAfter = readEnergy();
    auto end = std::chrono::high_resolution_clock::now();

    q.memcpy(C, C_dev, n * n * sizeof(double)).wait();

    double timeTaken = std::chrono::duration<double, std::milli>(end - start).count();
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    cl::sycl::free(A_dev, q);
    cl::sycl::free(B_dev, q);
    cl::sycl::free(C_dev, q);

    return {timeTaken, energyConsumed};
}
#endif

int main()
{
    settings s = getSettings();
    std::pair<double, double> results;
    std::vector<int> sizes;
    for (int i = 1024; i <= 8192; i += 1024)
        sizes.push_back(i);

    double *A_ref, *B_ref, *C_ref, *A, *B, *C_out;
    setupArrays(&A_ref, &B_ref, &C_ref, sizes[0]);

    referenceMultiplication(A_ref, B_ref, C_ref, sizes[0]);

    int EventSet = PAPI_NULL;
    long long values[7];

    if (setupPAPI(EventSet) != PAPI_OK)
    {
        std::cerr << "PAPI setup failed!" << std::endl;
        return 1;
    }

#ifdef USE_OMP
    for (int size : sizes)
    {
        setupArrays(&A, &B, &C_out, size);
        std::cout << "Running Sequential for size: " << size << std::endl;

        PAPI_start(EventSet);
        results = sequentialVersion(size, 128, A, B, C_out);
        PAPI_stop(EventSet, values);

        saveResult("Sequential", size, 16, 128, results.first, results.second, values);

        if (size == sizes[0])
        {
            if (!matricesEqual(C_out, C_ref, size))
            {
                std::cerr << "Error: Sequential result does not match reference!" << std::endl;
                cleanupPAPI(EventSet);
                return 1;
            }
        }

        freeArrays(A, B, C_out);
    }

    for (int size : sizes)
    {
        setupArrays(&A, &B, &C_out, size);
        std::cout << "Running Parallel for size: " << size << std::endl;

        PAPI_start(EventSet);
        results = parallelVersion(size, 16, 128, A, B, C_out);
        PAPI_stop(EventSet, values);

        saveResult("Parallel", size, 16, 128, results.first, results.second, values);

        if (size == sizes[0])
        {
            if (!matricesEqual(C_out, C_ref, size))
            {
                std::cerr << "Error: Parallel result does not match reference!" << std::endl;
                cleanupPAPI(EventSet);
                return 1;
            }
        }
        freeArrays(A, B, C_out);
    }
#endif

#ifdef USE_SYCL
    // Run SYCL (once per platform/device)
    try
    {
        std::vector<cl::sycl::platform> platforms = cl::sycl::platform::get_platforms();
        if (platforms.empty())
        {
            std::cerr << "No SYCL platforms found. Skipping SYCL version.\n";
        }
        else
        {
            cl::sycl::platform platform = platforms.at(s.platformIndex);
            cl::sycl::device device = platform.get_devices().at(s.deviceIndex);
            cl::sycl::queue q(device);

            for (int size : sizes)
            {
                setupArrays(&A, &B, &C_out, size);
                std::string label = device.is_gpu() ? "SYCL_GPU" : "SYCL_CPU";
                std::cout << "Running SYCL for size: " << size
                          << " on: " << device.get_info<cl::sycl::info::device::name>() << std::endl;

                PAPI_start(EventSet);
                results = SYCLVersion(size, 16, q, A, B, C_out);
                PAPI_stop(EventSet, values);

                saveResult(label, size, 16, 16, results.first, results.second, values);

                if (size == sizes[0])
                {
                    if (!matricesEqual(C_out, C_ref, size))
                    {
                        std::cerr << "Error: SYCL result does not match reference!" << std::endl;
                        cleanupPAPI(EventSet);
                        return 1;
                    }
                }

                freeArrays(A, B, C_out);
            }
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "SYCL error: " << e.what() << std::endl;
    }
#endif

    cleanupPAPI(EventSet);
    freeArrays(A_ref, B_ref, C_ref);
    freeArrays(A, B, C_out);

    while (true)
    {
        std::cout << "\a" << std::flush;
        sleep(1);
    }

    return 0;
}
