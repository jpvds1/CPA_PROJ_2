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
#include <CL/sycl.hpp>

using namespace cl::sycl;

std::pair<double, double> sequentialVersion(const int n, const int blockSize)
{
    double *A, *B, *C;
    setupArrays(&A, &B, &C, n);

    auto start = std::chrono::high_resolution_clock::now();
    long long energyBefore = readEnergy();

    for (int i = 0; i < n; i += blockSize) {
        int iMax = std::min(i + blockSize, n);
        for (int k = 0; k < n; k += blockSize) {
            int kMax = std::min(k + blockSize, n);
            for (int j = 0; j < n; j += blockSize) {
                int jMax = std::min(j + blockSize, n);

                for (int ii = i; ii < iMax; ++ii) {
                    for (int kk = k; kk < kMax; ++kk) {
                        double a_val = A[ii * n + kk];
                        // Use compiler vectorization hints
                        #pragma omp simd aligned(C, B: 64)
                        for (int jj = j; jj < jMax; ++jj) {
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

    freeArrays(A, B, C);
    return {timeTaken, energyConsumed};
}

std::pair<double, double> parallelVersion(const int n, const int cores, const int blockSize)
{
    double *A, *B, *C;
    setupArrays(&A, &B, &C, n);
    omp_set_num_threads(cores);

    auto start = std::chrono::high_resolution_clock::now();
    long long energyBefore = readEnergy();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            for (int k = 0; k < n; k += blockSize) {

                int iEnd = std::min(i + blockSize, n);
                int jEnd = std::min(j + blockSize, n);
                int kEnd = std::min(k + blockSize, n);

                for (int ii = i; ii < iEnd; ++ii) {
                    for (int jj = j; jj < jEnd; ++jj) {
                        double sum = 0.0;
                        for (int kk = k; kk < kEnd; ++kk)
                            sum += A[ii * n + kk] * B[kk * n + jj];
                        #pragma omp atomic
                        C[ii * n + jj] += sum;
                    }
                }
            }
        }
    }

    long long energyAfter = readEnergy();
    auto end = std::chrono::high_resolution_clock::now();

    double timeTaken = std::chrono::duration<double, std::milli>(end - start).count();
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    freeArrays(A, B, C);
    return {timeTaken, energyConsumed};
}

std::pair<double, double> SYCLVersion(const int n, const int blockSize, queue& q) {
    // Host arrays
    double *A, *B, *C;
    setupArrays(&A, &B, &C, n);

    // Device allocations using USM
    double* A_dev = malloc_device<double>(n * n, q);
    double* B_dev = malloc_device<double>(n * n, q);
    double* C_dev = malloc_device<double>(n * n, q);

    // Copy input matrices to device
    q.memcpy(A_dev, A, n * n * sizeof(double)).wait();
    q.memcpy(B_dev, B, n * n * sizeof(double)).wait();

    auto start = std::chrono::high_resolution_clock::now();
    long long energyBefore = readEnergy();

    // Round up global size to multiples of blockSize
    size_t globalSize = ((n + blockSize - 1) / blockSize) * blockSize;

    q.submit([&](handler& h) {
        local_accessor<double, 2> A_tile({blockSize, blockSize}, h);
        local_accessor<double, 2> B_tile({blockSize, blockSize}, h);

        h.parallel_for(nd_range<2>({globalSize, globalSize}, {blockSize, blockSize}),
            [=](nd_item<2> item) {
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                int li = item.get_local_id(0);
                int lj = item.get_local_id(1);

                double sum = 0.0;

                // Loop over tiles of input matrices
                for (int t = 0; t < n; t += blockSize) {
                    // Load tiles into local memory with bounds check
                    if (row < n && (t + lj) < n)
                        A_tile[li][lj] = A_dev[row * n + t + lj];
                    else
                        A_tile[li][lj] = 0.0;

                    if ((t + li) < n && col < n)
                        B_tile[li][lj] = B_dev[(t + li) * n + col];
                    else
                        B_tile[li][lj] = 0.0;

                    item.barrier(access::fence_space::local_space);

                    // Unroll the multiply-accumulate loop to improve ILP
                    #pragma unroll
                    for (int k = 0; k < blockSize; ++k) {
                        sum += A_tile[li][k] * B_tile[k][lj];
                    }

                    item.barrier(access::fence_space::local_space);
                }

                if (row < n && col < n)
                    C_dev[row * n + col] = sum;
            });
    });

    q.wait();

    long long energyAfter = readEnergy();
    auto end = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    q.memcpy(C, C_dev, n * n * sizeof(double)).wait();

    double timeTaken = std::chrono::duration<double, std::milli>(end - start).count();
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    freeArrays(A, B, C);
    free(A_dev, q);
    free(B_dev, q);
    free(C_dev, q);

    return {timeTaken, energyConsumed};
}


int main() {
    settings s = getSettings();
    if (s.errorCode != 0) {
        std::cerr << "Error in settings. Exiting." << std::endl;
        return s.errorCode;
    }

    std::pair<double, double> results;
    std::vector<int> sizes;
    for (int i = 1024; i <= 8192; i += 1024)
        sizes.push_back(i);

    int EventSet = PAPI_NULL;
    long long values[7];

    if (setupPAPI(EventSet) != PAPI_OK) {
        std::cerr << "PAPI setup failed!" << std::endl;
        return 1;
    }

    switch (s.algorithmChoice) {
        case 0: {
            for (int size : sizes) {
                std::cout << "Running Sequential for size: " << size << std::endl;

                PAPI_start(EventSet);
                results = sequentialVersion(size, s.blockSize);
                PAPI_stop(EventSet, values);

                saveResult("Sequential", size, s.cores, s.blockSize, results.first, results.second, values);
            }
            break;
        }
        case 1: {
            for (int size : sizes) {
                std::cout << "Running Parallel for size: " << size << std::endl;

                PAPI_start(EventSet);
                results = parallelVersion(size, s.cores, s.blockSize);
                PAPI_stop(EventSet, values);

                saveResult("Parallel", size, s.cores, s.blockSize, results.first, results.second, values);
            }
            break;
        }
        case 2: {
            for (int size : sizes) {
                queue q = queue(platform::get_platforms()[s.platformIndex].get_devices()[s.deviceIndex]);
                if (q.get_device().is_gpu() || q.get_device().is_cpu()) {
                    std::cout << "SYCL running for size: " << size << " on: " << q.get_device().get_info<info::device::name>() << "\n";

                    PAPI_start(EventSet);
                    results = SYCLVersion(size, s.blockSize, q);
                    PAPI_stop(EventSet, values);

                    if (q.get_device().is_gpu()) {
                        saveResult("SYCL_GPU", size, s.cores, s.blockSize, results.first, results.second, values);
                    } else {
                        saveResult("SYCL_CPU", size, s.cores, s.blockSize, results.first, results.second, values);
                    }
                } else {
                    std::cerr << "No suitable device found. Exiting." << std::endl;
                    return 1;
                }
            }
            break;
        }
        default:
            std::cerr << "Invalid algorithm choice. Exiting." << std::endl;
            cleanupPAPI(EventSet);
            return 1;
    }

    return 0;
}
