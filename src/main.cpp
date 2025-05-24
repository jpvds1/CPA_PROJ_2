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

    for (int i = 0; i < n; i += blockSize)
        for (int j = 0; j < n; j += blockSize)
            for (int k = 0; k < n; k += blockSize)
                for (int ii = i; ii < std::min(i + blockSize, n); ++ii)
                    for (int jj = j; jj < std::min(j + blockSize, n); ++jj)
                        for (int kk = k; kk < std::min(k + blockSize, n); ++kk)
                            C[ii * n + jj] += A[ii * n + kk] * B[kk * n + jj];

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

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            for (int k = 0; k < n; k += blockSize) {
                for (int ii = i; ii < std::min(i + blockSize, n); ++ii) {
                    for (int jj = j; jj < std::min(j + blockSize, n); ++jj) {
                        double temp = C[ii * n + jj];
                        for (int kk = k; kk < std::min(k + blockSize, n); ++kk) {
                            temp += A[ii * n + kk] * B[kk * n + jj];
                        }
                        C[ii * n + jj] = temp;
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
    double *A, *B, *C;
    setupArrays(&A, &B, &C, n);

    auto start = std::chrono::high_resolution_clock::now();
    long long energyBefore = readEnergy();

    buffer<double, 2> A_buf(A, range<2>(n, n));
    buffer<double, 2> B_buf(B, range<2>(n, n));
    buffer<double, 2> C_buf(C, range<2>(n, n));

    {
        host_accessor a_acc(A_buf, write_only);
        host_accessor b_acc(B_buf, write_only);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                a_acc[i][j] = A[i * n + j];
                b_acc[i][j] = B[i * n + j];
            }
    }

    q.submit([&](handler& h) {
        accessor A_acc(A_buf, h, read_only);
        accessor B_acc(B_buf, h, read_only);
        accessor C_acc(C_buf, h, write_only, no_init);

        local_accessor<double, 2> A_tile({static_cast<size_t>(blockSize), static_cast<size_t>(blockSize)}, h);
        local_accessor<double, 2> B_tile({static_cast<size_t>(blockSize), static_cast<size_t>(blockSize)}, h);

        h.parallel_for(nd_range<2>({(size_t)n, (size_t)n}, {(size_t)blockSize, (size_t)blockSize}), [=](nd_item<2> item) {
            int row = item.get_global_id(0);
            int col = item.get_global_id(1);
            double sum = 0.0;

            for (int t = 0; t < (n + blockSize - 1) / blockSize; ++t) {
                int a_col = t * blockSize + item.get_local_id(1);
                int b_row = t * blockSize + item.get_local_id(0);

                if (row < n && a_col < n)
                    A_tile[item.get_local_id(0)][item.get_local_id(1)] = A_acc[row][a_col];
                else
                    A_tile[item.get_local_id(0)][item.get_local_id(1)] = 0.0;

                if (b_row < n && col < n)
                    B_tile[item.get_local_id(0)][item.get_local_id(1)] = B_acc[b_row][col];
                else
                    B_tile[item.get_local_id(0)][item.get_local_id(1)] = 0.0;

                item.barrier(access::fence_space::local_space);

                for (int k = 0; k < blockSize; ++k)
                    sum += A_tile[item.get_local_id(0)][k] * B_tile[k][item.get_local_id(1)];

                item.barrier(access::fence_space::local_space);
            }

            if (row < n && col < n)
                C_acc[row][col] = sum;
        });
    });

    q.wait();

    {
        host_accessor c_acc(C_buf, read_only);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C[i * n + j] = c_acc[i][j];
    }

    long long energyAfter = readEnergy();
    auto end = std::chrono::high_resolution_clock::now();

    double timeTaken = std::chrono::duration<double, std::milli>(end - start).count();
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    freeArrays(A, B, C);
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
    for (int i = 200; i <= 1000; i += 200)
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

                    saveResult("SYCL", size, s.cores, s.blockSize, results.first, results.second, values);
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
