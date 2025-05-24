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

std::pair<double, double> sequentialVersion(const int n, const int blockSize,  queue& q)
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

std::pair<double, double> parallelVersion(const int n, const int cores, const int blockSize,  queue& q)
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

    long long energyAfter = readEnergy();
    auto end = std::chrono::high_resolution_clock::now();

    double timeTaken = std::chrono::duration<double, std::milli>(end - start).count();
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    freeArrays(A, B, C);
    return {timeTaken, energyConsumed};
}

int main() {
    int cores = omp_get_max_threads();
    int blockSize;
    int algorithmChoice;
    int platformIndex = 0;
    int deviceIndex = 0;

    std::cout << "Choose algorithm:\n0. Sequential\n1. Parallel\n2. SYCL\n> ";
    std::cin >> algorithmChoice;

    if (algorithmChoice < 0 || algorithmChoice > 2) {
        std::cerr << "Invalid choice. Exiting.\n";
        return 1;
    }

    if (algorithmChoice == 1) {
        std::cout << "Enter number of cores (default " << cores << "): ";
        std::cin >> cores;
        if (cores <= 0 || cores > omp_get_max_threads()) {
            cores = omp_get_max_threads();
        }
    }

    if (algorithmChoice == 2) {
        auto platforms = platform::get_platforms();
        std::cout << "Available platforms:\n";
        for (size_t i = 0; i < platforms.size(); ++i)
            std::cout << i << ": " << platforms[i].get_info<info::platform::name>() << "\n";
        std::cout << "Select platform index: ";
        std::cin >> platformIndex;

        auto devices = platforms[platformIndex].get_devices();
        std::cout << "Available devices:\n";
        for (size_t i = 0; i < devices.size(); ++i)
            std::cout << i << ": " << devices[i].get_info<info::device::name>() << "\n";
        std::cout << "Select device index: ";
        std::cin >> deviceIndex;
    }

    std::cout << "Enter block size (default 64): ";
    std::cin >> blockSize;
    if (blockSize <= 0) blockSize = 64;

    std::cout << "\nUsing block size: " << blockSize << "\n";
    std::cout << "Using " << cores << " cores\n";
    std::cout << "Running algorithm choice: " << algorithmChoice << "\n";

    // SYCL queue (optional, created only if needed)
    queue q;
    if (algorithmChoice == 2) {
        q = queue(platform::get_platforms()[platformIndex].get_devices()[deviceIndex]);
        std::cout << "SYCL running on: " << q.get_device().get_info<info::device::name>() << "\n";
    }

    switch (algorithmChoice) {
        case 0: {
            auto wrapper = [&](int n, int blockSize, int, int) {
                return sequentialVersion(n, blockSize, q);
            };
            caller("Sequential", wrapper, cores, 0, blockSize);
            break;
        }
        case 1: {
            auto wrapper = [&](int n, int blockSize, int cores, int) {
                return parallelVersion(n, cores, blockSize, q);
            };
            caller("Parallel", wrapper, cores, 0, blockSize);
            break;
        }
        case 2: {
            auto wrapper = [&](int n, int blockSize, int, int) {
                return SYCLVersion(n, blockSize, q);
            };
            caller("SYCL", wrapper, cores, 0, blockSize);
            break;
        }
        default:
            break;
    }

    return 0;
}
