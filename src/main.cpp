#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <papi.h>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <chrono>
#include "helpers.h"
#include <utility>

std::pair<double, double> sequentialVersion(const int n, const int cores, const int blockSize)
{
    clock_t startTime, endTime;
    long long energyBefore, energyAfter;
    int i, j, k, ii, jj, kk;

    double *A, *B, *C;
    setupArrays(&A, &B, &C, n);

    startTime = clock();
    energyBefore = readEnergy();

    for (int i = 0; i < n; i += blockSize)
        for (int j = 0; j < n; j += blockSize)
            for (int k = 0; k < n; k += blockSize)
                for (int ii = i; ii < std::min(i + blockSize, n); ++ii)
                    for (int jj = j; jj < std::min(j + blockSize, n); ++jj)
                        for (int kk = k; kk < std::min(k + blockSize, n); ++kk)
                            C[ii * n + jj] += A[ii * n + kk] * B[kk * n + jj];

    endTime = clock();
    energyAfter = readEnergy();

    double timeTaken = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    freeArrays(A, B, C);

    return {timeTaken, energyConsumed};
}

std::pair<double, double> parallelVersion(const int n, const int cores, const int blockSize)
{
    clock_t startTime, endTime;
    long long energyBefore, energyAfter;
    int i, j, k, ii, jj, kk;

    double *A, *B, *C;
    setupArrays(&A, &B, &C, n);

    omp_set_num_threads(cores);

    startTime = clock();
    energyBefore = readEnergy();

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

    endTime = clock();
    energyAfter = readEnergy();

    double timeTaken = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    freeArrays(A, B, C);

    return {timeTaken, energyConsumed};
}

int main()
{
    caller("Parallel", parallelVersion, 16);

    return 0;
}
