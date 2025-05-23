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

std::pair<double, double> sequentialVersion(const int n, const int cores)
{
    clock_t startTime, endTime;
    long long energyBefore, energyAfter;

    double *A, *B, *C;
    setupArrays(&A, &B, &C, n);

    startTime = clock();
    energyBefore = readEnergy();

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C[i * n + j] += A[i * n + k] * B[k * n + j];

    endTime = clock();
    energyAfter = readEnergy();

    double timeTaken = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    double energyConsumed = (energyAfter - energyBefore) / 1e6;

    freeArrays(A, B, C);

    return {timeTaken, energyConsumed};
}

std::pair<double, double>  parallelVersion(const int n, const int cores)
{
    clock_t startTime, endTime;
    long long energyBefore, energyAfter;

    double *A, *B, *C;
    setupArrays(&A, &B, &C, n);

    omp_set_num_threads(cores);

    startTime = clock();
    energyBefore = readEnergy();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C[i * n + j] += A[i * n + k] * B[k * n + j];

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
