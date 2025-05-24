#ifndef HELPERS_H
#define HELPERS_H

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
#include <utility>
#include <string>
#include <utility>
#include <functional>
#include <CL/sycl.hpp>

struct settings {
    int cores = 16;
    int blockSize;
    int algorithmChoice;
    int platformIndex;
    int deviceIndex;
    int errorCode = 0;
};

int setupPAPI(int &EventSet);
int cleanupPAPI(int &EventSet);
void setupArrays(double **A, double **B, double **C, const int size);
void freeArrays(double *A, double *B, double *C);
long long readEnergy();
void saveResult(std::string type, int size, int cores, int blockSize, double time, double consumed, long long values[]);
settings getSettings();

#endif
