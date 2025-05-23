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

// Function to read energy consumption from the system
long long readEnergy()
{
    std::ifstream file("/sys/class/powercap/intel-rapl:0/energy_uj");
    long long energy;
    file >> energy;
    return energy;
}

// Create and save a report
int createReport(const int time, const int energy)
{
    std::cout << "Time: " << time << " ms" << std::endl;
    std::cout << "Energy: " << energy << " J" << std::endl;

    return 0;
}

// Setup PAPI and return the EventSet
int setupPAPI(int &EventSet)
{
    int ret;

    ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_VER_CURRENT)
    {
        std::cerr << "PAPI library init error!" << std::endl;
        return ret;
    }

    ret = PAPI_create_eventset(&EventSet);
    if (ret != PAPI_OK)
        return ret;

    std::vector<int> events = {
        PAPI_L1_DCM,
        PAPI_L2_DCM,
        PAPI_FP_INS,
        PAPI_TOT_INS,
    };

    int i = 0;

    for (int e : events)
    {
        if (PAPI_query_event(e) != PAPI_OK)
        {
            std::cerr << "Event not supported on this system: " << e << std::endl;
            continue;
        }

        int ret = PAPI_add_event(EventSet, e);
        if (ret != PAPI_OK)
        {
            std::cerr << "Failed to add PAPI event: " << e << " (error " << ret << ")" << std::endl;
            continue;
        }

        i++;
    }

    return PAPI_OK;
}

// Cleanup PAPI
int cleanupPAPI(int &EventSet)
{
    std::vector<int> events = {
        PAPI_L1_DCM,
        PAPI_L2_DCM,
        PAPI_FP_INS,
        PAPI_TOT_INS,
    };

    int ret;
    for (int e : events)
    {
        ret = PAPI_remove_event(EventSet, e);
        if (ret != PAPI_OK)
            std::cerr << "Failed to remove event: " << e << std::endl;
    }

    ret = PAPI_destroy_eventset(&EventSet);
    if (ret != PAPI_OK)
        std::cerr << "Failed to destroy event set" << std::endl;

    PAPI_shutdown();
    return ret;
}

// Setup arrays for matrix multiplication
void setupArrays(double **A, double **B, double **C, const int size)
{
    *A = (double *)malloc(size * size * sizeof(double));
    *B = (double *)malloc(size * size * sizeof(double));
    *C = (double *)calloc(size * size, sizeof(double));

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            (*A)[i * size + j] = 1.0;

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            (*B)[i * size + j] = i + 1;
}

// Free allocated arrays
void freeArrays(double *A, double *B, double *C)
{
    free(A);
    free(B);
    free(C);
}

int caller(std::pair<double, double> (*func)(int))
{
    std::pair<double, double> results;

    std::vector<int> sizes;
    // for (int i = 1024; i <= 8192; i += 1024)
    //     sizes.push_back(i);
    for (int i = 200; i <= 1000; i += 200)
        sizes.push_back(i);

    int EventSet = PAPI_NULL;
    long long values[7];

    if (setupPAPI(EventSet) != PAPI_OK)
    {
        std::cerr << "PAPI setup failed!" << std::endl;
        return 1;
    }

    for (int size : sizes)
    {
        PAPI_start(EventSet);

        results = func(size);

        PAPI_stop(EventSet, values);

        createReport(results.first, results.second);
    }

    cleanupPAPI(EventSet);
    return 0;
}
