#include "helpers.h"

// Function to read energy consumption from the system
long long readEnergy()
{
    std::ifstream file("/sys/class/powercap/intel-rapl:0/energy_uj");
    long long energy;
    file >> energy;
    return energy;
}

// Save the results to a file
void saveResult(std::string type, int size, int cores, int blockSize, double time, double consumed, long long values[])
{
    std::string line = type + ',' +
                       std::to_string(size) + ',' +
                       std::to_string(cores) + ',' +
                       std::to_string(blockSize) + ',' +
                       std::to_string(time) + ',' +
                       std::to_string(consumed);

    for (int i = 0; i < 5; i++)
    {
        line += ',' + std::to_string(values[i]);
    }

    std::ofstream file("../data/results.txt", std::ios::app);

    if (file.is_open())
    {
        file << line << "\n";
        file.close();
        std::cout << line << std::endl;
    }
    else
    {
        std::cout << "Error: " << line << std::endl;
    }
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

int caller(std::string type,
           std::function<std::pair<double, double>(int, int, int, int)> func,
           const int cores,
           const int deviceChoice,
        const int blockSize)
{
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

    for (int size : sizes) {
        PAPI_start(EventSet);
        results = func(size, blockSize, cores, deviceChoice);
        PAPI_stop(EventSet, values);

        saveResult(type, size, cores, blockSize, results.first, results.second, values);
    }

    cleanupPAPI(EventSet);
    return 0;
}
