#include "helpers.h"

using namespace cl::sycl;

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

settings getSettings() {
    settings s;

    std::cout << "Choose algorithm:\n0. Sequential\n1. Parallel\n2. SYCL\n> ";
    std::cin >> s.algorithmChoice;

    if (s.algorithmChoice < 0 || s.algorithmChoice > 2) {
        std::cerr << "Invalid choice. Exiting.\n";
        s.errorCode = 1;
        return s;
    }

    if (s.algorithmChoice == 1) {
        std::cout << "Enter number of cores (default " << omp_get_max_threads() << "): ";
        std::cin >> s.cores;
        if (s.cores <= 0 || s.cores > omp_get_max_threads()) {
            s.cores = omp_get_max_threads();
        }
    }

    if (s.algorithmChoice == 2) {
        auto platforms = platform::get_platforms();
        std::cout << "Available platforms:\n";
        for (size_t i = 0; i < platforms.size(); ++i)
            std::cout << i << ": " << platforms[i].get_info<info::platform::name>() << "\n";
        std::cout << "Select platform index: ";
        std::cin >> s.platformIndex;

        auto devices = platforms[s.platformIndex].get_devices();
        std::cout << "Available devices:\n";
        for (size_t i = 0; i < devices.size(); ++i)
            std::cout << i << ": " << devices[i].get_info<info::device::name>() << "\n";
        std::cout << "Select device index: ";
        std::cin >> s.deviceIndex;
    }

    std::cout << "Enter block size (default 64): ";
    std::cin >> s.blockSize;
    if (s.blockSize <= 0) s.blockSize = 64;

    std::cout << "\nUsing block size: " << s.blockSize << "\n";
    if (s.algorithmChoice == 1) std::cout << "Using " << s.cores << " cores\n";
    if (s.algorithmChoice == 2) {
        std::cout << "Using SYCL on platform index: " << s.platformIndex
                  << ", device index: " << s.deviceIndex << "\n";
    }
    std::cout << "Running algorithm choice: " << s.algorithmChoice << "\n";

    return s;
}