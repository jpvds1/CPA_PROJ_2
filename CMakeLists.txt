cmake_minimum_required(VERSION 3.15)
project(matrix_project LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Add warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")

# Source files
set(SOURCES src/main.cpp src/helpers.cpp)

# Find and link PAPI (always used)
find_library(PAPI_LIB papi REQUIRED)
find_path(PAPI_INCLUDE_DIR papi.h REQUIRED)
include_directories(${PAPI_INCLUDE_DIR})

# Compiler-specific behavior
if(CMAKE_CXX_COMPILER MATCHES ".*acpp.*")
    message(STATUS "Using AdaptiveCpp (acpp) for SYCL build")
    add_executable(matrix_sycl ${SOURCES})
    target_compile_definitions(matrix_sycl PRIVATE USE_SYCL)
    target_link_libraries(matrix_sycl PRIVATE ${PAPI_LIB})
else()
    message(STATUS "Using standard compiler (g++, clang++) for OpenMP build")

    # OpenMP setup
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()

    add_executable(matrix_omp ${SOURCES})
    target_compile_definitions(matrix_omp PRIVATE USE_OMP)
    target_link_libraries(matrix_omp PRIVATE ${PAPI_LIB} OpenMP::OpenMP_CXX)
endif()

# Show final flags
message(STATUS "Final compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Final CXX flags: ${CMAKE_CXX_FLAGS}")
