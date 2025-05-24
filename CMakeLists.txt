cmake_minimum_required(VERSION 3.15)
project(matrix_benchmark LANGUAGES CXX)

# Use AdaptiveCpp (hipSYCL) compiler
set(CMAKE_CXX_COMPILER acpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Enable warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Find and link OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Find and link PAPI
find_library(PAPI_LIB papi REQUIRED)
find_path(PAPI_INCLUDE_DIR papi.h REQUIRED)

include_directories(${PAPI_INCLUDE_DIR})

# Sources
set(SOURCES
    src/main.cpp
    src/helpers.cpp
)

add_executable(matrix_benchmark ${SOURCES})

# Link libraries
target_link_libraries(matrix_benchmark PRIVATE ${PAPI_LIB} OpenMP::OpenMP_CXX)
