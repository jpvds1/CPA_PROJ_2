cmake_minimum_required(VERSION 3.10)
project(CPA_PROJ_2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

include_directories(src)

add_executable(matrix_benchmark
    src/main.cpp
    src/helpers.cpp
)

find_library(PAPI_LIB papi REQUIRED)
target_link_libraries(matrix_benchmark PRIVATE ${PAPI_LIB})