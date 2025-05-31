# CPA_PROJ_2



mkdir -p build_omp
cd build_omp
cmake -DCMAKE_CXX_COMPILER=g++ ..
make -j

mkdir -p build_sycl
cd build_sycl
cmake -DCMAKE_CXX_COMPILER=acpp ..
make -j
