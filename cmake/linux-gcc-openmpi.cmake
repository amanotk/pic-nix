# GCC with OpenMPI on Linux
set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_CXX_COMPILER "mpicxx" CACHE FILEPATH "C++ compiler")
set(CMAKE_CXX_FLAGS "-march=native -fopenmp -O3" CACHE STRING "C++ compiler flags")
