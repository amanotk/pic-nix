# LLVM-based Intel oneAPI with IntelMPI on Linux
set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_CXX_COMPILER "mpiicpc" CACHE FILEPATH "C++ compiler")
set(CMAKE_CXX_FLAGS "-cxx=icpx -xHost -qopenmp -O3" CACHE STRING "C++ compiler flags")
