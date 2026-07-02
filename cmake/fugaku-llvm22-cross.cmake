#
# LLVM-22.1.4 cross compiler on Fugaku Login Node
#
# Seting up the environment with the following commands is needed:
#
# $ module load LLVM/llvmorg-22.1.0
#
set(CMAKE_SYSTEM_NAME Fugaku)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(MPI_THREAD_MULTIPLE OFF CACHE BOOL "MPI thread multiple" FORCE)

set(CMAKE_CXX_COMPILER "mpiclang++" CACHE FILEPATH "C++ compiler")
set(CMAKE_CXX_FLAGS "-Wno-unused-command-line-argument \
    -mtune=a64fx -mcpu=a64fx -march=armv8.2-a+sve -msve-vector-bits=512 \
    -fopenmp -O3"
    CACHE STRING "C++ compiler flags")
