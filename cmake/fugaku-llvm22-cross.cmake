#
# LLVM-22.1.4 cross compiler on Fugaku Login Node
#
# On login node, setting up the environment with the following commands
# is needed for compilation:
#
# $ module load LLVM/llvmorg-22.1.0
#
# In a job script for the scheduler (PJM), the following is needed:
#
# module purge
# module load LLVM/llvmorg-22.1.0
# llio_transfer `find ${LLVM_BASEDIR}/lib64 -type f -name \*.so\*`
#
set(CMAKE_SYSTEM_NAME Fugaku)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(MPI_THREAD_MULTIPLE OFF CACHE BOOL "MPI thread multiple" FORCE)

set(CMAKE_CXX_COMPILER "mpiclang++" CACHE FILEPATH "C++ compiler")
set(CMAKE_CXX_FLAGS "-Wno-unused-command-line-argument \
    -mtune=a64fx -mcpu=a64fx -march=armv8.2-a+sve -msve-vector-bits=512 \
    -fopenmp -O3"
    CACHE STRING "C++ compiler flags")
