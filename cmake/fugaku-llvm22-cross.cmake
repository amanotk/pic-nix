#
# LLVM 22.1.0 cross compiler on Fugaku login node
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
# This file is both an initial cache (-C) and a toolchain file. CMake must
# load it before project() to recognize that target programs cannot run here.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR "aarch64")
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_FILE}" CACHE FILEPATH "Fugaku cross toolchain")
set(MPI_THREAD_MULTIPLE OFF CACHE BOOL "MPI thread multiple" FORCE)

set(CMAKE_C_COMPILER "mpiclang" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "mpiclang++" CACHE FILEPATH "C++ compiler")
set(CMAKE_C_FLAGS "-Wno-unused-command-line-argument -mtune=a64fx -mcpu=a64fx -march=armv8.2-a+sve -msve-vector-bits=512 -fopenmp -O3" CACHE STRING "C compiler flags")
set(CMAKE_CXX_FLAGS "-Wno-unused-command-line-argument -mtune=a64fx -mcpu=a64fx -march=armv8.2-a+sve -msve-vector-bits=512 -fopenmp -O3" CACHE STRING "C++ compiler flags")
set(CMAKE_CXX_STANDARD_LIBRARIES "-lstdc++fs" CACHE STRING "C++ standard libraries")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
