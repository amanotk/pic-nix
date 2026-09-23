#
# LLVM 23.1.0 cross compiler for Fugaku (configure and build on a login node).
#
# Load the matching toolchain before configuring:
#
#   module load LLVM/llvmorg-23.1.0
#
# This file doubles as an initial cache (-C) and a CMake toolchain file. The
# toolchain must load before project() so CMake knows target binaries cannot
# run on the x86_64 login node (for example, during ADIOS2 try_run checks).
# Load the same module in the compute-node job script. Shared LLVM runtime
# libraries may need staging there (see the site's llio_transfer guidance).
#
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR "aarch64")
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_FILE}" CACHE FILEPATH "Fugaku cross toolchain")
set(MPI_THREAD_MULTIPLE OFF CACHE BOOL "MPI thread multiple" FORCE)

set(CMAKE_C_COMPILER "mpiclang" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "mpiclang++" CACHE FILEPATH "C++ compiler")
set(CMAKE_C_FLAGS "-Wno-unused-command-line-argument -mtune=a64fx -mcpu=a64fx -march=armv8.2-a+sve -msve-vector-bits=512 -fopenmp -O3" CACHE STRING "C compiler flags")
set(CMAKE_CXX_FLAGS "-Wno-unused-command-line-argument -mtune=a64fx -mcpu=a64fx -march=armv8.2-a+sve -msve-vector-bits=512 -fopenmp -O3" CACHE STRING "C++ compiler flags")
# The Fugaku target sysroot uses GCC 8; ADIOS2 uses std::filesystem.
set(CMAKE_CXX_STANDARD_LIBRARIES "-lstdc++fs" CACHE STRING "C++ standard libraries")

# Host programs (CMake, flex, bison, etc.) run on the login node. Libraries,
# headers, and CMake packages must not resolve to x86_64 system installations.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
