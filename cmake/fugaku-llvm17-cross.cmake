#
# LLVM-17.0.2 cross compiler on Fugaku Login Node
#
# Seting up the environment with the following commands is needed:
#
# $ . /vol0004/apps/oss/llvm-v17.0.2/init.sh
#
set(CMAKE_SYSTEM_NAME Fugaku)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(MPI_THREAD_MULTIPLE OFF CACHE BOOL "MPI thread multiple" FORCE)
set(LLVM17_LIB "/vol0004/apps/oss/llvm-v17.0.2/login_node/lib")

set(CMAKE_CXX_COMPILER "mpiclang++" CACHE FILEPATH "C++ compiler")
set(CMAKE_CXX_FLAGS "-Wno-unused-command-line-argument \
    -march=armv8-a+sve -msve-vector-bits=512 -fopenmp -O3"
    CACHE STRING "C++ compiler flags")
set(CMAKE_EXE_LINKER_FLAGS "-L${LLVM17_LIB}" CACHE STRING "Linker flags")