#
# LLVM-14.0.1 cross compiler on Fugaku Login Node
#
# Seting up the environment with the following commands is needed:
#
# $ . /vol0004/apps/oss/llvm-v14.0.1/init.sh
#
set(CMAKE_SYSTEM_NAME Fugaku)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_CXX_COMPILER "mpiclang++" CACHE FILEPATH "C++ compiler")
set(CMAKE_CXX_FLAGS "-march=armv8-a+sve -msve-vector-bits=512 -fopenmp -O3" CACHE STRING "C++ compiler flags")
set(CMAKE_EXE_LINKER_FLAGS "-L/vol0004/apps/oss/llvm-v14.0.1/login_node/lib" CACHE STRING "Linker flags")
