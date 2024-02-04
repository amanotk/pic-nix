#
# GCC-12.2.0 native compiler on Fugaku Computing Node
#
# Seting up the environment with the following commands is needed:
#
# $ . /vol0004/apps/oss/spack/share/spack/setup-env.sh
# $ spack load gcc@12.2.0 /sxcx7km
# $ spack load fujitsu-mpi%gcc@12.2.0 /3wsaqfe
#
set(CMAKE_SYSTEM_NAME Fugaku)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_CXX_COMPILER "mpicxx" CACHE FILEPATH "C++ compiler")
set(CMAKE_CXX_FLAGS "-march=armv8-a+sve -msve-vector-bits=512 -fopenmp -O3" CACHE STRING "C++ compiler flags")
