#
# GCC-12.2.0 native compiler on Fugaku Computing Node
#
# Seting up the environment with the following commands is needed:
#
# $ . /vol0004/apps/oss/spack/share/spack/setup-env.sh
# $ spack load gcc@12.2.0
# $ spack load fujitsu-mpi%gcc@8.5.0
#
set(CMAKE_SYSTEM_NAME Fugaku)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(MPI_THREAD_MULTIPLE OFF CACHE BOOL "MPI thread multiple" FORCE)

set(CMAKE_CXX_COMPILER "mpicxx" CACHE FILEPATH "C++ compiler")
set(CMAKE_CXX_FLAGS "-mcpu=native -msve-vector-bits=512 -fopenmp -O3" CACHE STRING "C++ compiler flags")
