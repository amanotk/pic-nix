#
# GCC-10.3.1 cross compiler on Fugaku Login Node
#
# Follow the three steps for setting up the environment.
#
# 1) Copy /vol0004/apps/oss/gcc-arm-10.3.1/setup-env.sh as follows:
#
# $ cp /vol0004/apps/oss/gcc-arm-10.3.1/setup-env.sh ${HOME}/setup-gcc-10-aarch64.sh
#
# 2) Edit the file and remove "$FJMPI/inclue" from the INCLUDE-related variables as follows:
#
# export C_INCLUDE_PATH="$FJMPI/include/mpi/fujitsu:$C_INCLUDE_PATH"
# export CPLUS_INCLUDE_PATH="$FJMPI/include/mpi/fujitsu:$CPLUS_INCLUDE_PATH"
# export INCLUDE="$FJMPI/include/mpi/fujitsu:$INCLUDE"
#
# 3) Load the setup script with the following command:
#
# $ . ${HOME}/setup-gcc-10-aarch64.sh
#
set(CMAKE_SYSTEM_NAME Fugaku)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(MPI_THREAD_MULTIPLE OFF CACHE BOOL "MPI thread multiple" FORCE)
set(GCC10_LIB "/vol0004/apps/oss/spack-v0.19/opt/spack/linux-rhel8-a64fx/gcc-8.5.0/gcc-10.4.0-vuhiczfvddfc5n3tmny2e4ffde6cvsi5/lib64")

# cross compiler on login node
set(CMAKE_CXX_COMPILER "g++" CACHE FILEPATH "C++ compiler")
set(CMAKE_CXX_FLAGS "-Wno-psabi \
    -march=armv8-a+sve -msve-vector-bits=512 -fopenmp -O3"
    CACHE STRING "C++ compiler flags")
set(CMAKE_EXE_LINKER_FLAGS "-L${GCC10_LIB} -lstdc++" CACHE STRING "Linker flags")
set(CMAKE_CXX_LINK_EXECUTABLE "mpiFCCpx <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>" CACHE STRING "Linker")
