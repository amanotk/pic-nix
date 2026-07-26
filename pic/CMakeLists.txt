cmake_minimum_required(VERSION 3.20)

# project (standalone build)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pic CXX)
  option(BUILD_TESTING "Build tests" OFF)
  include(CTest)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT DEFINED PICNIX_DIR)
  set(PICNIX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

if(NOT DEFINED PICNIX_ENABLE_PETSC)
  option(PICNIX_ENABLE_PETSC "Enable PETSc-based solvers" OFF)
endif()

if(NOT DEFINED PICNIX_ENABLE_ASCENT)
  option(PICNIX_ENABLE_ASCENT "Enable Ascent in-situ diagnostics" OFF)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(NIX_BUILD_TESTING OFF CACHE BOOL "Build nix tests" FORCE)
  add_subdirectory(${PICNIX_DIR}/nix ${CMAKE_CURRENT_BINARY_DIR}/nix)
  add_subdirectory(${PICNIX_DIR}/elliptic ${CMAKE_CURRENT_BINARY_DIR}/elliptic)
endif()

# build static library
set(PIC_SOURCES
  pic_chunk.cpp
  pic_application.cpp
  pic_poisson.cpp
  pic_poisson_basic.cpp
  pic_poisson_factory.cpp
)

if(PICNIX_ENABLE_PETSC)
  list(APPEND PIC_SOURCES pic_poisson_petsc.cpp)
endif()

if(PICNIX_ENABLE_ASCENT)
  if(CMAKE_VERSION VERSION_LESS 3.23)
    message(FATAL_ERROR "PICNIX_ENABLE_ASCENT requires CMake 3.23 or newer")
  endif()

  enable_language(C)
  find_package(MPI REQUIRED COMPONENTS C CXX)
  find_package(Ascent CONFIG REQUIRED)

  if(NOT ASCENT_MPI_ENABLED)
    message(FATAL_ERROR "PIC-NIX requires an MPI-enabled Ascent installation")
  endif()

  if(NOT TARGET ascent::ascent_mpi)
    message(FATAL_ERROR "Ascent was found but ascent::ascent_mpi is unavailable")
  endif()

  if(NOT ASCENT_PYTHON_ENABLED)
    message(STATUS "Ascent Python extracts will be disabled: Ascent Python support is unavailable")
  endif()
  if(NOT ASCENT_VTKH_ENABLED)
    message(STATUS "Ascent render tests will be disabled: VTK-h support is unavailable")
  endif()

  list(APPEND PIC_SOURCES
    diag/ascent.cpp
    insitu/ascent_runtime.cpp
    insitu/blueprint_builder.cpp
  )
endif()

add_library(pic STATIC ${PIC_SOURCES})

# include directory
target_include_directories(pic PRIVATE ${PICNIX_DIR})
target_include_directories(pic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pic PUBLIC elliptic nix)
target_link_libraries(pic PRIVATE std::mdspan)
target_compile_definitions(pic PUBLIC PICNIX_ENABLE_PETSC=$<BOOL:${PICNIX_ENABLE_PETSC}>)

if(PICNIX_ENABLE_ASCENT)
  target_link_libraries(pic PUBLIC ascent::ascent_mpi)
  target_compile_definitions(pic PUBLIC PICNIX_ENABLE_ASCENT=1)
else()
  target_compile_definitions(pic PUBLIC PICNIX_ENABLE_ASCENT=0)
endif()

if(BUILD_TESTING)
  add_subdirectory(unittest)
endif()
