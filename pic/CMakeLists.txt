cmake_minimum_required(VERSION 3.23)

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

set(PICNIX_ASCENT_ROOT "" CACHE PATH "Ascent superbuild installation prefix")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(NIX_BUILD_TESTING OFF CACHE BOOL "Build nix tests" FORCE)
  add_subdirectory(${PICNIX_DIR}/nix ${CMAKE_CURRENT_BINARY_DIR}/nix)
  add_subdirectory(${PICNIX_DIR}/elliptic ${CMAKE_CURRENT_BINARY_DIR}/elliptic)
endif()

# build static library
set(PIC_SOURCES
  pic_chunk.cpp
  pic_application.cpp
  pic_performance.cpp
  pic_poisson.cpp
  pic_poisson_basic.cpp
  pic_poisson_factory.cpp
)

if(PICNIX_ENABLE_PETSC)
  list(APPEND PIC_SOURCES pic_poisson_petsc.cpp)
endif()

if(PICNIX_ENABLE_ASCENT)
  enable_language(C)
  find_package(MPI REQUIRED COMPONENTS C CXX)

  set(_picnix_ascent_hints)
  if(PICNIX_ASCENT_ROOT)
    list(APPEND _picnix_ascent_hints "${PICNIX_ASCENT_ROOT}")
    if(EXISTS "${PICNIX_ASCENT_ROOT}/ascent-checkout/lib/cmake/ascent/AscentConfig.cmake")
      list(APPEND _picnix_ascent_hints "${PICNIX_ASCENT_ROOT}/ascent-checkout")
    endif()
  endif()
  foreach(_picnix_prefix IN LISTS CMAKE_PREFIX_PATH)
    if(IS_ABSOLUTE "${_picnix_prefix}")
      set(_picnix_prefix_abs "${_picnix_prefix}")
    else()
      get_filename_component(_picnix_prefix_abs "${_picnix_prefix}" ABSOLUTE BASE_DIR "${PICNIX_DIR}")
    endif()
    if(EXISTS "${_picnix_prefix_abs}/ascent-checkout/lib/cmake/ascent/AscentConfig.cmake")
      list(APPEND _picnix_ascent_hints "${_picnix_prefix_abs}/ascent-checkout")
    endif()
  endforeach()
  find_package(Ascent CONFIG REQUIRED HINTS ${_picnix_ascent_hints})

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

  add_subdirectory(${PICNIX_DIR}/nix/diag/ascent ${CMAKE_CURRENT_BINARY_DIR}/nix_ascent)

  list(APPEND PIC_SOURCES
    diag/ascent.cpp
    diag/ascent/blueprint_builder.cpp
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
  target_link_libraries(pic PUBLIC nix::ascent)
  target_compile_definitions(pic PUBLIC PICNIX_ENABLE_ASCENT=1)
else()
  target_compile_definitions(pic PUBLIC PICNIX_ENABLE_ASCENT=0)
endif()

if(BUILD_TESTING)
  add_subdirectory(unittest)
endif()
