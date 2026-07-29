cmake_minimum_required(VERSION 3.20)

if(NOT TARGET nix_ascent)
  add_library(nix_ascent STATIC runtime.cpp)
  add_library(nix::ascent ALIAS nix_ascent)

  target_include_directories(nix_ascent
    PUBLIC
      ${PICNIX_DIR}
  )

  target_link_libraries(nix_ascent
    PUBLIC
      ascent::ascent_mpi
  )
endif()
