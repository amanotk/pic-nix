// -*- C++ -*-
#include "test_parallel.hpp"

#include "nix/nix.hpp"

#include <mpi.h>

#include <iostream>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

int get_mpi_size()
{
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

int get_mpi_rank()
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

bool require_mpi_size(int expected)
{
  if (get_mpi_size() != expected) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return false;
  }
  return true;
}

int main(int argc, char** argv)
{
  int thread_provided = -1;
  MPI_Init_thread(&argc, &argv, NIX_MPI_THREAD_LEVEL, &thread_provided);
  if (thread_provided < NIX_MPI_THREAD_LEVEL) {
    std::cerr << "MPI thread level is insufficient for tests." << std::endl;
    MPI_Finalize();
    return 1;
  }

  Catch::Session session;
  const int      parse_result = session.applyCommandLine(argc, argv);
  if (parse_result != 0) {
    MPI_Finalize();
    return parse_result;
  }

  const int result = session.run();
  MPI_Finalize();
  return result;
}
