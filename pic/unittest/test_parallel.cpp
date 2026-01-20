// -*- C++ -*-

#include "test_parallel.hpp"

#include <mpi.h>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

int get_mpi_size()
{
  int nprocess = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
  return nprocess;
}

int get_mpi_rank()
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

bool require_mpi_size(int expected)
{
  int nprocess = get_mpi_size();
  if (nprocess != expected) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return false;
  }
  return true;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  Catch::Session session;
  int            result = session.run(argc, argv);

  MPI_Finalize();
  return result;
}
