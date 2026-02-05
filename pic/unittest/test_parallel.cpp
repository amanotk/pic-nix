// -*- C++ -*-

#include "test_parallel.hpp"

#include "nix.hpp"

#include <iostream>
#include <mpi.h>
#if PICNIX_ENABLE_PETSC
#include <petsc.h>
#endif

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
  using namespace Catch::Clara;

  int thread_provided = -1;
  MPI_Init_thread(&argc, &argv, NIX_MPI_THREAD_LEVEL, &thread_provided);
  if (thread_provided < NIX_MPI_THREAD_LEVEL) {
    std::cerr << "MPI thread level is insufficient for tests." << std::endl;
    MPI_Finalize();
    return 1;
  }

#if PICNIX_ENABLE_PETSC
  PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif

  Catch::Session session;
  int            returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0) {
#if PICNIX_ENABLE_PETSC
    PetscFinalize();
#endif
    MPI_Finalize();
    return returnCode;
  }

  int result = session.run();

#if PICNIX_ENABLE_PETSC
  PetscFinalize();
#endif
  MPI_Finalize();
  return result;
}
