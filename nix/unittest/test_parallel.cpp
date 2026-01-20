// -*- C++ -*-

#include "test_parallel.hpp"

#include "nix.hpp"

#include <mpi.h>
#include <iostream>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

// global variable for MPI
int options_mpi_decomposition[3];

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

  // catch
  Catch::Session session;

  // custom command line options for MPI
  int  xdecomp = 0;
  int  ydecomp = 0;
  int  zdecomp = 0;
  auto cli = session.cli() | Opt(xdecomp, "xdecomp")["-X"]["--xdecomp"]("# decomposition in x") |
             Opt(ydecomp, "ydecomp")["-Y"]["--ydecomp"]("# decomposition in y") |
             Opt(zdecomp, "zdecomp")["-Z"]["--zdecomp"]("# decomposition in z");
  session.cli(cli);

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0) {
    MPI_Finalize();
    return returnCode;
  }

  // run
  options_mpi_decomposition[0] = zdecomp;
  options_mpi_decomposition[1] = ydecomp;
  options_mpi_decomposition[2] = xdecomp;
  int result                   = session.run();

  MPI_Finalize();
  return result;
}
