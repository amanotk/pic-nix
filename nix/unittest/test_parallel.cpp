// -*- C++ -*-

#include <iostream>
#include <mpi.h>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

// global variable for MPI
int options_mpi_decomposition[3];

int main(int argc, char** argv)
{
  using namespace Catch::clara;

  MPI_Init(&argc, &argv);

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
