// -*- C++-*-
#ifndef _ELLIPTIC_HPP_
#define _ELLIPTIC_HPP_

#include <memory>

#include "nix.hpp"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

#include "petsc_scatter.hpp"
#include "petsc_solver.hpp"

namespace elliptic
{

using namespace nix::typedefs;

class SolverInterface
{
};

template <typename T_chunkvec>
class Solver
{
public:
  using Interfacce   = SolverInterface;
  using PtrInterface = std::unique_ptr<Interfacce>;

public:
  Solver(int dims[3])
  {
  }

  int update_mapping(T_chunkvec& chunkvec)
  {
    return 0;
  }

  int copy_chunk_to_src(T_chunkvec& chunkvec)
  {
    return 0;
  }

  int copy_sol_to_chunk(T_chunkvec& chunkvec)
  {
    return 0;
  }
};

} // namespace elliptic

#endif
