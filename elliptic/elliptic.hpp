// -*- C++-*-
#ifndef _ELLIPTIC_HPP_
#define _ELLIPTIC_HPP_

#include <memory>

#include "nix.hpp"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

#include "chunk_accessor.hpp"

namespace elliptic
{

using namespace nix::typedefs;
using nix::Dims3D;

class SolverInterface
{
public:
  virtual ~SolverInterface()                             = default;
  virtual int update_mapping(ChunkAccessor& accessor)    = 0;
  virtual int copy_chunk_to_src(ChunkAccessor& accessor) = 0;
  virtual int copy_sol_to_chunk(ChunkAccessor& accessor) = 0;
};

class Solver
{
public:
  using Interface    = SolverInterface;
  using PtrInterface = std::unique_ptr<Interface>;

  Solver(Dims3D dims, PtrInterface interface) : dims(dims), interface(std::move(interface))
  {
  }

  int update_mapping(ChunkAccessor& accessor)
  {
    return interface->update_mapping(accessor);
  }

  int copy_chunk_to_src(ChunkAccessor& accessor)
  {
    return interface->copy_chunk_to_src(accessor);
  }

  int copy_sol_to_chunk(ChunkAccessor& accessor)
  {
    return interface->copy_sol_to_chunk(accessor);
  }

protected:
  Dims3D       dims;
  PtrInterface interface;
};

} // namespace elliptic

#endif
