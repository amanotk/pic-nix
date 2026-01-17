// -*- C++-*-
#ifndef _ELLIPTIC_HPP_
#define _ELLIPTIC_HPP_

#include <memory>

#include "nix.hpp"

#include "chunk_accessor.hpp"

namespace elliptic
{

using namespace nix::typedefs;

class SolverInterface
{
public:
  virtual ~SolverInterface()                             = default;
  virtual int update_mapping(ChunkAccessor& accessor)    = 0;
  virtual int copy_chunk_to_src(ChunkAccessor& accessor) = 0;
  virtual int copy_sol_to_chunk(ChunkAccessor& accessor) = 0;
  virtual int set_option(const nlohmann::json& config)   = 0;
  virtual int solve()                                    = 0;
  virtual int solve(ChunkAccessor& accessor)             = 0;
};

class Solver
{
public:
  using Interface    = SolverInterface;
  using PtrInterface = std::unique_ptr<Interface>;

  Solver(Dims3D dims, PtrInterface interface = nullptr)
      : dims(dims), interface(std::move(interface))
  {
  }

  void set_interface(PtrInterface interface)
  {
    this->interface = std::move(interface);
  }

  int update_mapping(ChunkAccessor& accessor)
  {
    if (interface == nullptr)
      return 1;
    return interface->update_mapping(accessor);
  }

  int copy_chunk_to_src(ChunkAccessor& accessor)
  {
    if (interface == nullptr)
      return 1;
    return interface->copy_chunk_to_src(accessor);
  }

  int copy_sol_to_chunk(ChunkAccessor& accessor)
  {
    if (interface == nullptr)
      return 1;
    return interface->copy_sol_to_chunk(accessor);
  }

  int set_option(const nlohmann::json& config)
  {
    if (interface == nullptr)
      return 1;
    return interface->set_option(config);
  }

  int solve(ChunkAccessor& accessor)
  {
    if (interface == nullptr)
      return 1;
    return interface->solve(accessor);
  }

  int solve()
  {
    if (interface == nullptr)
      return 1;
    return interface->solve();
  }

protected:
  Dims3D       dims;
  PtrInterface interface;
};

} // namespace elliptic

#endif
