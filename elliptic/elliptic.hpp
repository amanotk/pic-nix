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
  virtual int solve(ChunkAccessor& accessor)             = 0;
  virtual int scatter_forward()
  {
    return 0;
  }
  virtual int scatter_reverse()
  {
    return 0;
  }
};

class Solver
{
public:
  using Interface    = SolverInterface;
  using PtrInterface = std::shared_ptr<Interface>;

  Solver(PtrInterface interface = nullptr) : interface(std::move(interface))
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

    const int expected = accessor.get_num_grids_total();

    int       status = update_mapping(accessor);
    const int packed = copy_chunk_to_src(accessor);
    status |= (packed == expected) ? 0 : 1;
    status |= scatter_forward();
    status |= interface->solve(accessor);
    status |= scatter_reverse();
    const int unpacked = copy_sol_to_chunk(accessor);
    status |= (unpacked == expected) ? 0 : 1;
    return status;
  }

  int scatter_forward()
  {
    if (interface == nullptr)
      return 1;
    return interface->scatter_forward();
  }

  int scatter_reverse()
  {
    if (interface == nullptr)
      return 1;
    return interface->scatter_reverse();
  }

  PtrInterface get_interface()
  {
    return interface;
  }

  std::shared_ptr<const Interface> get_interface() const
  {
    return interface;
  }

protected:
  PtrInterface interface;
};

} // namespace elliptic

#endif
