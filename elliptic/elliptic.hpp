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

  Solver(PtrInterface interface = nullptr);

  void set_interface(PtrInterface interface);

  static int initialize();
  static int finalize();

  int update_mapping(ChunkAccessor& accessor);
  int copy_chunk_to_src(ChunkAccessor& accessor);
  int copy_sol_to_chunk(ChunkAccessor& accessor);
  int set_option(const nlohmann::json& config);
  int solve(ChunkAccessor& accessor);
  int scatter_forward();
  int scatter_reverse();

  PtrInterface                     get_interface();
  std::shared_ptr<const Interface> get_interface() const;

protected:
  PtrInterface       interface;
  inline static bool is_initialized = false;
  inline static bool is_finalized   = false;
};

} // namespace elliptic

#endif
