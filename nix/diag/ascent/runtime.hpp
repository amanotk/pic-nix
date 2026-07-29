// -*- C++ -*-
#ifndef _NIX_ASCENT_RUNTIME_HPP_
#define _NIX_ASCENT_RUNTIME_HPP_

#include <ascent/ascent.hpp>

#include <mpi.h>

#include <filesystem>

namespace nix
{
class AscentRuntime
{
public:
  void publish_execute(const conduit::Node& data, const std::filesystem::path& actions_path);
  void shutdown();

private:
  void open(const std::filesystem::path& actions_path);

  ascent::Ascent runtime;
  conduit::Node  actions;
  MPI_Comm       comm    = MPI_COMM_NULL;
  bool           is_open = false;
};
} // namespace nix

#endif
