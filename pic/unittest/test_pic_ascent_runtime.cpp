// -*- C++ -*-

#include "nix/diag/ascent/runtime.hpp"

#include <catch2/catch_test_macros.hpp>

#include <mpi.h>

#include <stdexcept>

TEST_CASE("AscentRuntime opens, executes, and closes")
{
  conduit::Node data;
  data["coordsets/coords/type"]    = "uniform";
  data["coordsets/coords/dims/i"]  = 2;
  data["coordsets/coords/dims/j"]  = 2;
  data["topologies/mesh/type"]     = "uniform";
  data["topologies/mesh/coordset"] = "coords";

  nix::AscentRuntime runtime;
  runtime.publish_execute(data, PICNIX_ASCENT_ACTIONS_FILE);
  runtime.shutdown();
  runtime.shutdown();

  REQUIRE(true);
}

TEST_CASE("AscentRuntime releases its communicator when actions loading fails")
{
  conduit::Node      data;
  nix::AscentRuntime runtime;

  REQUIRE_THROWS_AS(runtime.publish_execute(data, "missing-ascent-actions.yaml"),
                    std::runtime_error);
  runtime.shutdown();
}
