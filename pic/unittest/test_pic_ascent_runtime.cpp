// -*- C++ -*-

#include "../insitu/ascent_runtime.hpp"

#include <catch2/catch_test_macros.hpp>

#include <mpi.h>

TEST_CASE("AscentRuntime opens, executes, and closes")
{
  conduit::Node data;
  data["coordsets/coords/type"]    = "uniform";
  data["coordsets/coords/dims/i"]  = 2;
  data["coordsets/coords/dims/j"]  = 2;
  data["topologies/mesh/type"]     = "uniform";
  data["topologies/mesh/coordset"] = "coords";

  picnix::insitu::AscentRuntime runtime;
  runtime.publish_execute(data, PICNIX_ASCENT_ACTIONS_FILE);
  runtime.shutdown();
  runtime.shutdown();

  REQUIRE(true);
}
