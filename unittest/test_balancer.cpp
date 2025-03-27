// -*- C++ -*-

#include "balancer.hpp"
#include <iostream>

#include "catch.hpp"

using namespace nix;

TEST_CASE("access to chunkload")
{
  Balancer balancer(10);

  balancer.load(0) = 0.5;
  REQUIRE(balancer.load(0) == 0.5);

  balancer.load(1) = 1.5;
  REQUIRE(balancer.load(1) == 1.5);

  balancer.fill_load(-1.0);
  REQUIRE(balancer.load(2) == -1.0);
  REQUIRE(balancer.load(3) == -1.0);
}

TEST_CASE("assign_initial")
{
  const int nchunk_per_proc = 20;
  const int nprocess        = 10;
  const int nchunk          = nprocess * nchunk_per_proc;

  Balancer balancer(nchunk);

  SECTION("homogeneous load")
  {
    balancer.fill_load(1.0);

    auto boundary = balancer.assign_initial(nprocess);

    REQUIRE(balancer.is_boundary_ascending(boundary) == true);
    REQUIRE(balancer.is_boundary_optimum(boundary) == true);

    // also check deterministically
    for (int i = 0; i < nprocess; i++) {
      REQUIRE(boundary[i] == i * nchunk_per_proc);
    }
  }

  SECTION("inhomogeneous load")
  {
    std::random_device                      seed;
    std::mt19937                            engine(seed());
    std::uniform_real_distribution<float64> dist(0.5, 1.5);

    for (int i = 0; i < nchunk; i++) {
      balancer.load(i) = dist(engine);
    }

    auto boundary = balancer.assign_initial(nprocess);

    REQUIRE(balancer.is_boundary_ascending(boundary) == true);
    REQUIRE(balancer.is_boundary_optimum(boundary) == true);
  }
}

TEST_CASE("assign")
{
  const int nchunk_per_proc = 20;
  const int nprocess        = 10;
  const int nchunk          = nprocess * nchunk_per_proc;

  Balancer balancer(nchunk);

  std::vector<int> boundary(nprocess + 1);
  for (int i = 0; i < nprocess + 1; i++) {
    boundary[i] = i * nchunk_per_proc;
  }

  SECTION("homogeneous load")
  {
    balancer.fill_load(1.0);

    boundary = balancer.assign(boundary);

    REQUIRE(balancer.is_boundary_ascending(boundary) == true);

    // also check deterministically
    for (int i = 0; i < nprocess; i++) {
      REQUIRE(boundary[i] == i * nchunk_per_proc);
    }
  }

  SECTION("inhomogeneous load")
  {
    std::random_device                      seed;
    std::mt19937                            engine(seed());
    std::uniform_real_distribution<float64> dist(0.5, 1.5);

    for (int i = 0; i < nchunk; i++) {
      balancer.load(i) = dist(engine);
    }

    boundary = balancer.assign(boundary);

    REQUIRE(balancer.is_boundary_ascending(boundary) == true);
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
