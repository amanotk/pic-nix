// -*- C++ -*-

#include "balancer.hpp"
#include "chunkmap.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace nix;

namespace
{
float64 shock_imbalance(ChunkMap& chunkmap, int Cz, int Cy, int Cx, int dense_width, int nprocess,
                        int nassign)
{
  int      nchunk = Cz * Cy * Cx;
  Balancer balancer(nchunk);

  for (int id = 0; id < nchunk; id++) {
    auto [cz, cy, cx] = chunkmap.get_coordinate(id);
    balancer.load(id) = cx < dense_width ? 4.0 : 1.0;
  }

  std::vector<int> boundary(nprocess + 1);
  for (int rank = 0; rank <= nprocess; rank++) {
    boundary[rank] = rank * nchunk / nprocess;
  }

  for (int i = 0; i < nassign; i++) {
    boundary = balancer.assign(boundary);
  }

  std::vector<float64> load(nchunk);
  for (int id = 0; id < nchunk; id++) {
    load[id] = balancer.load(id);
  }
  auto rankload = balancer.get_rankload(boundary, load);
  auto meanload = std::accumulate(rankload.begin(), rankload.end(), 0.0) / nprocess;

  return *std::max_element(rankload.begin(), rankload.end()) / meanload;
}
} // namespace

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

TEST_CASE("axis-first ordering accelerates slab balancing")
{
  const int Cz          = 4;
  const int Cy          = 12;
  const int Cx          = 48;
  const int dense_width = 5;
  const int nprocess    = 96;
  const int nassign     = 4;

  ChunkMap gilbert(Cz, Cy, Cx);
  ChunkMap axis_first(Cz, Cy, Cx, sfc::SfcAxis::X);

  float64 gilbert_imbalance = shock_imbalance(gilbert, Cz, Cy, Cx, dense_width, nprocess, nassign);
  float64 axis_first_imbalance =
      shock_imbalance(axis_first, Cz, Cy, Cx, dense_width, nprocess, nassign);

  CAPTURE(gilbert_imbalance, axis_first_imbalance);
  REQUIRE(axis_first_imbalance < 1.2);
  REQUIRE(axis_first_imbalance < gilbert_imbalance);
}
