// -*- C++ -*-

#include "pic_performance.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <mpi.h>

TEST_CASE("PicPerformance validates sampling configuration")
{
  PicPerformance performance;

  REQUIRE(performance.configure(nix::json::object()));
  REQUIRE_FALSE(performance.begin_step(0, 1));

  nix::json config = {{"performance", {{"interval", 10}, {"offset", 3}}}};
  REQUIRE(performance.configure(config));
  REQUIRE_FALSE(performance.begin_step(2, 1));
  REQUIRE(performance.begin_step(3, 1));

  config["performance"]["enabled"] = false;
  REQUIRE(performance.configure(config));
  REQUIRE_FALSE(performance.begin_step(3, 1));

  config["performance"] = {{"interval", 0}};
  REQUIRE_FALSE(performance.configure(config));

  config["performance"] = {{"interval", 10}, {"offset", 10}};
  REQUIRE_FALSE(performance.configure(config));
}

TEST_CASE("PicPerformance aggregates rank and OpenMP summaries")
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  PicPerformance performance;
  nix::json      config = {{"performance", {{"interval", 1}}}};

  REQUIRE(performance.configure(config));
  REQUIRE(performance.begin_step(0, 2));

  int actual_threads = 0;
#pragma omp parallel num_threads(2)
  {
    int parallel_threads = 1;
    int thread           = 0;
#ifdef _OPENMP
    parallel_threads = omp_get_num_threads();
    thread           = omp_get_thread_num();
#endif
#pragma omp single
    {
      actual_threads = parallel_threads;
      performance.set_parallel_threads(parallel_threads);
    }

    if (thread == 0) {
      performance.record_chunk(PicPerformance::Phase::Advance, rank + 1.0);
    }
    performance.record_phase_wall(PicPerformance::Phase::Advance, rank + 2.0);
  }
  REQUIRE(actual_threads > 0);

  nix::json result = performance.finish_step(rank + 1.0, 0.1 * (rank + 1.0));
  if (rank != 0) {
    REQUIRE(result.empty());
    return;
  }

  auto push  = result["push"]["local"];
  auto phase = result["phase"]["advance"];

  REQUIRE(result["schema_version"] == 1);
  REQUIRE(push["size"] == size);
  REQUIRE(push["min"] == 1.0);
  REQUIRE(push["max"] == static_cast<nix::float64>(size));
  REQUIRE(push["mean"] == Catch::Approx(0.5 * (size + 1.0)));
  REQUIRE(push["median"] == Catch::Approx(0.5 * (size + 1.0)));
  REQUIRE(push["p95"] == Catch::Approx(1.0 + 0.95 * (size - 1.0)));
  REQUIRE(push["min_rank"] == 0);
  REQUIRE(push["max_rank"] == size - 1);

  REQUIRE(phase["wall"]["max"] == static_cast<nix::float64>(size + 1));
  REQUIRE(phase["max_chunk"]["max"] == static_cast<nix::float64>(size));
  REQUIRE(phase["omp_efficiency"]["min"] == Catch::Approx(0.5 / actual_threads));
  REQUIRE(phase["omp_efficiency"]["max"] < 1.0);
}
