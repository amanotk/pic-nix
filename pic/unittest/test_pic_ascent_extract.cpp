// -*- C++ -*-

#include "nix/diag/ascent/runtime.hpp"
#include "pic/diag/ascent/blueprint_builder.hpp"

#include "../pic_chunk.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <mpi.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace
{
std::unique_ptr<PicChunk> make_chunk(int rank)
{
  auto chunk =
      std::make_unique<PicChunk>(nix::Dims3D{1, 1, 2}, nix::Bool3D{false, false, true}, rank);
  chunk->set_boundary_margin(1);
  const int offset[3] = {0, 0, rank * 2};
  const int gdims[3]  = {1, 1, 4};
  chunk->set_global_context(offset, gdims);
  chunk->set_coordinate(1.0, 1.0, 0.5);
  auto data = chunk->get_internal_data();
  data.Ns   = 1;
  chunk->allocate();

  for (std::size_t iz = 0; iz < data.uf.shape(0); iz++) {
    for (std::size_t iy = 0; iy < data.uf.shape(1); iy++) {
      for (std::size_t ix = 0; ix < data.uf.shape(2); ix++) {
        for (std::size_t component = 0; component < data.uf.shape(3); component++) {
          data.uf(iz, iy, ix, component) = static_cast<float64>(rank + 1 + component);
        }
        for (std::size_t component = 0; component < data.uj.shape(3); component++) {
          data.uj(iz, iy, ix, component) = static_cast<float64>(10 + rank + component);
        }
      }
    }
  }
  data.um.fill(0.0);
  for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
    for (int component = 0; component < 14; component++) {
      data.um(data.Lbz, data.Lby, ix, 0, component) = static_cast<float64>(rank + 1 + component);
    }
  }

  for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        chunk->set_nb_id(dz, dy, dx, -1);
        chunk->set_nb_rank(dz, dy, dx, MPI_PROC_NULL);
      }
    }
  }
  chunk->set_nb_id(0, 0, 0, rank);
  chunk->set_nb_rank(0, 0, 0, rank);
  chunk->set_nb_id(0, 0, rank == 0 ? 1 : -1, 1 - rank);
  chunk->set_nb_rank(0, 0, rank == 0 ? 1 : -1, 1 - rank);

  data.up.resize(1);
  data.up[0] = std::make_shared<ParticleType>();
  data.up[0]->allocate(2, true);
  data.up[0]->set_Np_active(1);
  for (int component = 0; component < 7; component++) {
    data.up[0]->xu(0, component) = static_cast<float64>(100 * rank + component);
  }
  return chunk;
}
} // namespace

TEST_CASE("Ascent Python extract consumes PIC-NIX domains across two ranks")
{
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    SUCCEED("test requires exactly two MPI ranks");
    return;
  }

  auto                         chunk  = make_chunk(rank);
  const std::vector<PicChunk*> chunks = {chunk.get()};
  pic_ascent::BlueprintOptions options;
  options.raw             = true;
  options.particles       = true;
  const json  config      = {{"parameter", {{"case", "two-rank-protocol"}}}};
  auto        publication = pic_ascent::BlueprintBuilder::build(chunks, 0, 0.0, config, options);
  const auto* m00 = publication.node["domain_" + std::to_string(rank) + "/fields/um00/values/m00"]
                        .as_float64_ptr();
  REQUIRE(m00[0] == Catch::Approx(rank + 1.0));
  REQUIRE(m00[1] == Catch::Approx(rank + 1.0));

  nix::AscentRuntime runtime;
  runtime.publish_execute(publication.node, PICNIX_ASCENT_EXTRACT_ACTIONS_FILE);
  runtime.shutdown();

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::ifstream result("ascent_two_rank_result.txt");
    std::string   value;
    REQUIRE(static_cast<bool>(result));
    REQUIRE(std::getline(result, value));
    REQUIRE(value == "6.0 6.0");
  }
}
