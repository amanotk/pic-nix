// -*- C++ -*-

#include "../insitu/ascent_runtime.hpp"
#include "../insitu/blueprint_builder.hpp"

#include "../pic_chunk.hpp"

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
  chunk->set_coordinate(0.0, 0.0, static_cast<float64>(rank * 2));
  chunk->allocate();

  auto data = chunk->get_internal_data();
  data.uf.fill(0.0);
  data.uj.fill(0.0);
  data.phi.fill(static_cast<float64>(rank + 1));
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

  auto                             chunk  = make_chunk(rank);
  const std::vector<PicChunk*>     chunks = {chunk.get()};
  picnix::insitu::BlueprintOptions options;
  options.raw       = false;
  options.particles = false;
  auto publication  = picnix::insitu::BlueprintBuilder::build(chunks, 0, 0.0, options);

  picnix::insitu::AscentRuntime runtime;
  runtime.publish_execute(publication.node, PICNIX_ASCENT_EXTRACT_ACTIONS_FILE);
  runtime.shutdown();

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::ifstream result("ascent_two_rank_result.txt");
    std::string   value;
    REQUIRE(static_cast<bool>(result));
    REQUIRE(std::getline(result, value));
    REQUIRE(value == "6.0");
  }
}
