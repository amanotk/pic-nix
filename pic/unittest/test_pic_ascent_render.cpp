// -*- C++ -*-

#include "../insitu/ascent_runtime.hpp"
#include "../insitu/blueprint_builder.hpp"

#include "../pic_chunk.hpp"

#include <catch2/catch_test_macros.hpp>

#include <mpi.h>

#include <filesystem>
#include <memory>
#include <vector>

namespace
{
std::unique_ptr<PicChunk> make_render_chunk()
{
  auto chunk = std::make_unique<PicChunk>(nix::Dims3D{1, 2, 2}, nix::Bool3D{false, true, true}, 0);
  chunk->set_boundary_margin(1);
  const int offset[3] = {0, 0, 0};
  const int gdims[3]  = {1, 2, 2};
  chunk->set_global_context(offset, gdims);
  chunk->set_coordinate(0.0, 0.0, 0.0);
  chunk->allocate();

  auto data = chunk->get_internal_data();
  data.uf.fill(0.0);
  data.uj.fill(0.0);
  for (int iy = data.Lby; iy <= data.Uby; iy++) {
    for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
      data.phi(data.Lbz, iy, ix) = static_cast<float64>(ix + 2 * iy);
    }
  }
  return chunk;
}
} // namespace

TEST_CASE("Ascent renders a centered PIC-NIX scalar field")
{
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  REQUIRE(size == 1);

  auto                             chunk  = make_render_chunk();
  const std::vector<PicChunk*>     chunks = {chunk.get()};
  picnix::insitu::BlueprintOptions options;
  options.raw       = false;
  options.particles = false;
  auto publication  = picnix::insitu::BlueprintBuilder::build(chunks, 0, 0.0, options);

  picnix::insitu::AscentRuntime runtime;
  runtime.publish_execute(publication.node, PICNIX_ASCENT_RENDER_ACTIONS_FILE);
  runtime.shutdown();

  const auto image = std::filesystem::path("ascent_picnix_phi.png");
  REQUIRE(std::filesystem::exists(image));
  REQUIRE(std::filesystem::file_size(image) > 0);
}
