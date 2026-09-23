// -*- C++ -*-

#include "nix/diag/ascent/runtime.hpp"
#include "pic/diag/ascent/blueprint_builder.hpp"

#include "../pic_chunk.hpp"

#include <conduit_blueprint.hpp>

#include <catch2/catch_approx.hpp>
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
  chunk->set_coordinate(1.0, 0.5, 0.25);
  auto data = chunk->get_internal_data();
  data.Ns   = 1;
  chunk->allocate();

  data.xlim[0] = 1.0;
  data.ylim[0] = -2.0;
  data.uf.fill(0.0);
  for (std::size_t iy = 0; iy < data.uf.shape(1); iy++) {
    for (std::size_t ix = 0; ix < data.uf.shape(2); ix++) {
      data.uf(data.Lbz, iy, ix, 0) = static_cast<float64>(ix + 2 * iy);
      data.uf(data.Lbz, iy, ix, 1) = 1.0;
    }
  }
  for (int iy = data.Lby; iy <= data.Uby; iy++) {
    for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
      data.um(data.Lbz, iy, ix, 0, 0) = static_cast<float64>(ix + 2 * iy);
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

  auto                         chunk  = make_render_chunk();
  const std::vector<PicChunk*> chunks = {chunk.get()};
  pic_ascent::BlueprintOptions options;
  options.raw_fields    = false;
  options.raw_particles = false;
  auto  publication = pic_ascent::BlueprintBuilder::build(chunks, 0, 0.0, json::object(), options);
  auto& domain      = publication.node["domain_0"];
  conduit::Node info;
  REQUIRE(conduit::blueprint::mesh::verify(domain, info));
  REQUIRE(domain["coordsets/cell_coords/dims/i"].to_int() == 3);
  REQUIRE(domain["coordsets/cell_coords/dims/j"].to_int() == 3);
  REQUIRE(domain["coordsets/cell_coords/origin/x"].to_double() == Catch::Approx(1.0));
  REQUIRE(domain["coordsets/cell_coords/origin/y"].to_double() == Catch::Approx(-2.0));
  REQUIRE(domain["coordsets/cell_coords/spacing/dx"].to_double() == Catch::Approx(0.25));
  REQUIRE(domain["coordsets/cell_coords/spacing/dy"].to_double() == Catch::Approx(0.5));

  const double x_max = domain["coordsets/cell_coords/origin/x"].to_double() +
                       domain["coordsets/cell_coords/spacing/dx"].to_double() *
                           (domain["coordsets/cell_coords/dims/i"].to_int() - 1);
  const double y_max = domain["coordsets/cell_coords/origin/y"].to_double() +
                       domain["coordsets/cell_coords/spacing/dy"].to_double() *
                           (domain["coordsets/cell_coords/dims/j"].to_int() - 1);
  REQUIRE(x_max == Catch::Approx(1.5));
  REQUIRE(y_max == Catch::Approx(-1.0));

  nix::AscentRuntime runtime;
  runtime.publish_execute(publication.node, PICNIX_ASCENT_RENDER_ACTIONS_FILE);
  runtime.shutdown();

  const auto image = std::filesystem::path("ascent_picnix_E.png");
  REQUIRE(std::filesystem::exists(image));
  REQUIRE(std::filesystem::file_size(image) > 0);
  const auto moment_image = std::filesystem::path("ascent_picnix_M0.png");
  REQUIRE(std::filesystem::exists(moment_image));
  REQUIRE(std::filesystem::file_size(moment_image) > 0);
}
