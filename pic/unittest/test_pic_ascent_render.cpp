// -*- C++ -*-

#include "../insitu/ascent_runtime.hpp"
#include "../insitu/blueprint_builder.hpp"

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
  chunk->set_coordinate(0.0, 0.0, 0.0);
  chunk->allocate();

  auto data    = chunk->get_internal_data();
  data.delx    = 0.25;
  data.dely    = 0.5;
  data.xlim[0] = 1.0;
  data.ylim[0] = -2.0;
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
  options.raw               = false;
  options.particles         = false;
  auto          publication = picnix::insitu::BlueprintBuilder::build(chunks, 0, 0.0, options);
  auto&         domain      = publication.node["domain_0"];
  conduit::Node info;
  REQUIRE(conduit::blueprint::mesh::verify(domain, info));
  REQUIRE(domain["coordsets/cell_vertices/dims/i"].to_int() == 3);
  REQUIRE(domain["coordsets/cell_vertices/dims/j"].to_int() == 3);
  REQUIRE(domain["coordsets/cell_vertices/origin/x"].to_double() == Catch::Approx(1.0));
  REQUIRE(domain["coordsets/cell_vertices/origin/y"].to_double() == Catch::Approx(-2.0));
  REQUIRE(domain["coordsets/cell_vertices/spacing/dx"].to_double() == Catch::Approx(0.25));
  REQUIRE(domain["coordsets/cell_vertices/spacing/dy"].to_double() == Catch::Approx(0.5));

  const double x_max = domain["coordsets/cell_vertices/origin/x"].to_double() +
                       domain["coordsets/cell_vertices/spacing/dx"].to_double() *
                           (domain["coordsets/cell_vertices/dims/i"].to_int() - 1);
  const double y_max = domain["coordsets/cell_vertices/origin/y"].to_double() +
                       domain["coordsets/cell_vertices/spacing/dy"].to_double() *
                           (domain["coordsets/cell_vertices/dims/j"].to_int() - 1);
  REQUIRE(x_max == Catch::Approx(1.5));
  REQUIRE(y_max == Catch::Approx(-1.0));

  picnix::insitu::AscentRuntime runtime;
  runtime.publish_execute(publication.node, PICNIX_ASCENT_RENDER_ACTIONS_FILE);
  runtime.shutdown();

  const auto image = std::filesystem::path("ascent_picnix_phi.png");
  REQUIRE(std::filesystem::exists(image));
  REQUIRE(std::filesystem::file_size(image) > 0);
}
