// -*- C++ -*-

#include "../insitu/blueprint_builder.hpp"

#include <conduit_blueprint.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>

namespace
{
std::unique_ptr<PicChunk> make_chunk()
{
  auto chunk = std::make_unique<PicChunk>(nix::Dims3D{2, 2, 2}, nix::Bool3D{true, true, true}, 7);
  chunk->set_boundary_margin(1);
  const int offset[3] = {0, 0, 0};
  const int gdims[3]  = {2, 2, 2};
  chunk->set_global_context(offset, gdims);
  chunk->set_coordinate(0.0, 0.0, 0.0);
  chunk->allocate();

  auto data = chunk->get_internal_data();
  for (std::size_t iz = 0; iz < data.uf.shape(0); iz++) {
    for (std::size_t iy = 0; iy < data.uf.shape(1); iy++) {
      for (std::size_t ix = 0; ix < data.uf.shape(2); ix++) {
        for (std::size_t component = 0; component < data.uf.shape(3); component++) {
          data.uf(iz, iy, ix, component) = static_cast<float64>(component + 1);
        }
        for (std::size_t component = 0; component < data.uj.shape(3); component++) {
          data.uj(iz, iy, ix, component) = static_cast<float64>(10 + component);
        }
        data.phi(iz, iy, ix) = 20.0;
      }
    }
  }

  data.up.resize(1);
  data.up[0]    = std::make_shared<ParticleType>();
  data.up[0]->q = -1.0;
  data.up[0]->m = 2.0;
  data.up[0]->allocate(2, true);
  data.up[0]->set_Np_active(1);
  data.up[0]->xu(0, 0) = 0.25;

  return chunk;
}
} // namespace

TEST_CASE("BlueprintBuilder publishes a verifiable domain")
{
  auto                         chunk  = make_chunk();
  const std::vector<PicChunk*> chunks = {chunk.get()};

  picnix::insitu::BlueprintOptions options;
  options.particles         = true;
  auto          publication = picnix::insitu::BlueprintBuilder::build(chunks, 12, 3.5, options);
  auto&         domain      = publication.node["domain_7"];
  conduit::Node info;

  REQUIRE(conduit::blueprint::mesh::verify(domain, info));
  REQUIRE(domain["state/cycle"].to_int() == 12);
  REQUIRE(domain["state/time"].to_double() == Catch::Approx(3.5));
  REQUIRE(domain["picnix/schema_version"].to_int() == picnix::insitu::raw_schema_version);
  REQUIRE(domain["picnix/raw/uf/values"].as_float64_ptr() == chunk->get_internal_data().uf.data());
  REQUIRE(domain["picnix/raw/uf/components"].child(0).as_string() == "Ex");
  REQUIRE(domain["picnix/particles/species_000/np_active"].to_int() == 1);

  REQUIRE(domain["fields/E/values/x"].as_float64_ptr()[0] == Catch::Approx(1.0));
  REQUIRE(domain["fields/B/values/z"].as_float64_ptr()[0] == Catch::Approx(6.0));
  REQUIRE(domain["fields/J/values/x"].as_float64_ptr()[0] == Catch::Approx(11.0));
  REQUIRE(domain["fields/J/values/z"].as_float64_ptr()[0] == Catch::Approx(13.0));
  REQUIRE(domain["fields/rho/values"].as_float64_ptr()[0] == Catch::Approx(10.0));
  REQUIRE(domain["fields/phi/values"].as_float64_ptr()[0] == Catch::Approx(20.0));
}
