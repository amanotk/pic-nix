// -*- C++ -*-

#include "../insitu/blueprint_builder.hpp"

#include <conduit_blueprint.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>

namespace
{
std::unique_ptr<PicChunk> make_chunk(int id = 7, int x_offset = 0)
{
  auto chunk = std::make_unique<PicChunk>(nix::Dims3D{2, 2, 2}, nix::Bool3D{true, true, true}, id);
  chunk->set_boundary_margin(1);
  const int offset[3] = {0, 0, x_offset};
  const int gdims[3]  = {2, 2, 4};
  chunk->set_global_context(offset, gdims);
  chunk->set_coordinate(static_cast<float64>(x_offset), 0.0, 0.0);
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

TEST_CASE("BlueprintBuilder refreshes external pointers and preserves domain IDs")
{
  auto                         chunk        = make_chunk();
  const std::vector<PicChunk*> first_chunks = {chunk.get()};

  picnix::insitu::BlueprintOptions options;
  options.centered        = false;
  options.particles       = true;
  auto       first        = picnix::insitu::BlueprintBuilder::build(first_chunks, 0, 0.0, options);
  const auto first_uf_ptr = first.node["domain_7/picnix/raw/uf/values"].as_float64_ptr();
  const auto first_particle_ptr =
      first.node["domain_7/picnix/particles/species_000/values"].as_float64_ptr();

  auto data = chunk->get_internal_data();
  data.up[0]->allocate(16, true);
  data.up[0]->set_Np_active(4);
  data.uf.resize({data.uf.shape(0) + 1, data.uf.shape(1), data.uf.shape(2), data.uf.shape(3)});
  const std::vector<PicChunk*> refreshed_chunks = {chunk.get()};
  auto second = picnix::insitu::BlueprintBuilder::build(refreshed_chunks, 1, 0.0, options);

  REQUIRE(second.node["domain_7/picnix/raw/uf/values"].as_float64_ptr() == data.uf.data());
  REQUIRE(second.node["domain_7/picnix/raw/uf/values"].as_float64_ptr() != first_uf_ptr);
  REQUIRE(second.node["domain_7/picnix/particles/species_000/values"].as_float64_ptr() ==
          data.up[0]->xu.data());
  REQUIRE(second.node["domain_7/picnix/particles/species_000/values"].as_float64_ptr() !=
          first_particle_ptr);
  REQUIRE(second.node["domain_7/picnix/particles/species_000/np_allocated"].to_int() == 16);

  auto                         other            = make_chunk(9, 2);
  const std::vector<PicChunk*> reordered_chunks = {other.get(), chunk.get()};
  auto reordered = picnix::insitu::BlueprintBuilder::build(reordered_chunks, 2, 0.0, options);
  REQUIRE(reordered.node.has_path("domain_7"));
  REQUIRE(reordered.node.has_path("domain_9"));
}

TEST_CASE("BlueprintBuilder handles empty and null particle species")
{
  auto chunk = make_chunk();
  auto data  = chunk->get_internal_data();
  data.up.clear();

  picnix::insitu::BlueprintOptions options;
  options.raw                         = true;
  options.centered                    = false;
  options.particles                   = true;
  const std::vector<PicChunk*> chunks = {chunk.get()};
  auto publication = picnix::insitu::BlueprintBuilder::build(chunks, 0, 0.0, options);
  REQUIRE(publication.node.has_path("domain_7"));
  REQUIRE_FALSE(publication.node["domain_7"].has_path("picnix/particles"));

  data.up.resize(1);
  data.up[0].reset();
  publication = picnix::insitu::BlueprintBuilder::build(chunks, 1, 0.0, options);
  REQUIRE(publication.node.has_path("domain_7/picnix/particles/species_000"));
  REQUIRE(publication.node["domain_7/picnix/particles/species_000/np_active"].to_int() == 0);
}
