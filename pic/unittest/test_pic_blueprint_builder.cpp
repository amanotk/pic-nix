// -*- C++ -*-

#include "pic/diag/ascent/blueprint_builder.hpp"
#include "pic/diag/ascent/field_schema.hpp"

#include <conduit_blueprint.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <mpi.h>

namespace
{
std::unique_ptr<PicChunk> make_chunk(nix::Dims3D dims    = {2, 2, 2},
                                     nix::Bool3D has_dim = {true, true, true}, int id = 7,
                                     int boundary_margin = 1, int species_count = 2)
{
  auto chunk = std::make_unique<PicChunk>(dims, has_dim, id);
  chunk->set_boundary_margin(boundary_margin);
  const int offset[3] = {0, 0, 0};
  const int gdims[3]  = {dims[0], dims[1], dims[2]};
  chunk->set_global_context(offset, gdims);
  chunk->set_coordinate(0.75, 0.5, 0.25);
  auto data = chunk->get_internal_data();
  data.Ns   = species_count;
  chunk->allocate();

  for (std::size_t iz = 0; iz < data.uf.shape(0); iz++) {
    for (std::size_t iy = 0; iy < data.uf.shape(1); iy++) {
      for (std::size_t ix = 0; ix < data.uf.shape(2); ix++) {
        for (std::size_t component = 0; component < data.uf.shape(3); component++) {
          data.uf(iz, iy, ix, component) = static_cast<float64>(component + 1);
        }
        for (std::size_t component = 0; component < data.uj.shape(3); component++) {
          data.uj(iz, iy, ix, component) = 10000.0 * iz + 100.0 * iy + ix + 0.01 * component;
        }
      }
    }
  }
  for (int iz = data.Lbz; iz <= data.Ubz; iz++) {
    for (int iy = data.Lby; iy <= data.Uby; iy++) {
      for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
        for (int species = 0; species < species_count; species++) {
          for (int component = 0; component < 14; component++) {
            data.um(iz, iy, ix, species, component) =
                1.0e6 * species + 1.0e4 * component + 100.0 * iz + 10.0 * iy + ix;
          }
        }
      }
    }
  }

  for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        const int index = 9 * (dz + 1) + 3 * (dy + 1) + (dx + 1);
        chunk->set_nb_id(dz, dy, dx, 100 + index);
        chunk->set_nb_rank(dz, dy, dx, 200 + index);
      }
    }
  }
  chunk->set_nb_id(0, 0, 0, id);
  chunk->set_nb_rank(0, 0, 0, 3);
  chunk->set_nb_id(-1, -1, -1, -1);
  chunk->set_nb_rank(-1, -1, -1, MPI_PROC_NULL);

  data.up.resize(species_count);
  data.up[0] = std::make_shared<ParticleType>();
  data.up[0]->allocate(3, true);
  data.up[0]->set_Np_active(2);
  for (int particle = 0; particle < 3; particle++) {
    for (int component = 0; component < 7; component++) {
      data.up[0]->xu(particle, component) = 100.0 * particle + component;
    }
  }

  return chunk;
}

const json configuration = {
    {"application", {{"basedir", "output"}}},
    {"parameter", {{"Nx", 8}}},
};
} // namespace

TEST_CASE("BlueprintBuilder defaults to the centered protocol dataset")
{
  pic_ascent::BlueprintOptions defaults;
  REQUIRE(defaults.centered);
  REQUIRE_FALSE(defaults.raw);
  REQUIRE_FALSE(defaults.particles);

  auto                         chunk  = make_chunk();
  const std::vector<PicChunk*> chunks = {chunk.get()};
  auto          publication = pic_ascent::BlueprintBuilder::build(chunks, 12, 3.5, configuration);
  auto&         domain      = publication.node["domain_7"];
  conduit::Node info;

  REQUIRE(conduit::blueprint::mesh::verify(domain, info));
  REQUIRE(domain["state/cycle"].to_int() == 12);
  REQUIRE(domain["state/time"].to_double() == Catch::Approx(3.5));
  REQUIRE(domain["coordsets/cell_coords/dims/i"].to_int() == 3);
  REQUIRE(domain["coordsets/cell_coords/dims/j"].to_int() == 3);
  REQUIRE(domain["coordsets/cell_coords/dims/k"].to_int() == 3);
  REQUIRE(domain["topologies/cell_mesh/coordset"].as_string() == "cell_coords");

  REQUIRE(domain["fields/E/values/x"].as_float64_ptr()[0] == Catch::Approx(1.0));
  REQUIRE(domain["fields/B/values/z"].as_float64_ptr()[0] == Catch::Approx(6.0));
  REQUIRE(domain["fields/um00/values/m00"].as_float64_ptr()[0] == Catch::Approx(111.0));
  REQUIRE(domain["fields/um01/values/m13"].as_float64_ptr()[0] == Catch::Approx(1130111.0));

  REQUIRE(domain["pic/schema_version"].to_int() == pic_ascent::schema_version);
  REQUIRE(domain["pic/boundary_margin"].to_int() == 1);
  REQUIRE(json::parse(domain["pic/config"].as_string()) == configuration);

  REQUIRE_FALSE(domain.has_path("fields/J"));
  REQUIRE_FALSE(domain.has_path("fields/rho"));
  REQUIRE_FALSE(domain.has_path("fields/uf"));
  REQUIRE_FALSE(domain.has_path("coordsets/raw_storage_coords"));
  REQUIRE_FALSE(domain.has_path("pic/neighbors"));
  REQUIRE_FALSE(domain.has_path("pic/particles"));
  REQUIRE_FALSE(domain.has_path("ascent_ghosts"));
  REQUIRE_FALSE(domain.has_path("adjsets"));
  REQUIRE_FALSE(domain.has_path("nestsets"));
}

TEST_CASE("BlueprintBuilder preserves the actual centered topology dimension")
{
  SECTION("1D")
  {
    auto chunk                                   = make_chunk({1, 1, 1}, {false, false, true});
    auto data                                    = chunk->get_internal_data();
    data.uf(data.Lbz, data.Lby, data.Lbx, 0)     = 2.0;
    data.uf(data.Lbz, data.Lby, data.Lbx + 1, 0) = 4.0;
    auto  publication = pic_ascent::BlueprintBuilder::build({chunk.get()}, 0, 0.0, configuration);
    auto& coords      = publication.node["domain_7/coordsets/cell_coords"];
    REQUIRE(coords["dims/i"].to_int() == 2);
    REQUIRE(coords.has_path("dims/i"));
    REQUIRE_FALSE(coords.has_path("dims/j"));
    REQUIRE_FALSE(coords.has_path("dims/k"));
    REQUIRE(publication.node["domain_7/fields/E/values/x"].as_float64_ptr()[0] ==
            Catch::Approx(3.0));
  }
  SECTION("2D")
  {
    auto  chunk       = make_chunk({1, 2, 3}, {false, true, true});
    auto  publication = pic_ascent::BlueprintBuilder::build({chunk.get()}, 0, 0.0, configuration);
    auto& coords      = publication.node["domain_7/coordsets/cell_coords"];
    REQUIRE(coords.has_path("dims/i"));
    REQUIRE(coords.has_path("dims/j"));
    REQUIRE_FALSE(coords.has_path("dims/k"));
  }
}

TEST_CASE("BlueprintBuilder verifies raw meshes in every supported dimension")
{
  const auto verify_raw = [](nix::Dims3D dims, nix::Bool3D has_dim) {
    auto                         chunk = make_chunk(dims, has_dim);
    pic_ascent::BlueprintOptions options;
    options.centered = false;
    options.raw      = true;
    auto publication =
        pic_ascent::BlueprintBuilder::build({chunk.get()}, 0, 0.0, configuration, options);
    conduit::Node info;
    REQUIRE(conduit::blueprint::mesh::verify(publication.node["domain_7"], info));
  };

  SECTION("1D")
  {
    verify_raw({1, 1, 3}, {false, false, true});
  }
  SECTION("2D")
  {
    verify_raw({1, 2, 3}, {false, true, true});
  }
  SECTION("3D")
  {
    verify_raw({2, 2, 3}, {true, true, true});
  }
}

TEST_CASE("BlueprintBuilder publishes zero-copy raw fields, neighbors, and active particles")
{
  auto chunk = make_chunk({1, 1, 3}, {false, false, true});
  auto data  = chunk->get_internal_data();

  pic_ascent::BlueprintOptions options;
  options.centered  = false;
  options.raw       = true;
  options.particles = true;
  auto publication =
      pic_ascent::BlueprintBuilder::build({chunk.get()}, 0, 0.0, configuration, options);
  auto&         domain = publication.node["domain_7"];
  conduit::Node info;

  REQUIRE(conduit::blueprint::mesh::verify(domain, info));
  REQUIRE(domain["coordsets/raw_storage_coords/dims/i"].to_int() == 6);
  REQUIRE_FALSE(domain["coordsets/raw_storage_coords"].has_path("dims/j"));
  REQUIRE(domain["fields/uf/values/Ex"].as_float64_ptr() == &data.uf(1, 1, 0, 0));
  REQUIRE(domain["fields/uj/values/Jz"].as_float64_ptr()[0] == Catch::Approx(data.uj(1, 1, 0, 3)));
  REQUIRE(domain["pic/neighbors/domain_ids"].dtype().number_of_elements() == 27);
  REQUIRE(domain["pic/neighbors/domain_ids"].as_int32_ptr()[0] == -1);
  REQUIRE(domain["pic/neighbors/domain_ids"].as_int32_ptr()[13] == 7);
  REQUIRE(domain["pic/neighbors/neighbor_ranks"].as_int32_ptr()[13] == 3);

  auto& particles = domain["pic/particles/particle00/xu"];
  REQUIRE(particles.dtype().is_float64());
  REQUIRE(particles.dtype().number_of_elements() == 14);
  REQUIRE(particles.as_float64_ptr() == data.up[0]->xu.data());
  REQUIRE_FALSE(domain.has_path("pic/particles/particle01"));
  REQUIRE_FALSE(domain.has_path("fields/um"));

  data.uf(1, 1, 0, 0) = 42.5;
  REQUIRE(domain["fields/uf/values/Ex"].as_float64_ptr()[0] == Catch::Approx(42.5));
}

TEST_CASE("BlueprintBuilder assigns shared metadata to the first local domain")
{
  auto first  = make_chunk({1, 1, 2}, {false, false, true}, 17);
  auto second = make_chunk({1, 1, 2}, {false, false, true}, 4);
  auto publication =
      pic_ascent::BlueprintBuilder::build({first.get(), second.get()}, 0, 0.0, configuration);

  REQUIRE(publication.node["domain_17"].has_path("pic/schema_version"));
  REQUIRE(publication.node["domain_17"].has_path("pic/boundary_margin"));
  REQUIRE(publication.node["domain_17"].has_path("pic/config"));
  REQUIRE_FALSE(publication.node["domain_4"].has_path("pic/schema_version"));
  REQUIRE_FALSE(publication.node["domain_4"].has_path("pic/boundary_margin"));
  REQUIRE_FALSE(publication.node["domain_4"].has_path("pic/config"));

  auto mismatch = make_chunk({1, 1, 2}, {false, false, true}, 9, 2);
  REQUIRE_THROWS_AS(
      pic_ascent::BlueprintBuilder::build({first.get(), mismatch.get()}, 0, 0.0, configuration),
      std::invalid_argument);
  REQUIRE(
      pic_ascent::BlueprintBuilder::build({}, 0, 0.0, configuration).node.number_of_children() ==
      0);
}

TEST_CASE("BlueprintBuilder publishes particles independently of raw fields")
{
  auto                         chunk = make_chunk();
  pic_ascent::BlueprintOptions options;
  options.raw       = false;
  options.particles = true;
  auto publication =
      pic_ascent::BlueprintBuilder::build({chunk.get()}, 0, 0.0, configuration, options);
  auto& domain = publication.node["domain_7"];
  REQUIRE(domain.has_path("pic/particles/particle00/xu"));
  REQUIRE_FALSE(domain.has_path("fields/uf"));

  options.centered = false;
  REQUIRE_THROWS_AS(
      pic_ascent::BlueprintBuilder::build({chunk.get()}, 0, 0.0, configuration, options),
      std::invalid_argument);
}
