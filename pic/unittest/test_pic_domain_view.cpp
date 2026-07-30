// -*- C++ -*-

#include "pic/diag/ascent/domain_view.hpp"
#include "pic/diag/ascent/field_schema.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <mpi.h>

namespace
{
std::unique_ptr<PicChunk> make_chunk(nix::Dims3D dims, nix::Bool3D has_dim, int id)
{
  auto chunk = std::make_unique<PicChunk>(dims, has_dim, id);
  chunk->set_boundary_margin(2);
  const int offset[3] = {2, 3, 4};
  const int gdims[3]  = {dims[0] * 2, dims[1] * 2, dims[2] * 2};
  chunk->set_global_context(offset, gdims);
  chunk->set_coordinate(0.3, 0.2, 0.1);
  auto data = chunk->get_internal_data();
  data.Ns   = 1;
  chunk->allocate();

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

  data.up.resize(1);
  data.up[0] = std::make_shared<ParticleType>();
  data.up[0]->allocate(3, true);
  data.up[0]->set_Np_active(2);

  return chunk;
}
} // namespace

TEST_CASE("DomainView captures normalized PIC publication data")
{
  auto chunk = make_chunk({2, 3, 4}, {true, true, true}, 17);
  auto view  = pic_ascent::DomainView(*chunk, 9, 1.25);
  auto data  = chunk->get_internal_data();

  REQUIRE(view.domain_id == 17);
  REQUIRE(view.dimension == 3);
  REQUIRE(view.cycle == 9);
  REQUIRE(view.time == 1.25);
  REQUIRE(view.global_cell_shape_zyx == std::array<int, 3>{4, 6, 8});
  REQUIRE(view.local_cell_shape_zyx == std::array<int, 3>{2, 3, 4});
  REQUIRE(view.global_offset_zyx == std::array<int, 3>{2, 3, 4});
  REQUIRE(view.active_lower_zyx == std::array<int, 3>{2, 2, 2});
  REQUIRE(view.active_upper_zyx == std::array<int, 3>{3, 4, 5});
  REQUIRE(view.active_axes_zyx == std::array<bool, 3>{true, true, true});
  REQUIRE(view.boundary_margin == 2);
  REQUIRE(view.spacing_xyz == std::array<float64, 3>{0.1, 0.2, 0.3});
  REQUIRE(view.physical_origin_xyz[0] == Catch::Approx(0.4));
  REQUIRE(view.physical_origin_xyz[1] == Catch::Approx(0.6));
  REQUIRE(view.physical_origin_xyz[2] == Catch::Approx(0.6));

  REQUIRE(view.uf.data == data.uf.data());
  REQUIRE(view.uf.shape_zyx == std::array<std::size_t, 3>{6, 7, 8});
  REQUIRE(view.uf.component_count == 6);
  REQUIRE(view.uj.component_count == 4);
  REQUIRE(view.neighbor_domain_ids[0] == -1);
  REQUIRE(view.neighbor_ranks[0] == MPI_PROC_NULL);
  REQUIRE(view.neighbor_domain_ids[13] == 17);
  REQUIRE(view.neighbor_ranks[13] == 3);

  REQUIRE(view.particles.size() == 1);
  REQUIRE(view.particles[0].data == data.up[0]->xu.data());
  REQUIRE(view.particles[0].np_active == 2);
}

TEST_CASE("DomainView selects singleton center planes for inactive dimensions")
{
  SECTION("1D")
  {
    auto chunk = make_chunk({1, 1, 4}, {false, false, true}, 1);
    auto view  = pic_ascent::DomainView(*chunk, 0, 0.0);
    auto data  = chunk->get_internal_data();
    REQUIRE(view.dimension == 1);
    REQUIRE(view.uf.shape_zyx == std::array<std::size_t, 3>{1, 1, 8});
    REQUIRE(view.uf.data == &data.uf(2, 2, 0, 0));
  }

  SECTION("2D")
  {
    auto chunk = make_chunk({1, 3, 4}, {false, true, true}, 2);
    auto view  = pic_ascent::DomainView(*chunk, 0, 0.0);
    auto data  = chunk->get_internal_data();
    REQUIRE(view.dimension == 2);
    REQUIRE(view.uf.shape_zyx == std::array<std::size_t, 3>{1, 7, 8});
    REQUIRE(view.uf.data == &data.uf(2, 0, 0, 0));
  }
}

TEST_CASE("Field schema defines canonical protocol components")
{
  REQUIRE(pic_ascent::schema_version == 1);
  REQUIRE(pic_ascent::uf_components[0] == "Ex");
  REQUIRE(pic_ascent::uf_components[5] == "Bz");
  REQUIRE(pic_ascent::uj_components[0] == "rho");
  REQUIRE(pic_ascent::uj_components[3] == "Jz");
  REQUIRE(pic_ascent::moment_components[0] == "m00");
  REQUIRE(pic_ascent::moment_components[13] == "m13");
}
