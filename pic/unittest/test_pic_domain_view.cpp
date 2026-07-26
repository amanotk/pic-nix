// -*- C++ -*-

#include "../insitu/domain_view.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>

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
  chunk->allocate();

  auto data = chunk->get_internal_data();
  data.up.resize(1);
  data.up[0]    = std::make_shared<ParticleType>();
  data.up[0]->q = -1.0;
  data.up[0]->m = 2.0;
  data.up[0]->allocate(3, true);
  data.up[0]->set_Np_active(2);

  return chunk;
}
} // namespace

TEST_CASE("DomainView captures chunk metadata and raw descriptors")
{
  auto chunk = make_chunk({2, 3, 4}, {true, true, true}, 17);
  auto view  = picnix::insitu::DomainView(*chunk, 9, 1.25);

  REQUIRE(view.domain_id == 17);
  REQUIRE(view.dimension == 3);
  REQUIRE(view.cycle == 9);
  REQUIRE(view.time == 1.25);
  REQUIRE(view.global_cell_shape_zyx == std::array<int, 3>{4, 6, 8});
  REQUIRE(view.local_cell_shape_zyx == std::array<int, 3>{2, 3, 4});
  REQUIRE(view.global_offset_zyx == std::array<int, 3>{2, 3, 4});
  REQUIRE(view.active_lower_zyx == std::array<int, 3>{2, 2, 2});
  REQUIRE(view.active_upper_zyx == std::array<int, 3>{3, 4, 5});
  REQUIRE(view.allocated_shape_zyx == std::array<int, 3>{6, 7, 8});
  REQUIRE(view.ghost_width == 2);
  REQUIRE(view.spacing_xyz == std::array<float64, 3>{0.1, 0.2, 0.3});
  REQUIRE(view.physical_origin_xyz[0] == Catch::Approx(0.4));
  REQUIRE(view.physical_origin_xyz[1] == Catch::Approx(0.6));
  REQUIRE(view.physical_origin_xyz[2] == Catch::Approx(0.6));

  REQUIRE(view.uf.data == chunk->get_internal_data().uf.data());
  REQUIRE(view.uf.shape == std::vector<std::size_t>{6, 7, 8, 6});
  REQUIRE(view.uf.components[0] == "Ex");
  REQUIRE(view.uj.components[3] == "Jz");
  REQUIRE(view.um.components[13] == "zx");
  REQUIRE(view.phi.location == "cell");

  REQUIRE(view.particles.size() == 1);
  REQUIRE(view.particles[0].np_active == 2);
  REQUIRE(view.particles[0].np_allocated == 3);
  REQUIRE(view.particles[0].shape == std::vector<std::size_t>{3, 7});
  REQUIRE(view.particles[0].components[6] == "id_bits");
  REQUIRE(view.particles[0].id_encoding == "int64_bits_in_float64_slot");
}

TEST_CASE("DomainView reports reduced-dimensional chunk shapes")
{
  SECTION("1D")
  {
    auto chunk = make_chunk({1, 1, 4}, {false, false, true}, 1);
    auto view  = picnix::insitu::DomainView(*chunk, 0, 0.0);
    REQUIRE(view.dimension == 1);
    REQUIRE(view.local_cell_shape_zyx == std::array<int, 3>{1, 1, 4});
  }

  SECTION("2D")
  {
    auto chunk = make_chunk({1, 3, 4}, {false, true, true}, 2);
    auto view  = picnix::insitu::DomainView(*chunk, 0, 0.0);
    REQUIRE(view.dimension == 2);
    REQUIRE(view.local_cell_shape_zyx == std::array<int, 3>{1, 3, 4});
  }
}
