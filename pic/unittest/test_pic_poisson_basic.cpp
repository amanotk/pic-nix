// -*- C++ -*-

#include "pic_poisson_basic.hpp"

#include "pic_poisson_test_support.hpp"
#include "test_parallel.hpp"

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace
{

using namespace pic_poisson_test;

nix::json make_default_poisson_option()
{
  nix::json option;
  option["poisson_basic"]["max_iter"] = 2000;
  return option;
}

} // namespace

TEST_CASE("PicPoissonBasic solves 1D periodic Poisson", "[np=2]")
{
  if (!require_mpi_size(2)) {
    return;
  }

  const float64     tol             = 1.0e-10;
  const int         mz              = 0;
  const int         my              = 0;
  const int         mx              = 3;
  const nix::Dims3D global_dims     = {1, 1, 16};
  const nix::Dims3D chunk_dims      = {1, 1, 8};
  const nix::Bool3D has_dim         = {false, false, true};
  const auto        proc_dims       = std::array<int, 3>{1, 1, 2};
  const auto        chunk_grid_dims = make_chunk_grid_dims(global_dims, chunk_dims);
  nix::json         config          = make_default_config();

  REQUIRE(chunk_grid_dims[0] % proc_dims[0] == 0);
  REQUIRE(chunk_grid_dims[1] % proc_dims[1] == 0);
  REQUIRE(chunk_grid_dims[2] % proc_dims[2] == 0);

  std::vector<std::unique_ptr<PicChunk>> chunkvec;
  initialize_chunkvec(chunkvec, get_mpi_rank(), proc_dims, global_dims, chunk_dims, has_dim, mz, my,
                      mx, config);

  PicPoissonBasic poisson(global_dims, 1.0);
  poisson.set_option(make_default_poisson_option());
  poisson.bind_chunks(chunkvec);
  auto accessor = poisson.get_accessor();

  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  REQUIRE(poisson.solve(accessor) == 0);
  poisson.copy_sol_to_chunk(accessor);

  require_rms_error_below(chunkvec, mz, my, mx, global_dims, tol);
}

TEST_CASE("PicPoissonBasic solves 2D periodic Poisson", "[np=4]")
{
  if (!require_mpi_size(4)) {
    return;
  }

  const float64     tol             = 1.0e-10;
  const int         mz              = 0;
  const int         my              = 2;
  const int         mx              = 3;
  const nix::Dims3D global_dims     = {1, 32, 24};
  const nix::Dims3D chunk_dims      = {1, 8, 6};
  const nix::Bool3D has_dim         = {false, true, true};
  const auto        proc_dims       = std::array<int, 3>{1, 2, 2};
  const auto        chunk_grid_dims = make_chunk_grid_dims(global_dims, chunk_dims);
  nix::json         config          = make_default_config();

  REQUIRE(chunk_grid_dims[0] % proc_dims[0] == 0);
  REQUIRE(chunk_grid_dims[1] % proc_dims[1] == 0);
  REQUIRE(chunk_grid_dims[2] % proc_dims[2] == 0);

  std::vector<std::unique_ptr<PicChunk>> chunkvec;
  initialize_chunkvec(chunkvec, get_mpi_rank(), proc_dims, global_dims, chunk_dims, has_dim, mz, my,
                      mx, config);

  PicPoissonBasic poisson(global_dims, 1.0);
  poisson.set_option(make_default_poisson_option());
  poisson.bind_chunks(chunkvec);
  auto accessor = poisson.get_accessor();

  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  REQUIRE(poisson.solve(accessor) == 0);
  poisson.copy_sol_to_chunk(accessor);

  require_rms_error_below(chunkvec, mz, my, mx, global_dims, tol);
}

TEST_CASE("PicPoissonBasic solves 3D periodic Poisson", "[np=8]")
{
  if (!require_mpi_size(8)) {
    return;
  }

  const float64     tol             = 1.0e-10;
  const int         mz              = 2;
  const int         my              = 3;
  const int         mx              = 4;
  const nix::Dims3D global_dims     = {32, 32, 32};
  const nix::Dims3D chunk_dims      = {8, 8, 8};
  const nix::Bool3D has_dim         = {true, true, true};
  const auto        proc_dims       = std::array<int, 3>{2, 2, 2};
  const auto        chunk_grid_dims = make_chunk_grid_dims(global_dims, chunk_dims);
  nix::json         config          = make_default_config();

  REQUIRE(chunk_grid_dims[0] % proc_dims[0] == 0);
  REQUIRE(chunk_grid_dims[1] % proc_dims[1] == 0);
  REQUIRE(chunk_grid_dims[2] % proc_dims[2] == 0);

  std::vector<std::unique_ptr<PicChunk>> chunkvec;
  initialize_chunkvec(chunkvec, get_mpi_rank(), proc_dims, global_dims, chunk_dims, has_dim, mz, my,
                      mx, config);

  PicPoissonBasic poisson(global_dims, 1.0);
  poisson.set_option(make_default_poisson_option());
  poisson.bind_chunks(chunkvec);
  auto accessor = poisson.get_accessor();

  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  REQUIRE(poisson.solve(accessor) == 0);
  poisson.copy_sol_to_chunk(accessor);

  require_rms_error_below(chunkvec, mz, my, mx, global_dims, tol);
}

TEST_CASE("PicPoissonBasic phi ghost exchange", "[np=2]")
{
  if (!require_mpi_size(2)) {
    return;
  }

  const float64     tol             = 1.0e-10;
  const int         rank            = get_mpi_rank();
  const int         mz              = 0;
  const int         my              = 0;
  const int         mx              = 2;
  const nix::Dims3D global_dims     = {1, 1, 16};
  const nix::Dims3D chunk_dims      = {1, 1, 4};
  const nix::Bool3D has_dim         = {false, false, true};
  const auto        proc_dims       = std::array<int, 3>{1, 1, 2};
  const auto        chunk_grid_dims = make_chunk_grid_dims(global_dims, chunk_dims);
  nix::json         config          = make_default_config();

  REQUIRE(chunk_grid_dims[0] % proc_dims[0] == 0);
  REQUIRE(chunk_grid_dims[1] % proc_dims[1] == 0);
  REQUIRE(chunk_grid_dims[2] % proc_dims[2] == 0);

  std::vector<std::unique_ptr<PicChunk>> chunkvec;
  initialize_chunkvec(chunkvec, rank, proc_dims, global_dims, chunk_dims, has_dim, mz, my, mx,
                      config);

  PicPoissonBasic poisson(global_dims, 1.0);
  poisson.set_option(make_default_poisson_option());
  poisson.bind_chunks(chunkvec);
  auto accessor = poisson.get_accessor();

  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  REQUIRE(poisson.solve(accessor) == 0);
  poisson.copy_sol_to_chunk(accessor);

  for (auto& chunk : chunkvec) {
    chunk->set_boundary_pack(BoundaryPhi);
    chunk->set_boundary_begin(BoundaryPhi);
  }
  for (auto& chunk : chunkvec) {
    chunk->set_boundary_end(BoundaryPhi);
    chunk->set_boundary_unpack(BoundaryPhi);
  }

  for (const auto& base_chunk : chunkvec) {
    auto* chunk = dynamic_cast<TestPicChunk*>(base_chunk.get());
    REQUIRE(chunk != nullptr);
    auto          data   = chunk->get_internal_data();
    const float64 lx     = static_cast<float64>(global_dims[2]) * data.delx;
    const float64 ly     = static_cast<float64>(global_dims[1]) * data.dely;
    const float64 lz     = static_cast<float64>(global_dims[0]) * data.delz;
    const float64 kx     = static_cast<float64>(mx) * nix::math::pi2 / lx;
    const float64 ky     = static_cast<float64>(my) * nix::math::pi2 / ly;
    const float64 kz     = static_cast<float64>(mz) * nix::math::pi2 / lz;
    const float64 kappax = compute_kappa_component(kx, data.delx);
    const float64 kappay = compute_kappa_component(ky, data.dely);
    const float64 kappaz = compute_kappa_component(kz, data.delz);
    const float64 kappa2_sum =
        kappax * kappax + kappay * kappay + kappaz * kappaz + static_cast<float64>(1.0e-32);

    for (int ix = 0; ix < data.Lbx; ++ix) {
      auto          coords   = chunk->get_coordinates(0, 0, ix);
      const float64 x        = coords[2];
      const float64 y        = coords[1];
      const float64 z        = coords[0];
      const float64 expected = analytic_solution(kz, ky, kx, kappa2_sum, z, y, x);
      const float64 diff     = std::abs(data.phi(0, 0, ix) - expected);
      REQUIRE(diff < tol);
    }
    for (int ix = data.Ubx + 1; ix <= data.Ubx + data.boundary_margin; ++ix) {
      auto          coords   = chunk->get_coordinates(0, 0, ix);
      const float64 x        = coords[2];
      const float64 y        = coords[1];
      const float64 z        = coords[0];
      const float64 expected = analytic_solution(kz, ky, kx, kappa2_sum, z, y, x);
      const float64 diff     = std::abs(data.phi(0, 0, ix) - expected);
      REQUIRE(diff < tol);
    }
  }
}
