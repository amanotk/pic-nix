// -*- C++ -*-

#include "pic_poisson_basic.hpp"
#include "pic_poisson_basic_primitives.hpp"

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

int primitive_index(int iz, int iy, int ix, int ny, int nx)
{
  return iz * ny * nx + iy * nx + ix;
}

bool is_interior(const pic_poisson_basic_primitives::OperatorChunkData& cw, int iz, int iy, int ix)
{
  return cw.Lbz <= iz && iz <= cw.Ubz && cw.Lby <= iy && iy <= cw.Uby && cw.Lbx <= ix &&
         ix <= cw.Ubx;
}

bool is_interior(const pic_poisson_basic_primitives::PreconditionerChunkData& cw, int iz, int iy,
                 int ix)
{
  return cw.Lbz <= iz && iz <= cw.Ubz && cw.Lby <= iy && iy <= cw.Uby && cw.Lbx <= ix &&
         ix <= cw.Ubx;
}

void require_operator_result(const pic_poisson_basic_primitives::OperatorChunkData& cw,
                             const std::vector<float64>& in, const std::vector<float64>& out,
                             float64 sentinel, int ndim)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 dy2  = 1.0 / (cw.dely * cw.dely);
  const float64 dz2  = 1.0 / (cw.delz * cw.delz);
  const float64 ofdx = -dx2;
  const float64 ofdy = -dy2;
  const float64 ofdz = -dz2;

  for (int iz = 0; iz < cw.nz; ++iz) {
    for (int iy = 0; iy < cw.ny; ++iy) {
      for (int ix = 0; ix < cw.nx; ++ix) {
        const int idx = primitive_index(iz, iy, ix, cw.ny, cw.nx);
        if (!is_interior(cw, iz, iy, ix)) {
          REQUIRE((out[idx] == sentinel));
          continue;
        }

        const int idx_xm   = primitive_index(iz, iy, ix - 1, cw.ny, cw.nx);
        const int idx_xp   = primitive_index(iz, iy, ix + 1, cw.ny, cw.nx);
        float64   expected = (2.0 * dx2) * in[idx] + ofdx * (in[idx_xm] + in[idx_xp]);

        if (ndim >= 2) {
          const int idx_ym = primitive_index(iz, iy - 1, ix, cw.ny, cw.nx);
          const int idx_yp = primitive_index(iz, iy + 1, ix, cw.ny, cw.nx);
          expected += (2.0 * dy2) * in[idx] + ofdy * (in[idx_ym] + in[idx_yp]);
        }
        if (ndim >= 3) {
          const int idx_zm = primitive_index(iz - 1, iy, ix, cw.ny, cw.nx);
          const int idx_zp = primitive_index(iz + 1, iy, ix, cw.ny, cw.nx);
          expected += (2.0 * dz2) * in[idx] + ofdz * (in[idx_zm] + in[idx_zp]);
        }

        REQUIRE((std::abs(out[idx] - expected) < 1.0e-12));
      }
    }
  }
}

void apply_preconditioner_reference(const pic_poisson_basic_primitives::PreconditionerChunkData& cw,
                                    std::vector<float64>& z, int ndim, bool forward)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 dy2  = 1.0 / (cw.dely * cw.dely);
  const float64 dz2  = 1.0 / (cw.delz * cw.delz);
  const float64 ofdx = -dx2;
  const float64 ofdy = -dy2;
  const float64 ofdz = -dz2;

  const int iz_beg = forward ? cw.Lbz : cw.Ubz;
  const int iz_end = forward ? cw.Ubz : cw.Lbz;
  const int iy_beg = forward ? cw.Lby : cw.Uby;
  const int iy_end = forward ? cw.Uby : cw.Lby;
  const int ix_beg = forward ? cw.Lbx : cw.Ubx;
  const int ix_end = forward ? cw.Ubx : cw.Lbx;
  const int step   = forward ? 1 : -1;

  for (int iz = iz_beg; forward ? (iz <= iz_end) : (iz >= iz_end); iz += step) {
    for (int iy = iy_beg; forward ? (iy <= iy_end) : (iy >= iy_end); iy += step) {
      for (int ix = ix_beg; forward ? (ix <= ix_end) : (ix >= ix_end); ix += step) {
        const int idx    = primitive_index(iz, iy, ix, cw.ny, cw.nx);
        const int idx_xm = primitive_index(iz, iy, ix - 1, cw.ny, cw.nx);
        const int idx_xp = primitive_index(iz, iy, ix + 1, cw.ny, cw.nx);
        float64   diag   = 2.0 * dx2;
        float64   sum    = cw.r[idx];
        sum -= ofdx * (z[idx_xm] + z[idx_xp]);

        if (ndim >= 2) {
          const int idx_ym = primitive_index(iz, iy - 1, ix, cw.ny, cw.nx);
          const int idx_yp = primitive_index(iz, iy + 1, ix, cw.ny, cw.nx);
          diag += 2.0 * dy2;
          sum -= ofdy * (z[idx_ym] + z[idx_yp]);
        }
        if (ndim >= 3) {
          const int idx_zm = primitive_index(iz - 1, iy, ix, cw.ny, cw.nx);
          const int idx_zp = primitive_index(iz + 1, iy, ix, cw.ny, cw.nx);
          diag += 2.0 * dz2;
          sum -= ofdz * (z[idx_zm] + z[idx_zp]);
        }

        const float64 z_new = (diag > 0.0) ? sum / diag : cw.r[idx];
        z[idx]              = (1.0 - cw.omega) * z[idx] + cw.omega * z_new;
      }
    }
  }
}

void require_preconditioner_result(const pic_poisson_basic_primitives::PreconditionerChunkData& cw,
                                   const std::vector<float64>& z_before,
                                   const std::vector<float64>& z_after, int ndim, bool forward)
{
  std::vector<float64> expected = z_before;
  apply_preconditioner_reference(cw, expected, ndim, forward);

  for (int iz = 0; iz < cw.nz; ++iz) {
    for (int iy = 0; iy < cw.ny; ++iy) {
      for (int ix = 0; ix < cw.nx; ++ix) {
        const int idx = primitive_index(iz, iy, ix, cw.ny, cw.nx);
        if (!is_interior(cw, iz, iy, ix)) {
          REQUIRE((z_after[idx] == z_before[idx]));
          continue;
        }
        REQUIRE((std::abs(z_after[idx] - expected[idx]) < 1.0e-12));
      }
    }
  }
}

} // namespace

TEST_CASE("PicPoissonBasic apply_operator_1d primitive stencil", "[np=2]")
{
  const float64        sentinel = -1234.5;
  std::vector<float64> in(3 * 3 * 8, 0.0);
  std::vector<float64> out(in.size(), sentinel);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float64>(i + 1) * 0.25;
  }

  pic_poisson_basic_primitives::OperatorChunkData cw;
  cw.in   = in.data();
  cw.out  = out.data();
  cw.nz   = 3;
  cw.ny   = 3;
  cw.nx   = 8;
  cw.Lbz  = 1;
  cw.Ubz  = 1;
  cw.Lby  = 1;
  cw.Uby  = 1;
  cw.Lbx  = 2;
  cw.Ubx  = 5;
  cw.delx = 0.7;

  pic_poisson_basic_primitives::apply_operator_1d_primitive(cw);
  require_operator_result(cw, in, out, sentinel, 1);

  std::vector<float64> in2 = in;
  std::vector<float64> out2(out.size(), sentinel);
  for (size_t i = 0; i < in2.size(); ++i) {
    in2[i] = in2[i] * 1.5 - 0.2;
  }
  cw.in  = in2.data();
  cw.out = out2.data();
  pic_poisson_basic_primitives::apply_operator_1d_primitive(cw);
  require_operator_result(cw, in2, out2, sentinel, 1);
}

TEST_CASE("PicPoissonBasic apply_operator_2d primitive stencil", "[np=2]")
{
  const float64        sentinel = -4321.0;
  std::vector<float64> in(3 * 6 * 7, 0.0);
  std::vector<float64> out(in.size(), sentinel);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = 0.1 + static_cast<float64>(i) * 0.125;
  }

  pic_poisson_basic_primitives::OperatorChunkData cw;
  cw.in   = in.data();
  cw.out  = out.data();
  cw.nz   = 3;
  cw.ny   = 6;
  cw.nx   = 7;
  cw.Lbz  = 1;
  cw.Ubz  = 1;
  cw.Lby  = 1;
  cw.Uby  = 4;
  cw.Lbx  = 1;
  cw.Ubx  = 5;
  cw.delx = 0.4;
  cw.dely = 1.7;

  pic_poisson_basic_primitives::apply_operator_2d_primitive(cw);
  require_operator_result(cw, in, out, sentinel, 2);
}

TEST_CASE("PicPoissonBasic apply_operator_3d primitive stencil", "[np=2]")
{
  const float64        sentinel = -77.0;
  std::vector<float64> in(6 * 5 * 7, 0.0);
  std::vector<float64> out(in.size(), sentinel);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float64>(static_cast<int>(i % 11) - 3) * 0.3;
  }

  pic_poisson_basic_primitives::OperatorChunkData cw;
  cw.in   = in.data();
  cw.out  = out.data();
  cw.nz   = 6;
  cw.ny   = 5;
  cw.nx   = 7;
  cw.Lbz  = 1;
  cw.Ubz  = 4;
  cw.Lby  = 1;
  cw.Uby  = 3;
  cw.Lbx  = 1;
  cw.Ubx  = 5;
  cw.delx = 0.5;
  cw.dely = 0.8;
  cw.delz = 1.3;

  pic_poisson_basic_primitives::apply_operator_3d_primitive(cw);
  require_operator_result(cw, in, out, sentinel, 3);
}

TEST_CASE("PicPoissonBasic preconditioner forward 1d primitive", "[np=2]")
{
  std::vector<float64> r(3 * 3 * 8, 0.0);
  std::vector<float64> z(3 * 3 * 8, 0.0);
  for (size_t i = 0; i < r.size(); ++i) {
    r[i] = static_cast<float64>(i + 3) * 0.2;
    z[i] = static_cast<float64>(i % 7) * 0.05;
  }
  const std::vector<float64> z_before = z;

  pic_poisson_basic_primitives::PreconditionerChunkData cw;
  cw.r     = r.data();
  cw.z     = z.data();
  cw.nz    = 3;
  cw.ny    = 3;
  cw.nx    = 8;
  cw.Lbz   = 1;
  cw.Ubz   = 1;
  cw.Lby   = 1;
  cw.Uby   = 1;
  cw.Lbx   = 2;
  cw.Ubx   = 5;
  cw.delx  = 0.7;
  cw.omega = 1.2;

  pic_poisson_basic_primitives::preconditioner_forward_1d_primitive(cw);
  require_preconditioner_result(cw, z_before, z, 1, true);
}

TEST_CASE("PicPoissonBasic preconditioner backward 1d primitive", "[np=2]")
{
  std::vector<float64> r(3 * 3 * 8, 0.0);
  std::vector<float64> z(3 * 3 * 8, 0.0);
  for (size_t i = 0; i < r.size(); ++i) {
    r[i] = static_cast<float64>(i + 1) * 0.3;
    z[i] = static_cast<float64>(i % 5) * 0.08;
  }
  const std::vector<float64> z_before = z;

  pic_poisson_basic_primitives::PreconditionerChunkData cw;
  cw.r     = r.data();
  cw.z     = z.data();
  cw.nz    = 3;
  cw.ny    = 3;
  cw.nx    = 8;
  cw.Lbz   = 1;
  cw.Ubz   = 1;
  cw.Lby   = 1;
  cw.Uby   = 1;
  cw.Lbx   = 2;
  cw.Ubx   = 5;
  cw.delx  = 0.7;
  cw.omega = 1.1;

  pic_poisson_basic_primitives::preconditioner_backward_1d_primitive(cw);
  require_preconditioner_result(cw, z_before, z, 1, false);
}

TEST_CASE("PicPoissonBasic preconditioner forward 2d primitive", "[np=2]")
{
  std::vector<float64> r(3 * 6 * 7, 0.0);
  std::vector<float64> z(3 * 6 * 7, 0.0);
  for (size_t i = 0; i < r.size(); ++i) {
    r[i] = 0.1 + static_cast<float64>(i) * 0.1;
    z[i] = static_cast<float64>(static_cast<int>(i % 9) - 3) * 0.04;
  }
  const std::vector<float64> z_before = z;

  pic_poisson_basic_primitives::PreconditionerChunkData cw;
  cw.r     = r.data();
  cw.z     = z.data();
  cw.nz    = 3;
  cw.ny    = 6;
  cw.nx    = 7;
  cw.Lbz   = 1;
  cw.Ubz   = 1;
  cw.Lby   = 1;
  cw.Uby   = 4;
  cw.Lbx   = 1;
  cw.Ubx   = 5;
  cw.delx  = 0.4;
  cw.dely  = 1.7;
  cw.omega = 1.25;

  pic_poisson_basic_primitives::preconditioner_forward_2d_primitive(cw);
  require_preconditioner_result(cw, z_before, z, 2, true);
}

TEST_CASE("PicPoissonBasic preconditioner backward 2d primitive", "[np=2]")
{
  std::vector<float64> r(3 * 6 * 7, 0.0);
  std::vector<float64> z(3 * 6 * 7, 0.0);
  for (size_t i = 0; i < r.size(); ++i) {
    r[i] = 0.2 + static_cast<float64>(i) * 0.07;
    z[i] = static_cast<float64>(i % 11) * 0.03;
  }
  const std::vector<float64> z_before = z;

  pic_poisson_basic_primitives::PreconditionerChunkData cw;
  cw.r     = r.data();
  cw.z     = z.data();
  cw.nz    = 3;
  cw.ny    = 6;
  cw.nx    = 7;
  cw.Lbz   = 1;
  cw.Ubz   = 1;
  cw.Lby   = 1;
  cw.Uby   = 4;
  cw.Lbx   = 1;
  cw.Ubx   = 5;
  cw.delx  = 0.4;
  cw.dely  = 1.7;
  cw.omega = 1.15;

  pic_poisson_basic_primitives::preconditioner_backward_2d_primitive(cw);
  require_preconditioner_result(cw, z_before, z, 2, false);
}

TEST_CASE("PicPoissonBasic preconditioner forward 3d primitive", "[np=2]")
{
  std::vector<float64> r(6 * 5 * 7, 0.0);
  std::vector<float64> z(6 * 5 * 7, 0.0);
  for (size_t i = 0; i < r.size(); ++i) {
    r[i] = static_cast<float64>(i + 4) * 0.06;
    z[i] = static_cast<float64>(static_cast<int>(i % 13) - 6) * 0.02;
  }
  const std::vector<float64> z_before = z;

  pic_poisson_basic_primitives::PreconditionerChunkData cw;
  cw.r     = r.data();
  cw.z     = z.data();
  cw.nz    = 6;
  cw.ny    = 5;
  cw.nx    = 7;
  cw.Lbz   = 1;
  cw.Ubz   = 4;
  cw.Lby   = 1;
  cw.Uby   = 3;
  cw.Lbx   = 1;
  cw.Ubx   = 5;
  cw.delx  = 0.5;
  cw.dely  = 0.8;
  cw.delz  = 1.3;
  cw.omega = 1.3;

  pic_poisson_basic_primitives::preconditioner_forward_3d_primitive(cw);
  require_preconditioner_result(cw, z_before, z, 3, true);
}

TEST_CASE("PicPoissonBasic preconditioner backward 3d primitive", "[np=2]")
{
  std::vector<float64> r(6 * 5 * 7, 0.0);
  std::vector<float64> z(6 * 5 * 7, 0.0);
  for (size_t i = 0; i < r.size(); ++i) {
    r[i] = static_cast<float64>(i + 2) * 0.09;
    z[i] = static_cast<float64>(i % 17) * 0.015;
  }
  const std::vector<float64> z_before = z;

  pic_poisson_basic_primitives::PreconditionerChunkData cw;
  cw.r     = r.data();
  cw.z     = z.data();
  cw.nz    = 6;
  cw.ny    = 5;
  cw.nx    = 7;
  cw.Lbz   = 1;
  cw.Ubz   = 4;
  cw.Lby   = 1;
  cw.Uby   = 3;
  cw.Lbx   = 1;
  cw.Ubx   = 5;
  cw.delx  = 0.5;
  cw.dely  = 0.8;
  cw.delz  = 1.3;
  cw.omega = 1.05;

  pic_poisson_basic_primitives::preconditioner_backward_3d_primitive(cw);
  require_preconditioner_result(cw, z_before, z, 3, false);
}

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

TEST_CASE("PicPoissonBasic preserves phi on non-convergence", "[np=2]")
{
  if (!require_mpi_size(2)) {
    return;
  }

  const int         rank            = get_mpi_rank();
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
  initialize_chunkvec(chunkvec, rank, proc_dims, global_dims, chunk_dims, has_dim, mz, my, mx,
                      config);

  std::vector<std::vector<float64>> phi_before;
  phi_before.reserve(chunkvec.size());
  for (size_t ichunk = 0; ichunk < chunkvec.size(); ++ichunk) {
    auto data = chunkvec[ichunk]->get_internal_data();
    for (size_t i = 0; i < data.phi.size(); ++i) {
      data.phi.data()[i] = 10.0 * static_cast<float64>(rank + 1) +
                           static_cast<float64>(ichunk + 1) + 0.01 * static_cast<float64>(i + 1);
    }
    phi_before.emplace_back(data.phi.data(), data.phi.data() + data.phi.size());
  }

  nix::json option;
  option["poisson_basic"]["max_iter"] = 0;

  PicPoissonBasic poisson(global_dims, 1.0);
  poisson.set_option(option);
  poisson.bind_chunks(chunkvec);
  auto accessor = poisson.get_accessor();

  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  REQUIRE(poisson.solve(accessor) != 0);

  for (size_t ichunk = 0; ichunk < chunkvec.size(); ++ichunk) {
    auto data = chunkvec[ichunk]->get_internal_data();
    REQUIRE(phi_before[ichunk].size() == data.phi.size());
    for (size_t i = 0; i < data.phi.size(); ++i) {
      REQUIRE(data.phi.data()[i] == phi_before[ichunk][i]);
    }
  }
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
