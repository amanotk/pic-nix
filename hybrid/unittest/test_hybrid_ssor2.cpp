// -*- C++ -*-
#include "engine/ssor2.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
void fill_boundary(nix::Array4D<nix::float64>& field, int Lbx, int Ubx, int Lby, int Uby, int Lbz,
                   int Ubz, nix::float64 ex, nix::float64 ey, nix::float64 ez)
{
  for (int iz = Lbz - 1; iz <= Ubz + 1; ++iz) {
    for (int iy = Lby - 1; iy <= Uby + 1; ++iy) {
      field(iz, iy, Lbx - 1, 0) = ex;
      field(iz, iy, Ubx + 1, 0) = ex;
      field(iz, iy, Lbx - 1, 1) = ey;
      field(iz, iy, Ubx + 1, 1) = ey;
      field(iz, iy, Lbx - 1, 2) = ez;
      field(iz, iy, Ubx + 1, 2) = ez;
    }
  }
  for (int iz = Lbz - 1; iz <= Ubz + 1; ++iz) {
    for (int ix = Lbx - 1; ix <= Ubx + 1; ++ix) {
      field(iz, Lby - 1, ix, 0) = ex;
      field(iz, Uby + 1, ix, 0) = ex;
      field(iz, Lby - 1, ix, 1) = ey;
      field(iz, Uby + 1, ix, 1) = ey;
      field(iz, Lby - 1, ix, 2) = ez;
      field(iz, Uby + 1, ix, 2) = ez;
    }
  }
  for (int iy = Lby - 1; iy <= Uby + 1; ++iy) {
    for (int ix = Lbx - 1; ix <= Ubx + 1; ++ix) {
      field(Lbz - 1, iy, ix, 0) = ex;
      field(Ubz + 1, iy, ix, 0) = ex;
      field(Lbz - 1, iy, ix, 1) = ey;
      field(Ubz + 1, iy, ix, 1) = ey;
      field(Lbz - 1, iy, ix, 2) = ez;
      field(Ubz + 1, iy, ix, 2) = ez;
    }
  }
}

nix::Array4D<nix::float64> make_constant_source(int nz, int ny, int nx, nix::float64 coeff,
                                                nix::float64 vx, nix::float64 vy, nix::float64 vz)
{
  nix::Array4D<nix::float64> src = xt::zeros<nix::float64>({nz, ny, nx, 4});
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        src(iz, iy, ix, 0) = coeff;
        src(iz, iy, ix, 1) = vx * coeff;
        src(iz, iy, ix, 2) = vy * coeff;
        src(iz, iy, ix, 3) = vz * coeff;
      }
    }
  }
  return src;
}
} // namespace

TEST_CASE("SSOR2 solver converges on manufactured constant solution")
{
  constexpr int interior_nz = 4;
  constexpr int interior_ny = 4;
  constexpr int interior_nx = 4;
  constexpr int total_nz    = interior_nz + 2;
  constexpr int total_ny    = interior_ny + 2;
  constexpr int total_nx    = interior_nx + 2;

  nix::Array4D<nix::float64> field = xt::zeros<nix::float64>({total_nz, total_ny, total_nx, 6});
  nix::Array4D<nix::float64> source =
      make_constant_source(total_nz, total_ny, total_nx, 1.5, 2.0, -3.0, 1.0);
  fill_boundary(field, 1, interior_nx, 1, interior_ny, 1, interior_nz, 2.0, -3.0, 1.0);

  hybrid::engine::Ssor2Workspace workspace = {field, source,      1, interior_nx,
                                              1,     interior_ny, 1, interior_nz};
  const auto coeff = hybrid::engine::compute_ssor2_coefficients(20.0, 0.5, 2.0, 4.0);
  const hybrid::engine::Ssor2Config config{100, 1.0e-5};

  const auto stats = hybrid::engine::solve_ssor2(workspace, coeff, config);

  REQUIRE(stats.converged);
  for (int iz = 1; iz <= interior_nz; ++iz) {
    for (int iy = 1; iy <= interior_ny; ++iy) {
      for (int ix = 1; ix <= interior_nx; ++ix) {
        REQUIRE(field(iz, iy, ix, 0) == Catch::Approx(2.0).margin(5.0e-8));
        REQUIRE(field(iz, iy, ix, 1) == Catch::Approx(-3.0).margin(5.0e-8));
        REQUIRE(field(iz, iy, ix, 2) == Catch::Approx(1.0).margin(5.0e-8));
      }
    }
  }
}

TEST_CASE("SSOR2 solver converges to zero on constant source with zero boundary")
{
  constexpr int interior_nz = 4;
  constexpr int interior_ny = 4;
  constexpr int interior_nx = 4;
  constexpr int total_nz    = interior_nz + 2;
  constexpr int total_ny    = interior_ny + 2;
  constexpr int total_nx    = interior_nx + 2;

  nix::Array4D<nix::float64>     field = xt::zeros<nix::float64>({total_nz, total_ny, total_nx, 6});
  nix::Array4D<nix::float64>     source = xt::ones<nix::float64>({total_nz, total_ny, total_nx, 4});
  hybrid::engine::Ssor2Workspace workspace = {field, source,      1, interior_nx,
                                              1,     interior_ny, 1, interior_nz};
  const auto coeff = hybrid::engine::compute_ssor2_coefficients(20.0, 0.5, 2.0, 4.0);
  const hybrid::engine::Ssor2Config config{200, 1.0e-5};

  const auto stats = hybrid::engine::solve_ssor2(workspace, coeff, config);

  REQUIRE(stats.converged);
}

TEST_CASE("Legacy SSOR2 applies one globally synchronized iteration sequence")
{
  nix::Array4D<nix::float64>      field                = xt::zeros<nix::float64>({3, 3, 3, 6});
  nix::Array4D<nix::float64>      source               = xt::ones<nix::float64>({3, 3, 3, 4});
  hybrid::engine::OhmSystemView   system               = {field, source, 1, 1, 1, 1, 1, 1};
  int                             system_operations    = 0;
  int                             exchanges            = 0;
  int                             reductions           = 0;
  int                             recorded_iterations  = 0;
  bool                            recorded_convergence = false;
  hybrid::engine::OhmSolveContext context              = {
                   [&](const hybrid::engine::OhmSystemOperation& operation) {
        ++system_operations;
        operation(system);
      },
                   [&]() { ++exchanges; },
                   [&](nix::float64, nix::float64) {
        ++reductions;
        return reductions == 1 ? std::pair<nix::float64, nix::float64>{1.0, 1.0}
                                            : std::pair<nix::float64, nix::float64>{1.0e-12, 1.0};
      },
                   [&](int iteration, const hybrid::engine::OhmSolveStats& stats) {
        recorded_iterations  = iteration;
        recorded_convergence = stats.converged;
      },
  };
  const auto coeff = hybrid::engine::compute_ssor2_coefficients(20.0, 0.5, 2.0, 4.0);
  hybrid::engine::LegacySsor2 solver(coeff, {10, 1.0e-5});

  const auto stats = solver.solve(context);

  REQUIRE(stats.converged);
  REQUIRE(stats.iterations == 2);
  REQUIRE(stats.residual_norm == Catch::Approx(1.0e-6));
  REQUIRE(stats.source_norm == Catch::Approx(1.0));
  REQUIRE(stats.relative_residual == Catch::Approx(1.0e-6));
  REQUIRE(system_operations == 6);
  REQUIRE(exchanges == 4);
  REQUIRE(reductions == 2);
  REQUIRE(recorded_iterations == 2);
  REQUIRE(recorded_convergence);
}

TEST_CASE("Legacy SSOR2 reports iteration exhaustion as failure")
{
  nix::Array4D<nix::float64>      field   = xt::zeros<nix::float64>({3, 3, 3, 6});
  nix::Array4D<nix::float64>      source  = xt::ones<nix::float64>({3, 3, 3, 4});
  hybrid::engine::OhmSystemView   system  = {field, source, 1, 1, 1, 1, 1, 1};
  hybrid::engine::OhmSolveContext context = {
      [&](const hybrid::engine::OhmSystemOperation& operation) { operation(system); },
      []() {},
      [](nix::float64, nix::float64) {
        return std::pair<nix::float64, nix::float64>{1.0, 1.0};
      },
      [](int, const hybrid::engine::OhmSolveStats&) {},
  };
  const auto coeff = hybrid::engine::compute_ssor2_coefficients(20.0, 0.5, 2.0, 4.0);
  hybrid::engine::LegacySsor2 solver(coeff, {2, 1.0e-5});

  const auto stats = solver.solve(context);

  REQUIRE_FALSE(stats.converged);
  REQUIRE(stats.iterations == 2);
}
