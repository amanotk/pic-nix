// -*- C++ -*-
#include "engine/ohm_bridge.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
constexpr double tolerance = 1.0e-13;
}

TEST_CASE("fluid_to_moment matches legacy combined fluid moment")
{
  const hybrid::engine::FluidState fluid        = {1.2, 0.3,  -0.2, 0.4,  0.9485643171120767,
                                                   0.8, -0.1, 0.5,  -0.3, 0.6};
  const double                     electron_qm  = -2;
  const double                     ion_qm       = 0.5;
  const double                     electron_qm2 = nix::math::pi4 * electron_qm * electron_qm;
  const double                     ion_qm2      = nix::math::pi4 * ion_qm * ion_qm;
  const auto                       m = hybrid::engine::fluid_to_moment(fluid, ion_qm, electron_qm);
  REQUIRE(m[hybrid::moment_component::density] ==
          Catch::Approx(electron_qm2 * 1.2 + ion_qm2 * 0.8));
  REQUIRE(m[hybrid::moment_component::momentum_x] ==
          Catch::Approx(electron_qm2 * 1.2 * 0.3 + ion_qm2 * 0.8 * (-0.1)));
  REQUIRE(m[hybrid::moment_component::stress_xx] ==
          Catch::Approx(nix::math::pi4 * electron_qm * (1.2 * 0.3 * 0.3 + 0.9485643171120767) +
                        nix::math::pi4 * ion_qm * (0.8 * (-0.1) * (-0.1) + 0.6)));
  REQUIRE(m[hybrid::moment_component::stress_xy] ==
          Catch::Approx(nix::math::pi4 * electron_qm * 1.2 * 0.3 * (-0.2) +
                        nix::math::pi4 * ion_qm * 0.8 * (-0.1) * 0.5));
}

TEST_CASE("accelerated fluid + kinetic moment accumulation fills ohm_moment")
{
  constexpr int nz = 3;
  constexpr int ny = 3;
  constexpr int nx = 3;

  nix::Array4D<nix::float64> ohm =
      xt::zeros<nix::float64>({nz, ny, nx, hybrid::num_moment_components});
  nix::Array5D<nix::float64> kin =
      xt::zeros<nix::float64>({nz, ny, nx, 2, hybrid::num_moment_components});

  kin(1, 1, 1, 0, hybrid::moment_component::density)    = 0.1;
  kin(1, 1, 1, 0, hybrid::moment_component::momentum_x) = 0.02;
  kin(1, 1, 1, 0, hybrid::moment_component::stress_xx)  = 0.03;
  kin(1, 1, 1, 1, hybrid::moment_component::density)    = 0.2;
  kin(1, 1, 1, 1, hybrid::moment_component::stress_xx)  = 0.04;

  const hybrid::engine::FluidState fluid       = {1.0, 0.5, 0, 0, 0.3, 0.7, 0.2, 0, 0, 0.4};
  const double                     electron_qm = -2;
  const double                     ion_qm      = 0.5;
  hybrid::engine::accumulate_fluid_moment(fluid, ion_qm, electron_qm, ohm, 1, 1, 1);
  hybrid::engine::accumulate_kinetic_moments(kin, ohm, 1, 1, 1, {1.5, -0.25});

  REQUIRE(
      ohm(1, 1, 1, hybrid::moment_component::density) ==
      Catch::Approx(nix::math::pi4 * (4 * 1.0 + 0.25 * 0.7 + 1.5 * 1.5 * 0.1 + 0.25 * 0.25 * 0.2)));
  REQUIRE(ohm(1, 1, 1, hybrid::moment_component::momentum_x) ==
          Catch::Approx(nix::math::pi4 * (4 * 1.0 * 0.5 + 0.25 * 0.7 * 0.2 + 1.5 * 1.5 * 0.02)));
  REQUIRE(ohm(1, 1, 1, hybrid::moment_component::momentum_y) == Catch::Approx(0.0));
  REQUIRE(
      ohm(1, 1, 1, hybrid::moment_component::stress_xx) ==
      Catch::Approx(nix::math::pi4 * (-2 * (1.0 * 0.5 * 0.5 + 0.3) + 0.5 * (0.7 * 0.2 * 0.2 + 0.4) +
                                      1.5 * 0.03 - 0.25 * 0.04)));
  REQUIRE_THROWS_AS(hybrid::engine::accumulate_kinetic_moments(kin, ohm, 1, 1, 1, {1.5}),
                    std::invalid_argument);
}

TEST_CASE("solve_ssor2_electric adapts DataContainer arrays to solver workspace")
{
  constexpr int interior_nz = 2;
  constexpr int interior_ny = 2;
  constexpr int interior_nx = 2;
  constexpr int total_nz    = interior_nz + 2;
  constexpr int total_ny    = interior_ny + 2;
  constexpr int total_nx    = interior_nx + 2;

  nix::Array4D<nix::float64> field  = xt::zeros<nix::float64>({total_nz, total_ny, total_nx, 6});
  nix::Array4D<nix::float64> source = xt::zeros<nix::float64>({total_nz, total_ny, total_nx, 4});
  nix::float64               coeff  = 1.5;
  for (int iz = 0; iz < total_nz; ++iz) {
    for (int iy = 0; iy < total_ny; ++iy) {
      for (int ix = 0; ix < total_nx; ++ix) {
        source(iz, iy, ix, 0) = coeff;
        source(iz, iy, ix, 1) = 2.0 * coeff;
        source(iz, iy, ix, 2) = -3.0 * coeff;
        source(iz, iy, ix, 3) = 1.0 * coeff;
      }
    }
  }
  // fill boundary
  for (int iz = 0; iz < total_nz; ++iz) {
    for (int iy = 0; iy < total_ny; ++iy) {
      field(iz, iy, 0, 0)            = 2.0;
      field(iz, iy, 0, 1)            = -3.0;
      field(iz, iy, 0, 2)            = 1.0;
      field(iz, iy, total_nx - 1, 0) = 2.0;
      field(iz, iy, total_nx - 1, 1) = -3.0;
      field(iz, iy, total_nx - 1, 2) = 1.0;
    }
  }
  for (int iz = 0; iz < total_nz; ++iz) {
    for (int ix = 0; ix < total_nx; ++ix) {
      field(iz, 0, ix, 0)            = 2.0;
      field(iz, 0, ix, 1)            = -3.0;
      field(iz, 0, ix, 2)            = 1.0;
      field(iz, total_ny - 1, ix, 0) = 2.0;
      field(iz, total_ny - 1, ix, 1) = -3.0;
      field(iz, total_ny - 1, ix, 2) = 1.0;
    }
  }
  for (int iy = 0; iy < total_ny; ++iy) {
    for (int ix = 0; ix < total_nx; ++ix) {
      field(0, iy, ix, 0)            = 2.0;
      field(0, iy, ix, 1)            = -3.0;
      field(0, iy, ix, 2)            = 1.0;
      field(total_nz - 1, iy, ix, 0) = 2.0;
      field(total_nz - 1, iy, ix, 1) = -3.0;
      field(total_nz - 1, iy, ix, 2) = 1.0;
    }
  }

  const auto stats =
      hybrid::engine::solve_ssor2_electric(field, source, 1, interior_nx, 1, interior_ny, 1,
                                           interior_nz, 20.0, 0.5, 2.0, 4.0, 100, 1.0e-5);

  REQUIRE(stats.converged);
  for (int iz = 1; iz <= interior_nz; ++iz) {
    for (int iy = 1; iy <= interior_ny; ++iy) {
      for (int ix = 1; ix <= interior_nx; ++ix) {
        REQUIRE(field(iz, iy, ix, 0) == Catch::Approx(2.0).margin(1.0e-6));
        REQUIRE(field(iz, iy, ix, 1) == Catch::Approx(-3.0).margin(1.0e-6));
        REQUIRE(field(iz, iy, ix, 2) == Catch::Approx(1.0).margin(1.0e-6));
      }
    }
  }
}
