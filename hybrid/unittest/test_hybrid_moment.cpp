// -*- C++ -*-
#include "engine/filter.hpp"
#include "engine/moment.hpp"
#include "hybrid_chunk.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <numeric>

namespace
{
hybrid::HybridChunk make_moment_chunk(int species = 1, bool one_dimensional = false)
{
  const nix::Dims3D dims = one_dimensional ? nix::Dims3D{1, 1, 6} : nix::Dims3D{6, 6, 6};
  const nix::Bool3D has_dim =
      one_dimensional ? nix::Bool3D{false, false, true} : nix::Bool3D{true, true, true};
  const int           offset[3] = {0, 0, 0};
  const int           global[3] = {dims[0], dims[1], dims[2]};
  nix::json           config    = {{"Ns", species},
                                   {"cc", 10.0},
                                   {"gamma", 5.0 / 3.0},
                                   {"delh", 1.0},
                                   {"option", nix::json::object()}};
  hybrid::HybridChunk chunk(dims, has_dim);
  chunk.set_global_context(offset, global);
  chunk.setup(config);
  return chunk;
}

} // namespace

TEST_CASE("single particle deposits the ten legacy raw moments")
{
  auto  chunk       = make_moment_chunk();
  auto  data        = chunk.get_internal_data();
  auto& particle    = *data.particles[0];
  particle.m        = 2.0;
  particle.q        = 1.0;
  particle.Np       = 1;
  particle.xu(0, 0) = 2.5;
  particle.xu(0, 1) = 2.5;
  particle.xu(0, 2) = 2.5;
  particle.xu(0, 3) = 1.1;
  particle.xu(0, 4) = -2.3;
  particle.xu(0, 5) = 3.7;
  hybrid::engine::deposit_moments(data);
  const int                                               iz            = data.Lbz + 2;
  const int                                               iy            = data.Lby + 2;
  const int                                               ix            = data.Lbx + 2;
  const double                                            center_weight = 2.0 * 0.75 * 0.75 * 0.75;
  const std::array<double, hybrid::num_moment_components> value = {1,    1.1,   -2.3,  3.7,  1.21,
                                                                   5.29, 13.69, -2.53, 4.07, -8.51};
  for (int component = 0; component < hybrid::num_moment_components; ++component) {
    REQUIRE(data.moment_kinetic(iz, iy, ix, 0, component) ==
            Catch::Approx(center_weight * value[component]).margin(1.0e-14));
  }

  for (int component = 0; component < hybrid::num_moment_components; ++component) {
    double total = 0;
    for (int jz = 0; jz < static_cast<int>(data.moment_kinetic.shape()[0]); ++jz) {
      for (int jy = 0; jy < static_cast<int>(data.moment_kinetic.shape()[1]); ++jy) {
        for (int jx = 0; jx < static_cast<int>(data.moment_kinetic.shape()[2]); ++jx) {
          total += data.moment_kinetic(jz, jy, jx, 0, component);
        }
      }
    }
    REQUIRE(total == Catch::Approx(2.0 * value[component]).epsilon(1.0e-13).margin(1.0e-14));
  }
}

TEST_CASE("deposition includes an outbound particle before migration")
{
  auto  chunk       = make_moment_chunk();
  auto  data        = chunk.get_internal_data();
  auto& particle    = *data.particles[0];
  particle.m        = 1;
  particle.q        = 1;
  particle.Np       = 1;
  particle.xu(0, 0) = particle.xmax + 0.2;
  particle.xu(0, 1) = 2.5;
  particle.xu(0, 2) = 2.5;

  hybrid::engine::deposit_moments(data);
  double total_mass = 0;
  for (auto value : xt::view(data.moment_kinetic, xt::all(), xt::all(), xt::all(), 0,
                             hybrid::moment_component::density)) {
    total_mass += value;
  }
  REQUIRE(total_mass == Catch::Approx(1.0).margin(1.0e-14));
  REQUIRE(data.moment_kinetic(data.Lbz + 2, data.Lby + 2, data.Ubx + 2, 0,
                              hybrid::moment_component::density) > 0);
}

TEST_CASE("deposition preserves unsorted particle buffers and ID association")
{
  auto  chunk    = make_moment_chunk();
  auto  data     = chunk.get_internal_data();
  auto& particle = *data.particles[0];
  particle.m     = 1;
  particle.q     = 1;
  particle.Np    = 3;
  particle.xu.fill(0);
  particle.xv.fill(-7);
  for (int ip = 0; ip < particle.Np; ++ip) {
    particle.xu(ip, 0) = 4.5 - ip;
    particle.xu(ip, 1) = 2.5;
    particle.xu(ip, 2) = 2.5;
    particle.xu(ip, 3) = ip + 1;
    const int64_t id   = 100 + ip;
    std::memcpy(&particle.xu(ip, 6), &id, sizeof(id));
  }
  const auto accepted  = particle.xu;
  const auto temporary = particle.xv;

  hybrid::engine::deposit_moments(data);

  REQUIRE(xt::all(xt::equal(particle.xu, accepted)));
  REQUIRE(xt::all(xt::equal(particle.xv, temporary)));
  double total_mass = 0;
  for (auto value : xt::view(data.moment_kinetic, xt::all(), xt::all(), xt::all(), 0,
                             hybrid::moment_component::density)) {
    total_mass += value;
  }
  REQUIRE(total_mass == Catch::Approx(3.0).margin(1.0e-14));
}

TEST_CASE("kinetic current uses each species charge-to-mass ratio")
{
  auto chunk           = make_moment_chunk(2);
  auto data            = chunk.get_internal_data();
  data.particles[0]->q = 2;
  data.particles[0]->m = 4;
  data.particles[1]->q = -3;
  data.particles[1]->m = 2;
  for (int component = 0; component < hybrid::num_current_components; ++component) {
    xt::view(data.moment_kinetic, xt::all(), xt::all(), xt::all(), 0, component)
        .fill(component + 1);
    xt::view(data.moment_kinetic, xt::all(), xt::all(), xt::all(), 1, component)
        .fill(2 * (component + 1));
  }

  hybrid::engine::derive_current(data);
  for (int component = 0; component < hybrid::num_current_components; ++component) {
    const double expected = 0.5 * (component + 1) - 1.5 * 2 * (component + 1);
    REQUIRE(data.current_kinetic(0, 0, 0, component) == expected);
  }
}

TEST_CASE("binomial moment filter has the legacy one- and two-pass impulse response")
{
  auto chunk = make_moment_chunk();
  auto data  = chunk.get_internal_data();
  data.moment_kinetic.fill(0);
  const int iz                          = data.Lbz + 2;
  const int iy                          = data.Lby + 2;
  const int ix                          = data.Lbx + 2;
  data.moment_kinetic(iz, iy, ix, 0, 0) = 1;

  hybrid::engine::filter_moments_once(data);
  REQUIRE(data.moment_kinetic(iz, iy, ix, 0, 0) == 0.125);
  REQUIRE(data.moment_kinetic(iz, iy, ix + 1, 0, 0) == 0.0625);
  hybrid::engine::filter_moments_once(data);
  REQUIRE(data.moment_kinetic(iz, iy, ix, 0, 0) ==
          Catch::Approx(0.375 * 0.375 * 0.375).margin(1.0e-15));
}

TEST_CASE("binomial filter collapses inactive dimensions")
{
  auto chunk = make_moment_chunk(1, true);
  auto data  = chunk.get_internal_data();
  data.moment_kinetic.fill(0);
  const int ix                                      = data.Lbx + 2;
  data.moment_kinetic(data.Lbz, data.Lby, ix, 0, 0) = 1;

  hybrid::engine::filter_moments_once(data);
  REQUIRE(data.moment_kinetic(data.Lbz, data.Lby, ix, 0, 0) == 0.5);
  REQUIRE(data.moment_kinetic(data.Lbz, data.Lby, ix + 1, 0, 0) == 0.25);
}

TEST_CASE("electric filter has the legacy two-pass response and preserves magnetic field")
{
  auto chunk = make_moment_chunk();
  auto data  = chunk.get_internal_data();
  data.work_field_cell.fill(0);
  const int iz                                                          = data.Lbz + 2;
  const int iy                                                          = data.Lby + 2;
  const int ix                                                          = data.Lbx + 2;
  data.work_field_cell(iz, iy, ix, hybrid::field_component::electric_x) = 1;
  xt::view(data.work_field_cell, xt::all(), xt::all(), xt::all(),
           hybrid::field_component::magnetic_x)
      .fill(7);

  hybrid::engine::filter_electric_once(data);
  REQUIRE(data.work_field_cell(iz, iy, ix, hybrid::field_component::electric_x) == 0.125);
  REQUIRE(data.work_field_cell(iz, iy, ix + 1, hybrid::field_component::electric_x) == 0.0625);
  hybrid::engine::filter_electric_once(data);
  REQUIRE(data.work_field_cell(iz, iy, ix, hybrid::field_component::electric_x) ==
          Catch::Approx(0.375 * 0.375 * 0.375).margin(1.0e-15));
  REQUIRE(xt::all(xt::equal(xt::view(data.work_field_cell, xt::all(), xt::all(), xt::all(),
                                     hybrid::field_component::magnetic_x),
                            7)));
}
