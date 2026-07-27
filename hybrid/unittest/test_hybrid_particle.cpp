// -*- C++ -*-
#include "engine/interpolation.hpp"
#include "engine/moment.hpp"
#include "engine/particle.hpp"
#include "hybrid_chunk.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
hybrid::HybridChunk make_particle_chunk()
{
  const nix::Dims3D   dims      = {4, 4, 4};
  const nix::Bool3D   has_dim   = {true, true, true};
  const int           offset[3] = {0, 0, 0};
  const int           global[3] = {4, 4, 4};
  nix::json           config    = {{"Ns", 1},
                                   {"cc", 10.0},
                                   {"gamma", 5.0 / 3.0},
                                   {"delh", 1.0},
                                   {"option", nix::json::object()}};
  hybrid::HybridChunk chunk(dims, has_dim);
  chunk.set_global_context(offset, global);
  chunk.setup(config);
  return chunk;
}

hybrid::HybridChunk make_particle_chunk_1d()
{
  const nix::Dims3D   dims      = {1, 1, 4};
  const nix::Bool3D   has_dim   = {false, false, true};
  const int           offset[3] = {0, 0, 0};
  const int           global[3] = {1, 1, 4};
  nix::json           config    = {{"Ns", 1},
                                   {"cc", 10.0},
                                   {"gamma", 5.0 / 3.0},
                                   {"delh", 1.0},
                                   {"option", nix::json::object()}};
  hybrid::HybridChunk chunk(dims, has_dim);
  chunk.set_global_context(offset, global);
  chunk.setup(config);
  return chunk;
}

void set_particle(nix::XtensorParticle& particle, const std::array<double, 3>& position,
                  const std::array<double, 3>& velocity, int64_t id)
{
  particle.Np = 1;
  for (int component = 0; component < 3; ++component) {
    particle.xu(0, component)     = position[component];
    particle.xu(0, component + 3) = velocity[component];
  }
  std::memcpy(&particle.xu(0, 6), &id, sizeof(id));
}

int64_t particle_id(const nix::float64& slot)
{
  int64_t id = 0;
  std::memcpy(&id, &slot, sizeof(id));
  return id;
}
} // namespace

TEST_CASE("second-order particle shape has compact normalized support")
{
  using hybrid::engine::shape2;
  REQUIRE(shape2(0.0, 0.0, 1.0) == 0.75);
  REQUIRE(shape2(0.0, 0.5, 1.0) == 0.5);
  REQUIRE(shape2(0.0, 1.0, 1.0) == 0.125);
  REQUIRE(shape2(0.0, 1.5, 1.0) == 0.0);
  REQUIRE(shape2(0.0, 1.5001, 1.0) == 0.0);

  for (double position : {-0.49, 0.0, 0.23, 0.5, 0.91}) {
    double sum = 0;
    for (int grid = -3; grid <= 3; ++grid) {
      sum += shape2(position, static_cast<double>(grid), 1.0);
    }
    REQUIRE(sum == Catch::Approx(1.0).margin(1.0e-14));
  }
}

TEST_CASE("collocated interpolation reproduces uniform and linear fields")
{
  auto                           chunk    = make_particle_chunk();
  auto                           data     = chunk.get_internal_data();
  auto&                          particle = *data.particles[0];
  const hybrid::engine::Position position = {1.2, 1.7, 2.1};
  const auto                     anchor   = hybrid::engine::particle_cell(particle, position);

  for (int component = 0; component < hybrid::num_field_components; ++component) {
    for (int iz = 0; iz < static_cast<int>(data.field_cell.shape()[0]); ++iz) {
      for (int iy = 0; iy < static_cast<int>(data.field_cell.shape()[1]); ++iy) {
        for (int ix = 0; ix < static_cast<int>(data.field_cell.shape()[2]); ++ix) {
          const double x = particle.xmin + particle.delx * (ix - particle.Lbx + 0.5);
          const double y = particle.ymin + particle.dely * (iy - particle.Lby + 0.5);
          const double z = particle.zmin + particle.delz * (iz - particle.Lbz + 0.5);
          data.field_cell(iz, iy, ix, component) = component + 2.0 * x + 3.0 * y + 5.0 * z;
        }
      }
    }
  }
  data.background_cell.fill(0);
  for (int component = 0; component < 3; ++component) {
    xt::view(data.background_cell, xt::all(), xt::all(), xt::all(), component).fill(10 + component);
  }

  const auto   field = hybrid::engine::interpolate_collocated(data.field_cell, data.background_cell,
                                                              particle, anchor, position);
  const double linear_value = 2.0 * position[0] + 3.0 * position[1] + 5.0 * position[2];
  for (int component = 0; component < 3; ++component) {
    REQUIRE(field[component] == Catch::Approx(component + linear_value).margin(1.0e-13));
    REQUIRE(field[component + 3] ==
            Catch::Approx(component + 3 + linear_value + 10 + component).margin(1.0e-13));
  }
}

TEST_CASE("particle push samples a nonuniform field after the first half drift")
{
  auto  chunk    = make_particle_chunk();
  auto  data     = chunk.get_internal_data();
  auto& particle = *data.particles[0];
  particle.q     = 2.0;
  particle.m     = 4.0;
  set_particle(particle, {1.8, 1.5, 1.5}, {1.0, 0.0, 0.0}, 3);
  data.field_cell.fill(0);
  data.background_cell.fill(0);
  for (int iz = 0; iz < static_cast<int>(data.field_cell.shape()[0]); ++iz) {
    for (int iy = 0; iy < static_cast<int>(data.field_cell.shape()[1]); ++iy) {
      for (int ix = 0; ix < static_cast<int>(data.field_cell.shape()[2]); ++ix) {
        const double x                 = particle.xmin + particle.delx * (ix - particle.Lbx + 0.5);
        data.field_cell(iz, iy, ix, 0) = x;
      }
    }
  }

  const hybrid::engine::Position initial = {1.8, 1.5, 1.5};
  REQUIRE(hybrid::engine::particle_cell(particle, initial)[2] == particle.Lbx + 1);
  const double cfl = hybrid::engine::push_particles(data, data.field_cell, 0.4);
  REQUIRE(particle.xu(0, 3) == Catch::Approx(1.4).margin(1.0e-14));
  REQUIRE(particle.xu(0, 0) == Catch::Approx(2.28).margin(1.0e-14));
  REQUIRE(cfl == Catch::Approx(0.56).margin(1.0e-14));
}

TEST_CASE("Buneman-Boris push preserves rollback state and raw ID bits")
{
  auto  chunk      = make_particle_chunk();
  auto  data       = chunk.get_internal_data();
  auto& particle   = *data.particles[0];
  particle.q       = 2.0;
  particle.m       = 1.0;
  const int64_t id = 0x7ff8000000000042LL;
  set_particle(particle, {1.5, 1.5, 1.5}, {0.1, 0.0, 0.0}, id);
  data.field_cell.fill(0);
  data.background_cell.fill(0);
  xt::view(data.field_cell, xt::all(), xt::all(), xt::all(), 0).fill(1.0);

  const double cfl = hybrid::engine::push_particles(data, data.field_cell, 0.2);
  REQUIRE(particle.xu(0, 3) == Catch::Approx(0.5).margin(1.0e-14));
  REQUIRE(particle.xu(0, 0) == Catch::Approx(1.56).margin(1.0e-14));
  REQUIRE(cfl == Catch::Approx(0.1).margin(1.0e-14));
  REQUIRE(particle_id(particle.xu(0, 6)) == id);
  REQUIRE(particle_id(particle.xv(0, 6)) == id);

  hybrid::engine::rollback_particles(data);
  REQUIRE(particle.xu(0, 0) == 1.5);
  REQUIRE(particle.xu(0, 3) == 0.1);
  REQUIRE(particle_id(particle.xu(0, 6)) == id);
  REQUIRE(particle.xv(0, 0) == Catch::Approx(1.56).margin(1.0e-14));
}

TEST_CASE("first-corrector deposition preserves a bitwise accepted particle snapshot")
{
  auto  chunk    = make_particle_chunk();
  auto  data     = chunk.get_internal_data();
  auto& particle = *data.particles[0];
  particle.q     = 0;
  particle.m     = 1;
  particle.Np    = 3;
  particle.xu.fill(0);
  particle.xv.fill(-1);
  for (int ip = 0; ip < particle.Np; ++ip) {
    particle.xu(ip, 0) = 3.5 - ip;
    particle.xu(ip, 1) = 1.5;
    particle.xu(ip, 2) = 1.5;
    particle.xu(ip, 3) = 0.1 * (ip + 1);
    const int64_t id   = 20 + ip;
    std::memcpy(&particle.xu(ip, 6), &id, sizeof(id));
  }
  std::vector<nix::float64> accepted(static_cast<size_t>(particle.Np) * nix::Particle::Nc);
  std::memcpy(accepted.data(), particle.xu.data(), accepted.size() * sizeof(nix::float64));
  data.field_cell.fill(0);
  data.background_cell.fill(0);

  hybrid::engine::push_particles(data, data.field_cell, 0.1);
  hybrid::engine::deposit_moments(data);
  hybrid::engine::rollback_particles(data);

  REQUIRE(std::memcmp(particle.xu.data(), accepted.data(),
                      accepted.size() * sizeof(nix::float64)) == 0);
  for (int ip = 0; ip < particle.Np; ++ip) {
    REQUIRE(particle_id(particle.xu(ip, 6)) == 20 + ip);
  }
}

TEST_CASE("Buneman-Boris magnetic rotation matches the legacy formula")
{
  auto  chunk    = make_particle_chunk();
  auto  data     = chunk.get_internal_data();
  auto& particle = *data.particles[0];
  particle.q     = 1.0;
  particle.m     = 1.0;
  set_particle(particle, {1.5, 1.5, 1.5}, {1.0, 0.0, 0.0}, 9);
  data.field_cell.fill(0);
  data.background_cell.fill(0);
  xt::view(data.background_cell, xt::all(), xt::all(), xt::all(), 2).fill(10.0);

  hybrid::engine::push_particles(data, data.field_cell, 0.2);
  const double magnetic_impulse = 0.1;
  REQUIRE(particle.xu(0, 3) == Catch::Approx((1.0 - magnetic_impulse * magnetic_impulse) /
                                             (1.0 + magnetic_impulse * magnetic_impulse))
                                   .margin(1.0e-14));
  REQUIRE(particle.xu(0, 4) ==
          Catch::Approx(-2.0 * magnetic_impulse / (1.0 + magnetic_impulse * magnetic_impulse))
              .margin(1.0e-14));
  REQUIRE(particle.xu(0, 5) == 0.0);
}

TEST_CASE("Buneman-Boris combined electric and magnetic impulse matches legacy arithmetic")
{
  hybrid::engine::Velocity         velocity = {0.3, -0.2, 0.1};
  const hybrid::engine::FieldValue impulse  = {0.05, -0.03, 0.02, 0.1, -0.2, 0.15};
  hybrid::engine::buneman_boris(velocity, impulse);

  REQUIRE(velocity[0] == Catch::Approx(0.35156177156177154).margin(1.0e-15));
  REQUIRE(velocity[1] == Catch::Approx(-0.3413519813519814).margin(1.0e-15));
  REQUIRE(velocity[2] == Catch::Approx(0.06382284382284384).margin(1.0e-15));
}

TEST_CASE("particle CFL uses the maximum post-push component displacement")
{
  auto  chunk    = make_particle_chunk();
  auto  data     = chunk.get_internal_data();
  auto& particle = *data.particles[0];
  particle.q     = 0;
  particle.m     = 1;
  particle.delx  = 0.5;
  particle.dely  = 2.0;
  particle.delz  = 4.0;
  set_particle(particle, {1.5, 1.5, 1.5}, {1.0, 0.0, 0.0}, 1);
  REQUIRE(hybrid::engine::particle_cfl(data, 0.25) == 0.5);
  particle.xu(0, 3) = 0.0;
  particle.xu(0, 4) = -2.0;
  REQUIRE(hybrid::engine::particle_cfl(data, 0.25) == 0.25);
  particle.xu(0, 4) = 0.0;
  particle.xu(0, 5) = 0.5;
  REQUIRE(hybrid::engine::particle_cfl(data, 0.25) == 0.03125);
}

TEST_CASE("particle push enforces the cell CFL precondition")
{
  auto  chunk    = make_particle_chunk();
  auto  data     = chunk.get_internal_data();
  auto& particle = *data.particles[0];
  particle.q     = 0;
  particle.m     = 1;
  set_particle(particle, {1.5, 1.5, 1.5}, {4.0, 0.0, 0.0}, 1);
  REQUIRE_THROWS_AS(hybrid::engine::push_particles(data, data.field_cell, 0.25),
                    std::runtime_error);
  REQUIRE(particle.xu(0, 0) == 1.5);
}

TEST_CASE("post-push CFL failure restores every particle atomically")
{
  auto  chunk    = make_particle_chunk();
  auto  data     = chunk.get_internal_data();
  auto& particle = *data.particles[0];
  particle.q     = 1;
  particle.m     = 1;
  particle.resize(2);
  particle.Np = 2;
  set_particle(particle, {1.5, 1.5, 1.5}, {-1.0, 0.0, 0.0}, 11);
  particle.Np = 2;
  for (int component = 0; component < 3; ++component) {
    particle.xu(1, component)     = component == 0 ? 2.5 : 1.5;
    particle.xu(1, component + 3) = component == 0 ? 4.5 : 0.0;
  }
  const int64_t second_id = 12;
  std::memcpy(&particle.xu(1, 6), &second_id, sizeof(second_id));
  particle.xv.fill(-99);
  data.field_cell.fill(0);
  data.background_cell.fill(0);
  xt::view(data.field_cell, xt::all(), xt::all(), xt::all(), 0).fill(5.0);

  REQUIRE_THROWS_AS(hybrid::engine::push_particles(data, data.field_cell, 0.2), std::runtime_error);
  REQUIRE(particle.xu(0, 0) == 1.5);
  REQUIRE(particle.xu(0, 3) == -1.0);
  REQUIRE(particle_id(particle.xu(0, 6)) == 11);
  REQUIRE(particle.xu(1, 0) == 2.5);
  REQUIRE(particle.xu(1, 3) == 4.5);
  REQUIRE(particle_id(particle.xu(1, 6)) == second_id);
}

TEST_CASE("inactive axes use the center field plane but retain transverse drift and CFL")
{
  auto  chunk    = make_particle_chunk_1d();
  auto  data     = chunk.get_internal_data();
  auto& particle = *data.particles[0];
  particle.q     = 0;
  particle.m     = 1;
  set_particle(particle, {1.2, 0.0, 0.0}, {0.0, 2.0, -3.0}, 1);
  data.field_cell.fill(-100);
  data.background_cell.fill(0);
  for (int ix = 0; ix < static_cast<int>(data.field_cell.shape()[2]); ++ix) {
    for (int component = 0; component < hybrid::num_field_components; ++component) {
      data.field_cell(data.Lbz, data.Lby, ix, component) = component + 1;
    }
  }

  const hybrid::engine::Position position = {1.2, 17.0, -23.0};
  const auto                     anchor   = hybrid::engine::particle_cell(particle, position);
  REQUIRE(anchor[0] == data.Lbz);
  REQUIRE(anchor[1] == data.Lby);
  const auto field = hybrid::engine::interpolate_collocated(data.field_cell, data.background_cell,
                                                            particle, anchor, position);
  for (int component = 0; component < hybrid::num_field_components; ++component) {
    REQUIRE(field[component] == Catch::Approx(component + 1).margin(1.0e-14));
  }

  const double cfl = hybrid::engine::push_particles(data, data.field_cell, 0.1);
  REQUIRE(particle.xu(0, 1) == Catch::Approx(0.2).margin(1.0e-14));
  REQUIRE(particle.xu(0, 2) == Catch::Approx(-0.3).margin(1.0e-14));
  REQUIRE(cfl == Catch::Approx(0.3).margin(1.0e-14));
}
