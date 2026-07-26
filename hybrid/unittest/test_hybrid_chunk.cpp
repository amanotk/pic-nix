// -*- C++ -*-
#include "hybrid_chunk.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace
{
template <typename Array>
void fill_sequence(Array& array, nix::float64 offset)
{
  for (size_t i = 0; i < array.size(); ++i) {
    array.data()[i] = offset + static_cast<nix::float64>(i);
  }
}

template <typename Array>
bool arrays_equal(const Array& lhs, const Array& rhs)
{
  return lhs.shape() == rhs.shape() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename Array>
bool array_is_zero(const Array& array)
{
  return std::all_of(array.begin(), array.end(), [](auto value) { return value == 0; });
}

hybrid::HybridChunk make_chunk()
{
  const nix::Dims3D dims{2, 3, 4};
  const nix::Bool3D has_dim{true, true, true};
  const int         offset[3] = {0, 0, 0};
  const int         global[3] = {2, 3, 4};
  nix::json         config    = {
                 {"Ns", 2},
                 {"cc", 10000.0},
                 {"gamma", 5.0 / 3.0},
                 {"delh", 0.25},
                 {"option", {{"cell_load", 2.0}}},
  };

  hybrid::HybridChunk chunk(dims, has_dim, 7);
  chunk.set_global_context(offset, global);
  chunk.setup(config);
  return chunk;
}
} // namespace

TEST_CASE("HybridChunk allocates the legacy state layout")
{
  auto chunk = make_chunk();
  auto data  = chunk.get_internal_data();

  REQUIRE(data.boundary_margin == hybrid::boundary_margin);
  REQUIRE(data.order == hybrid::particle_order);
  REQUIRE(data.num_species == 2);
  REQUIRE(hybrid::fluid_component::electron_density == 0);
  REQUIRE(hybrid::fluid_component::ion_pressure == 9);
  REQUIRE(hybrid::field_component::electric_x == 0);
  REQUIRE(hybrid::field_component::magnetic_z == 5);
  REQUIRE(hybrid::moment_component::density == 0);
  REQUIRE(hybrid::moment_component::stress_yz == 9);
  REQUIRE(hybrid::current_component::charge == 0);
  REQUIRE(hybrid::current_component::current_z == 3);
  REQUIRE(data.eta.shape() == std::array<size_t, 3>{6, 7, 8});
  REQUIRE(data.chi.shape() == std::array<size_t, 3>{6, 7, 8});
  REQUIRE(data.fluid.shape() == std::array<size_t, 4>{6, 7, 8, 10});
  REQUIRE(data.field_cell.shape() == std::array<size_t, 4>{6, 7, 8, 6});
  REQUIRE(data.field_staggered.shape() == std::array<size_t, 4>{6, 7, 8, 6});
  REQUIRE(data.moment_kinetic.shape() == std::array<size_t, 5>{6, 7, 8, 2, 10});
  REQUIRE(data.current_kinetic.shape() == std::array<size_t, 4>{6, 7, 8, 4});
  REQUIRE(data.background_cell.shape() == std::array<size_t, 4>{6, 7, 8, 3});
  REQUIRE(data.background_x_face.shape() == std::array<size_t, 4>{6, 7, 8, 3});
  REQUIRE(data.background_y_face.shape() == std::array<size_t, 4>{6, 7, 8, 3});
  REQUIRE(data.background_z_face.shape() == std::array<size_t, 4>{6, 7, 8, 3});
  REQUIRE(data.phase_cell.shape() == std::array<size_t, 5>{6, 7, 8, 3, 3});
  REQUIRE(data.phase_face.shape() == std::array<size_t, 5>{6, 7, 8, 3, 3});
  REQUIRE(data.curl_b.shape() == std::array<size_t, 4>{6, 7, 8, 3});
  REQUIRE(data.ohm_moment.shape() == std::array<size_t, 4>{6, 7, 8, 10});
  REQUIRE(data.resistive_field.shape() == std::array<size_t, 4>{6, 7, 8, 3});
  REQUIRE(data.fluid_flux.shape() == std::array<size_t, 5>{6, 7, 8, 3, 5});
  REQUIRE(data.field_flux.shape() == std::array<size_t, 4>{6, 7, 8, 6});
  REQUIRE(data.work_fluid.shape() == std::array<size_t, 4>{6, 7, 8, 10});
  REQUIRE(data.work_field_cell.shape() == std::array<size_t, 4>{6, 7, 8, 6});
  REQUIRE(data.work_field_staggered.shape() == std::array<size_t, 4>{6, 7, 8, 6});
  REQUIRE(data.solver_left.shape() == std::array<size_t, 4>{6, 7, 8, 16});
  REQUIRE(data.solver_right.shape() == std::array<size_t, 4>{6, 7, 8, 16});
  REQUIRE(data.solver_field_x.shape() == std::array<size_t, 4>{6, 7, 8, 6});
  REQUIRE(data.solver_field_y.shape() == std::array<size_t, 4>{6, 7, 8, 6});
  REQUIRE(data.solver_field_z.shape() == std::array<size_t, 4>{6, 7, 8, 6});
  REQUIRE(data.ohm_source.shape() == std::array<size_t, 4>{6, 7, 8, 4});
  REQUIRE(data.filter_scratch.shape() == std::array<size_t, 3>{6, 7, 8});
  REQUIRE(data.particles.size() == 2);
  REQUIRE(data.particles[0]->Np == 0);
  REQUIRE(data.particles[0]->Np_total == nix::Particle::alloc_unit);
  REQUIRE(data.particles[0]->q == 0);
  REQUIRE(data.particles[0]->m == 0);
  REQUIRE(data.load[hybrid::LoadCell] == 2.0);
  REQUIRE(data.load[hybrid::LoadParticle] == 0.0);
}

TEST_CASE("HybridChunk serialization preserves accepted state only")
{
  auto source = make_chunk();
  auto src    = source.get_internal_data();

  fill_sequence(src.eta, 10);
  fill_sequence(src.chi, 20);
  fill_sequence(src.fluid, 30);
  fill_sequence(src.field_cell, 40);
  fill_sequence(src.field_staggered, 50);
  fill_sequence(src.moment_kinetic, 60);
  fill_sequence(src.current_kinetic, 70);
  fill_sequence(src.background_cell, 80);
  fill_sequence(src.background_x_face, 90);
  fill_sequence(src.background_y_face, 100);
  fill_sequence(src.background_z_face, 110);

  fill_sequence(src.phase_cell, 120);
  fill_sequence(src.phase_face, 130);
  fill_sequence(src.curl_b, 140);
  fill_sequence(src.ohm_moment, 150);
  fill_sequence(src.resistive_field, 160);
  fill_sequence(src.fluid_flux, 170);
  fill_sequence(src.field_flux, 180);
  fill_sequence(src.work_fluid, 190);
  fill_sequence(src.work_field_cell, 200);
  fill_sequence(src.work_field_staggered, 210);
  fill_sequence(src.solver_left, 220);
  fill_sequence(src.solver_right, 230);
  fill_sequence(src.solver_field_x, 240);
  fill_sequence(src.solver_field_y, 250);
  fill_sequence(src.solver_field_z, 260);
  fill_sequence(src.ohm_source, 270);
  fill_sequence(src.filter_scratch, 280);

  for (int species = 0; species < src.num_species; ++species) {
    auto& particle = *src.particles[species];
    particle.Np    = 3;
    particle.q     = 2.0 + species;
    particle.m     = 4.0 + species;
    fill_sequence(particle.xu, 300 + 10 * species);
    fill_sequence(particle.xv, 400 + 10 * species);
    for (size_t i = 0; i < particle.gindex.size(); ++i) {
      particle.gindex.data()[i] = static_cast<nix::int32>(i + species);
    }
    for (size_t i = 0; i < particle.pindex.size(); ++i) {
      particle.pindex.data()[i] = static_cast<nix::int32>(2 * i + species);
    }
    for (size_t i = 0; i < particle.pcount.size(); ++i) {
      particle.pcount.data()[i] = static_cast<nix::int32>(3 * i + species);
    }
  }
  src.load[hybrid::LoadCell]     = 3.5;
  src.load[hybrid::LoadParticle] = 7.5;

  const int size = source.pack(nullptr, 0);
  REQUIRE(source.get_size_byte() == size);
  std::vector<uint8_t> buffer(size);
  REQUIRE(source.pack(buffer.data(), 0) == size);

  const nix::Dims3D   dims{2, 3, 4};
  const nix::Bool3D   has_dim{true, true, true};
  hybrid::HybridChunk restored(dims, has_dim);
  REQUIRE(restored.unpack(buffer.data(), 0) == size);
  auto dst = restored.get_internal_data();

  REQUIRE(restored.get_id() == source.get_id());
  REQUIRE(dst.order == src.order);
  REQUIRE(dst.num_species == src.num_species);
  REQUIRE(dst.light_speed == src.light_speed);
  REQUIRE(dst.adiabatic_index == src.adiabatic_index);
  REQUIRE(dst.load == src.load);
  REQUIRE(dst.option == src.option);
  REQUIRE(arrays_equal(dst.eta, src.eta));
  REQUIRE(arrays_equal(dst.chi, src.chi));
  REQUIRE(arrays_equal(dst.fluid, src.fluid));
  REQUIRE(arrays_equal(dst.field_cell, src.field_cell));
  REQUIRE(arrays_equal(dst.field_staggered, src.field_staggered));
  REQUIRE(arrays_equal(dst.moment_kinetic, src.moment_kinetic));
  REQUIRE(arrays_equal(dst.current_kinetic, src.current_kinetic));
  REQUIRE(arrays_equal(dst.background_cell, src.background_cell));
  REQUIRE(arrays_equal(dst.background_x_face, src.background_x_face));
  REQUIRE(arrays_equal(dst.background_y_face, src.background_y_face));
  REQUIRE(arrays_equal(dst.background_z_face, src.background_z_face));

  REQUIRE(dst.particles.size() == src.particles.size());
  for (int species = 0; species < dst.num_species; ++species) {
    const auto& lhs = *src.particles[species];
    const auto& rhs = *dst.particles[species];
    REQUIRE(rhs.Np_total == lhs.Np_total);
    REQUIRE(rhs.Np == lhs.Np);
    REQUIRE(rhs.Ng == lhs.Ng);
    REQUIRE(rhs.q == lhs.q);
    REQUIRE(rhs.m == lhs.m);
    REQUIRE(rhs.has_xdim == lhs.has_xdim);
    REQUIRE(rhs.has_ydim == lhs.has_ydim);
    REQUIRE(rhs.has_zdim == lhs.has_zdim);
    REQUIRE(rhs.Lbx == lhs.Lbx);
    REQUIRE(rhs.Ubx == lhs.Ubx);
    REQUIRE(rhs.Lby == lhs.Lby);
    REQUIRE(rhs.Uby == lhs.Uby);
    REQUIRE(rhs.Lbz == lhs.Lbz);
    REQUIRE(rhs.Ubz == lhs.Ubz);
    REQUIRE(rhs.delx == lhs.delx);
    REQUIRE(rhs.dely == lhs.dely);
    REQUIRE(rhs.delz == lhs.delz);
    REQUIRE(rhs.xmin == lhs.xmin);
    REQUIRE(rhs.xmax == lhs.xmax);
    REQUIRE(rhs.ymin == lhs.ymin);
    REQUIRE(rhs.ymax == lhs.ymax);
    REQUIRE(rhs.zmin == lhs.zmin);
    REQUIRE(rhs.zmax == lhs.zmax);
    REQUIRE(rhs.xmin_global == lhs.xmin_global);
    REQUIRE(rhs.xmax_global == lhs.xmax_global);
    REQUIRE(rhs.ymin_global == lhs.ymin_global);
    REQUIRE(rhs.ymax_global == lhs.ymax_global);
    REQUIRE(rhs.zmin_global == lhs.zmin_global);
    REQUIRE(rhs.zmax_global == lhs.zmax_global);
    REQUIRE(arrays_equal(rhs.xu, lhs.xu));
    REQUIRE(arrays_equal(rhs.xv, lhs.xv));
    REQUIRE(arrays_equal(rhs.gindex, lhs.gindex));
    REQUIRE(arrays_equal(rhs.pindex, lhs.pindex));
    REQUIRE(arrays_equal(rhs.pcount, lhs.pcount));
  }

  REQUIRE(array_is_zero(dst.phase_cell));
  REQUIRE(array_is_zero(dst.phase_face));
  REQUIRE(array_is_zero(dst.curl_b));
  REQUIRE(array_is_zero(dst.ohm_moment));
  REQUIRE(array_is_zero(dst.resistive_field));
  REQUIRE(array_is_zero(dst.fluid_flux));
  REQUIRE(array_is_zero(dst.field_flux));
  REQUIRE(array_is_zero(dst.work_fluid));
  REQUIRE(array_is_zero(dst.work_field_cell));
  REQUIRE(array_is_zero(dst.work_field_staggered));
  REQUIRE(array_is_zero(dst.solver_left));
  REQUIRE(array_is_zero(dst.solver_right));
  REQUIRE(array_is_zero(dst.solver_field_x));
  REQUIRE(array_is_zero(dst.solver_field_y));
  REQUIRE(array_is_zero(dst.solver_field_z));
  REQUIRE(array_is_zero(dst.ohm_source));
  REQUIRE(array_is_zero(dst.filter_scratch));
  REQUIRE(restored.get_size_byte() == size);

  restored.reset_load();
  REQUIRE(dst.load[hybrid::LoadCell] == 2.0);
  REQUIRE(dst.load[hybrid::LoadParticle] == 0.25);
  for (auto& particle : dst.particles) {
    for (int ip = 0; ip < particle->Np; ++ip) {
      particle->xu(ip, 0) = 0.5 * (particle->xmin + particle->xmax);
      particle->xu(ip, 1) = 0.5 * (particle->ymin + particle->ymax);
      particle->xu(ip, 2) = 0.5 * (particle->zmin + particle->zmax);
    }
    particle->count(0, particle->Np - 1, true, restored.get_order());
    particle->sort();
    REQUIRE(particle->Np == 3);
  }
}

TEST_CASE("HybridChunk rejects serialization during an active exchange")
{
  auto chunk       = make_chunk();
  auto buffer      = chunk.get_mpi_buffer(hybrid::BoundaryCopy10);
  buffer->sendwait = true;

  REQUIRE_FALSE(chunk.exchanges_idle());
  REQUIRE_THROWS_AS(chunk.get_size_byte(), std::runtime_error);

  buffer->sendwait = false;
  REQUIRE(chunk.exchanges_idle());
  REQUIRE_NOTHROW(chunk.get_size_byte());
}
