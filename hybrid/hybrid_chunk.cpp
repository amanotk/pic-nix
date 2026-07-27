// -*- C++ -*-
#include "hybrid_chunk.hpp"

#include "hybrid_halo.hpp"

#include "nix/xtensor/xtensor_halo3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace hybrid
{
HybridChunk::HybridChunk(nix::Dims3D dims, nix::Bool3D has_dim, int id)
    : base_type(dims, has_dim, id), order(particle_order), num_species(0), light_speed(0),
      adiabatic_index(0)
{
  load.resize(NumLoadModes);
  mpibufvec.resize(NumBoundaryModes);
  for (auto& mpibuf : mpibufvec) {
    mpibuf = std::make_shared<MpiBuffer>();
  }
  reset_load();
}

HybridChunk::DataContainer HybridChunk::get_internal_data()
{
  // clang-format off
  return {boundary_margin,
          Lbx,
          Ubx,
          Lby,
          Uby,
          Lbz,
          Ubz,
          order,
          num_species,
          light_speed,
          adiabatic_index,
          load,
          option,
          eta,
          chi,
          fluid,
          field_cell,
          field_staggered,
          moment_kinetic,
          current_kinetic,
          background_cell,
          background_x_face,
          background_y_face,
          background_z_face,
          particles,
          phase_cell,
          phase_face,
          curl_b,
          ohm_moment,
          resistive_field,
          fluid_flux,
          field_flux,
          work_fluid,
          work_field_cell,
          work_field_staggered,
          solver_left,
          solver_right,
          solver_field_x,
          solver_field_y,
          solver_field_z,
          ohm_source,
          filter_scratch};
  // clang-format on
}

int HybridChunk::get_order() const
{
  return order;
}

int HybridChunk::get_num_species() const
{
  return num_species;
}

int64_t HybridChunk::get_size_byte() const
{
  return const_cast<HybridChunk*>(this)->pack(nullptr, 0);
}

int HybridChunk::pack(void* buffer, int address)
{
  if (!exchanges_idle()) {
    throw std::runtime_error("cannot serialize HybridChunk during an active boundary exchange");
  }

  int count = base_type::pack(buffer, address);

  count += nix::memcpy_count(buffer, &order, sizeof(order), count, 0);
  count += nix::memcpy_count(buffer, &num_species, sizeof(num_species), count, 0);
  count += nix::memcpy_count(buffer, &light_speed, sizeof(light_speed), count, 0);
  count += nix::memcpy_count(buffer, &adiabatic_index, sizeof(adiabatic_index), count, 0);

  auto pack_array = [&](auto& array) {
    using value_type = typename std::decay_t<decltype(array)>::value_type;
    count += nix::memcpy_count(buffer, array.data(), array.size() * sizeof(value_type), count, 0);
  };

  pack_array(eta);
  pack_array(chi);
  pack_array(fluid);
  pack_array(field_cell);
  pack_array(field_staggered);
  pack_array(moment_kinetic);
  pack_array(current_kinetic);
  pack_array(background_cell);
  pack_array(background_x_face);
  pack_array(background_y_face);
  pack_array(background_z_face);

  for (auto& particle : particles) {
    count = particle->pack(buffer, count);
  }

  return count;
}

int HybridChunk::unpack(void* buffer, int address)
{
  int count = base_type::unpack(buffer, address);

  count += nix::memcpy_count(&order, buffer, sizeof(order), 0, count);
  count += nix::memcpy_count(&num_species, buffer, sizeof(num_species), 0, count);
  count += nix::memcpy_count(&light_speed, buffer, sizeof(light_speed), 0, count);
  count += nix::memcpy_count(&adiabatic_index, buffer, sizeof(adiabatic_index), 0, count);

  allocate();

  auto unpack_array = [&](auto& array) {
    using value_type = typename std::decay_t<decltype(array)>::value_type;
    count += nix::memcpy_count(array.data(), buffer, array.size() * sizeof(value_type), 0, count);
  };

  unpack_array(eta);
  unpack_array(chi);
  unpack_array(fluid);
  unpack_array(field_cell);
  unpack_array(field_staggered);
  unpack_array(moment_kinetic);
  unpack_array(current_kinetic);
  unpack_array(background_cell);
  unpack_array(background_x_face);
  unpack_array(background_y_face);
  unpack_array(background_z_face);

  particles.resize(num_species);
  for (auto& particle : particles) {
    particle = std::make_shared<nix::XtensorParticle>();
    count    = particle->unpack(buffer, count);
  }

  return count;
}

void HybridChunk::allocate()
{
  const size_t nz = dims[0] + 2 * boundary_margin;
  const size_t ny = dims[1] + 2 * boundary_margin;
  const size_t nx = dims[2] + 2 * boundary_margin;
  const size_t ns = num_species;

  eta.resize({nz, ny, nx});
  chi.resize({nz, ny, nx});
  fluid.resize({nz, ny, nx, num_fluid_components});
  field_cell.resize({nz, ny, nx, num_field_components});
  field_staggered.resize({nz, ny, nx, num_field_components});
  moment_kinetic.resize({nz, ny, nx, ns, num_moment_components});
  current_kinetic.resize({nz, ny, nx, num_current_components});
  background_cell.resize({nz, ny, nx, num_vector_components});
  background_x_face.resize({nz, ny, nx, num_vector_components});
  background_y_face.resize({nz, ny, nx, num_vector_components});
  background_z_face.resize({nz, ny, nx, num_vector_components});

  phase_cell.resize({nz, ny, nx, num_phase_directions, num_phase_branches});
  phase_face.resize({nz, ny, nx, num_phase_directions, num_phase_branches});
  curl_b.resize({nz, ny, nx, num_vector_components});
  ohm_moment.resize({nz, ny, nx, num_moment_components});
  resistive_field.resize({nz, ny, nx, num_vector_components});

  fluid_flux.resize({nz, ny, nx, num_phase_directions, num_conserved_components});
  field_flux.resize({nz, ny, nx, num_field_components});
  work_fluid.resize({nz, ny, nx, num_fluid_components});
  work_field_cell.resize({nz, ny, nx, num_field_components});
  work_field_staggered.resize({nz, ny, nx, num_field_components});
  solver_left.resize({nz, ny, nx, num_reconstructed_components});
  solver_right.resize({nz, ny, nx, num_reconstructed_components});
  solver_field_x.resize({nz, ny, nx, num_field_components});
  solver_field_y.resize({nz, ny, nx, num_field_components});
  solver_field_z.resize({nz, ny, nx, num_field_components});
  ohm_source.resize({nz, ny, nx, num_ohm_source_components});
  filter_scratch.resize({nz, ny, nx});

  eta.fill(0);
  chi.fill(0);
  fluid.fill(0);
  field_cell.fill(0);
  field_staggered.fill(0);
  moment_kinetic.fill(0);
  current_kinetic.fill(0);
  background_cell.fill(0);
  background_x_face.fill(0);
  background_y_face.fill(0);
  background_z_face.fill(0);
  phase_cell.fill(0);
  phase_face.fill(0);
  curl_b.fill(0);
  ohm_moment.fill(0);
  resistive_field.fill(0);
  fluid_flux.fill(0);
  field_flux.fill(0);
  work_fluid.fill(0);
  work_field_cell.fill(0);
  work_field_staggered.fill(0);
  solver_left.fill(0);
  solver_right.fill(0);
  solver_field_x.fill(0);
  solver_field_y.fill(0);
  solver_field_z.fill(0);
  ohm_source.fill(0);
  filter_scratch.fill(0);
}

void HybridChunk::allocate_mpi_buffers()
{
  set_mpi_buffer(mpibufvec[BoundaryCopy10], 0, 0, sizeof(nix::float64) * num_fluid_components);
  set_mpi_buffer(mpibufvec[BoundaryCopy6], 0, 0, sizeof(nix::float64) * num_field_components);
  set_mpi_buffer(mpibufvec[BoundaryCopy3], 0, 0, sizeof(nix::float64) * num_vector_components);
  set_mpi_buffer(mpibufvec[BoundaryCopy9], 0, 0,
                 sizeof(nix::float64) * num_phase_directions * num_phase_branches);
  set_mpi_buffer(mpibufvec[BoundaryMomentAccum], 0, 0,
                 sizeof(nix::float64) * num_species * num_moment_components);
  set_mpi_buffer(mpibufvec[BoundaryMomentCopy], 0, 0,
                 sizeof(nix::float64) * num_species * num_moment_components);
}

bool HybridChunk::exchanges_idle() const
{
  return std::all_of(mpibufvec.begin(), mpibufvec.end(),
                     [](const auto& mpibuf) { return !mpibuf->sendwait && !mpibuf->recvwait; });
}

namespace
{
void validate_rank4_mode(const nix::Array4D<nix::float64>& array, BoundaryMode mode)
{
  int expected = 0;
  switch (mode) {
  case BoundaryCopy10:
    expected = num_fluid_components;
    break;
  case BoundaryCopy6:
    expected = num_field_components;
    break;
  case BoundaryCopy3:
    expected = num_vector_components;
    break;
  default:
    throw std::invalid_argument("invalid rank-4 Hybrid boundary mode");
  }
  if (array.shape()[3] != static_cast<size_t>(expected)) {
    throw std::invalid_argument("rank-4 Hybrid boundary component count does not match mode");
  }
}

void validate_rank5_mode(const nix::Array5D<nix::float64>& array, BoundaryMode mode,
                         int num_species)
{
  if (mode == BoundaryCopy9) {
    if (array.shape()[3] != static_cast<size_t>(num_phase_directions) ||
        array.shape()[4] != static_cast<size_t>(num_phase_branches)) {
      throw std::invalid_argument("rank-5 Hybrid boundary shape does not match phase mode");
    }
    return;
  }
  if (mode != BoundaryMomentAccum && mode != BoundaryMomentCopy) {
    throw std::invalid_argument("invalid rank-5 Hybrid boundary mode");
  }
  if (array.shape()[3] != static_cast<size_t>(num_species) ||
      array.shape()[4] != static_cast<size_t>(num_moment_components)) {
    throw std::invalid_argument("rank-5 Hybrid boundary shape does not match moment mode");
  }
}
} // namespace

void HybridChunk::boundary_pack(nix::Array4D<nix::float64>& array, BoundaryMode mode)
{
  validate_rank4_mode(array, mode);
  auto halo = nix::XtensorHaloField3D<HybridChunk>(array, *this);
  pack_bc_exchange(mpibufvec[mode], halo);
}

void HybridChunk::boundary_unpack(nix::Array4D<nix::float64>& array, BoundaryMode mode)
{
  validate_rank4_mode(array, mode);
  auto halo = nix::XtensorHaloField3D<HybridChunk>(array, *this);
  unpack_bc_exchange(mpibufvec[mode], halo);
}

void HybridChunk::boundary_begin(nix::Array4D<nix::float64>& array, BoundaryMode mode)
{
  validate_rank4_mode(array, mode);
  auto halo = nix::XtensorHaloField3D<HybridChunk>(array, *this);
  begin_bc_exchange(mpibufvec[mode], halo);
}

void HybridChunk::boundary_end(nix::Array4D<nix::float64>& array, BoundaryMode mode)
{
  validate_rank4_mode(array, mode);
  auto halo = nix::XtensorHaloField3D<HybridChunk>(array, *this);
  end_bc_exchange(mpibufvec[mode], halo);
}

void HybridChunk::boundary_pack(nix::Array5D<nix::float64>& array, BoundaryMode mode)
{
  validate_rank5_mode(array, mode, num_species);
  if (mode == BoundaryMomentAccum) {
    auto halo = nix::XtensorHaloMoment3D<HybridChunk>(array, *this);
    pack_bc_exchange(mpibufvec[mode], halo);
  } else {
    auto halo = Rank5CopyHalo3D<HybridChunk>(array, *this);
    pack_bc_exchange(mpibufvec[mode], halo);
  }
}

void HybridChunk::boundary_unpack(nix::Array5D<nix::float64>& array, BoundaryMode mode)
{
  validate_rank5_mode(array, mode, num_species);
  if (mode == BoundaryMomentAccum) {
    auto halo = nix::XtensorHaloMoment3D<HybridChunk>(array, *this);
    unpack_bc_exchange(mpibufvec[mode], halo);
  } else {
    auto halo = Rank5CopyHalo3D<HybridChunk>(array, *this);
    unpack_bc_exchange(mpibufvec[mode], halo);
  }
}

void HybridChunk::boundary_begin(nix::Array5D<nix::float64>& array, BoundaryMode mode)
{
  validate_rank5_mode(array, mode, num_species);
  if (mode == BoundaryMomentAccum) {
    auto halo = nix::XtensorHaloMoment3D<HybridChunk>(array, *this);
    begin_bc_exchange(mpibufvec[mode], halo);
  } else {
    auto halo = Rank5CopyHalo3D<HybridChunk>(array, *this);
    begin_bc_exchange(mpibufvec[mode], halo);
  }
}

void HybridChunk::boundary_end(nix::Array5D<nix::float64>& array, BoundaryMode mode)
{
  validate_rank5_mode(array, mode, num_species);
  if (mode == BoundaryMomentAccum) {
    auto halo = nix::XtensorHaloMoment3D<HybridChunk>(array, *this);
    end_bc_exchange(mpibufvec[mode], halo);
  } else {
    auto halo = Rank5CopyHalo3D<HybridChunk>(array, *this);
    end_bc_exchange(mpibufvec[mode], halo);
  }
}

void HybridChunk::particle_boundary_pack()
{
  auto halo = nix::XtensorHaloParticle3D<HybridChunk>(particles, *this);
  pack_bc_exchange(mpibufvec[BoundaryParticle], halo);
}

void HybridChunk::particle_boundary_unpack()
{
  auto halo = nix::XtensorHaloParticle3D<HybridChunk>(particles, *this);
  unpack_bc_exchange(mpibufvec[BoundaryParticle], halo);
}

void HybridChunk::particle_boundary_begin()
{
  auto halo = nix::XtensorHaloParticle3D<HybridChunk>(particles, *this);
  begin_bc_exchange(mpibufvec[BoundaryParticle], halo);
}

void HybridChunk::particle_boundary_end()
{
  auto halo = nix::XtensorHaloParticle3D<HybridChunk>(particles, *this);
  end_bc_exchange(mpibufvec[BoundaryParticle], halo);
}

bool HybridChunk::particle_boundary_probe(bool wait)
{
  if (!wait) {
    return probe_bc_exchange(mpibufvec[BoundaryParticle]);
  }
  while (!probe_bc_exchange(mpibufvec[BoundaryParticle])) {
  }
  return true;
}

HybridChunk::ParticleDisplacement HybridChunk::get_max_particle_displacement() const
{
  ParticleDisplacement              result;
  const std::array<nix::float64, 3> widths = {zlim[2], ylim[2], xlim[2]};
  const std::array<bool, 3>         active = {has_zdim(), has_ydim(), has_xdim()};

  for (int is = 0; is < static_cast<int>(particles.size()); ++is) {
    const auto& particle = *particles[is];
    for (int ip = 0; ip < particle.Np; ++ip) {
      nix::float64 ratio = 0;
      for (int dim = 0; dim < 3; ++dim) {
        if (active[dim]) {
          const int component = 2 - dim;
          ratio =
              std::max(ratio, std::abs(particle.xu(ip, component) - particle.xv(ip, component)) /
                                  widths[dim]);
        }
      }
      if (ratio > result.ratio) {
        result.ratio    = ratio;
        result.species  = is;
        result.particle = ip;
        std::memcpy(&result.id, &particle.xu(ip, 6), sizeof(result.id));
        for (int dim = 0; dim < 3; ++dim) {
          result.before[dim] = particle.xv(ip, dim);
          result.after[dim]  = particle.xu(ip, dim);
        }
      }
    }
  }
  return result;
}

void HybridChunk::prepare_particle_migration()
{
  for (auto& particle : particles) {
    particle->count(0, particle->Np - 1, true, order);
  }
}

void HybridChunk::reset_load()
{
  const int num_cells = dims[0] * dims[1] * dims[2];

  load[LoadCell]     = option.value("cell_load", 1.0);
  load[LoadParticle] = 0;
  for (const auto& particle : particles) {
    load[LoadParticle] += static_cast<nix::float64>(particle->Np) / num_cells;
  }
}

void HybridChunk::setup(nix::json& config)
{
  const nix::float64 delh = config["delh"].get<nix::float64>();

  option = config.value("option", nix::json::object());
  if (!option.is_object()) {
    option = nix::json::object();
  }
  order           = particle_order;
  num_species     = config["Ns"].get<int>();
  light_speed     = config["cc"].get<nix::float64>();
  adiabatic_index = config["gamma"].get<nix::float64>();
  if (num_species <= 0) {
    throw std::invalid_argument("Hybrid setup requires at least one species");
  }
  if (!std::isfinite(delh) || delh <= 0) {
    throw std::invalid_argument("Hybrid setup requires positive finite grid spacing");
  }
  if (!std::isfinite(light_speed) || light_speed <= 0) {
    throw std::invalid_argument("Hybrid setup requires positive finite light speed");
  }
  if (!std::isfinite(adiabatic_index) || adiabatic_index <= 1) {
    throw std::invalid_argument("Hybrid setup requires a finite adiabatic index above one");
  }

  const bool   has_beam = config.contains("Npc");
  int          Npc      = 0;
  int          Mpc      = 0;
  nix::float64 mie      = 100.0;
  nix::float64 tie      = 1.0;
  nix::float64 betae    = 1.0;
  nix::float64 nb       = 0.02;
  nix::float64 vcpa     = 1.0;
  nix::float64 vcpe     = 1.0;
  nix::float64 vbd      = 10.0;
  nix::float64 vbpa     = 1.0;
  nix::float64 vbpe     = 1.0;
  if (has_beam) {
    for (const char* name :
         {"Mpc", "mie", "tie", "betae", "nb", "vcpa", "vcpe", "vbd", "vbpa", "vbpe"}) {
      if (!config.contains(name)) {
        throw std::invalid_argument(std::string("Hybrid beam setup requires parameter ") + name);
      }
    }
    Npc   = config["Npc"].get<int>();
    Mpc   = config["Mpc"].get<int>();
    mie   = config["mie"].get<nix::float64>();
    tie   = config["tie"].get<nix::float64>();
    betae = config["betae"].get<nix::float64>();
    nb    = config["nb"].get<nix::float64>();
    vcpa  = config["vcpa"].get<nix::float64>();
    vcpe  = config["vcpe"].get<nix::float64>();
    vbd   = config["vbd"].get<nix::float64>();
    vbpa  = config["vbpa"].get<nix::float64>();
    vbpe  = config["vbpe"].get<nix::float64>();

    if (num_species != 2) {
      throw std::invalid_argument("Hybrid beam setup requires exactly two kinetic species");
    }
    if (Npc != 4) {
      throw std::invalid_argument(
          "Hybrid beam setup currently supports exactly four particles per cell");
    }
    if (Mpc < Npc) {
      throw std::invalid_argument("Hybrid beam particle capacity must cover particles per cell");
    }
    if (!std::isfinite(mie) || mie <= 0 || !std::isfinite(tie) || tie <= 0 ||
        !std::isfinite(betae) || betae <= 0) {
      throw std::invalid_argument(
          "Hybrid beam mass ratio and temperatures must be positive and finite");
    }
    if (!std::isfinite(nb) || nb <= 0 || nb >= 1) {
      throw std::invalid_argument(
          "Hybrid beam density fraction must be finite and between zero and one");
    }
    if (!std::isfinite(vcpa) || vcpa < 0 || !std::isfinite(vcpe) || vcpe < 0 ||
        !std::isfinite(vbpa) || vbpa < 0 || !std::isfinite(vbpe) || vbpe < 0 ||
        !std::isfinite(vbd)) {
      throw std::invalid_argument(
          "Hybrid beam drift must be finite and thermal speeds must be finite and nonnegative");
    }
  }

  set_boundary_margin(hybrid::boundary_margin);
  set_coordinate(delh, delh, delh);

  allocate();

  // --- Derive beam parameters from config ---
  const nix::float64 roe = 1.0 / mie;
  const nix::float64 roi = 1.0 - nb;
  const nix::float64 rob = nb;
  const nix::float64 pre = betae;
  const nix::float64 b0  = std::sqrt(nix::math::pi4);
  const nix::float64 bx  = 1.0 * b0;

  const nix::float64 qe      = -light_speed / std::sqrt(nix::math::pi4);
  const nix::float64 me      = 1.0 / mie;
  const nix::float64 qme     = qe / me;
  const nix::float64 qmk_ion = -qme / mie;

  const std::array<nix::float64, 2> species_mass = {roi / std::max(1, Npc), rob / std::max(1, Npc)};
  const std::array<nix::float64, 2> species_drift = {-vbd * nb, vbd};
  const std::array<nix::float64, 2> species_th_pa = {vcpa, vbpa};
  const std::array<nix::float64, 2> species_th_pe = {vcpe, vbpe};

  // --- Particle initialization (beam config only) ---
  particles.resize(num_species);
  if (has_beam) {
    const int global_nx = gdims[2];
    const int global_ny = gdims[1];

    const auto local_xrange  = get_xrange();
    const auto local_yrange  = get_yrange();
    const auto local_zrange  = get_zrange();
    const auto global_xrange = get_xrange_global();
    const auto global_yrange = get_yrange_global();
    const auto global_zrange = get_zrange_global();
    const auto local_xmin    = std::get<0>(local_xrange);
    const auto local_ymin    = std::get<0>(local_yrange);
    const auto local_zmin    = std::get<0>(local_zrange);
    const auto global_xmin   = std::get<0>(global_xrange);
    const auto global_ymin   = std::get<0>(global_yrange);
    const auto global_zmin   = std::get<0>(global_zrange);

    const int local_nx   = dims[2];
    const int local_ny   = dims[1];
    const int local_nz   = dims[0];
    const int local_np   = Npc * local_nx * local_ny * local_nz;
    const int global_ix0 = static_cast<int>(std::llround((local_xmin - global_xmin) / delh));
    const int global_iy0 = static_cast<int>(std::llround((local_ymin - global_ymin) / delh));
    const int global_iz0 = static_cast<int>(std::llround((local_zmin - global_zmin) / delh));

    for (int species = 0; species < num_species; ++species) {
      particles[species] = std::make_shared<nix::XtensorParticle>(local_np, *this);
      auto& p            = *particles[species];
      p.q                = qmk_ion * species_mass[species];
      p.m                = species_mass[species];
      p.Np               = local_np;

      const nix::float64 drift      = species_drift[species];
      const nix::float64 thermal_pa = species_th_pa[species];
      const nix::float64 thermal_pe = species_th_pe[species];

      for (int ip = 0; ip < local_np; ++ip) {
        const int local_particle = ip % Npc;
        const int local_cell     = ip / Npc;
        const int local_iz       = local_cell / (local_ny * local_nx);
        const int local_iy       = (local_cell / local_nx) % local_ny;
        const int local_ix       = local_cell % local_nx;
        const int global_ix      = global_ix0 + local_ix;
        const int global_iy      = global_iy0 + local_iy;
        const int global_iz      = global_iz0 + local_iz;

        p.xu(ip, 0) = local_xmin + delh * (local_ix + (local_particle + 0.5) / Npc);
        p.xu(ip, 1) = local_ymin + delh * (local_iy + 0.5);
        p.xu(ip, 2) = local_zmin + delh * (local_iz + 0.5);

        const std::array<nix::float64, 4> vx_sample = {1.0, -1.0, 0.0, 0.0};
        const std::array<nix::float64, 4> vy_sample = {0.0, 0.0, 1.0, -1.0};
        p.xu(ip, 3) = vx_sample[local_particle] * thermal_pa + drift;
        p.xu(ip, 4) = vy_sample[local_particle] * thermal_pe;
        p.xu(ip, 5) = (local_particle < 2 ? thermal_pe : -thermal_pe);

        const int    cell = (global_iz * global_ny + global_iy) * global_nx + global_ix;
        std::int64_t id   = static_cast<std::int64_t>(cell * Npc + local_particle);
        std::memcpy(&p.xu(ip, 6), &id, sizeof(std::int64_t));
      }
      p.count(0, p.Np - 1, true, order);
      p.sort();
    }
  } else {
    for (auto& particle : particles) {
      particle    = std::make_shared<nix::XtensorParticle>(0, *this);
      particle->q = 0;
      particle->m = 0;
    }
  }

  // --- Fluid and field initialization ---
  for (int iz = 0; iz < static_cast<int>(fluid.shape()[0]); ++iz) {
    for (int iy = 0; iy < static_cast<int>(fluid.shape()[1]); ++iy) {
      for (int ix = 0; ix < static_cast<int>(fluid.shape()[2]); ++ix) {
        if (has_beam) {
          fluid(iz, iy, ix, fluid_component::electron_density)    = roe;
          fluid(iz, iy, ix, fluid_component::electron_velocity_x) = 0;
          fluid(iz, iy, ix, fluid_component::electron_velocity_y) = 0;
          fluid(iz, iy, ix, fluid_component::electron_velocity_z) = 0;
          fluid(iz, iy, ix, fluid_component::electron_pressure)   = pre;
          fluid(iz, iy, ix, fluid_component::ion_density)         = 0;
          fluid(iz, iy, ix, fluid_component::ion_velocity_x)      = 0;
          fluid(iz, iy, ix, fluid_component::ion_velocity_y)      = 0;
          fluid(iz, iy, ix, fluid_component::ion_velocity_z)      = 0;
          fluid(iz, iy, ix, fluid_component::ion_pressure)        = 0;

          field_cell(iz, iy, ix, field_component::electric_x) = 0;
          field_cell(iz, iy, ix, field_component::electric_y) = 0;
          field_cell(iz, iy, ix, field_component::electric_z) = 0;
          field_cell(iz, iy, ix, field_component::magnetic_x) = bx;
          field_cell(iz, iy, ix, field_component::magnetic_y) = 0;
          field_cell(iz, iy, ix, field_component::magnetic_z) = 0;

          field_staggered(iz, iy, ix, field_component::electric_x) = 0;
          field_staggered(iz, iy, ix, field_component::electric_y) = 0;
          field_staggered(iz, iy, ix, field_component::electric_z) = 0;
          field_staggered(iz, iy, ix, field_component::magnetic_x) = bx;
          field_staggered(iz, iy, ix, field_component::magnetic_y) = 0;
          field_staggered(iz, iy, ix, field_component::magnetic_z) = 0;
        }
      }
    }
  }

  allocate_mpi_buffers();
  reset_load();
}
} // namespace hybrid
