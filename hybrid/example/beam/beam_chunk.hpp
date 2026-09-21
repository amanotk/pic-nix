// -*- C++ -*-
#ifndef _HYBRID_BEAM_CHUNK_HPP_
#define _HYBRID_BEAM_CHUNK_HPP_

#include "hybrid/hybrid_chunk.hpp"

#include "nix/random.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace hybrid::beam
{
class BeamChunk : public HybridChunk
{
public:
  using HybridChunk::HybridChunk;

  void setup(nix::json& config) override
  {
    HybridChunk::setup(config);

    int random_seed = option.value("random_seed", 42);
    // incorporate chunk ID so each chunk gets a distinct, reproducible sequence
    random_seed += 1000 * myid;
    int          Npc         = config["Npc"].get<int>();
    int          Mpc         = config["Mpc"].get<int>();
    nix::float64 mie         = config["mie"].get<nix::float64>();
    nix::float64 tie         = config["tie"].get<nix::float64>();
    nix::float64 betae       = config["betae"].get<nix::float64>();
    nix::float64 nb          = config["nb"].get<nix::float64>();
    nix::float64 vcpa        = config["vcpa"].get<nix::float64>();
    nix::float64 vcpe        = config["vcpe"].get<nix::float64>();
    nix::float64 vbd         = config["vbd"].get<nix::float64>();
    nix::float64 vbpa        = config["vbpa"].get<nix::float64>();
    nix::float64 vbpe        = config["vbpe"].get<nix::float64>();

    if (num_species != 2) {
      throw std::invalid_argument("Hybrid beam setup requires exactly two kinetic species");
    }
    if (Npc <= 0 || Mpc < Npc) {
      throw std::invalid_argument("Hybrid beam particle counts must be positive with Mpc >= Npc");
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
      throw std::invalid_argument("Hybrid beam thermal speeds must be finite and nonnegative");
    }

    const nix::float64 roe = 1.0 / mie;
    const nix::float64 roi = 1.0 - nb;
    const nix::float64 rob = nb;
    const nix::float64 pre = betae;
    const nix::float64 b0  = std::sqrt(nix::math::pi4);
    const nix::float64 bx  = 1.0 * b0;

    const nix::float64 qmk = light_speed / std::sqrt(nix::math::pi4);

    const std::array<nix::float64, 2> species_mass  = {roi / Npc, rob / Npc};
    const std::array<nix::float64, 2> species_drift = {-vbd * nb, vbd};
    const std::array<nix::float64, 2> species_vt_pa = {vcpa, vbpa};
    const std::array<nix::float64, 2> species_vt_pe = {vcpe, vbpe};

    const auto [local_xmin, local_xmax] = get_xrange();
    const auto [local_ymin, local_ymax] = get_yrange();
    const auto [local_zmin, local_zmax] = get_zrange();
    const auto local_xlen               = local_xmax - local_xmin;
    const auto local_ylen               = local_ymax - local_ymin;
    const auto local_zlen               = local_zmax - local_zmin;
    const int  local_np                 = Npc * dims[2] * dims[1] * dims[0];

    std::mt19937_64   mtp(random_seed);
    std::mt19937_64   mtv(random_seed + 1);
    nix::rand_uniform uniform(0.0, 1.0);
    nix::rand_normal  normal(0.0, 1.0);

    particles.clear();
    particles.reserve(num_species);

    for (int species = 0; species < num_species; ++species) {
      particles.push_back(std::make_shared<nix::XtensorParticle>(local_np * 2, *this));
      auto& p = *particles.back();
      p.q     = qmk * species_mass[species];
      p.m     = species_mass[species];
      p.Np    = local_np;

      const nix::float64 drift   = species_drift[species];
      const nix::float64 vt_pa   = species_vt_pa[species];
      const nix::float64 vt_pe   = species_vt_pe[species];
      const nix::int64   id_base = static_cast<nix::int64>(myid) * local_np;

      for (int ip = 0; ip < local_np; ++ip) {
        p.xu(ip, 0) = local_xmin + uniform(mtp) * local_xlen;
        p.xu(ip, 1) = local_ymin + uniform(mtp) * local_ylen;
        p.xu(ip, 2) = local_zmin + uniform(mtp) * local_zlen;

        p.xu(ip, 3) = normal(mtv) * vt_pa + drift;
        p.xu(ip, 4) = normal(mtv) * vt_pe;
        p.xu(ip, 5) = normal(mtv) * vt_pe;

        nix::int64 id = id_base + ip;
        std::memcpy(&p.xu(ip, 6), &id, sizeof(nix::int64));
      }
      p.count(0, p.Np - 1, true, order);
      p.sort();
    }

    for (int iz = 0; iz < static_cast<int>(fluid.shape()[0]); ++iz) {
      for (int iy = 0; iy < static_cast<int>(fluid.shape()[1]); ++iy) {
        for (int ix = 0; ix < static_cast<int>(fluid.shape()[2]); ++ix) {
          fluid(iz, iy, ix, hybrid::fluid_component::electron_density)    = roe;
          fluid(iz, iy, ix, hybrid::fluid_component::electron_velocity_x) = 0;
          fluid(iz, iy, ix, hybrid::fluid_component::electron_velocity_y) = 0;
          fluid(iz, iy, ix, hybrid::fluid_component::electron_velocity_z) = 0;
          fluid(iz, iy, ix, hybrid::fluid_component::electron_pressure)   = pre;
          fluid(iz, iy, ix, hybrid::fluid_component::ion_density)         = 0;
          fluid(iz, iy, ix, hybrid::fluid_component::ion_velocity_x)      = 0;
          fluid(iz, iy, ix, hybrid::fluid_component::ion_velocity_y)      = 0;
          fluid(iz, iy, ix, hybrid::fluid_component::ion_velocity_z)      = 0;
          fluid(iz, iy, ix, hybrid::fluid_component::ion_pressure)        = 0;

          field_cell(iz, iy, ix, hybrid::field_component::electric_x) = 0;
          field_cell(iz, iy, ix, hybrid::field_component::electric_y) = 0;
          field_cell(iz, iy, ix, hybrid::field_component::electric_z) = 0;
          field_cell(iz, iy, ix, hybrid::field_component::magnetic_x) = bx;
          field_cell(iz, iy, ix, hybrid::field_component::magnetic_y) = 0;
          field_cell(iz, iy, ix, hybrid::field_component::magnetic_z) = 0;

          field_staggered(iz, iy, ix, hybrid::field_component::electric_x) = 0;
          field_staggered(iz, iy, ix, hybrid::field_component::electric_y) = 0;
          field_staggered(iz, iy, ix, hybrid::field_component::electric_z) = 0;
          field_staggered(iz, iy, ix, hybrid::field_component::magnetic_x) = bx;
          field_staggered(iz, iy, ix, hybrid::field_component::magnetic_y) = 0;
          field_staggered(iz, iy, ix, hybrid::field_component::magnetic_z) = 0;
        }
      }
    }
  }
};
} // namespace hybrid::beam

#endif
