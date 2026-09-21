// -*- C++ -*-
#ifndef _HYBRID_ENGINE_PARTICLE_HPP_
#define _HYBRID_ENGINE_PARTICLE_HPP_

#include "interpolation.hpp"

#include "hybrid_chunk.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace hybrid::engine
{
using Velocity = std::array<nix::float64, 3>;

inline void buneman_boris(Velocity& velocity, const FieldValue& impulse)
{
  velocity[0] += impulse[0];
  velocity[1] += impulse[1];
  velocity[2] += impulse[2];

  const nix::float64 scale =
      2.0 / (1.0 + impulse[3] * impulse[3] + impulse[4] * impulse[4] + impulse[5] * impulse[5]);
  const Velocity rotated = {
      velocity[0] + velocity[1] * impulse[5] - velocity[2] * impulse[4],
      velocity[1] + velocity[2] * impulse[3] - velocity[0] * impulse[5],
      velocity[2] + velocity[0] * impulse[4] - velocity[1] * impulse[3],
  };

  velocity[0] += (rotated[1] * impulse[5] - rotated[2] * impulse[4]) * scale + impulse[0];
  velocity[1] += (rotated[2] * impulse[3] - rotated[0] * impulse[5]) * scale + impulse[1];
  velocity[2] += (rotated[0] * impulse[4] - rotated[1] * impulse[3]) * scale + impulse[2];
}

inline nix::float64 particle_cfl(const HybridChunk::DataContainer& data, nix::float64 time_step)
{
  nix::float64 result = 0;
  for (const auto& particle : data.particles) {
    for (int ip = 0; ip < particle->Np; ++ip) {
      result = std::max({result, std::abs(particle->xu(ip, 3) * time_step / particle->delx),
                         std::abs(particle->xu(ip, 4) * time_step / particle->dely),
                         std::abs(particle->xu(ip, 5) * time_step / particle->delz)});
    }
  }
  return result;
}

inline nix::float64 push_particles(HybridChunk::DataContainer&       data,
                                   const nix::Array4D<nix::float64>& field, nix::float64 time_step)
{
  if (data.light_speed == 0) {
    throw std::invalid_argument("Hybrid particle push requires nonzero light speed");
  }

  for (const auto& particle : data.particles) {
    if (particle->m == 0) {
      throw std::invalid_argument("Hybrid particle push requires nonzero particle mass");
    }
    for (int ip = 0; ip < particle->Np; ++ip) {
      const nix::float64 initial_cfl =
          std::max({std::abs(particle->xu(ip, 3) * time_step / particle->delx),
                    std::abs(particle->xu(ip, 4) * time_step / particle->dely),
                    std::abs(particle->xu(ip, 5) * time_step / particle->delz)});
      if (initial_cfl >= 1.0) {
        throw std::runtime_error("Hybrid particle push requires cell CFL below one");
      }
    }
  }

  for (auto& particle : data.particles) {
    if (particle->Np > 0) {
      std::memcpy(particle->xv.data(), particle->xu.data(),
                  static_cast<size_t>(particle->Np) * nix::Particle::Nc * sizeof(nix::float64));
    }
  }

  nix::float64 max_cfl = 0;
  try {
    for (auto& particle : data.particles) {
      const nix::float64 charge_impulse = 0.5 * time_step * particle->q / particle->m;
      for (int ip = 0; ip < particle->Np; ++ip) {
        Position        position = {particle->xu(ip, 0), particle->xu(ip, 1), particle->xu(ip, 2)};
        Velocity        velocity = {particle->xu(ip, 3), particle->xu(ip, 4), particle->xu(ip, 5)};
        const GridIndex anchor   = particle_cell(*particle, position);

        for (int component = 0; component < 3; ++component) {
          position[component] += 0.5 * time_step * velocity[component];
        }
        FieldValue impulse =
            interpolate_collocated(field, data.background_cell, *particle, anchor, position);
        for (int component = 0; component < 3; ++component) {
          impulse[component] *= charge_impulse;
          impulse[component + 3] *= charge_impulse / data.light_speed;
        }
        buneman_boris(velocity, impulse);
        for (int component = 0; component < 3; ++component) {
          position[component] += 0.5 * time_step * velocity[component];
          particle->xu(ip, component)     = position[component];
          particle->xu(ip, component + 3) = velocity[component];
        }
        const nix::float64 current_cfl =
            std::max({std::abs(velocity[0] * time_step / particle->delx),
                      std::abs(velocity[1] * time_step / particle->dely),
                      std::abs(velocity[2] * time_step / particle->delz)});
        if (current_cfl >= 1.0) {
          throw std::runtime_error("Hybrid particle push requires cell CFL below one");
        }
        max_cfl = std::max(max_cfl, current_cfl);
      }
    }
  } catch (...) {
    for (auto& particle : data.particles) {
      particle->swap();
    }
    throw;
  }
  return max_cfl;
}

inline void rollback_particles(HybridChunk::DataContainer& data)
{
  for (auto& particle : data.particles) {
    particle->swap();
  }
}
} // namespace hybrid::engine

#endif
