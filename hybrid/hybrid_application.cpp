// -*- C++ -*-
#include "hybrid_application.hpp"

#include "hybrid_chunk.hpp"
#include "hybrid_diag.hpp"

#include "engine/field.hpp"
#include "engine/filter.hpp"
#include "engine/fluid.hpp"
#include "engine/mc2.hpp"
#include "engine/moment.hpp"
#include "engine/ohm_bridge.hpp"
#include "engine/ohm_source.hpp"
#include "engine/particle.hpp"
#include "engine/pcc2.hpp"
#include "engine/phasespeed.hpp"
#include "engine/ssor2.hpp"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace hybrid
{
nix::Application::PtrChunk HybridApplicationInterface::create_chunk(nix::Dims3D dims,
                                                                    nix::Bool3D has_dim, int id)
{
  return std::make_unique<HybridChunk>(dims, has_dim, id);
}

HybridApplication::HybridApplication(int argc, char** argv, PtrInterface interface)
    : base_type(argc, argv, std::move(interface))
{
}

void HybridApplication::initialize(int argc, char** argv)
{
  base_type::initialize(argc, argv);
  for (int mode = 0; mode < NumBoundaryModes; ++mode) {
    for (int iz = 0; iz < 3; ++iz) {
      for (int iy = 0; iy < 3; ++iy) {
        for (int ix = 0; ix < 3; ++ix) {
          MPI_Comm_dup(MPI_COMM_WORLD, &mpicommvec(mode, iz, iy, ix));
        }
      }
    }
  }
  communicators_initialized = true;
}

void HybridApplication::finalize()
{
  if (communicators_initialized) {
    for (int mode = 0; mode < NumBoundaryModes; ++mode) {
      for (int iz = 0; iz < 3; ++iz) {
        for (int iy = 0; iy < 3; ++iy) {
          for (int ix = 0; ix < 3; ++ix) {
            MPI_Comm_free(&mpicommvec(mode, iz, iy, ix));
          }
        }
      }
    }
    communicators_initialized = false;
  }
  base_type::finalize();
}

void HybridApplication::set_chunk_communicators()
{
  for (auto& chunk_ptr : chunkvec) {
    auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
    for (int mode = 0; mode < NumBoundaryModes; ++mode) {
      for (int iz = 0; iz < 3; ++iz) {
        for (int iy = 0; iy < 3; ++iy) {
          for (int ix = 0; ix < 3; ++ix) {
            chunk.set_mpi_communicator(mode, iz, iy, ix, mpicommvec(mode, iz, iy, ix));
          }
        }
      }
    }
  }
}

void HybridApplication::restore_accepted_halos()
{
  auto exchange_rank4 = [&](auto get_array, BoundaryMode mode) {
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      chunk.boundary_pack(get_array(data), mode);
      chunk.boundary_begin(get_array(data), mode);
    }
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      chunk.boundary_end(get_array(data), mode);
      chunk.boundary_unpack(get_array(data), mode);
    }
  };
  auto exchange_rank5 = [&](auto get_array, BoundaryMode mode) {
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      chunk.boundary_pack(get_array(data), mode);
      chunk.boundary_begin(get_array(data), mode);
    }
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      chunk.boundary_end(get_array(data), mode);
      chunk.boundary_unpack(get_array(data), mode);
    }
  };

  exchange_rank4(
      [](auto& data) -> auto& { return data.fluid; }, BoundaryCopy10);
  exchange_rank4(
      [](auto& data) -> auto& { return data.field_cell; }, BoundaryCopy6);
  exchange_rank4(
      [](auto& data) -> auto& { return data.field_staggered; }, BoundaryCopy6);
  exchange_rank5(
      [](auto& data) -> auto& { return data.moment_kinetic; }, BoundaryMomentCopy);
}

void HybridApplication::setup_chunks()
{
  base_type::setup_chunks();
  set_chunk_communicators();
  restore_accepted_halos();
}

bool HybridApplication::rebalance()
{
  if (!base_type::rebalance()) {
    return false;
  }
  set_chunk_communicators();
  restore_accepted_halos();
  return true;
}

void HybridApplication::migrate_particles()
{
  HybridChunk::ParticleDisplacement local;
  for (const auto& chunk_ptr : chunkvec) {
    auto candidate = static_cast<const HybridChunk&>(*chunk_ptr).get_max_particle_displacement();
    if (candidate.ratio > local.ratio) {
      local = candidate;
    }
  }

  struct {
    double ratio;
    int    rank;
  } local_max{local.ratio, thisrank}, global_max{};
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

  static_assert(std::is_trivially_copyable_v<HybridChunk::ParticleDisplacement>);
  HybridChunk::ParticleDisplacement global = local;
  MPI_Bcast(&global, sizeof(global), MPI_BYTE, global_max.rank, MPI_COMM_WORLD);
  assert_mpi(global.ratio <= 1.0,
             fmt::format("particle crossed more than one chunk: ratio={}, species={}, id={}, "
                         "before=({}, {}, {}), after=({}, {}, {})",
                         global.ratio, global.species, global.id, global.before[0],
                         global.before[1], global.before[2], global.after[0], global.after[1],
                         global.after[2]));

  for (auto& chunk_ptr : chunkvec) {
    static_cast<HybridChunk&>(*chunk_ptr).prepare_particle_migration();
  }
  for (auto& chunk_ptr : chunkvec) {
    auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
    chunk.particle_boundary_pack();
    chunk.particle_boundary_begin();
  }
  for (auto& chunk_ptr : chunkvec) {
    static_cast<HybridChunk&>(*chunk_ptr).particle_boundary_probe(true);
  }
  for (auto& chunk_ptr : chunkvec) {
    auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
    chunk.particle_boundary_end();
    chunk.particle_boundary_unpack();
    chunk.reset_load();
  }
}

void HybridApplication::update_kinetic_moments()
{
  for (auto& chunk_ptr : chunkvec) {
    auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
    auto  data  = chunk.get_internal_data();
    engine::deposit_moments(data);
    chunk.boundary_pack(data.moment_kinetic, BoundaryMomentAccum);
    chunk.boundary_begin(data.moment_kinetic, BoundaryMomentAccum);
  }
  for (auto& chunk_ptr : chunkvec) {
    auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
    auto  data  = chunk.get_internal_data();
    chunk.boundary_end(data.moment_kinetic, BoundaryMomentAccum);
    chunk.boundary_unpack(data.moment_kinetic, BoundaryMomentAccum);
  }

  for (auto& chunk_ptr : chunkvec) {
    auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
    auto  data  = chunk.get_internal_data();
    chunk.boundary_pack(data.moment_kinetic, BoundaryMomentCopy);
    chunk.boundary_begin(data.moment_kinetic, BoundaryMomentCopy);
  }
  for (auto& chunk_ptr : chunkvec) {
    auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
    auto  data  = chunk.get_internal_data();
    chunk.boundary_end(data.moment_kinetic, BoundaryMomentCopy);
    chunk.boundary_unpack(data.moment_kinetic, BoundaryMomentCopy);
  }

  for (int pass = 0; pass < 2; ++pass) {
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      engine::filter_moments_once(data);
      chunk.boundary_pack(data.moment_kinetic, BoundaryMomentCopy);
      chunk.boundary_begin(data.moment_kinetic, BoundaryMomentCopy);
    }
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      chunk.boundary_end(data.moment_kinetic, BoundaryMomentCopy);
      chunk.boundary_unpack(data.moment_kinetic, BoundaryMomentCopy);
    }
  }

  for (auto& chunk_ptr : chunkvec) {
    auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
    engine::derive_current(data);
  }
}

void HybridApplication::require_kinetic_particles() const
{
  int local_count = 0;
  for (const auto& chunk_ptr : chunkvec) {
    const auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
    for (const auto& particle : data.particles) {
      local_count += particle->Np;
    }
  }
  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (global_count == 0) {
    throw std::invalid_argument("Hybrid application push requires kinetic particles");
  }
}

bool HybridApplication::is_push_needed()
{
  return curtime < argparser->get_physical_time_max();
}

void HybridApplication::push()
{
  try {
    // Extract global parameters from first chunk
    auto&              first_chunk     = static_cast<HybridChunk&>(*chunkvec.front());
    auto               first_data      = first_chunk.get_internal_data();
    const nix::float64 light_speed     = first_data.light_speed;
    const nix::float64 adiabatic_index = first_data.adiabatic_index;
    const nix::float64 dt              = cfgparser->get_delt();
    require_kinetic_particles();

    const std::filesystem::path diagnostic_root = "diagnostics";
    const auto                  write_snapshot  = [&](int step, nix::float64 time) {
      const auto step_dir = diagnostic_root / "snapshots" / ("step_" + std::to_string(step));
      int        directory_ready = 1;
      if (thisrank == 0) {
        try {
          std::filesystem::remove_all(step_dir);
        } catch (const std::filesystem::filesystem_error&) {
          directory_ready = 0;
        }
      }
      MPI_Bcast(&directory_ready, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (directory_ready == 0) {
        throw std::runtime_error("Failed to clear Hybrid diagnostic step directory");
      }
      MPI_Barrier(MPI_COMM_WORLD);

      for (const auto& chunk_ptr : chunkvec) {
        auto&                     chunk  = static_cast<HybridChunk&>(*chunk_ptr);
        auto                      data   = chunk.get_internal_data();
        const auto                offset = chunk.get_offset();
        const auto                dims   = chunk.get_dims();
        std::vector<nix::float64> particle_mass;
        std::vector<nix::float64> particle_charge;
        particle_mass.reserve(data.particles.size());
        particle_charge.reserve(data.particles.size());
        for (const auto& particle : data.particles) {
          particle_mass.push_back(particle->m);
          particle_charge.push_back(particle->q);
        }
        const diag::SnapshotMetadata metadata{
            thisrank,
            chunk.get_id(),
            {offset[0], offset[1], offset[2]},
            {dims[0], dims[1], dims[2]},
            {ndims[0], ndims[1], ndims[2]},
            step,
            time,
            dt,
            std::move(particle_mass),
            std::move(particle_charge),
        };
        diag::write_diagnostics(data, diagnostic_root, metadata);
      }
    };

    if (curstep == 0) {
      write_snapshot(curstep, curtime);
    }

    // Copy accepted state to working arrays
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      for (int iz = 0; iz < static_cast<int>(data.fluid.shape()[0]); ++iz) {
        for (int iy = 0; iy < static_cast<int>(data.fluid.shape()[1]); ++iy) {
          for (int ix = 0; ix < static_cast<int>(data.fluid.shape()[2]); ++ix) {
            for (int comp = 0; comp < num_fluid_components; ++comp) {
              data.work_fluid(iz, iy, ix, comp) = data.fluid(iz, iy, ix, comp);
            }
            for (int comp = 0; comp < num_field_components; ++comp) {
              data.work_field_cell(iz, iy, ix, comp)      = data.field_cell(iz, iy, ix, comp);
              data.work_field_staggered(iz, iy, ix, comp) = data.field_staggered(iz, iy, ix, comp);
            }
          }
        }
      }
    }

    // All subsequent stages reuse the same snapshot of accepted phase speed.
    // Charge-to-mass ratios: legacy beam sets qmi=0 (ion fluid inactive) and
    // qme = qe/me = -cc * mie / sqrt(4*pi).
    const nix::float64                 mie = cfgparser->get_parameter().value("mie", 100.0);
    const nix::float64                 qme = -light_speed * mie / std::sqrt(nix::math::pi4);
    const engine::PhaseSpeedParameters phase_params{
        light_speed,
        adiabatic_index,
        0.0,
        qme,
        first_chunk.get_delx(),
        first_chunk.get_dely(),
        first_chunk.get_delz(),
    };

    const nix::float64 electron_entropy =
        first_data.fluid(first_data.Lbz, first_data.Lby, first_data.Lbx,
                         fluid_component::electron_pressure) /
        std::pow(first_data.fluid(first_data.Lbz, first_data.Lby, first_data.Lbx,
                                  fluid_component::electron_density),
                 adiabatic_index);
    const engine::FluidParameters fluid_parameters{
        light_speed,
        adiabatic_index,
        phase_params.electron_charge_to_mass,
        phase_params.ion_charge_to_mass,
        electron_entropy,
    };

    const engine::OhmSolverCoefficients ssor2_coeff = engine::compute_ssor2_coefficients(
        light_speed, phase_params.spacing_x, phase_params.spacing_y, phase_params.spacing_z);

    const engine::Ssor2Config ssor2_config{100, 1.0e-5};

    std::ostringstream ssor_log;
    int                ohm_stage_index = 0;

    const engine::GridSpacing grid_spacing{phase_params.spacing_x, phase_params.spacing_y,
                                           phase_params.spacing_z};

    const engine::FluxSpacing flux_spacing{phase_params.spacing_x, phase_params.spacing_y,
                                           phase_params.spacing_z};

    update_kinetic_moments();

    // --- Phase speed evaluation ---
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      for (int iz = 0; iz < static_cast<int>(data.phase_cell.shape()[0]); ++iz) {
        for (int iy = 0; iy < static_cast<int>(data.phase_cell.shape()[1]); ++iy) {
          for (int ix = 0; ix < static_cast<int>(data.phase_cell.shape()[2]); ++ix) {
            engine::FluidState  local_fluid = {};
            engine::FieldState  local_field = {};
            engine::VectorState local_bg    = {};
            for (int c = 0; c < num_fluid_components; ++c) {
              local_fluid[c] = data.fluid(iz, iy, ix, c);
            }
            for (int c = 0; c < num_field_components; ++c) {
              local_field[c] = data.field_cell(iz, iy, ix, c);
            }
            for (int c = 0; c < num_vector_components; ++c) {
              local_bg[c] = data.background_cell(iz, iy, ix, c);
            }
            std::vector<engine::KineticPhaseMoment> kinetic_moments;
            kinetic_moments.reserve(static_cast<size_t>(data.num_species));
            for (int species = 0; species < data.num_species; ++species) {
              engine::KineticPhaseMoment kinetic = {};
              for (int c = 0; c < num_moment_components; ++c) {
                kinetic.moment[c] = data.moment_kinetic(iz, iy, ix, species, c);
              }
              kinetic.charge_to_mass = data.particles[species]->q / data.particles[species]->m;
              kinetic_moments.push_back(kinetic);
            }
            const auto phase =
                kinetic_moments.empty()
                    ? engine::default_phase_speed(local_fluid, local_field, local_bg, phase_params)
                    : engine::default_phase_speed(local_fluid, local_field, local_bg,
                                                  kinetic_moments, phase_params);
            for (int dir = 0; dir < num_phase_directions; ++dir) {
              for (int branch = 0; branch < num_phase_branches; ++branch) {
                data.phase_cell(iz, iy, ix, dir, branch) = phase[3 * dir + branch];
              }
            }
          }
        }
      }
    }

    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      chunk.boundary_pack(data.phase_cell, BoundaryCopy9);
      chunk.boundary_begin(data.phase_cell, BoundaryCopy9);
    }
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      chunk.boundary_end(data.phase_cell, BoundaryCopy9);
      chunk.boundary_unpack(data.phase_cell, BoundaryCopy9);
    }

    // Interpolate phase_cell → phase_face (max of two adjacent cells per direction)
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      // x-direction faces
      for (int iz = 0; iz < static_cast<int>(data.phase_face.shape()[0]); ++iz) {
        for (int iy = 0; iy < static_cast<int>(data.phase_face.shape()[1]); ++iy) {
          for (int ix = data.Lbx - 1; ix <= data.Ubx; ++ix) {
            const auto left                   = data.phase_cell(iz, iy, ix, 0, 0);
            const auto right                  = data.phase_cell(iz, iy, ix + 1, 0, 0);
            data.phase_face(iz, iy, ix, 0, 0) = std::max(left, right);
            data.phase_face(iz, iy, ix, 0, 1) =
                std::max(data.phase_cell(iz, iy, ix, 0, 1), data.phase_cell(iz, iy, ix + 1, 0, 1));
          }
        }
      }
      // y-direction faces
      for (int iz = 0; iz < static_cast<int>(data.phase_face.shape()[0]); ++iz) {
        for (int iy = data.Lby - 1; iy <= data.Uby; ++iy) {
          for (int ix = 0; ix < static_cast<int>(data.phase_face.shape()[2]); ++ix) {
            data.phase_face(iz, iy, ix, 1, 0) =
                std::max(data.phase_cell(iz, iy, ix, 1, 0), data.phase_cell(iz, iy + 1, ix, 1, 0));
            data.phase_face(iz, iy, ix, 1, 1) =
                std::max(data.phase_cell(iz, iy, ix, 1, 1), data.phase_cell(iz, iy + 1, ix, 1, 1));
          }
        }
      }
      // z-direction faces
      for (int iz = data.Lbz - 1; iz <= data.Ubz; ++iz) {
        for (int iy = 0; iy < static_cast<int>(data.phase_face.shape()[1]); ++iy) {
          for (int ix = 0; ix < static_cast<int>(data.phase_face.shape()[2]); ++ix) {
            data.phase_face(iz, iy, ix, 2, 0) =
                std::max(data.phase_cell(iz, iy, ix, 2, 0), data.phase_cell(iz + 1, iy, ix, 2, 0));
            data.phase_face(iz, iy, ix, 2, 1) =
                std::max(data.phase_cell(iz, iy, ix, 2, 1), data.phase_cell(iz + 1, iy, ix, 2, 1));
          }
        }
      }
    }

    // --- PCC2 stages ---
    for (engine::Pcc2Stage stage                 = engine::Pcc2Stage::PredictorField;
         stage != engine::Pcc2Stage::Idle; stage = engine::pcc2_next_stage(stage)) {
      if (stage == engine::Pcc2Stage::Commit)
        break;

      if (engine::pcc2_is_field_stage(stage)) {
        // Reconstruct cell states to faces and compute independent directional HLL fluxes.
        for (auto& chunk_ptr : chunkvec) {
          auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
          engine::compute_mc2_face_fluxes(data, dt, fluid_parameters);
          engine::compute_mc2_edge_electric(data);
        }

        // Edge electric values need two ghost cells for the Lb-1 CT update.
        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          chunk.boundary_pack(data.field_flux, BoundaryCopy6);
          chunk.boundary_begin(data.field_flux, BoundaryCopy6);
        }
        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          chunk.boundary_end(data.field_flux, BoundaryCopy6);
          chunk.boundary_unpack(data.field_flux, BoundaryCopy6);
        }

        for (auto& chunk_ptr : chunkvec) {
          auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
          for (int iz = data.Lbz - 1; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby - 1; iy <= data.Uby; ++iy) {
              for (int ix = data.Lbx - 1; ix <= data.Ubx; ++ix) {
                engine::FieldState baseline = {}, edge = {}, x_minus = {}, y_minus = {},
                                   z_minus = {};
                for (int component = 0; component < num_field_components; ++component) {
                  baseline[component] = data.field_staggered(iz, iy, ix, component);
                  edge[component]     = data.field_flux(iz, iy, ix, component);
                  x_minus[component]  = data.field_flux(iz, iy, ix - 1, component);
                  y_minus[component]  = data.field_flux(iz, iy - 1, ix, component);
                  z_minus[component]  = data.field_flux(iz - 1, iy, ix, component);
                }
                for (int component = 0; component < num_vector_components; ++component) {
                  data.work_field_staggered(iz, iy, ix, component) = edge[component];
                }
                const auto magnetic = engine::constrained_transport_magnetic(
                    baseline, edge, x_minus, y_minus, z_minus, grid_spacing, light_speed, dt);
                for (int component = 0; component < num_vector_components; ++component) {
                  data.work_field_staggered(iz, iy, ix, field_component::magnetic_x + component) =
                      magnetic[component];
                }
              }
            }
          }
        }

        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          chunk.boundary_pack(data.work_field_staggered, BoundaryCopy6);
          chunk.boundary_begin(data.work_field_staggered, BoundaryCopy6);
        }
        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          chunk.boundary_end(data.work_field_staggered, BoundaryCopy6);
          chunk.boundary_unpack(data.work_field_staggered, BoundaryCopy6);
        }

        for (auto& chunk_ptr : chunkvec) {
          auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
          for (int iz = data.Lbz - 1; iz <= data.Ubz + 1; ++iz) {
            for (int iy = data.Lby - 1; iy <= data.Uby + 1; ++iy) {
              for (int ix = data.Lbx - 1; ix <= data.Ubx + 1; ++ix) {
                data.work_field_cell(iz, iy, ix, field_component::magnetic_x) =
                    engine::magnetic_face_to_cell(
                        field_component::magnetic_x,
                        data.work_field_staggered(iz, iy, ix, field_component::magnetic_x),
                        data.work_field_staggered(iz, iy, ix - 1, field_component::magnetic_x));
                data.work_field_cell(iz, iy, ix, field_component::magnetic_y) =
                    engine::magnetic_face_to_cell(
                        field_component::magnetic_y,
                        data.work_field_staggered(iz, iy, ix, field_component::magnetic_y),
                        data.work_field_staggered(iz, iy - 1, ix, field_component::magnetic_y));
                data.work_field_cell(iz, iy, ix, field_component::magnetic_z) =
                    engine::magnetic_face_to_cell(
                        field_component::magnetic_z,
                        data.work_field_staggered(iz, iy, ix, field_component::magnetic_z),
                        data.work_field_staggered(iz - 1, iy, ix, field_component::magnetic_z));
              }
            }
          }

          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
                engine::FieldState xp = {}, xm = {}, yp = {}, ym = {}, zp = {}, zm = {};
                for (int c = 0; c < num_field_components; ++c) {
                  xp[c] = data.work_field_cell(iz, iy, ix + 1, c);
                  xm[c] = data.work_field_cell(iz, iy, ix - 1, c);
                  yp[c] = data.work_field_cell(iz, iy + 1, ix, c);
                  ym[c] = data.work_field_cell(iz, iy - 1, ix, c);
                  zp[c] = data.work_field_cell(iz + 1, iy, ix, c);
                  zm[c] = data.work_field_cell(iz - 1, iy, ix, c);
                }
                const auto curl =
                    engine::curl_magnetic(xp, xm, yp, ym, zp, zm, grid_spacing, light_speed);
                for (int c = 0; c < num_vector_components; ++c) {
                  data.curl_b(iz, iy, ix, c) = curl[c];
                }
              }
            }
          }

          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
                engine::FluidState   local_fluid    = {};
                engine::FieldState   accepted_field = {};
                engine::FieldState   working_field  = {};
                engine::VectorState  bg             = {};
                engine::CurrentState cur            = {};
                for (int c = 0; c < num_fluid_components; ++c) {
                  local_fluid[c] = data.fluid(iz, iy, ix, c);
                }
                for (int c = 0; c < num_field_components; ++c) {
                  accepted_field[c] = data.field_cell(iz, iy, ix, c);
                  working_field[c]  = data.work_field_cell(iz, iy, ix, c);
                }
                for (int c = 0; c < num_vector_components; ++c) {
                  bg[c] = data.background_cell(iz, iy, ix, c);
                }
                for (int c = 0; c < num_current_components; ++c) {
                  cur[c] = data.current_kinetic(iz, iy, ix, c);
                }

                const auto uc = engine::conservative(local_fluid, accepted_field, fluid_parameters);
                const auto rh = engine::fluid_rhs(dt, working_field, cur, bg, fluid_parameters);

                engine::ConservedState fx_minus = {}, fx_plus = {}, fy_minus = {}, fy_plus = {},
                                       fz_minus = {}, fz_plus = {};
                for (int c = 0; c < num_conserved_components; ++c) {
                  fx_minus[c] = data.fluid_flux(iz, iy, ix - 1, 0, c);
                  fx_plus[c]  = data.fluid_flux(iz, iy, ix, 0, c);
                  fy_minus[c] = data.fluid_flux(iz, iy - 1, ix, 1, c);
                  fy_plus[c]  = data.fluid_flux(iz, iy, ix, 1, c);
                  fz_minus[c] = data.fluid_flux(iz - 1, iy, ix, 2, c);
                  fz_plus[c]  = data.fluid_flux(iz, iy, ix, 2, c);
                }
                const auto vc = engine::advance_conserved_fluid(
                    uc, fx_minus, fx_plus, fy_minus, fy_plus, fz_minus, fz_plus, rh, flux_spacing);

                engine::FieldState  work_field = {};
                engine::VectorState curl_vec   = {};
                for (int c = 0; c < num_field_components; ++c) {
                  work_field[c] = data.work_field_cell(iz, iy, ix, c);
                }
                for (int c = 0; c < num_vector_components; ++c) {
                  curl_vec[c] = data.curl_b(iz, iy, ix, c);
                }
                const auto primitive =
                    engine::primitive(vc, work_field, curl_vec, cur, fluid_parameters);
                for (int c = 0; c < num_fluid_components; ++c) {
                  data.work_fluid(iz, iy, ix, c) = primitive[c];
                }
              }
            }
          }
        }

        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          chunk.boundary_pack(data.work_fluid, BoundaryCopy10);
          chunk.boundary_pack(data.work_field_cell, BoundaryCopy6);
          chunk.boundary_begin(data.work_fluid, BoundaryCopy10);
          chunk.boundary_begin(data.work_field_cell, BoundaryCopy6);
        }
        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          chunk.boundary_end(data.work_fluid, BoundaryCopy10);
          chunk.boundary_end(data.work_field_cell, BoundaryCopy6);
          chunk.boundary_unpack(data.work_fluid, BoundaryCopy10);
          chunk.boundary_unpack(data.work_field_cell, BoundaryCopy6);
        }
      }

      if (engine::pcc2_is_ohm_stage(stage)) {
        // Ohm solve: accumulate moments, construct source, SSOR2, copy E back
        ++ohm_stage_index;
        bool ssor_converged        = false;
        int  ssor_total_iterations = 0;

        for (auto& chunk_ptr : chunkvec) {
          auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
          data.ohm_moment.fill(0);
          std::vector<nix::float64> kinetic_charge_to_mass;
          kinetic_charge_to_mass.reserve(data.particles.size());
          for (const auto& particle : data.particles) {
            kinetic_charge_to_mass.push_back(particle->q / particle->m);
          }
          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
                engine::FluidState wf = {};
                for (int c = 0; c < num_fluid_components; ++c) {
                  wf[c] = data.work_fluid(iz, iy, ix, c);
                }
                engine::accumulate_fluid_moment(wf, phase_params.ion_charge_to_mass,
                                                phase_params.electron_charge_to_mass,
                                                data.ohm_moment, iz, iy, ix);
                engine::accumulate_kinetic_moments(data.moment_kinetic, data.ohm_moment, iz, iy, ix,
                                                   kinetic_charge_to_mass);
              }
            }
          }
        }

        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          chunk.boundary_pack(data.ohm_moment, BoundaryCopy10);
          chunk.boundary_begin(data.ohm_moment, BoundaryCopy10);
        }
        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          chunk.boundary_end(data.ohm_moment, BoundaryCopy10);
          chunk.boundary_unpack(data.ohm_moment, BoundaryCopy10);
        }

        for (auto& chunk_ptr : chunkvec) {
          auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
                std::array<nix::float64, num_moment_components> mom   = {};
                engine::FieldState                              field = {};
                engine::VectorState                             bg    = {};
                std::array<nix::float64, num_moment_components> pxp = {}, pxm = {}, pyp = {},
                                                                pym = {}, pzp = {}, pzm = {};
                for (int c = 0; c < num_moment_components; ++c) {
                  mom[c] = data.ohm_moment(iz, iy, ix, c);
                  pxp[c] = data.ohm_moment(iz, iy, ix + 1, c);
                  pxm[c] = data.ohm_moment(iz, iy, ix - 1, c);
                  pyp[c] = data.ohm_moment(iz, iy + 1, ix, c);
                  pym[c] = data.ohm_moment(iz, iy - 1, ix, c);
                  pzp[c] = data.ohm_moment(iz + 1, iy, ix, c);
                  pzm[c] = data.ohm_moment(iz - 1, iy, ix, c);
                }
                for (int c = 0; c < num_field_components; ++c) {
                  field[c] = data.work_field_cell(iz, iy, ix, c);
                }
                for (int c = 0; c < num_vector_components; ++c) {
                  bg[c] = data.background_cell(iz, iy, ix, c);
                }
                const auto src = engine::construct_ohm_source(
                    mom, field, bg, light_speed, phase_params.spacing_x, phase_params.spacing_y,
                    phase_params.spacing_z, pxp, pxm, pyp, pym, pzp, pzm);
                for (int c = 0; c < num_ohm_source_components; ++c) {
                  data.ohm_source(iz, iy, ix, c) = src[c];
                }
              }
            }
          }
        }

        const auto exchange_electric = [&]() {
          for (auto& chunk_ptr : chunkvec) {
            auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
            auto  data  = chunk.get_internal_data();
            chunk.boundary_pack(data.work_field_cell, BoundaryCopy6);
            chunk.boundary_begin(data.work_field_cell, BoundaryCopy6);
          }
          for (auto& chunk_ptr : chunkvec) {
            auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
            auto  data  = chunk.get_internal_data();
            chunk.boundary_end(data.work_field_cell, BoundaryCopy6);
            chunk.boundary_unpack(data.work_field_cell, BoundaryCopy6);
          }
        };
        engine::OhmSolveContext solve_context = {
            [&](const engine::OhmSystemOperation& operation) {
              for (auto& chunk_ptr : chunkvec) {
                auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
                engine::OhmSystemView system = {data.work_field_cell,
                                                data.ohm_source,
                                                data.Lbx,
                                                data.Ubx,
                                                data.Lby,
                                                data.Uby,
                                                data.Lbz,
                                                data.Ubz};
                operation(system);
              }
            },
            exchange_electric,
            [&](nix::float64 error_sum, nix::float64 norm_sum) {
              const nix::float64 local_residual[2]  = {error_sum, norm_sum};
              nix::float64       global_residual[2] = {};
              MPI_Allreduce(local_residual, global_residual, 2, MPI_DOUBLE, MPI_SUM,
                            MPI_COMM_WORLD);
              return std::pair{global_residual[0], global_residual[1]};
            },
            [&](int iteration, const engine::OhmSolveStats& stats) {
              ssor_log << "iter=" << iteration << ", error=" << stats.relative_residual << "\n";
            },
        };
        engine::LegacySsor2 solver(ssor2_coeff, ssor2_config);
        const auto          solve_stats = solver.solve(solve_context);
        ssor_total_iterations           = solve_stats.iterations;
        ssor_converged                  = solve_stats.converged;
        ssor_log << "# Ohm stage " << ohm_stage_index << ": iterations=" << ssor_total_iterations
                 << " converged=" << (ssor_converged ? "true" : "false") << "\n";
        if (!ssor_converged) {
          throw std::runtime_error("Hybrid SSOR2 failed to converge at Ohm stage " +
                                   std::to_string(ohm_stage_index));
        }

        // Final E-field halo exchange after Ohm solve (width 2 per legacy)
        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          chunk.boundary_pack(data.work_field_cell, BoundaryCopy6);
          chunk.boundary_begin(data.work_field_cell, BoundaryCopy6);
        }
        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          chunk.boundary_end(data.work_field_cell, BoundaryCopy6);
          chunk.boundary_unpack(data.work_field_cell, BoundaryCopy6);
        }

        for (int pass = 0; pass < 2; ++pass) {
          for (auto& chunk_ptr : chunkvec) {
            auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
            engine::filter_electric_once(data);
          }
          for (auto& chunk_ptr : chunkvec) {
            auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
            auto  data  = chunk.get_internal_data();
            chunk.boundary_pack(data.work_field_cell, BoundaryCopy6);
            chunk.boundary_begin(data.work_field_cell, BoundaryCopy6);
          }
          for (auto& chunk_ptr : chunkvec) {
            auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
            auto  data  = chunk.get_internal_data();
            chunk.boundary_end(data.work_field_cell, BoundaryCopy6);
            chunk.boundary_unpack(data.work_field_cell, BoundaryCopy6);
          }
        }
      }

      if (engine::pcc2_is_average_stage(stage)) {
        // 50/50 average working ← accepted
        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
                for (int c = 0; c < num_fluid_components; ++c) {
                  data.work_fluid(iz, iy, ix, c) =
                      0.5 * (data.work_fluid(iz, iy, ix, c) + data.fluid(iz, iy, ix, c));
                }
                for (int c = 0; c < num_field_components; ++c) {
                  data.work_field_cell(iz, iy, ix, c) =
                      0.5 * (data.work_field_cell(iz, iy, ix, c) + data.field_cell(iz, iy, ix, c));
                }
              }
            }
          }
        }
      }

      if (engine::pcc2_is_particle_stage(stage)) {
        const bool should_rollback = engine::pcc2_should_rollback_particles(stage);

        // Push particles using averaged work_field_cell.
        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();
          engine::push_particles(data, data.work_field_cell, dt);
        }

        // Deposit before sorting so xv remains the accepted particle snapshot.
        update_kinetic_moments();

        // Rollback on first corrector
        if (should_rollback) {
          for (auto& chunk_ptr : chunkvec) {
            auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
            engine::rollback_particles(data);
          }
        } else {
          // Final particle migration after second corrector
          migrate_particles();
        }
      }
    }

    // --- Commit ---
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      for (int iz = 0; iz < static_cast<int>(data.fluid.shape()[0]); ++iz) {
        for (int iy = 0; iy < static_cast<int>(data.fluid.shape()[1]); ++iy) {
          for (int ix = 0; ix < static_cast<int>(data.fluid.shape()[2]); ++ix) {
            for (int c = 0; c < num_fluid_components; ++c) {
              data.fluid(iz, iy, ix, c) = data.work_fluid(iz, iy, ix, c);
            }
            for (int c = 0; c < num_field_components; ++c) {
              data.field_cell(iz, iy, ix, c)      = data.work_field_cell(iz, iy, ix, c);
              data.field_staggered(iz, iy, ix, c) = data.work_field_staggered(iz, iy, ix, c);
            }
          }
        }
      }
    }

    // push() runs before Application advances its step and time counters.
    write_snapshot(curstep + 1, curtime + dt);
    if (thisrank == 0) {
      const auto ssor_dir = diagnostic_root / "ssor";
      std::filesystem::create_directories(ssor_dir);
      auto slog = diag::open_output(ssor_dir / ("step_" + std::to_string(curstep + 1) + ".log"));
      slog << ssor_log.str();
    }
  } catch (const std::exception& e) {
    std::cerr << "Hybrid push() failed: " << e.what() << std::endl;
    throw;
  }
}
} // namespace hybrid
