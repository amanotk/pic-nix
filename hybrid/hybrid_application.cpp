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

#include <sstream>
#include <type_traits>
#include <utility>

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

    // Initial state diagnostics (step 0)
    {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      for (size_t ic = 0; ic < chunkvec.size(); ++ic) {
        auto& chunk = static_cast<HybridChunk&>(*chunkvec[ic]);
        auto  data  = chunk.get_internal_data();
        if (curstep == 0) {
          diag::write_diagnostics(data, "diagnostics/initial", rank, static_cast<int>(ic));
        }
      }
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

    bool has_initial_particles = false;
    for (const auto& chunk_ptr : chunkvec) {
      auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
      for (const auto& particle : data.particles) {
        if (particle->Np > 0) {
          has_initial_particles = true;
          break;
        }
      }
      if (has_initial_particles) {
        break;
      }
    }
    if (has_initial_particles) {
      update_kinetic_moments();
    }

    // --- Phase speed evaluation ---
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
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

    // Interpolate phase_cell → phase_face (max of two adjacent cells per direction)
    for (auto& chunk_ptr : chunkvec) {
      auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
      auto  data  = chunk.get_internal_data();
      // x-direction faces
      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
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
      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby - 1; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
            data.phase_face(iz, iy, ix, 1, 0) =
                std::max(data.phase_cell(iz, iy, ix, 1, 0), data.phase_cell(iz, iy + 1, ix, 1, 0));
            data.phase_face(iz, iy, ix, 1, 1) =
                std::max(data.phase_cell(iz, iy, ix, 1, 1), data.phase_cell(iz, iy + 1, ix, 1, 1));
          }
        }
      }
      // z-direction faces
      for (int iz = data.Lbz - 1; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
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
        // --- Field update (push_field) ---
        for (auto& chunk_ptr : chunkvec) {
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();

          // MC2 flux for all 3 directions
          for (int dir = 0; dir < num_phase_directions; ++dir) {
            const int Lb1 = (dir == 0) ? data.Lbx - 1 : (dir == 1) ? data.Lby - 1 : data.Lbz - 1;
            const int Ub1 = (dir == 0) ? data.Ubx + 1 : (dir == 1) ? data.Uby + 1 : data.Ubz + 1;
            const int Lb2 = (dir == 0) ? data.Lbx : (dir == 1) ? data.Lby : data.Lbz;
            const int Ub2 = (dir == 0) ? data.Ubx : (dir == 1) ? data.Uby : data.Ubz;

            for (int iz = (dir == 2) ? Lb1 : data.Lbz; iz <= ((dir == 2) ? Ub1 : data.Ubz); ++iz) {
              for (int iy = (dir == 1) ? Lb1 : data.Lby; iy <= ((dir == 1) ? Ub1 : data.Uby);
                   ++iy) {
                for (int ix = (dir == 0) ? Lb1 : data.Lbx; ix <= ((dir == 0) ? Ub1 : data.Ubx);
                     ++ix) {
                  // Read fluid/field at left and right cells
                  const int liz = (dir == 2) ? iz : iz;
                  const int riz = (dir == 2) ? iz + 1 : iz;
                  const int liy = (dir == 1) ? iy : iy;
                  const int riy = (dir == 1) ? iy + 1 : iy;
                  const int lix = (dir == 0) ? ix : ix;
                  const int rix = (dir == 0) ? ix + 1 : ix;

                  const nix::float64 phase_max = data.phase_face(iz, iy, ix, dir, 0);
                  const nix::float64 phase_min = data.phase_face(iz, iy, ix, dir, 1);

                  engine::FluidState  left_f = {}, right_f = {};
                  engine::FieldState  left_eb = {}, right_eb = {};
                  engine::VectorState bg = {};
                  for (int c = 0; c < num_fluid_components; ++c) {
                    left_f[c]  = data.work_fluid(liz, liy, lix, c);
                    right_f[c] = data.work_fluid(riz, riy, rix, c);
                  }
                  for (int c = 0; c < num_field_components; ++c) {
                    left_eb[c]  = data.work_field_cell(liz, liy, lix, c);
                    right_eb[c] = data.work_field_cell(riz, riy, rix, c);
                  }
                  for (int c = 0; c < num_vector_components; ++c) {
                    bg[c] = data.background_cell(liz, liy, lix, c);
                  }

                  // MC2 reconstruct each component (fluid + relevant field)
                  // Per legacy, x-dir fluids: uc-1,uc,uc+1 → fl,fr
                  // For now, simple interpolation (no limiter) since cells are uniform
                  engine::FluidState fluid_left_rec  = left_f;
                  engine::FluidState fluid_right_rec = right_f;
                  engine::FieldState field_left_rec  = left_eb;
                  engine::FieldState field_right_rec = right_eb;

                  // HLL flux
                  const auto flux = engine::hll_fluid_flux(
                      dir, fluid_left_rec, field_left_rec, fluid_right_rec, field_right_rec, bg,
                      phase_max, phase_min, dt, fluid_parameters);

                  for (int c = 0; c < num_conserved_components; ++c) {
                    data.fluid_flux(iz, iy, ix, dir, c) = flux.flux[c];
                  }
                  for (int c = 0; c < num_field_components; ++c) {
                    data.solver_field_x(iz, iy, ix, c) = flux.field[c];
                  }
                }
              }
            }
          }

          // CT magnetic update and cell-B reconstruction
          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
                // Copy edge E from flux
                for (int c = 0; c < 3; ++c) {
                  data.work_field_staggered(iz, iy, ix, c) = data.solver_field_x(iz, iy, ix, c);
                }

                // Constrained transport for B
                engine::FieldState base_stag = {};
                engine::FieldState edge_e    = {};
                engine::FieldState x_e       = {};
                engine::FieldState y_e       = {};
                engine::FieldState z_e       = {};
                for (int c = 0; c < num_field_components; ++c) {
                  base_stag[c] = data.field_staggered(iz, iy, ix, c);
                  edge_e[c]    = data.solver_field_x(iz, iy, ix, c);
                  x_e[c]       = data.solver_field_x(iz, iy, ix - 1, c);
                  y_e[c]       = data.solver_field_x(iz, iy - 1, ix, c);
                  z_e[c]       = data.solver_field_x(iz - 1, iy, ix, c);
                }
                const auto new_b = engine::constrained_transport_magnetic(
                    base_stag, edge_e, x_e, y_e, z_e, grid_spacing, light_speed, dt);
                data.work_field_staggered(iz, iy, ix, field_component::magnetic_x) = new_b[0];
                data.work_field_staggered(iz, iy, ix, field_component::magnetic_y) = new_b[1];
                data.work_field_staggered(iz, iy, ix, field_component::magnetic_z) = new_b[2];

                // Cell-centered B from face values
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

                // Cell-centered E: keep previous SSOR solution as initial guess (do not overwrite)
              }
            }
          }

          // Curl_B
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

          // Conservative update + primitive recovery
          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
                engine::FluidState   local_fluid = {};
                engine::FieldState   local_field = {};
                engine::VectorState  bg          = {};
                engine::CurrentState cur         = {};
                for (int c = 0; c < num_fluid_components; ++c) {
                  local_fluid[c] = data.fluid(iz, iy, ix, c);
                }
                for (int c = 0; c < num_field_components; ++c) {
                  local_field[c] = data.field_cell(iz, iy, ix, c);
                }
                for (int c = 0; c < num_vector_components; ++c) {
                  bg[c] = data.background_cell(iz, iy, ix, c);
                }
                for (int c = 0; c < num_current_components; ++c) {
                  cur[c] = data.current_kinetic(iz, iy, ix, c);
                }

                const auto uc = engine::conservative(local_fluid, local_field, fluid_parameters);
                const auto rh = engine::fluid_rhs(dt, local_field, cur, bg, fluid_parameters);

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

                // Recover primitive
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

        // BC halo exchanges for work_fluid and work_field_cell
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
          auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
          auto  data  = chunk.get_internal_data();

          // Zero ohm_moment
          for (int iz = 0; iz < static_cast<int>(data.ohm_moment.shape()[0]); ++iz) {
            for (int iy = 0; iy < static_cast<int>(data.ohm_moment.shape()[1]); ++iy) {
              for (int ix = 0; ix < static_cast<int>(data.ohm_moment.shape()[2]); ++ix) {
                for (int c = 0; c < num_moment_components; ++c) {
                  data.ohm_moment(iz, iy, ix, c) = 0;
                }
              }
            }
          }

          // Accumulate fluid moments
          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
                engine::FluidState wf = {};
                for (int c = 0; c < num_fluid_components; ++c) {
                  wf[c] = data.work_fluid(iz, iy, ix, c);
                }
                engine::accumulate_fluid_moment(wf, data.ohm_moment, iz, iy, ix);
              }
            }
          }

          // Add kinetic moments
          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
                engine::accumulate_kinetic_moments(data.moment_kinetic, data.ohm_moment, iz, iy, ix,
                                                   data.num_species);
              }
            }
          }

          // Construct Ohm source (4 components: coeff, src_x, src_y, src_z)
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

          // SSOR2 solve for E-field with sweep-level E-field exchanges
          for (int ssor_iter = 1; ssor_iter <= ssor2_config.max_iterations; ++ssor_iter) {
            // Forward sweep
            for (auto& chunk_ptr : chunkvec) {
              auto&                  chunk = static_cast<HybridChunk&>(*chunk_ptr);
              auto                   data  = chunk.get_internal_data();
              engine::Ssor2Workspace ws    = {data.work_field_cell,
                                              data.ohm_source,
                                              data.resistive_field,
                                              data.Lbx,
                                              data.Ubx,
                                              data.Lby,
                                              data.Uby,
                                              data.Lbz,
                                              data.Ubz};
              engine::ssor2_forward_sweep(ws, ssor2_coeff);
            }
            // E-field exchange after forward sweep (width 1)
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

            // Backward sweep
            for (auto& chunk_ptr : chunkvec) {
              auto&                  chunk = static_cast<HybridChunk&>(*chunk_ptr);
              auto                   data  = chunk.get_internal_data();
              engine::Ssor2Workspace ws    = {data.work_field_cell,
                                              data.ohm_source,
                                              data.resistive_field,
                                              data.Lbx,
                                              data.Ubx,
                                              data.Lby,
                                              data.Uby,
                                              data.Lbz,
                                              data.Ubz};
              engine::ssor2_backward_sweep(ws, ssor2_coeff);
            }
            // E-field exchange after backward sweep (width 1)
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

            // Local residual
            nix::float64 error_sum = 0;
            nix::float64 norm_sum  = 0;
            for (auto& chunk_ptr : chunkvec) {
              auto&                  chunk = static_cast<HybridChunk&>(*chunk_ptr);
              auto                   data  = chunk.get_internal_data();
              engine::Ssor2Workspace ws    = {data.work_field_cell,
                                              data.ohm_source,
                                              data.resistive_field,
                                              data.Lbx,
                                              data.Ubx,
                                              data.Lby,
                                              data.Uby,
                                              data.Lbz,
                                              data.Ubz};
              auto [le, ln]                = engine::ssor2_local_residual(ws, ssor2_coeff);
              error_sum += le;
              norm_sum += ln;
            }

            const nix::float64 relative_error =
                std::sqrt(error_sum) / (std::sqrt(norm_sum) + 1.0e-32);
            ssor_log << "iter=" << ssor_iter << ", error=" << relative_error << "\n";
            if (relative_error < ssor2_config.tolerance) {
              ssor_converged = true;
            }
            if (relative_error < ssor2_config.tolerance ||
                ssor_iter >= ssor2_config.max_iterations) {
              ssor_total_iterations = ssor_iter;
              break;
            }
          }
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

        ssor_log << "# Ohm stage " << ohm_stage_index << ": iterations=" << ssor_total_iterations
                 << " converged=" << (ssor_converged ? "true" : "false") << "\n";
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

        // Skip particle stages when there are no active particles
        bool has_particles = false;
        for (const auto& chunk_ptr : chunkvec) {
          auto data = static_cast<HybridChunk&>(*chunk_ptr).get_internal_data();
          for (const auto& p : data.particles) {
            if (p->Np > 0) {
              has_particles = true;
              break;
            }
          }
          if (has_particles)
            break;
        }

        if (has_particles) {
          // Push particles using averaged work_field_cell
          for (auto& chunk_ptr : chunkvec) {
            auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
            auto  data  = chunk.get_internal_data();
            engine::push_particles(data, data.work_field_cell, dt);
            // Count and sort after push for moment deposition
            for (auto& particle : data.particles) {
              particle->count(0, particle->Np - 1, true, data.order);
              particle->sort();
            }
          }

          // Accumulate moments from pushed particles
          update_kinetic_moments();
        }

        // Rollback on first corrector
        if (should_rollback && has_particles) {
          for (auto& chunk_ptr : chunkvec) {
            auto& chunk = static_cast<HybridChunk&>(*chunk_ptr);
            auto  data  = chunk.get_internal_data();
            for (auto& particle : data.particles) {
              particle->swap();
              // Re-count and sort after rollback to restore original ordering
              particle->count(0, particle->Np - 1, true, data.order);
              particle->sort();
            }
          }
        }

        // Final particle migration after second corrector
        if (!should_rollback && has_particles) {
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

    // Final state diagnostics after commit
    {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0) {
        std::system("mkdir -p diagnostics");
        std::ofstream slog("diagnostics/debug.log");
        slog << ssor_log.str();
      }
      for (size_t ic = 0; ic < chunkvec.size(); ++ic) {
        auto& chunk = static_cast<HybridChunk&>(*chunkvec[ic]);
        auto  data  = chunk.get_internal_data();
        diag::write_diagnostics(data, "diagnostics/final", rank, static_cast<int>(ic));
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Hybrid push() failed: " << e.what() << std::endl;
    throw;
  }
}
} // namespace hybrid
