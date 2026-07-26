// -*- C++ -*-
#include "hybrid_application.hpp"

#include "hybrid_chunk.hpp"

#include "engine/filter.hpp"
#include "engine/moment.hpp"

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
  assert_mpi(false, "Hybrid physics is not implemented yet");
}
} // namespace hybrid
