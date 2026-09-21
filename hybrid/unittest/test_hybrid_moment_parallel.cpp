// -*- C++ -*-
#include "engine/filter.hpp"
#include "engine/moment.hpp"
#include "hybrid_application.hpp"
#include "test_hybrid_context.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

namespace
{
struct Communicators {
  std::vector<MPI_Comm> values;

  ~Communicators()
  {
    for (auto& value : values) {
      MPI_Comm_free(&value);
    }
  }
};

void bind_communicators(hybrid::HybridChunk& chunk, MPI_Comm parent, Communicators& communicators)
{
  communicators.values.resize(static_cast<size_t>(hybrid::NumBoundaryModes) * 27);
  for (int mode = 0; mode < hybrid::NumBoundaryModes; ++mode) {
    for (int iz = 0; iz < 3; ++iz) {
      for (int iy = 0; iy < 3; ++iy) {
        for (int ix = 0; ix < 3; ++ix) {
          const int index = mode * 27 + iz * 9 + iy * 3 + ix;
          MPI_Comm_dup(parent, &communicators.values[static_cast<size_t>(index)]);
          chunk.set_mpi_communicator(mode, iz, iy, ix,
                                     communicators.values[static_cast<size_t>(index)]);
        }
      }
    }
  }
}

void initialize_particles(hybrid::HybridChunk& chunk)
{
  auto       data              = chunk.get_internal_data();
  const auto dims              = chunk.get_dims();
  const auto offset            = chunk.get_offset();
  auto [xmin, xmax]            = chunk.get_xrange();
  auto [ymin, ymax]            = chunk.get_yrange();
  auto [zmin, zmax]            = chunk.get_zrange();
  const int        count       = dims[0] * dims[1] * dims[2];
  constexpr double fraction[2] = {0.35, 0.65};
  constexpr double mass[2]     = {1.5, 0.75};
  constexpr double charge[2]   = {2.0, -1.0};

  for (int species = 0; species < data.num_species; ++species) {
    auto& particle = *data.particles[species];
    particle.resize(count);
    particle.Np = count;
    particle.m  = mass[species];
    particle.q  = charge[species];
    int ip      = 0;
    for (int iz = 0; iz < dims[0]; ++iz) {
      for (int iy = 0; iy < dims[1]; ++iy) {
        for (int ix = 0; ix < dims[2]; ++ix, ++ip) {
          const int gz           = offset[0] + iz;
          const int gy           = offset[1] + iy;
          const int gx           = offset[2] + ix;
          const int global_index = (gz * 37 + gy) * 41 + gx;
          particle.xu(ip, 0)     = particle.has_xdim ? xmin + ix + fraction[species] : 0;
          particle.xu(ip, 1)     = particle.has_ydim ? ymin + iy + fraction[species] : 0;
          particle.xu(ip, 2)     = particle.has_zdim ? zmin + iz + fraction[species] : 0;
          particle.xu(ip, 3)     = 0.17 + 0.013 * global_index + 0.11 * species;
          particle.xu(ip, 4)     = -0.23 + 0.007 * global_index - 0.05 * species;
          particle.xu(ip, 5)     = 0.31 - 0.009 * global_index + 0.03 * species;
        }
      }
    }
    particle.count(0, particle.Np - 1, true, hybrid::particle_order);
    particle.sort();
  }
}

void run_pipeline(hybrid::HybridChunk& chunk)
{
  auto data = chunk.get_internal_data();
  hybrid::engine::deposit_moments(data);
  chunk.boundary_pack(data.moment_kinetic, hybrid::BoundaryMomentAccum);
  chunk.boundary_begin(data.moment_kinetic, hybrid::BoundaryMomentAccum);
  chunk.boundary_end(data.moment_kinetic, hybrid::BoundaryMomentAccum);
  chunk.boundary_unpack(data.moment_kinetic, hybrid::BoundaryMomentAccum);
  chunk.boundary_pack(data.moment_kinetic, hybrid::BoundaryMomentCopy);
  chunk.boundary_begin(data.moment_kinetic, hybrid::BoundaryMomentCopy);
  chunk.boundary_end(data.moment_kinetic, hybrid::BoundaryMomentCopy);
  chunk.boundary_unpack(data.moment_kinetic, hybrid::BoundaryMomentCopy);
  for (int pass = 0; pass < 2; ++pass) {
    hybrid::engine::filter_moments_once(data);
    chunk.boundary_pack(data.moment_kinetic, hybrid::BoundaryMomentCopy);
    chunk.boundary_begin(data.moment_kinetic, hybrid::BoundaryMomentCopy);
    chunk.boundary_end(data.moment_kinetic, hybrid::BoundaryMomentCopy);
    chunk.boundary_unpack(data.moment_kinetic, hybrid::BoundaryMomentCopy);
  }
  hybrid::engine::derive_current(data);
}

std::unique_ptr<hybrid::HybridChunk> make_reference(const HybridTestGrid& grid,
                                                    Communicators&        communicators)
{
  auto      reference = std::make_unique<hybrid::HybridChunk>(grid.gdims, grid.has_dim, 0);
  const int offset[3] = {0, 0, 0};
  reference->set_global_context(offset, grid.gdims.data());
  auto config = make_hybrid_test_config();
  reference->setup(config);
  for (int dirz = -1; dirz <= 1; ++dirz) {
    for (int diry = -1; diry <= 1; ++diry) {
      for (int dirx = -1; dirx <= 1; ++dirx) {
        reference->set_nb_id(dirz, diry, dirx, 0);
        reference->set_nb_rank(dirz, diry, dirx, 0);
      }
    }
  }
  bind_communicators(*reference, MPI_COMM_SELF, communicators);
  initialize_particles(*reference);
  run_pipeline(*reference);
  return reference;
}

void compare_with_reference(const hybrid::HybridChunk& chunk, hybrid::HybridChunk& reference)
{
  auto       local    = const_cast<hybrid::HybridChunk&>(chunk).get_internal_data();
  auto       expected = reference.get_internal_data();
  const auto dims     = chunk.get_dims();
  const auto offset   = chunk.get_offset();
  for (int iz = 0; iz < dims[0]; ++iz) {
    for (int iy = 0; iy < dims[1]; ++iy) {
      for (int ix = 0; ix < dims[2]; ++ix) {
        for (int species = 0; species < local.num_species; ++species) {
          for (int component = 0; component < hybrid::num_moment_components; ++component) {
            INFO("chunk=" << chunk.get_id() << " cell=" << iz << "," << iy << "," << ix
                          << " species=" << species << " component=" << component);
            REQUIRE(local.moment_kinetic(local.Lbz + iz, local.Lby + iy, local.Lbx + ix, species,
                                         component) ==
                    Catch::Approx(expected.moment_kinetic(
                                      expected.Lbz + offset[0] + iz, expected.Lby + offset[1] + iy,
                                      expected.Lbx + offset[2] + ix, species, component))
                        .epsilon(5.0e-13)
                        .margin(5.0e-14));
          }
        }
        for (int component = 0; component < hybrid::num_current_components; ++component) {
          REQUIRE(
              local.current_kinetic(local.Lbz + iz, local.Lby + iy, local.Lbx + ix, component) ==
              Catch::Approx(expected.current_kinetic(expected.Lbz + offset[0] + iz,
                                                     expected.Lby + offset[1] + iy,
                                                     expected.Lbx + offset[2] + ix, component))
                  .epsilon(5.0e-13)
                  .margin(5.0e-14));
        }
      }
    }
  }
}

void run_moment_pipeline_test(const HybridTestGrid& grid)
{
  auto context = build_hybrid_exchange_context(grid);
  initialize_particles(*context.chunk);
  run_pipeline(*context.chunk);

  Communicators reference_communicators;
  auto          reference = make_reference(grid, reference_communicators);
  compare_with_reference(*context.chunk, *reference);
}

class TestMomentApplication : public hybrid::HybridApplication
{
public:
  TestMomentApplication() : HybridApplication(0, nullptr, nullptr)
  {
  }

  void add_chunk(std::unique_ptr<hybrid::HybridChunk> chunk)
  {
    chunkvec.push_back(std::move(chunk));
  }

  void set_neighbors(std::unique_ptr<nix::ChunkMap>& chunkmap)
  {
    chunkvec.set_neighbors(chunkmap);
  }

  void bind(Communicators& communicators)
  {
    for (auto& chunk_ptr : chunkvec) {
      for (int mode = 0; mode < hybrid::NumBoundaryModes; ++mode) {
        for (int iz = 0; iz < 3; ++iz) {
          for (int iy = 0; iy < 3; ++iy) {
            for (int ix = 0; ix < 3; ++ix) {
              const int index = mode * 27 + iz * 9 + iy * 3 + ix;
              chunk_ptr->set_mpi_communicator(mode, iz, iy, ix,
                                              communicators.values[static_cast<size_t>(index)]);
            }
          }
        }
      }
    }
  }

  void update()
  {
    update_kinetic_moments();
  }

  const ChunkVec& chunks() const
  {
    return chunkvec;
  }
};

void run_multichunk_application_test()
{
  const int            rank       = get_mpi_rank();
  const HybridTestGrid grid       = {{1, 1, 12}, {false, false, true}, {1, 1, 4}, 2};
  const nix::Dims3D    local_dims = {1, 1, 3};
  auto                 chunkmap   = std::make_unique<nix::ChunkMap>(1, 1, 4);
  std::vector<int>     boundary   = {0, 2, 4};
  chunkmap->set_rank_boundary(boundary);

  TestMomentApplication application;
  for (int id = boundary[rank]; id < boundary[rank + 1]; ++id) {
    auto      chunk     = std::make_unique<hybrid::HybridChunk>(local_dims, grid.has_dim, id);
    const int offset[3] = {0, 0, id * local_dims[2]};
    chunk->set_global_context(offset, grid.gdims.data());
    auto config = make_hybrid_test_config();
    chunk->setup(config);
    initialize_particles(*chunk);
    application.add_chunk(std::move(chunk));
  }
  application.set_neighbors(chunkmap);

  Communicators communicators;
  communicators.values.resize(static_cast<size_t>(hybrid::NumBoundaryModes) * 27);
  for (auto& communicator : communicators.values) {
    MPI_Comm_dup(MPI_COMM_WORLD, &communicator);
  }
  application.bind(communicators);
  application.update();

  Communicators reference_communicators;
  auto          reference = make_reference(grid, reference_communicators);
  for (const auto& chunk : application.chunks()) {
    compare_with_reference(static_cast<const hybrid::HybridChunk&>(*chunk), *reference);
  }
}
} // namespace

TEST_CASE("Hybrid moment pipeline matches one-chunk reference in 1D")
{
  const HybridTestGrid grid = {{1, 1, 12}, {false, false, true}, {1, 1, 2}, 2};
  if (require_mpi_size(grid.mpi_size)) {
    run_moment_pipeline_test(grid);
  }
}

TEST_CASE("Hybrid moment pipeline matches one-chunk reference in 2D")
{
  const HybridTestGrid grid = {{1, 12, 12}, {false, true, true}, {1, 2, 2}, 4};
  if (require_mpi_size(grid.mpi_size)) {
    run_moment_pipeline_test(grid);
  }
}

TEST_CASE("Hybrid moment pipeline matches one-chunk reference in 3D")
{
  const HybridTestGrid grid = {{12, 12, 12}, {true, true, true}, {2, 2, 2}, 8};
  if (require_mpi_size(grid.mpi_size)) {
    run_moment_pipeline_test(grid);
  }
}

TEST_CASE("HybridApplication moment schedule supports multiple chunks per rank")
{
  if (require_mpi_size(2)) {
    run_multichunk_application_test();
  }
}
