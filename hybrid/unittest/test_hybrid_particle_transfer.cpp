// -*- C++ -*-
#include "test_hybrid_context.hpp"

#include "hybrid_application.hpp"

#include "nix/chunkmap.hpp"
#include "nix/chunkvector.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>

namespace
{
void store_id(nix::float64& slot, int64_t id)
{
  std::memcpy(&slot, &id, sizeof(id));
}

int64_t load_id(const nix::float64& slot)
{
  int64_t id = 0;
  std::memcpy(&id, &slot, sizeof(id));
  return id;
}

nix::float64 outgoing_position(int direction, nix::float64 lower, nix::float64 upper)
{
  if (direction < 0) {
    return lower - 0.1;
  }
  if (direction > 0) {
    return upper + 0.1;
  }
  return 0.5 * (lower + upper);
}

nix::float64 accepted_position(int direction, nix::float64 lower, nix::float64 upper)
{
  if (direction < 0) {
    return lower + 0.1;
  }
  if (direction > 0) {
    return upper - 0.1;
  }
  return 0.5 * (lower + upper);
}

void run_particle_test(const HybridTestGrid& grid, const std::array<int, 3>& direction)
{
  auto          context      = build_hybrid_exchange_context(grid);
  auto          data         = context.chunk->get_internal_data();
  const int     rank         = get_mpi_rank();
  constexpr int num_outgoing = 140;

  auto [xmin, xmax]                       = context.chunk->get_xrange();
  auto [ymin, ymax]                       = context.chunk->get_yrange();
  auto [zmin, zmax]                       = context.chunk->get_zrange();
  const std::array<nix::float64, 3> lower = {xmin, ymin, zmin};
  const std::array<nix::float64, 3> upper = {xmax, ymax, zmax};

  for (int is = 0; is < data.num_species; ++is) {
    auto& particle = *data.particles[is];
    particle.resize(num_outgoing + 1);
    particle.Np = num_outgoing + 1;
    particle.xu.fill(0);
    particle.xv.fill(0);
    for (int component = 0; component < 3; ++component) {
      particle.xu(0, component) = 0.5 * (lower[component] + upper[component]);
      particle.xv(0, component) = particle.xu(0, component);
    }
    store_id(particle.xu(0, 6), is * 100000 + rank * 1000);
    for (int ip = 1; ip <= num_outgoing; ++ip) {
      for (int component = 0; component < 3; ++component) {
        const int dir                  = direction[2 - component];
        particle.xu(ip, component)     = outgoing_position(dir, lower[component], upper[component]);
        particle.xv(ip, component)     = accepted_position(dir, lower[component], upper[component]);
        particle.xu(ip, component + 3) = rank * 1000.0 + ip * 10.0 + component;
      }
      store_id(particle.xu(ip, 6), is * 100000 + rank * 1000 + ip);
    }
  }

  const auto displacement = context.chunk->get_max_particle_displacement();
  REQUIRE(displacement.ratio < 1.0);
  context.chunk->prepare_particle_migration();
  context.chunk->particle_boundary_pack();
  context.chunk->particle_boundary_begin();
  REQUIRE(context.chunk->particle_boundary_probe(true));
  context.chunk->particle_boundary_end();
  context.chunk->particle_boundary_unpack();
  REQUIRE(context.chunk->exchanges_idle());

  const int source_rank = context.chunk->get_nb_rank(-direction[0], -direction[1], -direction[2]);
  int       local_count = 0;
  for (int is = 0; is < data.num_species; ++is) {
    const auto& particle = *data.particles[is];
    REQUIRE(particle.Np == num_outgoing + 1);
    REQUIRE(particle.Np_total > nix::Particle::alloc_unit);
    REQUIRE(particle.pindex(particle.Ng) == particle.Np);
    local_count += particle.Np;

    std::map<int64_t, int> records;
    for (int ip = 0; ip < particle.Np; ++ip) {
      records.emplace(load_id(particle.xu(ip, 6)), ip);
      REQUIRE(particle.xu(ip, 0) >= xmin);
      REQUIRE(particle.xu(ip, 0) < xmax);
      REQUIRE(particle.xu(ip, 1) >= ymin);
      REQUIRE(particle.xu(ip, 1) < ymax);
      REQUIRE(particle.xu(ip, 2) >= zmin);
      REQUIRE(particle.xu(ip, 2) < zmax);
    }
    REQUIRE(records.count(is * 100000 + rank * 1000) == 1);
    const int retained = records.at(is * 100000 + rank * 1000);
    for (int component = 0; component < 3; ++component) {
      REQUIRE(particle.xu(retained, component) ==
              Catch::Approx(0.5 * (lower[component] + upper[component])).margin(1.0e-12));
    }
    for (int source_ip = 1; source_ip <= num_outgoing; ++source_ip) {
      const int64_t id = is * 100000 + source_rank * 1000 + source_ip;
      REQUIRE(records.count(id) == 1);
      const int ip = records.at(id);
      for (int component = 0; component < 3; ++component) {
        const int  dir               = direction[2 - component];
        const auto expected_position = dir < 0   ? upper[component] - 0.1
                                       : dir > 0 ? lower[component] + 0.1
                                                 : 0.5 * (lower[component] + upper[component]);
        REQUIRE(particle.xu(ip, component) == Catch::Approx(expected_position).margin(1.0e-12));
        REQUIRE(particle.xu(ip, component + 3) ==
                source_rank * 1000.0 + source_ip * 10.0 + component);
      }
    }
  }
  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(global_count == grid.mpi_size * data.num_species * (num_outgoing + 1));

  auto& particle    = *data.particles[0];
  particle.Np       = 1;
  particle.xv(0, 0) = xmin + 0.25 * (xmax - xmin);
  particle.xu(0, 0) = particle.xv(0, 0) + 1.01 * (xmax - xmin);
  REQUIRE(context.chunk->get_max_particle_displacement().ratio > 1.0);
}

void run_multichunk_particle_test()
{
  constexpr int     num_chunks      = 4;
  constexpr int     chunks_per_rank = 2;
  const int         rank            = get_mpi_rank();
  const nix::Dims3D local_dims      = {1, 1, 6};
  const nix::Dims3D global_dims     = {1, 1, num_chunks * local_dims[2]};
  const nix::Bool3D has_dim         = {false, false, true};

  auto             chunkmap = std::make_unique<nix::ChunkMap>(1, 1, num_chunks);
  std::vector<int> boundary = {0, chunks_per_rank, num_chunks};
  chunkmap->set_rank_boundary(boundary);
  nix::ChunkVector<std::shared_ptr<hybrid::HybridChunk>> chunks;
  for (int id = boundary[rank]; id < boundary[rank + 1]; ++id) {
    auto               chunk  = std::make_shared<hybrid::HybridChunk>(local_dims, has_dim, id);
    std::array<int, 3> offset = {0, 0, id * local_dims[2]};
    chunk->set_global_context(offset.data(), global_dims.data());
    auto config = make_hybrid_test_config();
    chunk->setup(config);

    std::vector<unsigned char> buffer(static_cast<size_t>(chunk->get_size_byte()));
    REQUIRE(chunk->pack(buffer.data(), 0) == static_cast<int>(buffer.size()));
    auto restored = std::make_shared<hybrid::HybridChunk>(local_dims, has_dim, id);
    REQUIRE(restored->unpack(buffer.data(), 0) == static_cast<int>(buffer.size()));
    chunks.push_back(restored);
  }
  chunks.set_neighbors(chunkmap);

  std::vector<MPI_Comm> communicators(static_cast<size_t>(hybrid::NumBoundaryModes) * 27);
  for (int mode = 0; mode < hybrid::NumBoundaryModes; ++mode) {
    for (int iz = 0; iz < 3; ++iz) {
      for (int iy = 0; iy < 3; ++iy) {
        for (int ix = 0; ix < 3; ++ix) {
          const int index = mode * 27 + iz * 9 + iy * 3 + ix;
          MPI_Comm_dup(MPI_COMM_WORLD, &communicators[static_cast<size_t>(index)]);
          for (auto& chunk : chunks) {
            chunk->set_mpi_communicator(mode, iz, iy, ix,
                                        communicators[static_cast<size_t>(index)]);
          }
        }
      }
    }
  }

  for (auto& chunk : chunks) {
    auto  data        = chunk->get_internal_data();
    auto& particle    = *data.particles[0];
    auto [xmin, xmax] = chunk->get_xrange();
    particle.Np       = 2;
    particle.xu.fill(0);
    particle.xv.fill(0);
    particle.xu(0, 0) = 0.5 * (xmin + xmax);
    particle.xv(0, 0) = particle.xu(0, 0);
    particle.xu(1, 0) = xmax + 0.1;
    particle.xv(1, 0) = xmax - 0.1;
    store_id(particle.xu(0, 6), chunk->get_id() * 100 + 1);
    store_id(particle.xu(1, 6), chunk->get_id() * 100 + 2);
    chunk->prepare_particle_migration();
  }
  for (auto& chunk : chunks) {
    chunk->particle_boundary_pack();
    chunk->particle_boundary_begin();
  }
  for (auto& chunk : chunks) {
    REQUIRE(chunk->particle_boundary_probe(true));
  }
  for (auto& chunk : chunks) {
    chunk->particle_boundary_end();
    chunk->particle_boundary_unpack();
  }

  int local_count = 0;
  for (const auto& chunk : chunks) {
    auto        data     = chunk->get_internal_data();
    const auto& particle = *data.particles[0];
    REQUIRE(particle.Np == 2);
    local_count += particle.Np;
    std::array<int64_t, 2> ids = {load_id(particle.xu(0, 6)), load_id(particle.xu(1, 6))};
    std::sort(ids.begin(), ids.end());
    std::array<int64_t, 2> expected = {chunk->get_id() * 100 + 1,
                                       chunk->get_nb_id(0, 0, -1) * 100 + 2};
    std::sort(expected.begin(), expected.end());
    REQUIRE(ids == expected);
    REQUIRE(chunk->exchanges_idle());
  }
  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  REQUIRE(global_count == 2 * num_chunks);

  for (auto& communicator : communicators) {
    MPI_Comm_free(&communicator);
  }
}

class TestHybridApplication : public hybrid::HybridApplication
{
public:
  TestHybridApplication() : HybridApplication(0, nullptr, nullptr)
  {
    thisrank = get_mpi_rank();
  }

  hybrid::HybridChunk& add_chunk(std::unique_ptr<hybrid::HybridChunk> chunk)
  {
    auto& result = *chunk;
    chunkvec.push_back(std::move(chunk));
    return result;
  }

  void migrate()
  {
    migrate_particles();
  }

  void require_particles() const
  {
    require_kinetic_particles();
  }
};

void run_application_migration_test()
{
  const int         rank        = get_mpi_rank();
  const nix::Dims3D local_dims  = {1, 1, 6};
  const nix::Dims3D global_dims = {1, 1, 12};
  const nix::Bool3D has_dim     = {false, false, true};
  auto              chunkmap    = std::make_unique<nix::ChunkMap>(1, 1, 2);
  std::vector<int>  boundary    = {0, 1, 2};
  chunkmap->set_rank_boundary(boundary);

  auto               chunk  = std::make_unique<hybrid::HybridChunk>(local_dims, has_dim, rank);
  std::array<int, 3> offset = {0, 0, rank * local_dims[2]};
  chunk->set_global_context(offset.data(), global_dims.data());
  auto config = make_hybrid_test_config();
  chunk->setup(config);
  nix::ChunkVector<std::unique_ptr<nix::Chunk>> base_chunks;
  base_chunks.push_back(std::move(chunk));
  base_chunks.set_neighbors(chunkmap);
  chunk = std::unique_ptr<hybrid::HybridChunk>(
      static_cast<hybrid::HybridChunk*>(base_chunks[0].release()));

  std::vector<MPI_Comm> communicators(static_cast<size_t>(hybrid::NumBoundaryModes) * 27);
  for (int mode = 0; mode < hybrid::NumBoundaryModes; ++mode) {
    for (int iz = 0; iz < 3; ++iz) {
      for (int iy = 0; iy < 3; ++iy) {
        for (int ix = 0; ix < 3; ++ix) {
          const int index = mode * 27 + iz * 9 + iy * 3 + ix;
          MPI_Comm_dup(MPI_COMM_WORLD, &communicators[static_cast<size_t>(index)]);
          chunk->set_mpi_communicator(mode, iz, iy, ix, communicators[static_cast<size_t>(index)]);
        }
      }
    }
  }

  TestHybridApplication application;
  auto&                 installed = application.add_chunk(std::move(chunk));
  auto                  data      = installed.get_internal_data();
  auto&                 particle  = *data.particles[0];
  auto [xmin, xmax]               = installed.get_xrange();
  REQUIRE_THROWS_AS(application.require_particles(), std::invalid_argument);
  if (rank == 0) {
    particle.Np = 1;
  }
  REQUIRE_NOTHROW(application.require_particles());

  particle.Np       = 1;
  particle.xu(0, 0) = xmax + 0.1;
  particle.xv(0, 0) = xmax - 0.1;
  store_id(particle.xu(0, 6), rank * 100 + 7);
  application.migrate();
  auto restored = installed.get_internal_data();
  REQUIRE(restored.particles[0]->Np == 1);
  REQUIRE(load_id(restored.particles[0]->xu(0, 6)) == (1 - rank) * 100 + 7);
  REQUIRE(restored.particles[0]->xu(0, 0) == Catch::Approx(xmin + 0.1).margin(1.0e-12));
  REQUIRE(restored.load[hybrid::LoadParticle] == 1.0 / local_dims[2]);

  for (auto& communicator : communicators) {
    MPI_Comm_free(&communicator);
  }
}
} // namespace

TEST_CASE("Hybrid particle transfer 1D faces and periodic boundaries")
{
  const HybridTestGrid grid = {{1, 1, 12}, {false, false, true}, {1, 1, 2}, 2};
  if (!require_mpi_size(grid.mpi_size)) {
    return;
  }
  run_particle_test(grid, {0, 0, 1});
  run_particle_test(grid, {0, 0, -1});
}

TEST_CASE("Hybrid particle transfer 2D face and diagonal")
{
  const HybridTestGrid grid = {{1, 12, 12}, {false, true, true}, {1, 2, 2}, 4};
  if (!require_mpi_size(grid.mpi_size)) {
    return;
  }
  run_particle_test(grid, {0, 1, 0});
  run_particle_test(grid, {0, 1, 1});
}

TEST_CASE("Hybrid particle transfer 3D face and diagonal")
{
  const HybridTestGrid grid = {{12, 12, 12}, {true, true, true}, {2, 2, 2}, 8};
  if (!require_mpi_size(grid.mpi_size)) {
    return;
  }
  run_particle_test(grid, {1, 0, 0});
  run_particle_test(grid, {1, 1, 1});
}

TEST_CASE("Hybrid particle transfer supports multiple chunks per rank")
{
  if (require_mpi_size(2)) {
    run_multichunk_particle_test();
  }
}

TEST_CASE("HybridApplication schedules accepted-particle migration")
{
  if (require_mpi_size(2)) {
    run_application_migration_test();
  }
}
