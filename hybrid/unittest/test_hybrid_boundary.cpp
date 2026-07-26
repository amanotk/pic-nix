// -*- C++ -*-
#include "test_hybrid_context.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace
{
int direction(int index, int lower, int upper, bool active)
{
  if (!active) {
    return 0;
  }
  return (index > upper) - (index < lower);
}

template <typename Array>
void fill_copy_interior(Array& array, const hybrid::HybridChunk::DataContainer& data, int rank)
{
  array.fill(-1);
  const int trailing =
      static_cast<int>(array.size() / (array.shape()[0] * array.shape()[1] * array.shape()[2]));
  auto* values = array.data();
  for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        const size_t cell =
            (static_cast<size_t>(iz) * array.shape()[1] + iy) * array.shape()[2] + ix;
        for (int component = 0; component < trailing; ++component) {
          values[cell * trailing + component] = rank * 1000.0 + component;
        }
      }
    }
  }
}

template <typename Array>
void verify_copy(const Array& array, const HybridExchangeContext& context,
                 const hybrid::HybridChunk::DataContainer& data)
{
  const int trailing =
      static_cast<int>(array.size() / (array.shape()[0] * array.shape()[1] * array.shape()[2]));
  const auto* values  = array.data();
  const int   z_begin = context.has_dim[0] ? 0 : data.Lbz;
  const int   z_end   = context.has_dim[0] ? static_cast<int>(array.shape()[0]) - 1 : data.Ubz;
  const int   y_begin = context.has_dim[1] ? 0 : data.Lby;
  const int   y_end   = context.has_dim[1] ? static_cast<int>(array.shape()[1]) - 1 : data.Uby;
  const int   x_begin = context.has_dim[2] ? 0 : data.Lbx;
  const int   x_end   = context.has_dim[2] ? static_cast<int>(array.shape()[2]) - 1 : data.Ubx;

  for (int iz = z_begin; iz <= z_end; ++iz) {
    for (int iy = y_begin; iy <= y_end; ++iy) {
      for (int ix = x_begin; ix <= x_end; ++ix) {
        const int    dirz        = direction(iz, data.Lbz, data.Ubz, context.has_dim[0]);
        const int    diry        = direction(iy, data.Lby, data.Uby, context.has_dim[1]);
        const int    dirx        = direction(ix, data.Lbx, data.Ubx, context.has_dim[2]);
        const int    source_rank = context.chunk->get_nb_rank(dirz, diry, dirx);
        const size_t cell =
            (static_cast<size_t>(iz) * array.shape()[1] + iy) * array.shape()[2] + ix;
        for (int component = 0; component < trailing; ++component) {
          INFO("index=" << iz << "," << iy << "," << ix << " component=" << component);
          REQUIRE(values[cell * trailing + component] == source_rank * 1000.0 + component);
        }
      }
    }
  }
}

template <typename Array>
void exchange_copy(hybrid::HybridChunk& chunk, Array& array, hybrid::BoundaryMode mode)
{
  chunk.boundary_pack(array, mode);
  chunk.boundary_begin(array, mode);
  chunk.boundary_end(array, mode);
  chunk.boundary_unpack(array, mode);
  REQUIRE(chunk.exchanges_idle());
}

bool in_boundary_strip(int index, int lower, int upper)
{
  return index < lower + hybrid::boundary_margin || index > upper - hybrid::boundary_margin;
}

void run_boundary_test(const HybridTestGrid& grid)
{
  auto context = build_hybrid_exchange_context(grid);
  auto data    = context.chunk->get_internal_data();

  fill_copy_interior(data.fluid, data, get_mpi_rank());
  exchange_copy(*context.chunk, data.fluid, hybrid::BoundaryCopy10);
  verify_copy(data.fluid, context, data);

  fill_copy_interior(data.field_cell, data, get_mpi_rank());
  exchange_copy(*context.chunk, data.field_cell, hybrid::BoundaryCopy6);
  verify_copy(data.field_cell, context, data);
  fill_copy_interior(data.field_staggered, data, get_mpi_rank());
  exchange_copy(*context.chunk, data.field_staggered, hybrid::BoundaryCopy6);
  verify_copy(data.field_staggered, context, data);

  fill_copy_interior(data.background_cell, data, get_mpi_rank());
  exchange_copy(*context.chunk, data.background_cell, hybrid::BoundaryCopy3);
  verify_copy(data.background_cell, context, data);

  fill_copy_interior(data.moment_kinetic, data, get_mpi_rank());
  exchange_copy(*context.chunk, data.moment_kinetic, hybrid::BoundaryMomentCopy);
  verify_copy(data.moment_kinetic, context, data);

  data.moment_kinetic.fill(1);
  exchange_copy(*context.chunk, data.moment_kinetic, hybrid::BoundaryMomentAccum);
  for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        int factor = 1;
        factor *= context.has_dim[0] && in_boundary_strip(iz, data.Lbz, data.Ubz) ? 2 : 1;
        factor *= context.has_dim[1] && in_boundary_strip(iy, data.Lby, data.Uby) ? 2 : 1;
        factor *= context.has_dim[2] && in_boundary_strip(ix, data.Lbx, data.Ubx) ? 2 : 1;
        for (int is = 0; is < data.num_species; ++is) {
          for (int component = 0; component < hybrid::num_moment_components; ++component) {
            INFO("index=" << iz << "," << iy << "," << ix << " species=" << is
                          << " component=" << component);
            REQUIRE(data.moment_kinetic(iz, iy, ix, is, component) == factor);
          }
        }
      }
    }
  }
}

void run_dlb_rebind_test()
{
  constexpr int     num_chunks  = 4;
  const int         rank        = get_mpi_rank();
  const nix::Dims3D local_dims  = {1, 1, 6};
  const nix::Dims3D global_dims = {1, 1, num_chunks * local_dims[2]};
  const nix::Bool3D has_dim     = {false, false, true};
  auto              chunkmap    = std::make_unique<nix::ChunkMap>(1, 1, num_chunks);
  std::vector<int>  boundary    = {0, 2, 4};
  chunkmap->set_rank_boundary(boundary);

  std::vector<std::shared_ptr<hybrid::HybridChunk>> local_chunks;
  for (int id = boundary[rank]; id < boundary[rank + 1]; ++id) {
    auto               chunk  = std::make_shared<hybrid::HybridChunk>(local_dims, has_dim, id);
    std::array<int, 3> offset = {0, 0, id * local_dims[2]};
    chunk->set_global_context(offset.data(), global_dims.data());
    auto config = make_hybrid_test_config();
    chunk->setup(config);
    auto data = chunk->get_internal_data();
    fill_copy_interior(data.fluid, data, id);
    fill_copy_interior(data.moment_kinetic, data, id);
    local_chunks.push_back(chunk);
  }

  if (rank == 0) {
    auto&                      migrating = local_chunks.back();
    const int                  size      = static_cast<int>(migrating->get_size_byte());
    std::vector<unsigned char> buffer(static_cast<size_t>(size));
    REQUIRE(migrating->pack(buffer.data(), 0) == size);
    MPI_Send(&size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Send(buffer.data(), size, MPI_BYTE, 1, 1, MPI_COMM_WORLD);
    local_chunks.pop_back();
  } else {
    int size = 0;
    MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::vector<unsigned char> buffer(static_cast<size_t>(size));
    MPI_Recv(buffer.data(), size, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    auto migrated = std::make_shared<hybrid::HybridChunk>(local_dims, has_dim, 1);
    REQUIRE(migrated->unpack(buffer.data(), 0) == size);
    local_chunks.push_back(migrated);
  }

  boundary = {0, 1, 4};
  chunkmap->set_rank_boundary(boundary);
  nix::ChunkVector<std::shared_ptr<hybrid::HybridChunk>> neighbor_chunks;
  for (auto& chunk : local_chunks) {
    neighbor_chunks.push_back(chunk);
  }
  neighbor_chunks.set_neighbors(chunkmap);

  std::vector<MPI_Comm> communicators(static_cast<size_t>(hybrid::NumBoundaryModes) * 27);
  for (int mode = 0; mode < hybrid::NumBoundaryModes; ++mode) {
    for (int iz = 0; iz < 3; ++iz) {
      for (int iy = 0; iy < 3; ++iy) {
        for (int ix = 0; ix < 3; ++ix) {
          const int index = mode * 27 + iz * 9 + iy * 3 + ix;
          MPI_Comm_dup(MPI_COMM_WORLD, &communicators[static_cast<size_t>(index)]);
          for (auto& chunk : local_chunks) {
            chunk->set_mpi_communicator(mode, iz, iy, ix,
                                        communicators[static_cast<size_t>(index)]);
          }
        }
      }
    }
  }

  auto exchange_all = [&](auto get_array, hybrid::BoundaryMode mode) {
    for (auto& chunk : local_chunks) {
      auto data = chunk->get_internal_data();
      chunk->boundary_pack(get_array(data), mode);
      chunk->boundary_begin(get_array(data), mode);
    }
    for (auto& chunk : local_chunks) {
      auto data = chunk->get_internal_data();
      chunk->boundary_end(get_array(data), mode);
      chunk->boundary_unpack(get_array(data), mode);
    }
  };

  exchange_all(
      [](auto& data) -> auto& { return data.fluid; }, hybrid::BoundaryCopy10);
  exchange_all(
      [](auto& data) -> auto& { return data.moment_kinetic; }, hybrid::BoundaryMomentCopy);
  for (const auto& chunk : local_chunks) {
    auto data = chunk->get_internal_data();
    for (int ix = 0; ix < static_cast<int>(data.fluid.shape()[2]); ++ix) {
      const int dirx      = direction(ix, data.Lbx, data.Ubx, true);
      const int source_id = chunk->get_nb_id(0, 0, dirx);
      for (int component = 0; component < hybrid::num_fluid_components; ++component) {
        REQUIRE(data.fluid(data.Lbz, data.Lby, ix, component) == source_id * 1000.0 + component);
      }
      for (int is = 0; is < data.num_species; ++is) {
        for (int component = 0; component < hybrid::num_moment_components; ++component) {
          const int trailing = is * hybrid::num_moment_components + component;
          REQUIRE(data.moment_kinetic(data.Lbz, data.Lby, ix, is, component) ==
                  source_id * 1000.0 + trailing);
        }
      }
    }
    REQUIRE(chunk->exchanges_idle());
  }

  for (auto& chunk : local_chunks) {
    chunk->get_internal_data().moment_kinetic.fill(1);
  }
  exchange_all(
      [](auto& data) -> auto& { return data.moment_kinetic; }, hybrid::BoundaryMomentAccum);
  for (const auto& chunk : local_chunks) {
    auto data = chunk->get_internal_data();
    for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
      const int expected = in_boundary_strip(ix, data.Lbx, data.Ubx) ? 2 : 1;
      for (int is = 0; is < data.num_species; ++is) {
        for (int component = 0; component < hybrid::num_moment_components; ++component) {
          REQUIRE(data.moment_kinetic(data.Lbz, data.Lby, ix, is, component) == expected);
        }
      }
    }
  }

  for (auto& communicator : communicators) {
    MPI_Comm_free(&communicator);
  }
}
} // namespace

TEST_CASE("Hybrid halo exchange 1D after serialization")
{
  const HybridTestGrid grid = {{1, 1, 12}, {false, false, true}, {1, 1, 2}, 2};
  if (require_mpi_size(grid.mpi_size)) {
    run_boundary_test(grid);
  }
}

TEST_CASE("Hybrid halo exchange 2D after serialization")
{
  const HybridTestGrid grid = {{1, 12, 12}, {false, true, true}, {1, 2, 2}, 4};
  if (require_mpi_size(grid.mpi_size)) {
    run_boundary_test(grid);
  }
}

TEST_CASE("Hybrid halo exchange 3D after serialization")
{
  const HybridTestGrid grid = {{12, 12, 12}, {true, true, true}, {2, 2, 2}, 8};
  if (require_mpi_size(grid.mpi_size)) {
    run_boundary_test(grid);
  }
}

TEST_CASE("Hybrid fixed halos rebind after mock DLB migration")
{
  if (require_mpi_size(2)) {
    run_dlb_rebind_test();
  }
}
