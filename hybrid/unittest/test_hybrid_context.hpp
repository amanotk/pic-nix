// -*- C++ -*-
#ifndef HYBRID_UNITTEST_TEST_HYBRID_CONTEXT_HPP
#define HYBRID_UNITTEST_TEST_HYBRID_CONTEXT_HPP

#include "hybrid_chunk.hpp"
#include "test_parallel.hpp"

#include "nix/chunkmap.hpp"
#include "nix/chunkvector.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>
#include <vector>

struct HybridTestGrid {
  nix::Dims3D        gdims;
  nix::Bool3D        has_dim;
  std::array<int, 3> cdims;
  int                mpi_size;
};

struct HybridExchangeContext {
  std::shared_ptr<hybrid::HybridChunk> chunk;
  nix::Dims3D                          gdims;
  nix::Bool3D                          has_dim;
  nix::Dims3D                          local_dims;
  std::array<int, 3>                   cdims;
  std::array<int, 3>                   coord;
  std::vector<MPI_Comm>                communicators;

  HybridExchangeContext()                                        = default;
  HybridExchangeContext(const HybridExchangeContext&)            = delete;
  HybridExchangeContext& operator=(const HybridExchangeContext&) = delete;
  HybridExchangeContext(HybridExchangeContext&&)                 = default;
  HybridExchangeContext& operator=(HybridExchangeContext&&)      = default;

  ~HybridExchangeContext()
  {
    for (auto& communicator : communicators) {
      MPI_Comm_free(&communicator);
    }
  }
};

inline nix::json make_hybrid_test_config()
{
  return {{"delh", 1.0},
          {"Ns", 2},
          {"cc", 20.0},
          {"gamma", 5.0 / 3.0},
          {"option", {{"cell_load", 1.0}, {"buffer_ratio", 0.2}}}};
}

inline HybridExchangeContext build_hybrid_exchange_context(const HybridTestGrid& grid,
                                                           bool serialize_first = true)
{
  REQUIRE(get_mpi_size() == grid.mpi_size);
  const int rank = get_mpi_rank();

  nix::Dims3D local_dims = {grid.gdims[0] / grid.cdims[0], grid.gdims[1] / grid.cdims[1],
                            grid.gdims[2] / grid.cdims[2]};
  auto chunkmap = std::make_unique<nix::ChunkMap>(grid.cdims[0], grid.cdims[1], grid.cdims[2]);
  std::vector<int> boundary(grid.mpi_size + 1);
  for (int i = 0; i <= grid.mpi_size; ++i) {
    boundary[i] = i;
  }
  chunkmap->set_rank_boundary(boundary);

  const int id    = boundary[rank];
  auto      chunk = std::make_shared<hybrid::HybridChunk>(local_dims, grid.has_dim, id);
  nix::ChunkVector<std::shared_ptr<hybrid::HybridChunk>> chunks;
  chunks.push_back(chunk);
  chunks.set_neighbors(chunkmap);

  auto [cz, cy, cx]         = chunkmap->get_coordinate(id);
  std::array<int, 3> offset = {cz * local_dims[0], cy * local_dims[1], cx * local_dims[2]};
  chunk->set_global_context(offset.data(), grid.gdims.data());
  auto config = make_hybrid_test_config();
  chunk->setup(config);

  if (serialize_first) {
    std::vector<unsigned char> buffer(static_cast<size_t>(chunk->get_size_byte()));
    REQUIRE(chunk->pack(buffer.data(), 0) == static_cast<int>(buffer.size()));
    chunk = std::make_shared<hybrid::HybridChunk>(local_dims, grid.has_dim, id);
    REQUIRE(chunk->unpack(buffer.data(), 0) == static_cast<int>(buffer.size()));
    nix::ChunkVector<std::shared_ptr<hybrid::HybridChunk>> restored_chunks;
    restored_chunks.push_back(chunk);
    restored_chunks.set_neighbors(chunkmap);
  }

  HybridExchangeContext context;
  context.chunk      = chunk;
  context.gdims      = grid.gdims;
  context.has_dim    = grid.has_dim;
  context.local_dims = local_dims;
  context.cdims      = grid.cdims;
  context.coord      = {cz, cy, cx};
  context.communicators.resize(static_cast<size_t>(hybrid::NumBoundaryModes) * 27);
  for (int mode = 0; mode < hybrid::NumBoundaryModes; ++mode) {
    for (int iz = 0; iz < 3; ++iz) {
      for (int iy = 0; iy < 3; ++iy) {
        for (int ix = 0; ix < 3; ++ix) {
          const int index = mode * 27 + iz * 9 + iy * 3 + ix;
          MPI_Comm_dup(MPI_COMM_WORLD, &context.communicators[static_cast<size_t>(index)]);
          chunk->set_mpi_communicator(mode, iz, iy, ix,
                                      context.communicators[static_cast<size_t>(index)]);
        }
      }
    }
  }
  return context;
}

#endif
