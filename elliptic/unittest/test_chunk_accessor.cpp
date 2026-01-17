// -*- C++-*-
#include "thirdparty/catch.hpp"

#include <array>
#include <cassert>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "chunk_accessor.hpp"
#include "test_parallel_common.hpp"

using namespace nix::typedefs;
using namespace elliptic;

TEST_CASE("ChunkAccessor::flatten_index", "[np=1]")
{
  if (get_mpi_size() != 1) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const Dims3D dims{4, 5, 6};

  REQUIRE(ChunkAccessor::flatten_index(0, 0, 0, dims) == 0);
  REQUIRE(ChunkAccessor::flatten_index(0, 0, 1, dims) == 1);
  REQUIRE(ChunkAccessor::flatten_index(0, 1, 0, dims) == 6);
  REQUIRE(ChunkAccessor::flatten_index(1, 0, 0, dims) == 30);
  REQUIRE(ChunkAccessor::flatten_index(3, 4, 5, dims) == 119);
}

TEST_CASE("ChunkAccessor::build_global_index with 8 ranks", "[np=8]")
{
  if (get_mpi_size() != 8) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const int    rank                = get_mpi_rank();
  const int    num_chunks_per_rank = 8;
  const Dims3D global_dims{8, 12, 16};
  const Dims3D chunk_dims{2, 3, 4};
  const int    chunk_size = chunk_dims[0] * chunk_dims[1] * chunk_dims[2];

  /// test build_global_index
  auto [index, chunkvec] =
      get_index_and_chunkvec(rank, chunk_dims, global_dims, num_chunks_per_rank);
  MockChunkAccessor accessor(chunkvec);
  accessor.build_global_index(index, global_dims);

  for (size_t i = 0; i < chunkvec.size(); ++i) {
    auto offset = chunkvec[i]->get_offset();

    for (int iz = 0; iz < chunk_dims[0]; ++iz) {
      for (int iy = 0; iy < chunk_dims[1]; ++iy) {
        for (int ix = 0; ix < chunk_dims[2]; ++ix) {
          int jz         = iz + offset[0];
          int jy         = iy + offset[1];
          int jx         = ix + offset[2];
          int idx_local  = ChunkAccessor::flatten_index(iz, iy, ix, chunk_dims) + i * chunk_size;
          int idx_global = ChunkAccessor::flatten_index(jz, jy, jx, global_dims);

          REQUIRE(index[idx_local] == idx_global);
        }
      }
    }
  }
}
