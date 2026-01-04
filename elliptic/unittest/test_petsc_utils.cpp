// -*- C++-*-
#include "thirdparty/catch.hpp"

#include <array>
#include <memory>
#include <vector>

#include "mock_chunk.hpp"
#include "petsc_utils.hpp"

using namespace nix::typedefs;
using elliptic::PetscUtils;

int get_mpi_size()
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

int get_mpi_rank()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

TEST_CASE("PetrscUtils::flatten_index", "[np=1]")
{
  const std::array<int, 3> dims{4, 5, 6};

  CHECK(PetscUtils::flatten_index(0, 0, 0, dims) == 0);
  CHECK(PetscUtils::flatten_index(0, 0, 1, dims) == 1);
  CHECK(PetscUtils::flatten_index(0, 1, 0, dims) == 6);
  CHECK(PetscUtils::flatten_index(1, 0, 0, dims) == 30);
  CHECK(PetscUtils::flatten_index(3, 4, 5, dims) == 119);
}

TEST_CASE("PetscUtils::calc_global_index with 8 ranks", "[np=8]")
{
  if (get_mpi_size() != 8) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const std::array<int, 3> global_dims{4, 4, 4};
  const std::array<int, 3> chunk_dims{2, 2, 2};

  const int rank     = get_mpi_rank();
  const int rank_x   = rank % 2;
  const int rank_y   = (rank / 2) % 2;
  const int rank_z   = (rank / 4) % 2;
  const int offset_z = rank_z * chunk_dims[0];
  const int offset_y = rank_y * chunk_dims[1];
  const int offset_x = rank_x * chunk_dims[2];

  auto chunk = std::make_shared<MockChunk>();
  chunk->set_dims({chunk_dims[0], chunk_dims[1], chunk_dims[2]});
  chunk->set_offset({offset_z, offset_y, offset_x});

  std::vector<std::shared_ptr<MockChunk>> chunkvec{chunk};

  const int             chunk_size = chunk_dims[0] * chunk_dims[1] * chunk_dims[2];
  std::vector<PetscInt> index(static_cast<size_t>(chunk_size));

  PetscUtils::calc_global_index(chunkvec, global_dims, index);

  for (int iz = 0; iz < chunk_dims[2]; ++iz) {
    for (int iy = 0; iy < chunk_dims[1]; ++iy) {
      for (int ix = 0; ix < chunk_dims[0]; ++ix) {
        const int jx = offset_x + ix;
        const int jy = offset_y + iy;
        const int jz = offset_z + iz;

        const int idx_local  = PetscUtils::flatten_index(iz, iy, ix, chunk_dims);
        const int idx_global = PetscUtils::flatten_index(jz, jy, jx, global_dims);

        CHECK(index[idx_local] == idx_global);
      }
    }
  }
}
