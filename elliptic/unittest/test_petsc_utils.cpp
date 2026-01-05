// -*- C++-*-
#include "thirdparty/catch.hpp"

#include <array>
#include <cassert> // assert
#include <memory>
#include <tuple>   // std::tuple
#include <utility> // std::pair
#include <vector>

#include <petscao.h>   // AO, AOApplicationToPetsc
#include <petscdmda.h> // DMDAGetLocalInfo

#include "mock_chunk.hpp"
#include "petsc_utils.hpp"

using namespace nix::typedefs;
using elliptic::PetscUtils;

inline int get_mpi_size()
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

inline int get_mpi_rank()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

inline auto get_rank_dims(const std::array<int, 3>& chunk_dims,
                          const std::array<int, 3>& global_dims)
{
  return std::array<int, 3>{global_dims[0] / chunk_dims[0], global_dims[1] / chunk_dims[1],
                            global_dims[2] / chunk_dims[2]};
}

inline auto get_chunklist(int rank, const std::array<int, 3>& rank_dims, int num_chunks_per_rank)
{
  std::vector<int> id;
  std::vector<int> oz;
  std::vector<int> oy;
  std::vector<int> ox;

  id.reserve(static_cast<size_t>(num_chunks_per_rank));
  oz.reserve(static_cast<size_t>(num_chunks_per_rank));
  oy.reserve(static_cast<size_t>(num_chunks_per_rank));
  ox.reserve(static_cast<size_t>(num_chunks_per_rank));

  const int nx = rank_dims[2];
  const int ny = rank_dims[1];

  // global chunk id = cx*sx + cy*sy + cz*sz
  const int sx = 1;
  const int sy = nx;
  const int sz = nx * ny;

  for (int i = 0; i < num_chunks_per_rank; ++i) {
    const int id_global = rank * num_chunks_per_rank + i;

    // chunk coordinates
    const int cx = (id_global / sx) % nx;
    const int cy = (id_global / sy) % ny;
    const int cz = (id_global / sz);

    id.push_back(id_global);
    oz.push_back(cz);
    oy.push_back(cy);
    ox.push_back(cx);
  }

  return std::make_tuple(std::move(id), std::move(oz), std::move(oy), std::move(ox));
}

inline auto get_chunkvec(const std::array<int, 3>& dims, const std::vector<int>& id,
                         const std::vector<int>& oz, const std::vector<int>& oy,
                         const std::vector<int>& ox)
{
  std::vector<std::shared_ptr<MockChunk>> chunkvec;
  chunkvec.reserve(id.size());

  for (size_t i = 0; i < id.size(); ++i) {
    auto chunk = std::make_shared<MockChunk>();
    chunk->set_id(id[i]);
    chunk->set_dims({dims[0], dims[1], dims[2]});
    chunk->set_offset({oz[i] * dims[0], oy[i] * dims[1], ox[i] * dims[2]});
    chunkvec.push_back(std::move(chunk));
  }

  return chunkvec;
}

inline auto get_chunkvec(int rank, const std::array<int, 3>& chunk_dims,
                         const std::array<int, 3>& rank_dims, int num_chunks_per_rank)
{
  const auto [cids, offz, offy, offx] = get_chunklist(rank, rank_dims, num_chunks_per_rank);
  return get_chunkvec(chunk_dims, cids, offz, offy, offx);
}

inline auto get_index_and_chunkvec(int rank, const std::array<int, 3>& chunk_dims,
                                   const std::array<int, 3>& global_dims, int num_chunks_per_rank)
{
  const auto rank_dims = get_rank_dims(chunk_dims, global_dims);

  // check consistency
  const int size = get_mpi_size();
  assert(size * num_chunks_per_rank == (rank_dims[0] * rank_dims[1] * rank_dims[2]));

  const int chunk_size = chunk_dims[0] * chunk_dims[1] * chunk_dims[2];

  std::vector<int> index(chunk_size * num_chunks_per_rank);
  auto             chunkvec = get_chunkvec(rank, chunk_dims, rank_dims, num_chunks_per_rank);

  return std::make_tuple(std::move(index), std::move(chunkvec));
}

TEST_CASE("PetscUtils::setup_indexset_local", "[np=1]")
{
  if (get_mpi_size() != 1) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const int        local_size = 10;
  std::vector<int> index(local_size);

  PetscUtils petsc_utils(nullptr);

  petsc_utils.setup_indexset_local(index);
  petsc_utils.get_indexset_local(index);

  CHECK(index.size() == static_cast<size_t>(local_size));
  CHECK(index[0] == 0);
  CHECK(index[1] == 1);
  CHECK(index[5] == 5);
  CHECK(index[9] == 9);
}

TEST_CASE("PetscUtils::setup_indexset_global", "[np=8]")
{
  if (get_mpi_size() != 8) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const int                rank                = get_mpi_rank();
  const int                num_chunks_per_rank = 8;
  const std::array<int, 3> global_dims{8, 8, 8};
  const std::array<int, 3> chunk_dims{2, 2, 2};

  /// test setup_indexset_global
  auto [index_test, chunkvec] =
      get_index_and_chunkvec(rank, chunk_dims, global_dims, num_chunks_per_rank);
  PetscUtils::calc_global_index(chunkvec, global_dims, index_test);

  // index_test: natural ordering indices (to be converted)
  // index_true: PETSc ordering indices (ground truth)
  std::vector<int> index_true(index_test);

  // get PETSc ordering using AO as ground truth for comparison
  DM dm_obj;
  AO ao_obj;
  DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_BOX, global_dims[2], global_dims[1], global_dims[0], PETSC_DECIDE,
               PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &dm_obj);
  DMSetUp(dm_obj);
  DMDAGetAO(dm_obj, &ao_obj);
  AOApplicationToPetsc(ao_obj, static_cast<PetscInt>(index_true.size()), index_true.data());

  // convert index_test to PETSc ordering index
  PetscUtils petsc_utils(&dm_obj);
  petsc_utils.setup_indexset_global(index_test);
  petsc_utils.get_indexset_global(index_test);

  for (size_t i = 0; i < index_test.size(); ++i) {
    CHECK(index_true[i] == index_test[i]);
  }

  DMDestroy(&dm_obj);
}

TEST_CASE("PetrscUtils::flatten_index", "[np=1]")
{
  if (get_mpi_size() != 1) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

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

  const int                rank                = get_mpi_rank();
  const int                num_chunks_per_rank = 8;
  const std::array<int, 3> global_dims{8, 12, 16};
  const std::array<int, 3> chunk_dims{2, 3, 4};
  const int                chunk_size = chunk_dims[0] * chunk_dims[1] * chunk_dims[2];

  /// test calc_global_index
  auto [index, chunkvec] =
      get_index_and_chunkvec(rank, chunk_dims, global_dims, num_chunks_per_rank);
  PetscUtils::calc_global_index(chunkvec, global_dims, index);

  for (size_t i = 0; i < chunkvec.size(); ++i) {
    auto offset = chunkvec[i]->get_offset();

    for (int iz = 0; iz < chunk_dims[0]; ++iz) {
      for (int iy = 0; iy < chunk_dims[1]; ++iy) {
        for (int ix = 0; ix < chunk_dims[2]; ++ix) {
          int jz         = iz + offset[0];
          int jy         = iy + offset[1];
          int jx         = ix + offset[2];
          int idx_local  = PetscUtils::flatten_index(iz, iy, ix, chunk_dims) + i * chunk_size;
          int idx_global = PetscUtils::flatten_index(jz, jy, jx, global_dims);

          CHECK(index[idx_local] == idx_global);
        }
      }
    }
  }
}
