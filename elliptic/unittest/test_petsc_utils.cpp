// -*- C++-*-
#include "thirdparty/catch.hpp"

#include <array>
#include <cassert> // assert
#include <memory>
#include <tuple>   // std::tuple
#include <utility> // std::pair
#include <vector>

#include <petscao.h>
#include <petscdmda.h>

#include "petsc_utils.hpp"
#include "test_parallel_common.hpp"

using namespace nix::typedefs;
using elliptic::PetscUtils;

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
