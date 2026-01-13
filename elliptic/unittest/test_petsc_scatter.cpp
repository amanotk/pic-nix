// -*- C++-*-
#include "thirdparty/catch.hpp"

#include <array>
#include <cassert>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <petscao.h>
#include <petscdmda.h>
#include <petscvec.h>

#include "petsc_scatter.hpp"
#include "test_parallel_common.hpp"

using namespace nix::typedefs;
using elliptic::PetscScatter;

TEST_CASE("PetscScatter::setup_indexset_local", "[np=1]")
{
  if (get_mpi_size() != 1) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const int        local_size = 10;
  std::vector<int> index(local_size);

  PetscScatter scatter(nullptr);

  scatter.setup_indexset_local(index.size());
  scatter.get_indexset_local(index);

  CHECK(index.size() == static_cast<size_t>(local_size));
  CHECK(index[0] == 0);
  CHECK(index[1] == 1);
  CHECK(index[5] == 5);
  CHECK(index[9] == 9);
}

TEST_CASE("PetscScatter::setup_indexset_global", "[np=8]")
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
  PetscScatter::calc_global_index(chunkvec, global_dims, index_test);

  // index_test: natural ordering indices (to be converted)
  // index_true: PETSc ordering indices (ground truth)
  std::vector<int> index_true(index_test);

  // get PETSc ordering using AO as ground truth for comparison
  DM dm_obj = nullptr;
  AO ao_obj = nullptr;
  DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_BOX, global_dims[2], global_dims[1], global_dims[0], PETSC_DECIDE,
               PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &dm_obj);
  DMSetUp(dm_obj);
  DMDAGetAO(dm_obj, &ao_obj);
  AOApplicationToPetsc(ao_obj, static_cast<PetscInt>(index_true.size()), index_true.data());

  // convert index_test to PETSc ordering index
  PetscScatter scatter(&dm_obj);
  scatter.setup_indexset_global(index_test);
  scatter.get_indexset_global(index_test);

  for (size_t i = 0; i < index_test.size(); ++i) {
    CHECK(index_true[i] == index_test[i]);
  }

  DMDestroy(&dm_obj);
}

TEST_CASE("PetscScatter::flatten_index", "[np=1]")
{
  if (get_mpi_size() != 1) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const std::array<int, 3> dims{4, 5, 6};

  CHECK(PetscScatter::flatten_index(0, 0, 0, dims) == 0);
  CHECK(PetscScatter::flatten_index(0, 0, 1, dims) == 1);
  CHECK(PetscScatter::flatten_index(0, 1, 0, dims) == 6);
  CHECK(PetscScatter::flatten_index(1, 0, 0, dims) == 30);
  CHECK(PetscScatter::flatten_index(3, 4, 5, dims) == 119);
}

TEST_CASE("PetscScatter::calc_global_index with 8 ranks", "[np=8]")
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
  PetscScatter::calc_global_index(chunkvec, global_dims, index);

  for (size_t i = 0; i < chunkvec.size(); ++i) {
    auto offset = chunkvec[i]->get_offset();

    for (int iz = 0; iz < chunk_dims[0]; ++iz) {
      for (int iy = 0; iy < chunk_dims[1]; ++iy) {
        for (int ix = 0; ix < chunk_dims[2]; ++ix) {
          int jz         = iz + offset[0];
          int jy         = iy + offset[1];
          int jx         = ix + offset[2];
          int idx_local  = PetscScatter::flatten_index(iz, iy, ix, chunk_dims) + i * chunk_size;
          int idx_global = PetscScatter::flatten_index(jz, jy, jx, global_dims);

          CHECK(index[idx_local] == idx_global);
        }
      }
    }
  }
}

TEST_CASE("PetscScatter::scatter_forward_begin/end", "[np=8]")
{
  if (get_mpi_size() != 8) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const int                rank                = get_mpi_rank();
  const int                num_chunks_per_rank = 8;
  const std::array<int, 3> global_dims{8, 8, 8};
  const std::array<int, 3> chunk_dims{2, 2, 2};

  // deterministic index to value mapping
  auto index_to_val = [](const int index) { return static_cast<PetscScalar>(index) * 13 + 47; };

  // chunkvec and global index
  auto [index, chunkvec] =
      get_index_and_chunkvec(rank, chunk_dims, global_dims, num_chunks_per_rank);
  PetscScatter::calc_global_index(chunkvec, global_dims, index);

  // DMDA
  DM dm_obj = nullptr;
  DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_BOX, global_dims[2], global_dims[1], global_dims[0], PETSC_DECIDE,
               PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &dm_obj);
  DMSetUp(dm_obj);

  // vectors
  Vec vec_local  = nullptr;
  Vec vec_global = nullptr;
  DMCreateGlobalVector(dm_obj, &vec_global);
  VecSet(vec_global, static_cast<PetscScalar>(0.0));

  // buffer for local vector
  std::vector<float64> buf(index.size());
  for (size_t i = 0; i < index.size(); ++i) {
    buf[i] = index_to_val(index[i]);
  }

  // perform forward scatter
  {
    PetscScatter scatter(&dm_obj);

    scatter.setup_vector_local(buf, vec_local);
    scatter.setup_indexset_local(index.size());
    scatter.setup_indexset_global(index);
    scatter.setup_scatter(vec_local, vec_global);
    scatter.scatter_forward_begin(vec_local, vec_global);
    scatter.scatter_forward_end(vec_local, vec_global);
  }

  // verify results
  {
    DMDALocalInfo  info_obj;
    PetscScalar*** vec;
    DMDAVecGetArrayRead(dm_obj, vec_global, &vec);
    DMDAGetLocalInfo(dm_obj, &info_obj);

    for (int iz = info_obj.zs; iz < info_obj.zs + info_obj.zm; ++iz) {
      for (int iy = info_obj.ys; iy < info_obj.ys + info_obj.ym; ++iy) {
        for (int ix = info_obj.xs; ix < info_obj.xs + info_obj.xm; ++ix) {
          int         ii       = PetscScatter::flatten_index(iz, iy, ix, global_dims);
          PetscScalar expected = index_to_val(ii);
          CHECK(std::abs(vec[iz][iy][ix] - expected) < 1.0e-10);
        }
      }
    }
    DMDAVecRestoreArrayRead(dm_obj, vec_global, &vec);
  }

  VecDestroy(&vec_local);
  VecDestroy(&vec_global);
  DMDestroy(&dm_obj);
}

TEST_CASE("PetscScatter::scatter_reverse_begin/end", "[np=8]")
{
  if (get_mpi_size() != 8) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const int                rank                = get_mpi_rank();
  const int                num_chunks_per_rank = 8;
  const std::array<int, 3> global_dims{8, 8, 8};
  const std::array<int, 3> chunk_dims{2, 2, 2};

  // deterministic index to value mapping
  auto index_to_val = [](const int index) { return static_cast<PetscScalar>(index) * 13 + 47; };

  // chunkvec and global index
  auto [index, chunkvec] =
      get_index_and_chunkvec(rank, chunk_dims, global_dims, num_chunks_per_rank);
  PetscScatter::calc_global_index(chunkvec, global_dims, index);

  // DMDA
  DM dm_obj = nullptr;
  DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_BOX, global_dims[2], global_dims[1], global_dims[0], PETSC_DECIDE,
               PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &dm_obj);
  DMSetUp(dm_obj);

  // vectors
  Vec vec_local  = nullptr;
  Vec vec_global = nullptr;
  DMCreateGlobalVector(dm_obj, &vec_global);
  VecSet(vec_global, static_cast<PetscScalar>(0.0));

  // buffer for local vector
  std::vector<float64> buf(index.size());
  std::fill(buf.begin(), buf.end(), -1.0);

  // populate global vector
  {
    DMDALocalInfo  info_obj;
    PetscScalar*** vec;
    DMDAVecGetArrayWrite(dm_obj, vec_global, &vec);
    DMDAGetLocalInfo(dm_obj, &info_obj);

    for (int iz = info_obj.zs; iz < info_obj.zs + info_obj.zm; ++iz) {
      for (int iy = info_obj.ys; iy < info_obj.ys + info_obj.ym; ++iy) {
        for (int ix = info_obj.xs; ix < info_obj.xs + info_obj.xm; ++ix) {
          int ii = PetscScatter::flatten_index(iz, iy, ix, global_dims);

          vec[iz][iy][ix] = index_to_val(ii);
        }
      }
    }
    DMDAVecRestoreArrayWrite(dm_obj, vec_global, &vec);
    VecAssemblyBegin(vec_global);
    VecAssemblyEnd(vec_global);
  }

  // perform reverse scatter
  {
    PetscScatter     scatter(&dm_obj);
    std::vector<int> scatter_index(index);

    scatter.setup_vector_local(buf, vec_local);
    scatter.setup_indexset_local(scatter_index.size());
    scatter.setup_indexset_global(scatter_index);
    scatter.setup_scatter(vec_local, vec_global);
    scatter.scatter_reverse_begin(vec_local, vec_global);
    scatter.scatter_reverse_end(vec_local, vec_global);
  }

  // verify results
  for (size_t i = 0; i < index.size(); ++i) {
    PetscScalar expected = index_to_val(index[i]);
    CHECK(std::abs(buf[i] - expected) < 1.0e-10);
  }

  VecDestroy(&vec_local);
  VecDestroy(&vec_global);
  DMDestroy(&dm_obj);
}