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

#include "chunk_accessor.hpp"
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

  PetscScatter scatter(nullptr, {1, 1, 1});

  scatter.setup_indexset_local(index.size());
  scatter.get_indexset_local(index);

  REQUIRE(index.size() == static_cast<size_t>(local_size));
  REQUIRE(index[0] == 0);
  REQUIRE(index[1] == 1);
  REQUIRE(index[5] == 5);
  REQUIRE(index[9] == 9);
}

TEST_CASE("PetscScatter::setup_indexset_global", "[np=8]")
{
  if (get_mpi_size() != 8) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const int    rank                = get_mpi_rank();
  const int    num_chunks_per_rank = 8;
  const Dims3D global_dims{8, 8, 8};
  const Dims3D chunk_dims{2, 2, 2};

  /// test setup_indexset_global
  auto [index_test, chunkvec] =
      get_index_and_chunkvec(rank, chunk_dims, global_dims, num_chunks_per_rank);
  MockChunkAccessor accessor(chunkvec);
  accessor.build_global_index(index_test, global_dims);

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
  PetscScatter scatter(&dm_obj, global_dims);
  scatter.setup_indexset_global(index_test);
  scatter.get_indexset_global(index_test);

  for (size_t i = 0; i < index_test.size(); ++i) {
    REQUIRE(index_true[i] == index_test[i]);
  }

  DMDestroy(&dm_obj);
}

TEST_CASE("PetscScatter::scatter_forward_begin/end", "[np=8]")
{
  if (get_mpi_size() != 8) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const int    rank                = get_mpi_rank();
  const int    num_chunks_per_rank = 8;
  const Dims3D global_dims{8, 8, 8};
  const Dims3D chunk_dims{2, 2, 2};

  // deterministic index to value mapping
  auto index_to_val = [](const int index) { return static_cast<PetscScalar>(index) * 13 + 47; };

  // chunkvec and global index
  auto [index, chunkvec] =
      get_index_and_chunkvec(rank, chunk_dims, global_dims, num_chunks_per_rank);
  std::vector<float64> sol_buf(index.size());
  std::vector<float64> src_buf(index.size());
  MockChunkAccessor    accessor(chunkvec);

  // DMDA
  DM dm_obj = nullptr;
  DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_BOX, global_dims[2], global_dims[1], global_dims[0], PETSC_DECIDE,
               PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &dm_obj);
  DMSetUp(dm_obj);

  // vectors
  Vec vec_src    = nullptr;
  Vec vec_sol    = nullptr;
  Vec vec_global = nullptr;
  DMCreateGlobalVector(dm_obj, &vec_global);
  VecSet(vec_global, static_cast<PetscScalar>(0.0));

  // perform forward scatter
  {
    PetscScatter scatter(&dm_obj, global_dims);

    // scatter from src_buf
    accessor.build_global_index(index, global_dims);
    std::transform(index.begin(), index.end(), src_buf.begin(),
                   [index_to_val](int idx) { return index_to_val(idx); });

    scatter.setup_scatter(accessor, src_buf, sol_buf, vec_src, vec_sol, vec_global);
    scatter.scatter_forward_begin(vec_src, vec_global);
    scatter.scatter_forward_end(vec_src, vec_global);
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
          int         ii       = MockChunkAccessor::flatten_index(iz, iy, ix, global_dims);
          PetscScalar expected = index_to_val(ii);
          REQUIRE(std::abs(vec[iz][iy][ix] - expected) < 1.0e-10);
        }
      }
    }
    DMDAVecRestoreArrayRead(dm_obj, vec_global, &vec);
  }

  VecDestroy(&vec_src);
  VecDestroy(&vec_sol);
  VecDestroy(&vec_global);
  DMDestroy(&dm_obj);
}

TEST_CASE("PetscScatter::scatter_reverse_begin/end", "[np=8]")
{
  if (get_mpi_size() != 8) {
    SUCCEED("Skipping test because of incompatible MPI rank");
    return;
  }

  const int    rank                = get_mpi_rank();
  const int    num_chunks_per_rank = 8;
  const Dims3D global_dims{8, 8, 8};
  const Dims3D chunk_dims{2, 2, 2};

  // deterministic index to value mapping
  auto index_to_val = [](const int index) { return static_cast<PetscScalar>(index) * 13 + 47; };

  // chunkvec and global index
  auto [index, chunkvec] =
      get_index_and_chunkvec(rank, chunk_dims, global_dims, num_chunks_per_rank);
  std::vector<float64> sol_buf(index.size());
  std::vector<float64> src_buf(index.size());
  MockChunkAccessor    accessor(chunkvec);

  // DMDA
  DM dm_obj = nullptr;
  DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_BOX, global_dims[2], global_dims[1], global_dims[0], PETSC_DECIDE,
               PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, nullptr, &dm_obj);
  DMSetUp(dm_obj);

  // vectors
  Vec vec_src    = nullptr;
  Vec vec_sol    = nullptr;
  Vec vec_global = nullptr;
  DMCreateGlobalVector(dm_obj, &vec_global);
  VecSet(vec_global, static_cast<PetscScalar>(0.0));

  // populate global vector
  {
    DMDALocalInfo  info_obj;
    PetscScalar*** vec;
    DMDAVecGetArrayWrite(dm_obj, vec_global, &vec);
    DMDAGetLocalInfo(dm_obj, &info_obj);

    for (int iz = info_obj.zs; iz < info_obj.zs + info_obj.zm; ++iz) {
      for (int iy = info_obj.ys; iy < info_obj.ys + info_obj.ym; ++iy) {
        for (int ix = info_obj.xs; ix < info_obj.xs + info_obj.xm; ++ix) {
          int ii = MockChunkAccessor::flatten_index(iz, iy, ix, global_dims);

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
    PetscScatter scatter(&dm_obj, global_dims);

    accessor.build_global_index(index, global_dims);
    std::fill(sol_buf.begin(), sol_buf.end(), -1.0);

    scatter.setup_scatter(accessor, src_buf, sol_buf, vec_src, vec_sol, vec_global);
    scatter.scatter_reverse_begin(vec_sol, vec_global);
    scatter.scatter_reverse_end(vec_sol, vec_global);
  }

  // verify results
  for (size_t i = 0; i < index.size(); ++i) {
    PetscScalar expected = index_to_val(index[i]);
    REQUIRE(std::abs(sol_buf[i] - expected) < 1.0e-10);
  }

  VecDestroy(&vec_src);
  VecDestroy(&vec_sol);
  VecDestroy(&vec_global);
  DMDestroy(&dm_obj);
}