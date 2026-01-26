// -*- C++ -*-
#include "pic_poisson.hpp"

#include <array>
#include <cassert>

#include "elliptic/petsc_matrix_helpers.hpp"

#include <petscdmda.h>
#include <petscmat.h>

using namespace nix::typedefs;

PicPoisson::PicPoisson(const nix::Dims3D& global_dims, float64 delh)
    : PetscInterface(global_dims), global_dims_(global_dims), accessor_(), delx(delh), dely(delh),
      delz(delh)
{
  setup();
}

int PicPoisson::update_mapping(const ChunkVec& chunks)
{
  accessor_.set_chunks(chunks);
  return PetscInterface::update_mapping(accessor_);
}

int PicPoisson::scatter_rhs(const ChunkVec& chunks)
{
  update_mapping(chunks);
  copy_chunk_to_src(accessor_);
  scatter_forward_begin();
  scatter_forward_end();
  return 0;
}

int PicPoisson::copy_rhs_to_solution()
{
  return VecCopy(vector_src_g, vector_sol_g);
}

int PicPoisson::scatter_solution_to_chunks(const ChunkVec& chunks)
{
  update_mapping(chunks);
  scatter_reverse_begin();
  scatter_reverse_end();
  copy_sol_to_chunk(accessor_);
  return 0;
}

int PicPoisson::solve(const ChunkVec& chunks)
{
  update_mapping(chunks);
  copy_chunk_to_src(accessor_);
  scatter_forward_begin();
  scatter_forward_end();
  const int status = solve();
  scatter_reverse_begin();
  scatter_reverse_end();
  copy_sol_to_chunk(accessor_);
  return status;
}

int PicPoisson::solve(elliptic::ChunkAccessor& accessor)
{
  return solve();
}

int PicPoisson::solve()
{
  PetscErrorCode ierr = KSPSolve(ksp_obj, vector_src_g, vector_sol_g);
  if (ierr != 0) {
    ERROR << "KSPSolve failed with error code: " << ierr << std::endl;
  }
  return ierr;
}

int PicPoisson::set_option(const json& config)
{
  return PetscInterface::set_option(config);
}

float64 PicPoisson::get_residual_norm()
{
  Vec       vector_res_g;
  PetscReal res_norm;
  PetscReal src_norm;

  VecDuplicate(vector_src_g, &vector_res_g);
  MatMult(matrix, vector_sol_g, vector_res_g);
  VecAYPX(vector_res_g, -1.0, vector_src_g);
  VecNorm(vector_res_g, NORM_2, &res_norm);
  VecNorm(vector_src_g, NORM_2, &src_norm);
  VecDestroy(&vector_res_g);

  return static_cast<float64>(res_norm / (src_norm + 1.0e-32));
}

int PicPoisson::set_matrix()
{
  const bool    is_1d   = (global_dims_[0] == 1) && (global_dims_[1] == 1);
  const bool    is_2d   = (global_dims_[0] == 1) && (global_dims_[1] > 1);
  const bool    is_3d   = (global_dims_[0] > 1) && (global_dims_[1] > 1);
  const float64 dx2_inv = 1.0 / (delx * delx);
  const float64 dy2_inv = 1.0 / (dely * dely);
  const float64 dz2_inv = 1.0 / (delz * delz);
  const float64 ofdx    = -1.0 * dx2_inv;
  const float64 ofdy    = -1.0 * dy2_inv;
  const float64 ofdz    = -1.0 * dz2_inv;
  const float64 diag_1d = +2.0 * dx2_inv;
  const float64 diag_2d = +2.0 * dx2_inv + 2.0 * dy2_inv;
  const float64 diag_3d = +2.0 * dx2_inv + 2.0 * dy2_inv + 2.0 * dz2_inv;

  if (matrix != nullptr) {
    MatDestroy(&matrix);
  }
  DMCreateMatrix(dm_obj, &matrix);

  if (is_1d) {
    elliptic::build_poisson_matrix_1d(matrix, dm_obj, diag_1d, ofdx);
  } else if (is_2d) {
    elliptic::build_poisson_matrix_2d(matrix, dm_obj, diag_2d, ofdx, ofdy);
  } else if (is_3d) {
    elliptic::build_poisson_matrix_3d(matrix, dm_obj, diag_3d, ofdx, ofdy, ofdz);
  } else {
    ERROR << tfm::format("Invalid global dimensions for PicPoisson: %d %d %d", global_dims_[0],
                         global_dims_[1], global_dims_[2]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  set_nullspace();
  return 0;
}

void PicPoisson::set_nullspace()
{
  MatNullSpace ns;
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, nullptr, &ns);
  MatSetNullSpace(matrix, ns);
  MatNullSpaceDestroy(&ns);
}

void PicPoisson::PicChunkAccessor::set_chunks(const ChunkVec& chunks)
{
  chunks_ = chunks;

  if (chunks_.empty()) {
    chunk_dims_ = {0, 0, 0};
    chunk_size_ = 0;
    return;
  }

  auto dims   = chunks_.front()->get_dims();
  chunk_dims_ = {dims[0], dims[1], dims[2]};
  chunk_size_ = chunk_dims_[0] * chunk_dims_[1] * chunk_dims_[2];
}

void PicPoisson::PicChunkAccessor::build_global_index(std::vector<int>& index,
                                                      nix::Dims3D       dims) const
{
  assert(static_cast<int>(index.size()) >= get_num_grids_total());

  for (int i = 0; i < get_num_chunks(); ++i) {
    auto chunk   = chunks_[i];
    auto offset  = chunk->get_offset();
    auto data    = chunk->get_internal_data();
    auto lstride = std::array<int, 3>{chunk_dims_[1] * chunk_dims_[2], chunk_dims_[2], 1};

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          const int local_z = iz - data.Lbz;
          const int local_y = iy - data.Lby;
          const int local_x = ix - data.Lbx;
          const int jj =
              local_z * lstride[0] + local_y * lstride[1] + local_x * lstride[2] + i * chunk_size_;
          const int gz = offset[0] + local_z;
          const int gy = offset[1] + local_y;
          const int gx = offset[2] + local_x;
          index[jj]    = flatten_index(gz, gy, gx, dims);
        }
      }
    }
  }
}

int PicPoisson::PicChunkAccessor::pack(float64* buffer, int size)
{
  assert(size >= get_num_grids_total());

  auto lstride = std::array<int, 3>{chunk_dims_[1] * chunk_dims_[2], chunk_dims_[2], 1};
  int  count   = 0;

  for (int i = 0; i < get_num_chunks(); ++i) {
    auto data = chunks_[i]->get_internal_data();

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          const int jz = iz - data.Lbz;
          const int jy = iy - data.Lby;
          const int jx = ix - data.Lbx;
          const int jj = jz * lstride[0] + jy * lstride[1] + jx * lstride[2] + i * chunk_size_;

          buffer[jj] = data.uj(iz, iy, ix, 0);
          ++count;
        }
      }
    }
  }

  return count;
}

int PicPoisson::PicChunkAccessor::unpack(float64* buffer, int size)
{
  assert(size >= get_num_grids_total());

  auto lstride = std::array<int, 3>{chunk_dims_[1] * chunk_dims_[2], chunk_dims_[2], 1};
  int  count   = 0;

  for (int i = 0; i < get_num_chunks(); ++i) {
    auto data = chunks_[i]->get_internal_data();

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          const int jz = iz - data.Lbz;
          const int jy = iy - data.Lby;
          const int jx = ix - data.Lbx;
          const int jj = jz * lstride[0] + jy * lstride[1] + jx * lstride[2] + i * chunk_size_;

          data.phi(iz, iy, ix) = buffer[jj];
          ++count;
        }
      }
    }
  }

  return count;
}

int PicPoisson::PicChunkAccessor::get_num_chunks() const
{
  return static_cast<int>(chunks_.size());
}

int PicPoisson::PicChunkAccessor::get_num_grids_per_chunk() const
{
  return chunk_size_;
}

int PicPoisson::PicChunkAccessor::get_num_grids_total() const
{
  return get_num_chunks() * chunk_size_;
}
