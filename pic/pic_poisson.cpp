// -*- C++ -*-
#include "pic_poisson.hpp"

#include <array>
#include <cassert>

#include "elliptic/petsc_matrix_helpers.hpp"

#include <petscdmda.h>
#include <petscmat.h>

using namespace nix::typedefs;

PicPoisson::PicPoisson(const nix::Dims3D& global_dims, float64 delh)
    : PetscInterface(global_dims), global_dims_(global_dims), delx(delh), dely(delh), delz(delh)
{
  setup();
}

int PicPoisson::set_option(const json& config)
{
  return PetscInterface::set_option(config);
}

int PicPoisson::solve(elliptic::ChunkAccessor& accessor)
{
  PetscErrorCode ierr = KSPSolve(ksp_obj, vector_src_g, vector_sol_g);
  if (ierr != 0) {
    ERROR << "KSPSolve failed with error code: " << ierr << std::endl;
  }
  return ierr;
}

PicPoisson::PicChunkAccessor PicPoisson::get_accessor(ChunkVec& chunkvec)
{
  return PicChunkAccessor(chunkvec);
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

  return 0;
}

void PicPoisson::set_nullspace()
{
  MatNullSpace ns;
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, nullptr, &ns);
  MatSetNullSpace(matrix, ns);
  MatNullSpaceDestroy(&ns);
}

PicPoisson::PicChunkAccessor::PicChunkAccessor(const ChunkVec& chunks)
    : chunks_(chunks), chunk_dims_{0, 0, 0}, chunk_size_(0)
{
  if (chunks_.empty()) {
    return;
  }
  const auto& first_chunk = chunks_.front();
  auto        dims        = first_chunk->get_dims();
  chunk_dims_             = {dims[0], dims[1], dims[2]};
  chunk_size_             = chunk_dims_[0] * chunk_dims_[1] * chunk_dims_[2];
}

void PicPoisson::PicChunkAccessor::build_global_index(std::vector<int>& index,
                                                      nix::Dims3D       dims) const
{
  assert(static_cast<int>(index.size()) >= get_num_grids_total());

  for (int i = 0; i < get_num_chunks(); ++i) {
    auto chunk   = chunks_[i].get();
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
    auto chunk = chunks_[i].get();
    auto data  = chunk->get_internal_data();

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
    auto chunk = chunks_[i].get();
    auto data  = chunk->get_internal_data();

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
