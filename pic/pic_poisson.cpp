// -*- C++ -*-
#include "pic_poisson.hpp"

#include <array>
#include <cassert>

PicPoisson::PicPoisson(const nix::Dims3D& global_dims, float64 delh) : global_dims_(global_dims)
{
  const bool is_1d = (global_dims_[0] == 1) && (global_dims_[1] == 1);
  const bool is_2d = (global_dims_[0] == 1) && (global_dims_[1] > 1);
  const bool is_3d = (global_dims_[0] > 1) && (global_dims_[1] > 1);

  if (is_1d) {
    solver_ = std::make_unique<SolverAdapter<elliptic::PetscPoisson1D>>(global_dims_, delh);
  } else if (is_2d) {
    solver_ = std::make_unique<SolverAdapter<elliptic::PetscPoisson2D>>(global_dims_, delh);
  } else if (is_3d) {
    solver_ = std::make_unique<SolverAdapter<elliptic::PetscPoisson3D>>(global_dims_, delh);
  } else {
    ERROR << tfm::format("Invalid global dimensions for PicPoisson: %d %d %d", global_dims_[0],
                         global_dims_[1], global_dims_[2]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}

int PicPoisson::update_mapping(const ChunkVec& chunks)
{
  accessor_.set_chunks(chunks);
  return solver_->update_mapping(accessor_);
}

int PicPoisson::scatter_rhs(const ChunkVec& chunks)
{
  update_mapping(chunks);
  solver_->copy_chunk_to_src(accessor_);
  solver_->scatter_forward_begin();
  solver_->scatter_forward_end();
  return 0;
}

int PicPoisson::copy_rhs_to_solution()
{
  return solver_->copy_rhs_to_solution();
}

int PicPoisson::scatter_solution_to_chunks(const ChunkVec& chunks)
{
  update_mapping(chunks);
  solver_->scatter_reverse_begin();
  solver_->scatter_reverse_end();
  solver_->copy_sol_to_chunk(accessor_);
  return 0;
}

int PicPoisson::solve(const ChunkVec& chunks)
{
  update_mapping(chunks);
  solver_->copy_chunk_to_src(accessor_);
  solver_->scatter_forward_begin();
  solver_->scatter_forward_end();
  int status = solver_->solve();
  solver_->scatter_reverse_begin();
  solver_->scatter_reverse_end();
  solver_->copy_sol_to_chunk(accessor_);
  return status;
}

int PicPoisson::set_option(const json& config)
{
  return solver_->set_option(config);
}

float64 PicPoisson::get_residual_norm()
{
  return solver_->get_residual_norm();
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
