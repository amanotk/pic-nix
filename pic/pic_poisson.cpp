// -*- C++ -*-
#include "pic_poisson.hpp"

#include <array>
#include <cassert>

using namespace nix::typedefs;

PicPoisson::PicPoisson(const nix::Dims3D& global_dims, float64 delh)
    : global_dims(global_dims), delx(delh), dely(delh), delz(delh),
      chunk_views(), chunk_dims{0, 0, 0}
{
}

void PicPoisson::bind_chunks(AppChunkVec& chunkvec)
{
  bind_chunks_impl(chunkvec);
}

void PicPoisson::bind_chunks(PicChunkVec& chunkvec)
{
  bind_chunks_impl(chunkvec);
}

PicPoisson::PicChunkAccessor PicPoisson::get_accessor()
{
  return PicChunkAccessor(chunk_views, chunk_dims);
}

PicPoisson::PicChunkAccessor::PicChunkAccessor(const ChunkViewVec& chunks, nix::Dims3D chunk_dims)
    : chunkvec(chunks), chunk_dims(chunk_dims)
{
}

void PicPoisson::PicChunkAccessor::build_global_index(std::vector<int>& index,
                                                      nix::Dims3D       global_dims) const
{
  assert(static_cast<int>(index.size()) >= get_num_grids_total());

  for (int i = 0; i < get_num_chunks(); ++i) {
    auto chunk   = chunkvec[i];
    auto offset  = chunk->get_offset();
    auto data    = chunk->get_internal_data();
    auto lstride = std::array<int, 3>{chunk_dims[1] * chunk_dims[2], chunk_dims[2], 1};
    auto csize   = get_num_grids_per_chunk();

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          const int local_z = iz - data.Lbz;
          const int local_y = iy - data.Lby;
          const int local_x = ix - data.Lbx;
          const int jj =
              local_z * lstride[0] + local_y * lstride[1] + local_x * lstride[2] + i * csize;
          const int gz = offset[0] + local_z;
          const int gy = offset[1] + local_y;
          const int gx = offset[2] + local_x;
          index[jj]    = flatten_index(gz, gy, gx, global_dims);
        }
      }
    }
  }
}

int PicPoisson::PicChunkAccessor::pack(float64* buffer, int size)
{
  assert(size >= get_num_grids_total());

  auto lstride = std::array<int, 3>{chunk_dims[1] * chunk_dims[2], chunk_dims[2], 1};
  int  count   = 0;
  int  csize   = get_num_grids_per_chunk();

  for (int i = 0; i < get_num_chunks(); ++i) {
    auto chunk = chunkvec[i];
    auto data  = chunk->get_internal_data();

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          const int jz = iz - data.Lbz;
          const int jy = iy - data.Lby;
          const int jx = ix - data.Lbx;
          const int jj = jz * lstride[0] + jy * lstride[1] + jx * lstride[2] + i * csize;

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

  auto lstride = std::array<int, 3>{chunk_dims[1] * chunk_dims[2], chunk_dims[2], 1};
  int  count   = 0;
  int  csize   = get_num_grids_per_chunk();

  for (int i = 0; i < get_num_chunks(); ++i) {
    auto chunk = chunkvec[i];
    auto data  = chunk->get_internal_data();

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          const int jz = iz - data.Lbz;
          const int jy = iy - data.Lby;
          const int jx = ix - data.Lbx;
          const int jj = jz * lstride[0] + jy * lstride[1] + jx * lstride[2] + i * csize;

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
  return static_cast<int>(chunkvec.size());
}

int PicPoisson::PicChunkAccessor::get_num_grids_per_chunk() const
{
  return chunk_dims[0] * chunk_dims[1] * chunk_dims[2];
}

int PicPoisson::PicChunkAccessor::get_num_grids_total() const
{
  return get_num_chunks() * get_num_grids_per_chunk();
}

const PicPoisson::ChunkViewVec& PicPoisson::PicChunkAccessor::get_chunks() const
{
  return chunkvec;
}
