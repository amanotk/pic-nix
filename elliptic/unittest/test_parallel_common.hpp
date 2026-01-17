#ifndef _TEST_PARALLEL_COMMON_HPP_
#define _TEST_PARALLEL_COMMON_HPP_

#include "nix.hpp"
#include "xtensorall.hpp"
#include <algorithm>
#include <vector>

#include "chunk_accessor.hpp"

using namespace nix;
using namespace nix::typedefs;

class MockChunk;
using MockChunkPtr = std::shared_ptr<MockChunk>;
using MockChunkVec = std::vector<std::shared_ptr<MockChunk>>;

// MockChunk class for testing
class MockChunk
{
protected:
  int                     Nb;
  int                     Lbx;
  int                     Ubx;
  int                     Lby;
  int                     Uby;
  int                     Lbz;
  int                     Ubz;
  xt::xtensor<float64, 3> src;
  xt::xtensor<float64, 3> sol;

  // Members mimicking nix::Chunk
  int id;
  int dims[3];
  int offset[3];
  int global_dims[3];

public:
  struct DataContainer {
    int&                     Lbx;
    int&                     Ubx;
    int&                     Lby;
    int&                     Uby;
    int&                     Lbz;
    int&                     Ubz;
    xt::xtensor<float64, 3>& src;
    xt::xtensor<float64, 3>& sol;
  };

  MockChunk() : Nb(2), id(0)
  {
    std::fill(std::begin(dims), std::end(dims), 0);
    std::fill(std::begin(offset), std::end(offset), 0);
    std::fill(std::begin(global_dims), std::end(global_dims), 0);
  }

  MockChunk(const int dims[3], int id = 0) : Nb(2), id(id)
  {
    std::copy(dims, dims + 3, this->dims);
    std::fill(std::begin(offset), std::end(offset), 0);
    std::fill(std::begin(global_dims), std::end(global_dims), 0);

    size_t nz = dims[0] + 2 * Nb;
    size_t ny = dims[1] + 2 * Nb;
    size_t nx = dims[2] + 2 * Nb;

    Lbz = Nb;
    Ubz = Lbz + dims[0] - 1;
    Lby = Nb;
    Uby = Lby + dims[1] - 1;
    Lbx = Nb;
    Ubx = Lbx + dims[2] - 1;

    src.resize({nz, ny, nx});
    src.fill(0);
    sol.resize({nz, ny, nx});
    sol.fill(0);
  }

  void set_dims(Dims3D d)
  {
    if (d.size() < 3)
      return;
    std::copy(d.begin(), d.begin() + 3, dims);

    size_t nz = dims[0] + 2 * Nb;
    size_t ny = dims[1] + 2 * Nb;
    size_t nx = dims[2] + 2 * Nb;

    Lbz = Nb;
    Ubz = Lbz + dims[0] - 1;
    Lby = Nb;
    Uby = Lby + dims[1] - 1;
    Lbx = Nb;
    Ubx = Lbx + dims[2] - 1;

    src.resize({nz, ny, nx});
    src.fill(0);
    sol.resize({nz, ny, nx});
    sol.fill(0);
  }

  void set_offset(const std::vector<int>& o)
  {
    if (o.size() < 3)
      return;
    std::copy(o.begin(), o.begin() + 3, offset);
  }

  // Mimic nix::Chunk interface required by elliptic::Solver
  void set_global_context(const int offset[3], const int global_dims[3])
  {
    std::copy(offset, offset + 3, this->offset);
    std::copy(global_dims, global_dims + 3, this->global_dims);
  }

  const int* get_dims() const
  {
    return dims;
  }

  const int* get_offset() const
  {
    return offset;
  }

  DataContainer get_internal_data()
  {
    return DataContainer{Lbx, Ubx, Lby, Uby, Lbz, Ubz, src, sol};
  }

  void set_id(int new_id)
  {
    id = new_id;
  }

  int get_id() const
  {
    return id;
  }
};

// MockChunkAccessor class for testing
class MockChunkAccessor : public elliptic::ChunkAccessor
{
private:
  MockChunkVec chunkvec;

public:
  MockChunkAccessor()
  {
  }

  MockChunkAccessor(MockChunkVec& chunks) : chunkvec(chunks)
  {
  }

  virtual void build_global_index(std::vector<int>& index, Dims3D dims) const
  {
    assert(chunkvec.size() > 0);

    auto chunk_dims = chunkvec[0]->get_dims();
    int  chunk_size = chunk_dims[0] * chunk_dims[1] * chunk_dims[2];

    for (int i = 0; i < chunkvec.size(); ++i) {
      auto offset = chunkvec[i]->get_offset();

      for (int iz = 0; iz < chunk_dims[0]; ++iz) {
        for (int iy = 0; iy < chunk_dims[1]; ++iy) {
          for (int ix = 0; ix < chunk_dims[2]; ++ix) {
            int jz = iz + offset[0];
            int jy = iy + offset[1];
            int jx = ix + offset[2];
            int jj = flatten_index(iz, iy, ix, chunk_dims) + i * chunk_size;

            index[jj] = flatten_index(jz, jy, jx, dims);
          }
        }
      }
    }
  }

  virtual int pack(float64* buffer, int size)
  {
    assert(chunkvec.size() > 0);
    assert(size >= get_num_grids_total());

    auto             chunk_dims = chunkvec[0]->get_dims();
    int              chunk_size = get_num_grids_per_chunk();
    std::vector<int> lstride    = {chunk_dims[1] * chunk_dims[2], chunk_dims[2], 1};

    int count = 0;

    for (int i = 0; i < static_cast<int>(chunkvec.size()); ++i) {
      auto data = chunkvec[i]->get_internal_data();

      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
            int jz = iz - data.Lbz;
            int jy = iy - data.Lby;
            int jx = ix - data.Lbx;
            int jj = jz * lstride[0] + jy * lstride[1] + jx * lstride[2] + i * chunk_size;

            buffer[jj] = data.src(iz, iy, ix);
            ++count;
          }
        }
      }
    }

    return count;
  }

  virtual int unpack(float64* buffer, int size)
  {
    assert(chunkvec.size() > 0);
    assert(size >= get_num_grids_total());

    auto             chunk_dims = chunkvec[0]->get_dims();
    int              chunk_size = get_num_grids_per_chunk();
    std::vector<int> lstride    = {chunk_dims[1] * chunk_dims[2], chunk_dims[2], 1};

    int count = 0;

    for (int i = 0; i < static_cast<int>(chunkvec.size()); ++i) {
      auto data = chunkvec[i]->get_internal_data();

      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
            int jz = iz - data.Lbz;
            int jy = iy - data.Lby;
            int jx = ix - data.Lbx;
            int jj = jz * lstride[0] + jy * lstride[1] + jx * lstride[2] + i * chunk_size;

            data.sol(iz, iy, ix) = buffer[jj];
            ++count;
          }
        }
      }
    }

    return count;
  }

  virtual int get_num_chunks() const
  {
    return static_cast<int>(chunkvec.size());
  }

  virtual int get_num_grids_per_chunk() const
  {
    assert(chunkvec.size() > 0);
    auto chunk_dims = chunkvec[0]->get_dims();
    return chunk_dims[0] * chunk_dims[1] * chunk_dims[2];
  }

  virtual int get_num_grids_total() const
  {
    return get_num_chunks() * get_num_grids_per_chunk();
  }
};

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
  MockChunkVec chunkvec;
  chunkvec.reserve(id.size());

  for (size_t i = 0; i < id.size(); ++i) {
    auto chunk = std::make_unique<MockChunk>();
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

#endif //_TEST_PARALLEL_COMMON_HPP_
