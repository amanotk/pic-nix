#ifndef MOCK_CHUNK_HPP
#define MOCK_CHUNK_HPP

#include "nix.hpp"
#include "xtensorall.hpp"
#include <algorithm>
#include <vector>

using namespace nix;
using namespace nix::typedefs;

class MockChunk;
using MockChunkPtr = std::unique_ptr<MockChunk>;
using MockChunkVec = std::vector<std::unique_ptr<MockChunk>>;

// Mock Chunk class for testing
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
  xt::xtensor<float64, 4> uj;

  // Members mimicking nix::Chunk
  int dims[3];
  int id;
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
    xt::xtensor<float64, 4>& uj;
  };

  MockChunk() : Nb(2), id(0)
  {
    std::fill(std::begin(dims), std::end(dims), 0);
    std::fill(std::begin(offset), std::end(offset), 0);
    std::fill(std::begin(global_dims), std::end(global_dims), 0);

    // Minimal initialization
    uj.resize({(size_t)2 * Nb, (size_t)2 * Nb, (size_t)2 * Nb, 4});
    uj.fill(0);
  }

  MockChunk(const int dims[3], int id = 0) : Nb(2), id(id)
  {
    std::copy(dims, dims + 3, this->dims);
    std::fill(std::begin(offset), std::end(offset), 0);
    std::fill(std::begin(global_dims), std::end(global_dims), 0);

    size_t nz = dims[2] + 2 * Nb;
    size_t ny = dims[1] + 2 * Nb;
    size_t nx = dims[0] + 2 * Nb;

    Lbz = Nb;
    Ubz = Lbz + dims[2] - 1;
    Lby = Nb;
    Uby = Lby + dims[1] - 1;
    Lbx = Nb;
    Ubx = Lbx + dims[0] - 1;

    uj.resize({nz, ny, nx, 4});
    uj.fill(0);
  }

  void set_dims(const std::vector<int>& d)
  {
    if (d.size() < 3) return;
    std::copy(d.begin(), d.begin() + 3, dims);

    size_t nz = dims[2] + 2 * Nb;
    size_t ny = dims[1] + 2 * Nb;
    size_t nx = dims[0] + 2 * Nb;

    Lbz = Nb;
    Ubz = Lbz + dims[2] - 1;
    Lby = Nb;
    Uby = Lby + dims[1] - 1;
    Lbx = Nb;
    Ubx = Lbx + dims[0] - 1;

    uj.resize({nz, ny, nx, 4});
    uj.fill(0);
  }

  void set_offset(const std::vector<int>& o)
  {
    if (o.size() < 3) return;
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
    return DataContainer{Lbx, Ubx, Lby, Uby, Lbz, Ubz, uj};
  }

  xt::xtensor<float64, 4>& get_uj()
  {
    return uj;
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

MockChunkVec create_chunkvec_1d(int rank, int Cz, int Cy, int Cx, int npx)
{
  int global_dims[3] = {Cz, Cy, Cx * npx};
  int chunk_dims[3]  = {Cz, Cy, Cx};

  MockChunkVec chunkvec;
  // 1 chunk per rank
  auto chunk     = std::make_unique<MockChunk>(chunk_dims, rank);
  int  offset[3] = {0, 0, rank * Cx};
  chunk->set_global_context(offset, global_dims);
  chunkvec.emplace_back(std::move(chunk));

  return chunkvec;
}

MockChunkVec create_chunkvec_2d(int rank, int Cz, int Cy, int Cx, int npx, int npy)
{
  int global_dims[3] = {Cz, Cy * npy, Cx * npx};
  int chunk_dims[3]  = {Cz, Cy, Cx};

  MockChunkVec chunkvec;
  // 1 chunk per rank
  auto chunk = std::make_unique<MockChunk>(chunk_dims, rank);

  int rx        = rank % npx;
  int ry        = rank / npx;
  int offset[3] = {0, ry * Cy, rx * Cx};

  chunk->set_global_context(offset, global_dims);
  chunkvec.emplace_back(std::move(chunk));

  return chunkvec;
}

MockChunkVec create_chunkvec_3d(int rank, int Cz, int Cy, int Cx, int npx, int npy, int npz)
{
  int global_dims[3] = {Cz * npz, Cy * npy, Cx * npx};
  int chunk_dims[3]  = {Cz, Cy, Cx};

  MockChunkVec chunkvec;
  // 1 chunk per rank
  auto chunk = std::make_unique<MockChunk>(chunk_dims, rank);

  int rx        = rank % npx;
  int ry        = (rank / npx) % npy;
  int rz        = rank / (npx * npy);
  int offset[3] = {rz * Cz, ry * Cy, rx * Cx};

  chunk->set_global_context(offset, global_dims);
  chunkvec.emplace_back(std::move(chunk));

  return chunkvec;
}

#endif // MOCK_CHUNK_HPP
