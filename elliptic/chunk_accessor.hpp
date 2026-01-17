#ifndef _CHUNK_ACCESSOR_HPP_
#define _CHUNK_ACCESSOR_HPP_

#include "nix.hpp"
#include <cstddef>
#include <vector>

namespace elliptic
{
using namespace nix::typedefs;

class ChunkAccessor
{
public:
  virtual ~ChunkAccessor()                                                    = default;
  virtual void build_global_index(std::vector<int>& index, Dims3D dims) const = 0;
  virtual int  pack(float64* buffer, int size)                                = 0;
  virtual int  unpack(float64* buffer, int size)                              = 0;
  virtual int  get_num_chunks() const                                         = 0;
  virtual int  get_num_grids_per_chunk() const                                = 0;
  virtual int  get_num_grids_total() const                                    = 0;

  template <typename T_dims>
  static int flatten_index(int iz, int iy, int ix, const T_dims& dims)
  {
    const int stride_z = dims[1] * dims[2];
    const int stride_y = dims[2];
    const int stride_x = 1;

    return iz * stride_z + iy * stride_y + ix * stride_x;
  }
};

} // namespace elliptic

#endif
