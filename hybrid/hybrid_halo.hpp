// -*- C++ -*-
#ifndef _HYBRID_HALO_HPP_
#define _HYBRID_HALO_HPP_

#include "nix/halo3d.hpp"

#include "nix/array_types.hpp"

#include <algorithm>

namespace hybrid
{
template <typename Chunk>
class Rank5CopyHalo3D : public nix::Halo3D<nix::Array5D<nix::float64>, Chunk, true>
{
public:
  using Base = nix::Halo3D<nix::Array5D<nix::float64>, Chunk, true>;
  using Base::Base;
  using Base::chunk;
  using Base::data;

  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    if (iz == 1 && iy == 1 && ix == 1) {
      return false;
    }

    auto Iz   = xt::range(send_bound[0][0], send_bound[0][1] + 1);
    auto Iy   = xt::range(send_bound[1][0], send_bound[1][1] + 1);
    auto Ix   = xt::range(send_bound[2][0], send_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    auto* ptr = static_cast<nix::float64*>(mpibuf->get_send_buffer(iz, iy, ix));
    std::copy(view.begin(), view.end(), ptr);
    mpibuf->sendtype(iz, iy, ix) = MPI_BYTE;
    mpibuf->recvtype(iz, iy, ix) = MPI_BYTE;
    return true;
  }

  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    if (iz == 1 && iy == 1 && ix == 1) {
      return false;
    }
    if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL) {
      return false;
    }

    auto Iz   = xt::range(recv_bound[0][0], recv_bound[0][1] + 1);
    auto Iy   = xt::range(recv_bound[1][0], recv_bound[1][1] + 1);
    auto Ix   = xt::range(recv_bound[2][0], recv_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    auto* ptr = static_cast<nix::float64*>(mpibuf->get_recv_buffer(iz, iy, ix));
    std::copy(ptr, ptr + view.size(), view.begin());
    return true;
  }
};
} // namespace hybrid

#endif
