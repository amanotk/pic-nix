// -*- C++ -*-
#ifndef _XTENSOR_HALO3D_HPP_
#define _XTENSOR_HALO3D_HPP_

#include "halo3d.hpp"
#include "nix.hpp"
#include "xtensor_particle.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

using ParticlePtr = std::shared_ptr<XtensorParticle>;
using ParticleVec = std::vector<ParticlePtr>;

///
/// @brief Boundary Halo3D class for field
///
template <typename Chunk>
class XtensorHaloField3D : public Halo3D<xt::xtensor<float64, 4>, Chunk, true>
{
public:
  using Base = Halo3D<xt::xtensor<float64, 4>, Chunk, true>;
  using Base::Base; // constructor
  using Base::data;
  using Base::chunk;

  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    // packing
    auto Iz   = xt::range(send_bound[0][0], send_bound[0][1] + 1);
    auto Iy   = xt::range(send_bound[1][0], send_bound[1][1] + 1);
    auto Ix   = xt::range(send_bound[2][0], send_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_send_buffer(iz, iy, ix));
    std::copy(view.begin(), view.end(), ptr);

    // datatype
    mpibuf->sendtype(iz, iy, ix) = MPI_BYTE;
    mpibuf->recvtype(iz, iy, ix) = MPI_BYTE;

    return true;
  }

  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL)
      return false;

    // unpacking
    auto Iz   = xt::range(recv_bound[0][0], recv_bound[0][1] + 1);
    auto Iy   = xt::range(recv_bound[1][0], recv_bound[1][1] + 1);
    auto Ix   = xt::range(recv_bound[2][0], recv_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_recv_buffer(iz, iy, ix));
    std::copy(ptr, ptr + view.size(), view.begin());

    return true;
  }
};

///
/// @brief Boundary Halo3D class for current
///
template <typename Chunk>
class XtensorHaloCurrent3D : public Halo3D<xt::xtensor<float64, 4>, Chunk, true>
{
public:
  using Base = Halo3D<xt::xtensor<float64, 4>, Chunk, true>;
  using Base::Base; // constructor
  using Base::data;
  using Base::chunk;

  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    // packing
    auto Iz   = xt::range(recv_bound[0][0], recv_bound[0][1] + 1);
    auto Iy   = xt::range(recv_bound[1][0], recv_bound[1][1] + 1);
    auto Ix   = xt::range(recv_bound[2][0], recv_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_send_buffer(iz, iy, ix));
    std::copy(view.begin(), view.end(), ptr);

    // datatype
    mpibuf->sendtype(iz, iy, ix) = MPI_BYTE;
    mpibuf->recvtype(iz, iy, ix) = MPI_BYTE;

    return true;
  }

  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL)
      return false;

    // unpacking
    auto Iz   = xt::range(send_bound[0][0], send_bound[0][1] + 1);
    auto Iy   = xt::range(send_bound[1][0], send_bound[1][1] + 1);
    auto Ix   = xt::range(send_bound[2][0], send_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_recv_buffer(iz, iy, ix));
    std::transform(ptr, ptr + view.size(), view.begin(), view.begin(), std::plus<float64>());

    return true;
  }
};

///
/// @brief Boundary Halo3D class for moment
///
template <typename Chunk>
class XtensorHaloMoment3D : public Halo3D<xt::xtensor<float64, 5>, Chunk, true>
{
public:
  using Base = Halo3D<xt::xtensor<float64, 5>, Chunk, true>;
  using Base::Base; // constructor
  using Base::data;
  using Base::chunk;

  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    // packing
    auto Iz   = xt::range(recv_bound[0][0], recv_bound[0][1] + 1);
    auto Iy   = xt::range(recv_bound[1][0], recv_bound[1][1] + 1);
    auto Ix   = xt::range(recv_bound[2][0], recv_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_send_buffer(iz, iy, ix));
    std::copy(view.begin(), view.end(), ptr);

    // datatype
    mpibuf->sendtype(iz, iy, ix) = MPI_BYTE;
    mpibuf->recvtype(iz, iy, ix) = MPI_BYTE;

    return true;
  }

  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL)
      return false;

    // unpacking
    auto Iz   = xt::range(send_bound[0][0], send_bound[0][1] + 1);
    auto Iy   = xt::range(send_bound[1][0], send_bound[1][1] + 1);
    auto Ix   = xt::range(send_bound[2][0], send_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_recv_buffer(iz, iy, ix));
    std::transform(ptr, ptr + view.size(), view.begin(), view.begin(), std::plus<float64>());

    return true;
  }
};

///
/// @brief Boundary Halo3D class for particle
///
template <typename Chunk>
class XtensorHaloParticle3D : public Halo3D<ParticleVec, Chunk, false>
{
public:
  using Base = Halo3D<ParticleVec, Chunk, false>;
  using Base::data;
  using Base::chunk;

  static constexpr int32_t head_byte = sizeof(int32_t);
  static constexpr int32_t elem_byte = ParticlePtr::element_type::get_particle_size();

  int32_t              Ns;
  ParticleVec          particle;
  std::vector<int32_t> num_unpacked;

  XtensorHaloParticle3D(ParticleVec& data, Chunk& chunk)
      : Halo3D<ParticleVec, Chunk, false>(data, chunk)
  {
    Ns       = data.size();
    particle = data;
  }

  template <typename BufferPtr>
  void pre_pack(BufferPtr& mpibuf, int indexlb[3], int indexub[3])
  {
    auto [xmin, xmax] = chunk->get_xrange();
    auto [ymin, ymax] = chunk->get_yrange();
    auto [zmin, zmax] = chunk->get_zrange();

    std::array<size_t, 4>   shape = {static_cast<size_t>(Ns + 1), 3ul, 3ul, 3ul};
    xt::xtensor<int32_t, 4> send_count(shape);

    // initialize
    send_count.fill(0);

    //
    // count out-of-bounds particles
    //
    for (int is = 0; is < Ns; is++) {
      auto& xu = particle[is]->xu;
      for (int ip = 0; ip < particle[is]->Np; ip++) {
        int iz = (xu(ip, 2) >= zmax) - (xu(ip, 2) < zmin) + 1;
        int iy = (xu(ip, 1) >= ymax) - (xu(ip, 1) < ymin) + 1;
        int ix = (xu(ip, 0) >= xmax) - (xu(ip, 0) < xmin) + 1;

        // skip
        if (ix == 1 && iy == 1 && iz == 1)
          continue;

        send_count(is, iz, iy, ix)++;
        send_count(Ns, iz, iy, ix)++; // total number of send particles
      }
    }

    //
    // allocate buffer
    //
    {
      int bufsize = 0;

      mpibuf->bufsize.fill(0);
      mpibuf->bufaddr.fill(0);

      for (int iz = indexlb[0]; iz <= indexub[0]; iz++) {
        for (int iy = indexlb[1]; iy <= indexub[1]; iy++) {
          for (int ix = indexlb[2]; ix <= indexub[2]; ix++) {
            // skip
            if (iz == 1 && iy == 1 && ix == 1)
              continue;

            mpibuf->bufsize(iz, iy, ix) = elem_byte * send_count(Ns, iz, iy, ix) + head_byte * Ns;
            mpibuf->bufaddr(iz, iy, ix) = bufsize;
            bufsize += mpibuf->bufsize(iz, iy, ix);
          }
        }
      }

      mpibuf->sendbuf.resize(bufsize);
    }

    //
    // pack header
    //
    {
      for (int iz = indexlb[0]; iz <= indexub[0]; iz++) {
        for (int iy = indexlb[1]; iy <= indexub[1]; iy++) {
          for (int ix = indexlb[2]; ix <= indexub[2]; ix++) {
            // skip
            if (iz == 1 && iy == 1 && ix == 1)
              continue;

            int addr = mpibuf->bufaddr(iz, iy, ix);
            for (int is = 0; is < Ns; is++) {
              std::memcpy(mpibuf->sendbuf.get(addr), &send_count(is, iz, iy, ix), head_byte);
              addr += head_byte + elem_byte * send_count(is, iz, iy, ix);
            }
          }
        }
      }
    }

    //
    // pack out-of-bounds particles
    //
    {
      auto addr = mpibuf->bufaddr;

      for (int is = 0; is < Ns; is++) {
        // skip header
        for (int iz = indexlb[0]; iz <= indexub[0]; iz++) {
          for (int iy = indexlb[1]; iy <= indexub[1]; iy++) {
            for (int ix = indexlb[2]; ix <= indexub[2]; ix++) {
              addr(iz, iy, ix) += head_byte;
            }
          }
        }

        // pack particles
        auto& xu = particle[is]->xu;
        for (int ip = 0; ip < particle[is]->Np; ip++) {
          int iz = (xu(ip, 2) >= zmax) - (xu(ip, 2) < zmin) + 1;
          int iy = (xu(ip, 1) >= ymax) - (xu(ip, 1) < ymin) + 1;
          int ix = (xu(ip, 0) >= xmax) - (xu(ip, 0) < xmin) + 1;

          // skip
          if (ix == 1 && iy == 1 && iz == 1)
            continue;

          // pack
          std::memcpy(mpibuf->sendbuf.get(addr(iz, iy, ix)), &xu(ip, 0), elem_byte);
          addr(iz, iy, ix) += elem_byte;
          send_count(is, iz, iy, ix)--;
        }
      }
    }

    //
    // check if all particles are packed
    //
    {
      bool is_all_packed = true;

      for (int is = 0; is < Ns; is++) {
        for (int iz = indexlb[0]; iz <= indexub[0]; iz++) {
          for (int iy = indexlb[1]; iy <= indexub[1]; iy++) {
            for (int ix = indexlb[2]; ix <= indexub[2]; ix++) {
              is_all_packed = is_all_packed && (send_count(is, iz, iy, ix) == 0);
            }
          }
        }
      }

      if (is_all_packed == false) {
        ERROR << tfm::format("Some particles are not properly packed!");
      }
    }
  }

  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    // datatype
    mpibuf->sendtype(iz, iy, ix) = MPI_BYTE;
    mpibuf->recvtype(iz, iy, ix) = MPI_BYTE;

    return true;
  }

  template <typename BufferPtr>
  void post_pack(BufferPtr& mpibuf, int indexlb[3], int indexub[3])
  {
    // do nothing
  }

  template <typename BufferPtr>
  void pre_unpack(BufferPtr& mpibuf, int indexlb[3], int indexub[3])
  {
    std::array<size_t, 4>   shape = {static_cast<size_t>(Ns + 1), 3ul, 3ul, 3ul};
    xt::xtensor<int32_t, 4> recv_count(shape);

    // initialize
    recv_count.fill(0);

    //
    // unpack header
    //
    for (int iz = indexlb[0]; iz <= indexub[0]; iz++) {
      for (int iy = indexlb[1]; iy <= indexub[1]; iy++) {
        for (int ix = indexlb[2]; ix <= indexub[2]; ix++) {
          // skip
          if (iz == 1 && iy == 1 && ix == 1)
            continue;

          // skip null message
          if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL)
            continue;

          int rcnt = 0;
          int addr = mpibuf->bufaddr(iz, iy, ix);
          for (int is = 0; is < Ns; is++) {
            std::memcpy(&rcnt, mpibuf->recvbuf.get(addr), head_byte);
            addr += head_byte + elem_byte * rcnt;
            recv_count(is, iz, iy, ix) = rcnt;
            recv_count(Ns, iz, iy, ix) += rcnt;
          }
        }
      }
    }

    //
    // resize particle buffer if needed
    //
    {
      const float64 target = 1 + chunk->get_buffer_ratio();

      for (int is = 0; is < Ns; is++) {
        int np_next = particle[is]->Np;
        for (int iz = indexlb[0]; iz <= indexub[0]; iz++) {
          for (int iy = indexlb[1]; iy <= indexub[1]; iy++) {
            for (int ix = indexlb[2]; ix <= indexub[2]; ix++) {
              np_next += recv_count(is, iz, iy, ix);
            }
          }
        }

        if (np_next > particle[is]->Np_total) {
          // expand
          particle[is]->resize(target * np_next);
        } else if (target * np_next < particle[is]->Np_total) {
          // shrink
          particle[is]->resize(target * np_next);
        }
      }
    }

    // number of unpacked particles
    num_unpacked.resize(Ns, 0);
  }

  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL)
      return false;

    //
    // copy to the end of particle array
    //
    uint8_t* recvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));
    int      recvcnt = mpibuf->bufsize(iz, iy, ix);

    // check message size
    if (recvcnt < Ns * head_byte) {
      ERROR << tfm::format("Received message smaller than the header size: %d", recvcnt);
      return false;
    }

    // unpack
    for (int is = 0; is < Ns; is++) {
      int Np = particle[is]->Np;

      // header
      int rcnt;
      std::memcpy(&rcnt, recvptr, head_byte);
      recvptr += head_byte;
      recvcnt -= head_byte;

      // particles
      float64* ptcl = &particle[is]->xu(Np + num_unpacked[is], 0);
      std::memcpy(ptcl, recvptr, elem_byte * rcnt);
      recvptr += rcnt * elem_byte;
      recvcnt -= rcnt * elem_byte;

      // increment number of unpacked particles
      num_unpacked[is] += rcnt;
    }

    // check consistency
    if (recvcnt != 0) {
      ERROR << tfm::format("Unexpected message perhaps with wrong headers?");
      return false;
    }

    return true;
  }

  template <typename BufferPtr>
  void post_unpack(BufferPtr& mpibuf, int indexlb[3], int indexub[3])
  {
    //
    // set boundary condition and append count for received particles
    //
    for (int is = 0; is < Ns; is++) {
      int np_prev = particle[is]->Np;
      int np_next = particle[is]->Np + num_unpacked[is];
      particle[is]->set_boundary_periodic(np_prev, np_next);
      particle[is]->count(np_prev, np_next - 1, false, chunk->get_order());
      // now update number of particles
      particle[is]->Np = np_next;
    }

    //
    // sort particle array and discard out-of-range particles
    //
    for (int is = 0; is < Ns; is++) {
      particle[is]->sort();
    }
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
