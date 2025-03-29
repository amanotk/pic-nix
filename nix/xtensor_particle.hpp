// -*- C++ -*-
#ifndef _XTENSOR_PARTICLE_HPP_
#define _XTENSOR_PARTICLE_HPP_

#include "nix.hpp"
#include "particle.hpp"
#include "primitives.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

class XtensorParticle : public Particle
{
public:
  xt::xtensor<float64, 2> xu;     ///< particle array
  xt::xtensor<float64, 2> xv;     ///< temporary particle array
  xt::xtensor<int32, 1>   gindex; ///< index to grid for each particle
  xt::xtensor<int32, 1>   pindex; ///< index to first particle for each cell
  xt::xtensor<int32, 2>   pcount; ///< particle count for each cell

  XtensorParticle() : Particle()
  {
  }

  template <typename T_chunk>
  XtensorParticle(int Np_total, T_chunk& chunk) : Particle(Np_total, chunk)
  {
    allocate(Np_total, Ng);
  }

  ///
  /// @brief return size in byte
  /// @return size in byte
  ///
  int64_t get_size_byte()
  {
    int64_t size = 0;
    size += xu.size() * sizeof(float64);
    size += xv.size() * sizeof(float64);
    size += gindex.size() * sizeof(int32);
    size += pindex.size() * sizeof(int32);
    size += pcount.size() * sizeof(int32);
    return size;
  }

  ///
  /// @brief initial memory allocation
  ///
  void allocate(int Np_total, int Ng)
  {
    const size_t np = Np_total;
    const size_t ng = Ng;
    const size_t nc = Nc;

    xu.resize({np, nc});
    xv.resize({np, nc});
    gindex.resize({np});
    pindex.resize({ng + 1});
    pcount.resize({ng + 1, nix::simd_width});

    xu.fill(0);
    xv.fill(0);
    gindex.fill(0);
    pindex.fill(0);
    pcount.fill(0);
  }

  ///
  /// @brief resize particle array
  ///
  void resize(int newsize)
  {
    const size_t np = newsize;
    const size_t nc = Nc;

    //
    // Resize should not be performed either of the following conditions are met:
    //
    // (1) newsize is equal to the original (no need to resize)
    // (2) newsize is smaller than the current active particle number (erroneous!)
    //
    if (newsize == Np_total || newsize <= Np) {
      return;
    }

    //
    // The following implementation of resize is not ideal as it requires copy of buffer twice, one
    // from the original to the temporary and another from the temporary to the resized buffer.
    //

    {
      const size_t size = std::min(xu.size(), np * nc) * sizeof(float64);
      auto         tmp(xu);

      xu.resize({np, nc});
      std::memcpy(xu.data(), tmp.data(), size);
    }

    {
      const size_t size = std::min(xv.size(), np * nc) * sizeof(float64);
      auto         tmp(xv);

      xv.resize({np, nc});
      std::memcpy(xv.data(), tmp.data(), size);
    }

    {
      const size_t size = std::min(gindex.size(), np) * sizeof(int32);
      auto         tmp(gindex);

      gindex.resize({np});
      std::memcpy(gindex.data(), tmp.data(), size);
    }

    // set new total number of particles
    Np_total = newsize;
  }

  ///
  /// @brief swap pointer of particle array with temporary array
  ///
  void swap()
  {
    xu.storage().swap(xv.storage());
  }

  ///
  /// @brief pack data into buffer
  ///
  int pack(void* buffer, int address)
  {
    // scalar
    address += memcpy_count(buffer, &Np_total, sizeof(int), address, 0);
    address += memcpy_count(buffer, &Np, sizeof(int), address, 0);
    address += memcpy_count(buffer, &Ng, sizeof(int), address, 0);
    address += memcpy_count(buffer, &q, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &m, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &has_xdim, sizeof(bool), address, 0);
    address += memcpy_count(buffer, &has_ydim, sizeof(bool), address, 0);
    address += memcpy_count(buffer, &has_zdim, sizeof(bool), address, 0);
    address += memcpy_count(buffer, &Lbx, sizeof(int), address, 0);
    address += memcpy_count(buffer, &Ubx, sizeof(int), address, 0);
    address += memcpy_count(buffer, &Lby, sizeof(int), address, 0);
    address += memcpy_count(buffer, &Uby, sizeof(int), address, 0);
    address += memcpy_count(buffer, &Lbz, sizeof(int), address, 0);
    address += memcpy_count(buffer, &Ubz, sizeof(int), address, 0);
    address += memcpy_count(buffer, &delx, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &dely, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &delz, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &xmin, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &xmax, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &ymin, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &ymax, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &zmin, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &zmax, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &xmin_global, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &xmax_global, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &ymin_global, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &ymax_global, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &zmin_global, sizeof(float64), address, 0);
    address += memcpy_count(buffer, &zmax_global, sizeof(float64), address, 0);

    // array
    address += memcpy_count(buffer, xu.data(), xu.size() * sizeof(float64), address, 0);
    address += memcpy_count(buffer, xv.data(), xv.size() * sizeof(float64), address, 0);
    address += memcpy_count(buffer, gindex.data(), gindex.size() * sizeof(int), address, 0);
    address += memcpy_count(buffer, pindex.data(), pindex.size() * sizeof(int), address, 0);
    address += memcpy_count(buffer, pcount.data(), pcount.size() * sizeof(int), address, 0);

    return address;
  }

  ///
  /// @brief unpack data from buffer
  ///
  int unpack(void* buffer, int address)
  {
    // scalar
    address += memcpy_count(&Np_total, buffer, sizeof(int), 0, address);
    address += memcpy_count(&Np, buffer, sizeof(int), 0, address);
    address += memcpy_count(&Ng, buffer, sizeof(int), 0, address);
    address += memcpy_count(&q, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&m, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&has_xdim, buffer, sizeof(bool), 0, address);
    address += memcpy_count(&has_ydim, buffer, sizeof(bool), 0, address);
    address += memcpy_count(&has_zdim, buffer, sizeof(bool), 0, address);
    address += memcpy_count(&Lbx, buffer, sizeof(int), 0, address);
    address += memcpy_count(&Ubx, buffer, sizeof(int), 0, address);
    address += memcpy_count(&Lby, buffer, sizeof(int), 0, address);
    address += memcpy_count(&Uby, buffer, sizeof(int), 0, address);
    address += memcpy_count(&Lbz, buffer, sizeof(int), 0, address);
    address += memcpy_count(&Ubz, buffer, sizeof(int), 0, address);
    address += memcpy_count(&delx, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&dely, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&delz, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&xmin, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&xmax, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&ymin, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&ymax, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&zmin, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&zmax, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&xmin_global, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&xmax_global, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&ymin_global, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&ymax_global, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&zmin_global, buffer, sizeof(float64), 0, address);
    address += memcpy_count(&zmax_global, buffer, sizeof(float64), 0, address);

    // memory allocation before reading arrays
    allocate(Np_total, Ng);

    // array
    address += memcpy_count(xu.data(), buffer, xu.size() * sizeof(float64), 0, address);
    address += memcpy_count(xv.data(), buffer, xv.size() * sizeof(float64), 0, address);
    address += memcpy_count(gindex.data(), buffer, gindex.size() * sizeof(int), 0, address);
    address += memcpy_count(pindex.data(), buffer, pindex.size() * sizeof(int), 0, address);
    address += memcpy_count(pcount.data(), buffer, pcount.size() * sizeof(int), 0, address);

    return address;
  }

  ///
  /// reset particle count
  ///
  void reset_count()
  {
    pcount.fill(0);
  }

  ///
  /// @brief return flat grid index
  ///
  int flatindex(int iz, int iy, int ix)
  {
    const int stride_x = 1;
    const int stride_y = stride_x * (Ubx - Lbx + 2);
    const int stride_z = stride_y * (Uby - Lby + 2);

    return iz * stride_z + iy * stride_y + ix * stride_x;
  }

  ///
  /// @brief increment particle count
  ///
  /// @param[in] ip particle index
  /// @param[in] ii grid index (where the particle resides)
  ///
  void increment(int ip, int ii)
  {
    int jj = ip % nix::simd_width;

    gindex(ip) = ii;
    pcount(ii, jj)++;
  }

  ///
  /// @brief sort particle array
  ///
  /// This routine performs counting sort of particles using pre-computed pcount and gindex.
  /// Out-of-bounds particles are eliminated.
  ///
  void sort()
  {
    //
    // cumulative sum of particle count
    //
    for (int ii = 0; ii < Ng + 1; ii++) {
      for (int jj = 0; jj < nix::simd_width - 1; jj++) {
        pcount(ii, jj + 1) += pcount(ii, jj);
      }
    }
    for (int ii = 0; ii < Ng; ii++) {
      for (int jj = 0; jj < nix::simd_width; jj++) {
        pcount(ii + 1, jj) += pcount(ii, nix::simd_width - 1);
      }
    }

    //
    // first particle index for each cell
    //
    pindex(0) = 0;
    for (int ii = 0; ii < Ng; ii++) {
      pindex(ii + 1) = pcount(ii, nix::simd_width - 1);
    }

    //
    // particle address for rearrangement
    //
    for (int ii = 0; ii < Ng + 1; ii++) {
      for (int jj = nix::simd_width - 1; jj > 0; jj--) {
        pcount(ii, jj) = pcount(ii, jj - 1);
      }
    }
    for (int ii = 0; ii < Ng + 1; ii++) {
      pcount(ii, 0) = pindex(ii);
    }

    //
    // rearrange particles
    //
    {
      float64* v = xv.data();
      float64* u = xu.data();

      for (int ip = 0; ip < Np; ip++) {
        int ii = gindex(ip);
        int jj = ip % nix::simd_width;
        int jp = pcount(ii, jj);

        // copy particle to new address
        std::memcpy(&v[Nc * jp], &u[Nc * ip], Nc * sizeof(float64));

        // increment address
        pcount(ii, jj)++;
      }
    }

    // swap two particle arrays
    swap();

    // particles contained in the last index are discarded
    Np = pindex(Ng);
  }

  // count particles in each cell
  void count(int Lbp, int Ubp, bool reset, int order = 1)
  {
    using nix::primitives::digitize;

    // notice the half-grid offset of cell boundaries for odd-order shape functions
    const int is_odd        = (order % 2 == 1) ? 1 : 0;
    const int out_of_bounds = Ng;

    const float64 xoffset = xmin - 0.5 * delx * is_odd;
    const float64 yoffset = ymin - 0.5 * dely * is_odd;
    const float64 zoffset = zmin - 0.5 * delz * is_odd;
    const float64 rdx     = 1 / delx;
    const float64 rdy     = 1 / dely;
    const float64 rdz     = 1 / delz;

    // reset count
    if (reset) {
      reset_count();
    }

    for (int ip = Lbp; ip <= Ubp; ip++) {
      int ix = has_xdim ? digitize(xu(ip, 0), xoffset, rdx) : 0;
      int iy = has_ydim ? digitize(xu(ip, 1), yoffset, rdy) : 0;
      int iz = has_zdim ? digitize(xu(ip, 2), zoffset, rdz) : 0;
      int ii = flatindex(iz, iy, ix);

      // take care out-of-bounds particles
      ii = has_xdim && (xu(ip, 0) < xmin || xu(ip, 0) >= xmax) ? out_of_bounds : ii;
      ii = has_ydim && (xu(ip, 1) < ymin || xu(ip, 1) >= ymax) ? out_of_bounds : ii;
      ii = has_zdim && (xu(ip, 2) < zmin || xu(ip, 2) >= zmax) ? out_of_bounds : ii;

      increment(ip, ii);
    }
  }

  void set_boundary_periodic(int Lbp, int Ubp)
  {
    const float64 X1 = xmin_global;
    const float64 X2 = xmax_global;
    const float64 Y1 = ymin_global;
    const float64 Y2 = ymax_global;
    const float64 Z1 = zmin_global;
    const float64 Z2 = zmax_global;
    const float64 X  = has_xdim * (X2 - X1);
    const float64 Y  = has_ydim * (Y2 - Y1);
    const float64 Z  = has_zdim * (Z2 - Z1);

    for (int ip = Lbp; ip <= Ubp; ip++) {
      xu(ip, 0) += (xu(ip, 0) < X1) * X - (xu(ip, 0) >= X2) * X;
      xu(ip, 1) += (xu(ip, 1) < Y1) * Y - (xu(ip, 1) >= Y2) * Y;
      xu(ip, 2) += (xu(ip, 2) < Z1) * Z - (xu(ip, 2) >= Z2) * Z;
    }
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
