// -*- C++ -*-
#ifndef _XTENSOR_PARTICLE_HPP_
#define _XTENSOR_PARTICLE_HPP_

#include "nix.hpp"
#include "particle.hpp"
#include "primitives.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

template <int N_component>
class XtensorParticle : public Particle<N_component>
{
public:
  using Particle<N_component>::Particle; // inherit constructor
  using BaseParticle = Particle<N_component>;
  using BaseParticle::Nc;
  using BaseParticle::Np_total;
  using BaseParticle::Np;
  using BaseParticle::Ng;
  using BaseParticle::q;
  using BaseParticle::m;

  static constexpr int simd_width = nix_simd_width;

  xt::xtensor<float64, 2> xu;     ///< particle array
  xt::xtensor<float64, 2> xv;     ///< temporary particle array
  xt::xtensor<int32, 1>   gindex; ///< index to grid for each particle
  xt::xtensor<int32, 1>   pindex; ///< index to first particle for each cell
  xt::xtensor<int32, 2>   pcount; ///< particle count for each cell

  ///
  /// @brief Constructor
  ///
  /// @param[in] Np_total Total number of particle for memory allocation
  /// @param[in] Ng       Number of grid
  ///
  XtensorParticle(int Np_total, int Ng)
  {
    this->Np_total = Np_total;
    this->Ng       = Ng;
    Np             = 0;

    // allocate and initialize arrays
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
    const size_t ns = simd_width;

    xu.resize({np, nc});
    xv.resize({np, nc});
    gindex.resize({np});
    pindex.resize({ng + 1});
    pcount.resize({ng + 1, ns});

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
    const size_t np     = newsize;
    const size_t nc     = Nc;

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
    int count = address;

    count += memcpy_count(buffer, &Np_total, sizeof(int), count, 0);
    count += memcpy_count(buffer, &Np, sizeof(int), count, 0);
    count += memcpy_count(buffer, &Ng, sizeof(int), count, 0);
    count += memcpy_count(buffer, &q, sizeof(float64), count, 0);
    count += memcpy_count(buffer, &m, sizeof(float64), count, 0);
    count += memcpy_count(buffer, xu.data(), xu.size() * sizeof(float64), count, 0);
    count += memcpy_count(buffer, xv.data(), xv.size() * sizeof(float64), count, 0);
    count += memcpy_count(buffer, gindex.data(), gindex.size() * sizeof(int), count, 0);
    count += memcpy_count(buffer, pindex.data(), pindex.size() * sizeof(int), count, 0);
    count += memcpy_count(buffer, pcount.data(), pcount.size() * sizeof(int), count, 0);

    return count;
  }

  ///
  /// @brief unpack data from buffer
  ///
  int unpack(void* buffer, int address)
  {
    int count = address;

    count += memcpy_count(&Np_total, buffer, sizeof(int), 0, count);
    count += memcpy_count(&Np, buffer, sizeof(int), 0, count);
    count += memcpy_count(&Ng, buffer, sizeof(int), 0, count);
    count += memcpy_count(&q, buffer, sizeof(float64), 0, count);
    count += memcpy_count(&m, buffer, sizeof(float64), 0, count);

    // memory allocation before reading arrays
    allocate(Np_total, Ng);

    count += memcpy_count(xu.data(), buffer, xu.size() * sizeof(float64), 0, count);
    count += memcpy_count(xv.data(), buffer, xv.size() * sizeof(float64), 0, count);
    count += memcpy_count(gindex.data(), buffer, gindex.size() * sizeof(int), 0, count);
    count += memcpy_count(pindex.data(), buffer, pindex.size() * sizeof(int), 0, count);
    count += memcpy_count(pcount.data(), buffer, pcount.size() * sizeof(int), 0, count);

    return count;
  }

  ///
  /// reset particle count
  ///
  void reset_count()
  {
    pcount.fill(0);
  }

  ///
  /// @brief increment particle count
  ///
  /// @param[in] ip particle index
  /// @param[in] ii grid index (where the particle resides)
  ///
  void increment(int ip, int ii)
  {
    int jj = ip % simd_width;

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
      for (int jj = 0; jj < simd_width - 1; jj++) {
        pcount(ii, jj + 1) += pcount(ii, jj);
      }
    }
    for (int ii = 0; ii < Ng; ii++) {
      for (int jj = 0; jj < simd_width; jj++) {
        pcount(ii + 1, jj) += pcount(ii, simd_width - 1);
      }
    }

    //
    // first particle index for each cell
    //
    pindex(0) = 0;
    for (int ii = 0; ii < Ng; ii++) {
      pindex(ii + 1) = pcount(ii, simd_width - 1);
    }

    //
    // particle address for rearrangement
    //
    for (int ii = 0; ii < Ng + 1; ii++) {
      for (int jj = simd_width - 1; jj > 0; jj--) {
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
        int jj = ip % simd_width;
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
};

using ParticlePtr = std::shared_ptr<XtensorParticle<7>>;
using ParticleVec = std::vector<ParticlePtr>;

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
