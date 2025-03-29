// -*- C++ -*-
#ifndef _PARTICLE_HPP_
#define _PARTICLE_HPP_

#include "nix.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Particle
///
/// This class is a container for particles of single species. Member variables are intentionally
/// made public for enabling access from everywhere else.
///
class Particle
{
public:
  static constexpr int Nc = 7; ///< # component for each particle (including ID)

  int     Np_total; ///< # total particles
  int     Np;       ///< # particles in active
  int     Ng;       ///< # grid points
  float64 q;        ///< charge
  float64 m;        ///< mass

  bool    has_xdim;
  bool    has_ydim;
  bool    has_zdim;
  int     Lbx;
  int     Ubx;
  int     Lby;
  int     Uby;
  int     Lbz;
  int     Ubz;
  float64 delx;
  float64 dely;
  float64 delz;
  float64 xmin;
  float64 xmax;
  float64 ymin;
  float64 ymax;
  float64 zmin;
  float64 zmax;
  float64 xmin_global;
  float64 xmax_global;
  float64 ymin_global;
  float64 ymax_global;
  float64 zmin_global;
  float64 zmax_global;

  /// get single particle data size in bytes
  static constexpr int get_particle_size()
  {
    return Nc * sizeof(float64);
  }

  /// get number of total particles
  int get_Np_total() const
  {
    return Np_total;
  }

  /// get number of active particles
  int get_Np_active() const
  {
    return Np;
  }

  /// set number of active particles
  void set_Np_active(int Np)
  {
    this->Np = Np;
  }

  Particle() : Np_total(0), Np(0), Ng(0)
  {
  }

  template <typename T_chunk>
  Particle(int Np_total, T_chunk& chunk) : Np_total(Np_total), Np(0)
  {
    has_xdim                           = chunk.has_xdim();
    has_ydim                           = chunk.has_ydim();
    has_zdim                           = chunk.has_zdim();
    std::tie(Lbx, Ubx)                 = chunk.get_xbound();
    std::tie(Lby, Uby)                 = chunk.get_ybound();
    std::tie(Lbz, Ubz)                 = chunk.get_zbound();
    delx                               = chunk.get_delx();
    dely                               = chunk.get_dely();
    delz                               = chunk.get_delz();
    std::tie(xmin, xmax)               = chunk.get_xrange();
    std::tie(ymin, ymax)               = chunk.get_yrange();
    std::tie(zmin, zmax)               = chunk.get_zrange();
    std::tie(xmin_global, xmax_global) = chunk.get_xrange_global();
    std::tie(ymin_global, ymax_global) = chunk.get_yrange_global();
    std::tie(zmin_global, zmax_global) = chunk.get_zrange_global();

    // set grid number
    {
      int nb = chunk.get_boundary_margin();
      int nx = Ubx - Lbx + 2 * nb + 1;
      int ny = Uby - Lby + 2 * nb + 1;
      int nz = Ubz - Lbz + 2 * nb + 1;
      Ng     = nx * ny * nz;
    }
  }

  /// @brief return size in byte
  int64_t get_size_byte();

  /// @brief initial memory allocation
  void allocate(int Np_total, int Ng);

  /// @brief resize particle array
  void resize(int newsize);

  /// @brief swap pointer of particle array with temporary array
  void swap();

  /// @brief pack data into buffer
  int pack(void* buffer, int address);

  /// @brief unpack data from buffer
  int unpack(void* buffer, int address);

  /// reset particle count
  void reset_count();

  /// @brief return flat grid index
  int flatindex(int iz, int iy, int ix);

  /// @brief increment particle count
  void increment(int ip, int ii);

  /// @brief sort particle array
  void sort();

  /// @brief count particle in each cell
  void count(int Lbp, int Ubp, bool reset, int order = 1);

  /// @brief set periodic boundary condition
  void set_boundary_periodic(int Lbp, int Ubp);
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
