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
template <int N_component>
class Particle
{
public:
  static constexpr int Nc = N_component; ///< # component for each particle (including ID)

  int     Np_total; ///< # total particles
  int     Np;       ///< # particles in active
  int     Ng;       ///< # grid points (product in x, y, z dirs)
  float64 q;        ///< charge
  float64 m;        ///< mass

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

  /// @brief Default constructor
  Particle() : Np_total(0), Ng(0)
  {
  }

  /// @brief Constructor
  Particle(int Np_total, int Ng);

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

  /// @brief increment particle count
  void increment(int ip, int ii);

  /// @brief sort particle array
  void sort();
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
