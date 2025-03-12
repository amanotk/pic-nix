// -*- C++ -*-
#ifndef _ENGINE_POSITION_HPP_
#define _ENGINE_POSITION_HPP_

#include "nix/nix.hpp"
#include "nix/particle.hpp"

namespace engine
{
class Position
{
public:
  int     ns;
  float64 cc;

  template <typename T_data>
  Position(const T_data& data)
  {
    ns = data.Ns;
    cc = data.cc;
  }

  template <typename T_particle>
  void call_scalar_impl(const T_particle& up, const float64 delt)
  {
    float64 rc = 1 / cc;

    for (int is = 0; is < ns; is++) {
      for (int ip = 0; ip < up[is]->Np; ip++) {
        auto xu = &up[is]->xu(ip, 0);
        auto xv = &up[is]->xv(ip, 0);

        push(xv, xu, rc, delt);
      }
    }
  }

  template <typename T_particle>
  void call_vector_impl(const T_particle& up, const float64 delt)
  {
    const simd_i64 index = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

    float64 rc = 1 / cc;

    for (int is = 0; is < ns; is++) {
      int np_simd = (up[is]->Np / simd_f64::size) * simd_f64::size;

      //
      // vectorized loop
      //
      for (int ip = 0; ip < np_simd; ip += simd_f64::size) {
        // local SIMD register
        simd_f64 xu[ParticleType::Nc];
        simd_f64 xv[ParticleType::Nc];

        // load particles to SIMD register
        for (int i = 0; i < ParticleType::Nc; i++) {
          xu[i] = simd_f64::gather(&up[is]->xu(ip, i), index);
          xv[i] = simd_f64::gather(&up[is]->xv(ip, i), index);
        }

        push(xv, xu, simd_f64(rc), delt);

        // store particles to memory
        xu[0].scatter(&up[is]->xu(ip, 0), index);
        xu[1].scatter(&up[is]->xu(ip, 1), index);
        xu[2].scatter(&up[is]->xu(ip, 2), index);
        for (int i = 0; i < ParticleType::Nc; i++) {
          xv[i].scatter(&up[is]->xv(ip, i), index);
        }
      }

      //
      // scalar loop for reminder
      //
      for (int ip = np_simd; ip < up[is]->Np; ip++) {
        auto xu = &up[is]->xu(ip, 0);
        auto xv = &up[is]->xv(ip, 0);

        push(xv, xu, rc, delt);
      }
    }
  }

  template <typename T_float>
  void push(T_float xv[], T_float xu[], T_float rc, float64 delt)
  {
    // copy to temporary
    for (int i = 0; i < ParticleType::Nc; i++) {
      xv[i] = xu[i];
    }

    auto gm = lorentz_factor(xu[3], xu[4], xu[5], rc);
    auto dt = delt / gm;
    xu[0] += xu[3] * dt;
    xu[1] += xu[4] * dt;
    xu[2] += xu[5] * dt;
  }
};

class ScalarPosition : public Position
{
public:
  template <typename T_data>
  ScalarPosition(const T_data& data) : Position(data)
  {
  }

  template <typename T_particle>
  void operator()(const T_particle& up, const float64 delt)
  {
    this->call_scalar_impl(up, delt);
  }
};

class VectorPosition : public Position
{
public:
  template <typename T_data>
  VectorPosition(const T_data& data) : Position(data)
  {
  }

  template <typename T_particle>
  void operator()(const T_particle& up, const float64 delt)
  {
    this->call_vector_impl(up, delt);
  }
};

} // namespace engine

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif