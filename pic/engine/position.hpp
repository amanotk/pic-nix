// -*- C++ -*-
#ifndef _ENGINE_POSITION_HPP_
#define _ENGINE_POSITION_HPP_

#include "nix/nix.hpp"
#include "nix/particle.hpp"

namespace pic_engine
{
class BasePosition
{
public:
  int     ns;
  int     lbx;
  int     lby;
  int     lbz;
  int     ubx;
  int     uby;
  int     ubz;
  int     stride_x;
  int     stride_y;
  int     stride_z;
  float64 cc;
  float64 dx;
  float64 dy;
  float64 dz;
  float64 xmin;
  float64 ymin;
  float64 zmin;
  float64 xmax;
  float64 ymax;
  float64 zmax;

  template <typename T_data>
  BasePosition(const T_data& data)
  {
    ns       = data.Ns;
    cc       = data.cc;
    ns       = data.Ns;
    lbx      = data.Lbx;
    lby      = data.Lby;
    lbz      = data.Lbz;
    ubx      = data.Ubx;
    uby      = data.Uby;
    ubz      = data.Ubz;
    stride_x = 1;
    stride_y = stride_x * (data.Ubx - data.Lbx + 2);
    stride_z = stride_y * (data.Uby - data.Lby + 2);
    cc       = data.cc;
    dx       = data.delx;
    dy       = data.dely;
    dz       = data.delz;
    xmin     = data.xlim[0];
    ymin     = data.ylim[0];
    zmin     = data.zlim[0];
    xmax     = data.xlim[1];
    ymax     = data.ylim[1];
    zmax     = data.zlim[1];
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

class ScalarPosition : public BasePosition
{
public:
  using BasePosition::BasePosition; // inherit constructor

  template <typename T_particle>
  void operator()(const T_particle& up, const float64 delt)
  {
    this->call_scalar_impl(up, delt);
  }
};

class VectorPosition : public BasePosition
{
public:
  using BasePosition::BasePosition; // inherit constructor

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