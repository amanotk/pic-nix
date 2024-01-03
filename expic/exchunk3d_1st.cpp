// -*- C++ -*-
#include "exchunk3d.hpp"

//
// implementation specific to 1st-order shape function
//
#define DEFINE_MEMBER(type, name)                                                                  \
  template <>                                                                                      \
  type ExChunk3D<1>::name

using namespace nix::primitives;

DEFINE_MEMBER(void, push_velocity)(const float64 delt)
{
  const float64 rc    = 1 / cc;
  const float64 rdx   = 1 / delx;
  const float64 rdy   = 1 / dely;
  const float64 rdz   = 1 / delz;
  const float64 dx2   = 0.5 * delx;
  const float64 dy2   = 0.5 * dely;
  const float64 dz2   = 0.5 * delz;
  const float64 ximin = xlim[0] + dx2;
  const float64 xhmin = xlim[0];
  const float64 yimin = ylim[0] + dy2;
  const float64 yhmin = ylim[0];
  const float64 zimin = zlim[0] + dz2;
  const float64 zhmin = zlim[0];

  for (int is = 0; is < Ns; is++) {
    auto    ps  = up[is];
    float64 dt1 = 0.5 * ps->q / ps->m * delt;

    // loop over particle
    auto& xu = ps->xu;
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 wix[2] = {0};
      float64 whx[2] = {0};
      float64 wiy[2] = {0};
      float64 why[2] = {0};
      float64 wiz[2] = {0};
      float64 whz[2] = {0};
      float64 emf[6] = {0};

      float64 gam = lorentz_factor(xu(ip, 3), xu(ip, 4), xu(ip, 5), rc);
      float64 dt2 = dt1 * rc / gam;

      // grid indices
      int ix = digitize(xu(ip, 0), ximin, rdx);
      int hx = digitize(xu(ip, 0), xhmin, rdx);
      int iy = digitize(xu(ip, 1), yimin, rdy);
      int hy = digitize(xu(ip, 1), yhmin, rdy);
      int iz = digitize(xu(ip, 2), zimin, rdz);
      int hz = digitize(xu(ip, 2), zhmin, rdz);

      // weights
      shape<1>(xu(ip, 0), ximin + ix * delx, rdx, wix);
      shape<1>(xu(ip, 0), xhmin + hx * delx, rdx, whx);
      shape<1>(xu(ip, 1), yimin + iy * dely, rdy, wiy);
      shape<1>(xu(ip, 1), yhmin + hy * dely, rdy, why);
      shape<1>(xu(ip, 2), zimin + iz * delz, rdz, wiz);
      shape<1>(xu(ip, 2), zhmin + hz * delz, rdz, whz);

      //
      // calculate electromagnetic field at particle position
      //
      // * Ex => half-integer for x, full-integer for y, z
      // * Ey => half-integer for y, full-integer for z, x
      // * Ez => half-integer for z, full-integer for x, y
      // * Bx => half-integer for y, z, full-integer for x
      // * By => half-integer for z, x, full-integer for y
      // * Bz => half-integer for x, y, full-integer for z
      //
      ix += Lbx;
      iy += Lby;
      iz += Lbz;
      hx += Lbx;
      hy += Lby;
      hz += Lbz;
      emf[0] = interp3d1(uf, iz, iy, hx, 0, wiz, wiy, whx, dt1);
      emf[1] = interp3d1(uf, iz, hy, ix, 1, wiz, why, wix, dt1);
      emf[2] = interp3d1(uf, hz, iy, ix, 2, whz, wiy, wix, dt1);
      emf[3] = interp3d1(uf, hz, hy, ix, 3, whz, why, wix, dt2);
      emf[4] = interp3d1(uf, hz, iy, hx, 4, whz, wiy, whx, dt2);
      emf[5] = interp3d1(uf, iz, hy, hx, 5, wiz, why, whx, dt2);

      // push particle velocity
      push_buneman_boris(&xu(ip, 3), emf);
    }
  }
}

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
