// -*- C++ -*-
#include "exchunk3d.hpp"

//
// implementation specific to 1st-order shape function
//
#define DEFINE_MEMBER(type, name)                                                                  \
  template <>                                                                                      \
  type ExChunk3D<1>::name

using nix::digitize;
using nix::lorentz_factor;
using nix::push_buneman_boris;
using nix::shape1;
using nix::interp3d1;
using nix::esirkepov3d1;

DEFINE_MEMBER(void, push_velocity)(const float64 delt)
{
  const float64 rc    = 1 / cc;
  const float64 rdx   = 1 / delx;
  const float64 rdy   = 1 / dely;
  const float64 rdz   = 1 / delz;
  const float64 ximin = xlim[0] + 0.5 * delx;
  const float64 xhmin = xlim[0];
  const float64 yimin = ylim[0] + 0.5 * dely;
  const float64 yhmin = ylim[0];
  const float64 zimin = zlim[0] + 0.5 * delz;
  const float64 zhmin = zlim[0];

  for (int is = 0; is < Ns; is++) {
    ParticlePtr ps  = up[is];
    float64     dt1 = 0.5 * ps->q / ps->m * delt;

    // loop over particle
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 wix[2] = {0};
      float64 whx[2] = {0};
      float64 wiy[2] = {0};
      float64 why[2] = {0};
      float64 wiz[2] = {0};
      float64 whz[2] = {0};
      float64 emf[6] = {0};

      float64* xu  = &ps->xu(ip, 0);
      float64  gam = lorentz_factor(xu[3], xu[4], xu[5], rc);
      float64  dt2 = dt1 * rc / gam;

      // grid indices
      int ix = digitize(xu[0], ximin, rdx);
      int hx = digitize(xu[0], xhmin, rdx);
      int iy = digitize(xu[1], yimin, rdy);
      int hy = digitize(xu[1], yhmin, rdy);
      int iz = digitize(xu[2], zimin, rdz);
      int hz = digitize(xu[2], zhmin, rdz);

      // weights
      shape1(xu[0], ximin + ix * delx, rdx, wix);
      shape1(xu[0], xhmin + hx * delx, rdx, whx);
      shape1(xu[1], yimin + iy * dely, rdy, wiy);
      shape1(xu[1], yhmin + hy * dely, rdy, why);
      shape1(xu[2], zimin + iz * delz, rdz, wiz);
      shape1(xu[2], zhmin + hz * delz, rdz, whz);

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
      push_buneman_boris(&xu[3], emf);
    }
  }
}

DEFINE_MEMBER(void, push_position)(const float64 delt)
{
  const float64 rc = 1 / cc;

  for (int is = 0; is < Ns; is++) {
    ParticlePtr ps = up[is];

    // loop over particle
    for (int ip = 0; ip < ps->Np; ip++) {
      float64* xu  = &ps->xu(ip, 0);
      float64* xv  = &ps->xv(ip, 0);
      float64  gam = lorentz_factor(xu[3], xu[4], xu[5], rc);
      float64  dt  = delt / gam;

      // substitute to temporary
      std::memcpy(xv, xu, sizeof(float64) * Particle::Nc);

      // update position
      xu[0] += xu[3] * dt;
      xu[1] += xu[4] * dt;
      xu[2] += xu[5] * dt;
    }

    // count
    count_particle(ps, 0, ps->Np - 1, true);
  }
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  const float64 rdx   = 1 / delx;
  const float64 rdy   = 1 / dely;
  const float64 rdz   = 1 / delz;
  const float64 dxdt  = delx / delt;
  const float64 dydt  = dely / delt;
  const float64 dzdt  = delz / delt;
  const float64 ximin = xlim[0] + 0.5 * delx;
  const float64 yimin = ylim[0] + 0.5 * dely;
  const float64 zimin = zlim[0] + 0.5 * delz;

  // clear charge/current density
  uj.fill(0);

  for (int is = 0; is < Ns; is++) {
    ParticlePtr ps = up[is];
    float64     qs = ps->q;

    // loop over particle
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 ss[2][3][4]     = {0};
      float64 cur[4][4][4][4] = {0};

      float64* xv = &ps->xv(ip, 0);
      float64* xu = &ps->xu(ip, 0);

      //
      // -*- weights before move -*-
      //
      // grid indices
      int ix0 = digitize(xv[0], ximin, rdx);
      int iy0 = digitize(xv[1], yimin, rdy);
      int iz0 = digitize(xv[2], zimin, rdz);

      // weights
      shape1(xv[0], ximin + ix0 * delx, rdx, &ss[0][0][1]);
      shape1(xv[1], yimin + iy0 * dely, rdy, &ss[0][1][1]);
      shape1(xv[2], zimin + iz0 * delz, rdz, &ss[0][2][1]);

      //
      // -*- weights after move -*-
      //
      // grid indices
      int ix1 = digitize(xu[0], ximin, rdx);
      int iy1 = digitize(xu[1], yimin, rdy);
      int iz1 = digitize(xu[2], zimin, rdz);

      // weights
      shape1(xu[0], ximin + ix1 * delx, rdx, &ss[1][0][1 + ix1 - ix0]);
      shape1(xu[1], yimin + iy1 * dely, rdy, &ss[1][1][1 + iy1 - iy0]);
      shape1(xu[2], zimin + iz1 * delz, rdz, &ss[1][2][1 + iz1 - iz0]);

      //
      // -*- accumulate current via density decomposition -*-
      //
      esirkepov3d1(dxdt, dydt, dzdt, ss, cur);

      ix0 += Lbx;
      iy0 += Lby;
      iz0 += Lbz;
      for (int jz = 0; jz < 4; jz++) {
        for (int jy = 0; jy < 4; jy++) {
          for (int jx = 0; jx < 4; jx++) {
            uj(iz0 + jz - 1, iy0 + jy - 1, ix0 + jx - 1, 0) += qs * cur[jz][jy][jx][0];
            uj(iz0 + jz - 1, iy0 + jy - 1, ix0 + jx - 1, 1) += qs * cur[jz][jy][jx][1];
            uj(iz0 + jz - 1, iy0 + jy - 1, ix0 + jx - 1, 2) += qs * cur[jz][jy][jx][2];
            uj(iz0 + jz - 1, iy0 + jy - 1, ix0 + jx - 1, 3) += qs * cur[jz][jy][jx][3];
          }
        }
      }
    }
  }
}

DEFINE_MEMBER(void, deposit_moment)()
{
  //
  // This computes moment quantities for each grid points. Computed moments are
  // the mass density and the relativistic energy-momentum in laboratory frame.
  //
  const float64 rc    = 1 / cc;
  const float64 rdx   = 1 / delx;
  const float64 rdy   = 1 / dely;
  const float64 rdz   = 1 / delz;
  const float64 ximin = xlim[0] + 0.5 * delx;
  const float64 yimin = ylim[0] + 0.5 * dely;
  const float64 zimin = zlim[0] + 0.5 * delz;

  // clear moment
  um.fill(0);

  for (int is = 0; is < Ns; is++) {
    ParticlePtr ps = up[is];
    float64     ms = ps->m;

    // loop over particle
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 wx[2]            = {0};
      float64 wy[2]            = {0};
      float64 wz[2]            = {0};
      float64 mom[2][2][2][11] = {0};

      float64* xu = &ps->xu(ip, 0);

      // grid indices
      int ix = digitize(xu[0], ximin, rdx);
      int iy = digitize(xu[1], yimin, rdy);
      int iz = digitize(xu[2], zimin, rdz);

      // weights
      shape1(xu[0], ximin + ix * delx, rdx, wx);
      shape1(xu[1], yimin + iy * dely, rdy, wy);
      shape1(xu[2], zimin + iz * delz, rdz, wz);

      // deposit to local array (this step is not necessary for scalar version)
      for (int jz = 0; jz < 2; jz++) {
        for (int jy = 0; jy < 2; jy++) {
          for (int jx = 0; jx < 2; jx++) {
            float64 ww = ms * wz[jz] * wy[jy] * wx[jx];
            float64 gm = lorentz_factor(xu[3], xu[4], xu[5], rc);

            //
            //  0: mass density
            //  1: tx-component T^{0,1} (x-component of momentum density)
            //  2: ty-component T^{0,2} (y-component of momentum density)
            //  3: tz-component T^{0,3} (z-component of momentum density)
            //  4: tt-component T^{0,0} (energy density / c)
            //  5: xx-component T^{1,1}
            //  6: yy-component T^{2,2}
            //  7: zz-component T^{3,3}
            //  8: xy-component T^{1,2}
            //  9: xz-component T^{1,3}
            // 10: yz-component T^{2,3}
            //
            mom[jz][jy][jx][0]  = ww;
            mom[jz][jy][jx][1]  = ww * xu[3];
            mom[jz][jy][jx][2]  = ww * xu[4];
            mom[jz][jy][jx][3]  = ww * xu[5];
            mom[jz][jy][jx][4]  = ww * gm * cc;
            mom[jz][jy][jx][5]  = ww * xu[3] * xu[3] / gm;
            mom[jz][jy][jx][6]  = ww * xu[4] * xu[4] / gm;
            mom[jz][jy][jx][7]  = ww * xu[5] * xu[5] / gm;
            mom[jz][jy][jx][8]  = ww * xu[3] * xu[4] / gm;
            mom[jz][jy][jx][9]  = ww * xu[3] * xu[5] / gm;
            mom[jz][jy][jx][10] = ww * xu[4] * xu[5] / gm;
          }
        }
      }

      // deposit to global array
      ix += Lbx;
      iy += Lby;
      iz += Lbz;
      for (int jz = 0; jz < 2; jz++) {
        for (int jy = 0; jy < 2; jy++) {
          for (int jx = 0; jx < 2; jx++) {
            for (int k = 0; k < 11; k++) {
              um(iz + jz, iy + jy, ix + jx, is, k) += mom[jz][jy][jx][k];
            }
          }
        }
      }
    }
  }
}

template class ExChunk3D<1>;

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
