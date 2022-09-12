// -*- C++ -*-

//
// implementation specific to 1st-order shape function
//
#define DEFINE_MEMBER1(type, name)                                                                 \
  template <>                                                                                      \
  type ExChunk3D<1>::name

DEFINE_MEMBER1(void, push_velocity)(const float64 delt)
{
  const float64 rc    = 1 / cc;
  const float64 rdh   = 1 / delh;
  const float64 dh2   = 0.5 * delh;
  const float64 ximin = xlim[0] + dh2;
  const float64 xhmin = xlim[0];
  const float64 yimin = ylim[0] + dh2;
  const float64 yhmin = ylim[0];
  const float64 zimin = zlim[0] + dh2;
  const float64 zhmin = zlim[0];

  for (int is = 0; is < Ns; is++) {
    PtrParticle p = up[is];

    // loop over particle
    float64 dt1 = 0.5 * p->q / p->m * delt;
    for (int ip = 0; ip < p->Np; ip++) {
      float64 wix[2] = {0};
      float64 whx[2] = {0};
      float64 wiy[2] = {0};
      float64 why[2] = {0};
      float64 wiz[2] = {0};
      float64 whz[2] = {0};
      float64 emf[6] = {0};

      float64 *xu  = &p->xu(ip, 0);
      float64  gam = Particle::lorentz_factor(xu[3], xu[4], xu[5], rc);
      float64  dt2 = dt1 * rc / gam;

      // grid indices
      int ix = Particle::digitize(xu[0], ximin, rdh) + Lbx;
      int hx = Particle::digitize(xu[0], xhmin, rdh) + Lbx;
      int iy = Particle::digitize(xu[1], yimin, rdh) + Lby;
      int hy = Particle::digitize(xu[1], yhmin, rdh) + Lby;
      int iz = Particle::digitize(xu[2], zimin, rdh) + Lbz;
      int hz = Particle::digitize(xu[2], zhmin, rdh) + Lbz;

      // weights
      Particle::S1(xu[0], ximin + ix * delh, delh, wix);
      Particle::S1(xu[0], xhmin + hx * delh, delh, whx);
      Particle::S1(xu[1], yimin + iy * delh, delh, wiy);
      Particle::S1(xu[1], yhmin + hy * delh, delh, why);
      Particle::S1(xu[2], zimin + iz * delh, delh, wiz);
      Particle::S1(xu[2], zhmin + hz * delh, delh, whz);

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
      emf[0] = Particle::interp3d1(uf, iz, iy, hx, 0, wiz, wiy, whx, dt1);
      emf[1] = Particle::interp3d1(uf, iz, hy, ix, 1, wiz, why, wix, dt1);
      emf[2] = Particle::interp3d1(uf, hz, iy, ix, 2, whz, wiy, wix, dt1);
      emf[3] = Particle::interp3d1(uf, hz, hy, ix, 3, whz, why, wix, dt2);
      emf[4] = Particle::interp3d1(uf, hz, iy, hx, 4, whz, wiy, whx, dt2);
      emf[5] = Particle::interp3d1(uf, iz, hy, hx, 5, wiz, why, whx, dt2);

      // push particle velocity
      Particle::push_buneman_boris(&xu[3], emf);
    }
  }
}

DEFINE_MEMBER1(void, push_position)(const float64 delt)
{
  const float64 rc = 1 / cc;

  for (int is = 0; is < Ns; is++) {
    PtrParticle p = up[is];

    // loop over particle
    for (int ip = 0; ip < p->Np; ip++) {
      float64 *xu  = &p->xu(ip, 0);
      float64 *xv  = &p->xv(ip, 0);
      float64  gam = Particle::lorentz_factor(xu[3], xu[4], xu[5], rc);
      float64  dt  = delt / gam;

      // substitute to temporary
      std::memcpy(xv, xu, sizeof(float64) * Particle::Nc);

      // update position
      xu[0] += xu[3] * dt;
      xu[1] += xu[4] * dt;
      xu[2] += xu[5] * dt;
    }

    // count
    count_particle(p, 0, p->Np - 1, true);
  }
}

DEFINE_MEMBER1(void, deposit_current)(const float64 delt)
{
  const float64 rc    = 1 / cc;
  const float64 rdh   = 1 / delh;
  const float64 dh2   = 0.5 * delh;
  const float64 dhdt  = delh / delt;
  const float64 ximin = xlim[0] + dh2;
  const float64 yimin = ylim[0] + dh2;
  const float64 zimin = zlim[0] + dh2;

  // clear charge/current density
  uj.fill(0);

  for (int is = 0; is < Ns; is++) {
    PtrParticle p = up[is];

    // loop over particle
    for (int ip = 0; ip < p->Np; ip++) {
      float64 ss[2][3][4]     = {0};
      float64 cur[4][4][4][4] = {0};

      float64 *xv = &p->xv(ip, 0);
      float64 *xu = &p->xu(ip, 0);

      //
      // -*- weights before move -*-
      //
      // grid indices
      int ix0 = Particle::digitize(xv[0], ximin, rdh) + Lbx;
      int iy0 = Particle::digitize(xv[1], yimin, rdh) + Lby;
      int iz0 = Particle::digitize(xv[2], zimin, rdh) + Lbz;

      // weights
      Particle::S1(xv[0], ximin + ix0 * delh, delh, &ss[0][0][1]);
      Particle::S1(xv[1], yimin + iy0 * delh, delh, &ss[0][1][1]);
      Particle::S1(xv[2], zimin + iz0 * delh, delh, &ss[0][2][1]);

      //
      // -*- weights after move -*-
      //
      // grid indices
      int ix1 = Particle::digitize(xu[0], ximin, rdh) + Lbx;
      int iy1 = Particle::digitize(xu[1], yimin, rdh) + Lby;
      int iz1 = Particle::digitize(xu[2], zimin, rdh) + Lbz;

      // weights
      Particle::S1(xu[0], ximin + ix1 * delh, delh, &ss[1][0][1 + ix1 - ix0]);
      Particle::S1(xu[1], yimin + iy1 * delh, delh, &ss[1][1][1 + iy1 - iy0]);
      Particle::S1(xu[2], zimin + iz1 * delh, delh, &ss[1][2][1 + iz1 - iz0]);

      //
      // -*- accumulate current via density decomposition -*-
      //
      Particle::esirkepov3d1(dhdt, ss, cur);

      for (int jz = 0; jz < 4; jz++) {
        for (int jy = 0; jy < 4; jy++) {
          for (int jx = 0; jx < 4; jx++) {
            uj(iz0 + jz - 1, iy0 + jy - 1, ix0 + jx - 1, 0) += cur[jz][jy][jx][0];
            uj(iz0 + jz - 1, iy0 + jy - 1, ix0 + jx - 1, 1) += cur[jz][jy][jx][1];
            uj(iz0 + jz - 1, iy0 + jy - 1, ix0 + jx - 1, 2) += cur[jz][jy][jx][2];
            uj(iz0 + jz - 1, iy0 + jy - 1, ix0 + jx - 1, 3) += cur[jz][jy][jx][3];
          }
        }
      }
    }
  }
}
