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

  float64 etime = nix::wall_clock();

  for (int is = 0; is < Ns; is++) {
    PtrParticle ps  = up[is];
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

      float64 *xu  = &ps->xu(ip, 0);
      float64  gam = Particle::lorentz_factor(xu[3], xu[4], xu[5], rc);
      float64  dt2 = dt1 * rc / gam;

      // grid indices
      int ix = Particle::digitize(xu[0], ximin, rdh);
      int hx = Particle::digitize(xu[0], xhmin, rdh);
      int iy = Particle::digitize(xu[1], yimin, rdh);
      int hy = Particle::digitize(xu[1], yhmin, rdh);
      int iz = Particle::digitize(xu[2], zimin, rdh);
      int hz = Particle::digitize(xu[2], zhmin, rdh);

      // weights
      Particle::S1(xu[0], ximin + ix * delh, rdh, wix);
      Particle::S1(xu[0], xhmin + hx * delh, rdh, whx);
      Particle::S1(xu[1], yimin + iy * delh, rdh, wiy);
      Particle::S1(xu[1], yhmin + hy * delh, rdh, why);
      Particle::S1(xu[2], zimin + iz * delh, rdh, wiz);
      Particle::S1(xu[2], zhmin + hz * delh, rdh, whz);

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

  // store computation time
  this->load[LoadParticle] += nix::wall_clock() - etime;
}

DEFINE_MEMBER1(void, push_position)(const float64 delt)
{
  const float64 rc = 1 / cc;

  float64 etime = nix::wall_clock();

  for (int is = 0; is < Ns; is++) {
    PtrParticle ps = up[is];

    // loop over particle
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 *xu  = &ps->xu(ip, 0);
      float64 *xv  = &ps->xv(ip, 0);
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
    count_particle(ps, 0, ps->Np - 1, true);
  }

  // store computation time
  this->load[LoadParticle] += nix::wall_clock() - etime;
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

  float64 etime = nix::wall_clock();

  // clear charge/current density
  uj.fill(0);

  for (int is = 0; is < Ns; is++) {
    PtrParticle ps = up[is];
    float64     qs = ps->q;

    // loop over particle
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 ss[2][3][4]     = {0};
      float64 cur[4][4][4][4] = {0};

      float64 *xv = &ps->xv(ip, 0);
      float64 *xu = &ps->xu(ip, 0);

      //
      // -*- weights before move -*-
      //
      // grid indices
      int ix0 = Particle::digitize(xv[0], ximin, rdh);
      int iy0 = Particle::digitize(xv[1], yimin, rdh);
      int iz0 = Particle::digitize(xv[2], zimin, rdh);

      // weights
      Particle::S1(xv[0], ximin + ix0 * delh, rdh, &ss[0][0][1]);
      Particle::S1(xv[1], yimin + iy0 * delh, rdh, &ss[0][1][1]);
      Particle::S1(xv[2], zimin + iz0 * delh, rdh, &ss[0][2][1]);

      //
      // -*- weights after move -*-
      //
      // grid indices
      int ix1 = Particle::digitize(xu[0], ximin, rdh);
      int iy1 = Particle::digitize(xu[1], yimin, rdh);
      int iz1 = Particle::digitize(xu[2], zimin, rdh);

      // weights
      Particle::S1(xu[0], ximin + ix1 * delh, rdh, &ss[1][0][1 + ix1 - ix0]);
      Particle::S1(xu[1], yimin + iy1 * delh, rdh, &ss[1][1][1 + iy1 - iy0]);
      Particle::S1(xu[2], zimin + iz1 * delh, rdh, &ss[1][2][1 + iz1 - iz0]);

      //
      // -*- accumulate current via density decomposition -*-
      //
      Particle::esirkepov3d1(dhdt, ss, cur);

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

  // store computation time
  this->load[LoadCur] += nix::wall_clock() - etime;
}

DEFINE_MEMBER1(void, deposit_moment)()
{
  const float64 rc    = 1 / cc;
  const float64 rdh   = 1 / delh;
  const float64 dh2   = 0.5 * delh;
  const float64 ximin = xlim[0] + dh2;
  const float64 yimin = ylim[0] + dh2;
  const float64 zimin = zlim[0] + dh2;

  // clear moment
  um.fill(0);

  for (int is = 0; is < Ns; is++) {
    PtrParticle ps = up[is];
    float64     ms = ps->m;

    // loop over particle
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 wx[2]            = {0};
      float64 wy[2]            = {0};
      float64 wz[2]            = {0};
      float64 mom[2][2][2][10] = {0};

      float64 *xu = &ps->xu(ip, 0);

      // grid indices
      int ix = Particle::digitize(xu[0], ximin, rdh);
      int iy = Particle::digitize(xu[1], yimin, rdh);
      int iz = Particle::digitize(xu[2], zimin, rdh);

      // weights
      Particle::S1(xu[0], ximin + ix * delh, rdh, wx);
      Particle::S1(xu[1], yimin + iy * delh, rdh, wy);
      Particle::S1(xu[2], zimin + iz * delh, rdh, wz);

      // deposit to local array (this step is not necessary for scalar version)
      for (int jz = 0; jz < 2; jz++) {
        for (int jy = 0; jy < 2; jy++) {
          for (int jx = 0; jx < 2; jx++) {
            float64 ww = ms * wz[jz] * wy[jy] * wx[jx];

            // FIXME; requires relativistic correction
            mom[jz][jy][jx][0] = ww;
            mom[jz][jy][jx][1] = ww * xu[3];
            mom[jz][jy][jx][2] = ww * xu[4];
            mom[jz][jy][jx][3] = ww * xu[5];
            mom[jz][jy][jx][4] = ww * xu[3] * xu[3];
            mom[jz][jy][jx][5] = ww * xu[4] * xu[4];
            mom[jz][jy][jx][6] = ww * xu[5] * xu[5];
            mom[jz][jy][jx][7] = ww * xu[3] * xu[4];
            mom[jz][jy][jx][8] = ww * xu[3] * xu[5];
            mom[jz][jy][jx][9] = ww * xu[4] * xu[5];
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
            for (int k = 0; k < 10; k++) {
              um(iz + jz, iy + jy, ix + jx, is, k) += mom[jz][jy][jx][k];
            }
          }
        }
      }
    }
  }
}
