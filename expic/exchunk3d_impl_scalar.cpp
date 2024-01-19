// -*- C++ -*-

//
// Implementation for Scalar Version
//
// This file is to be included from exchunk3d.cpp
//

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Order>                                                                             \
  type ExChunk3D<Order>::name ## _impl_scalar

DEFINE_MEMBER(void, push_position)(const float64 delt)
{
  const float64 rc = 1 / cc;

  for (int is = 0; is < Ns; is++) {
    auto ps = up[is];

    // loop over particle
    auto& xu = ps->xu;
    auto& xv = ps->xv;
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 gam = lorentz_factor(xu(ip, 3), xu(ip, 4), xu(ip, 5), rc);
      float64 dt  = delt / gam;

      // substitute to temporary
      std::memcpy(&xv(ip, 0), &xu(ip, 0), ParticleType::get_particle_size());

      // update position
      xu(ip, 0) += xu(ip, 3) * dt;
      xu(ip, 1) += xu(ip, 4) * dt;
      xu(ip, 2) += xu(ip, 5) * dt;
    }

    // count
    this->count_particle(ps, 0, ps->Np - 1, true);
  }
}

DEFINE_MEMBER(void, push_velocity)(const float64 delt)
{
  constexpr int size   = Order + 2;
  constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  const float64 rc     = 1 / cc;
  const float64 rdx    = 1 / delx;
  const float64 rdy    = 1 / dely;
  const float64 rdz    = 1 / delz;
  const float64 dx2    = 0.5 * delx;
  const float64 dy2    = 0.5 * dely;
  const float64 dz2    = 0.5 * delz;
  const float64 ximin  = xlim[0] + dx2 * is_odd;
  const float64 xhmin  = xlim[0] + dx2 * is_odd - dx2;
  const float64 yimin  = ylim[0] + dy2 * is_odd;
  const float64 yhmin  = ylim[0] + dy2 * is_odd - dy2;
  const float64 zimin  = zlim[0] + dz2 * is_odd;
  const float64 zhmin  = zlim[0] + dz2 * is_odd - dz2;
  const float64 xigrid = ximin - dx2 * is_odd + dx2;
  const float64 xhgrid = xhmin - dx2 * is_odd + dx2;
  const float64 yigrid = yimin - dy2 * is_odd + dy2;
  const float64 yhgrid = yhmin - dy2 * is_odd + dy2;
  const float64 zigrid = zimin - dz2 * is_odd + dz2;
  const float64 zhgrid = zhmin - dz2 * is_odd + dz2;

  for (int is = 0; is < Ns; is++) {
    auto    ps  = up[is];
    float64 dt1 = 0.5 * ps->q / ps->m * delt;

    // loop over particle
    auto& xu = ps->xu;
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 wix[size] = {0};
      float64 whx[size] = {0};
      float64 wiy[size] = {0};
      float64 why[size] = {0};
      float64 wiz[size] = {0};
      float64 whz[size] = {0};

      float64 gam = lorentz_factor(xu(ip, 3), xu(ip, 4), xu(ip, 5), rc);
      float64 dt2 = dt1 * rc / gam;

      // grid indices
      int ix0 = digitize(xu(ip, 0), ximin, rdx);
      int hx0 = digitize(xu(ip, 0), xhmin, rdx);
      int iy0 = digitize(xu(ip, 1), yimin, rdy);
      int hy0 = digitize(xu(ip, 1), yhmin, rdy);
      int iz0 = digitize(xu(ip, 2), zimin, rdz);
      int hz0 = digitize(xu(ip, 2), zhmin, rdz);

      // weights
      shape<Order>(xu(ip, 0), xigrid + ix0 * delx, rdx, wix);
      shape<Order>(xu(ip, 0), xhgrid + hx0 * delx, rdx, whx);
      shape<Order>(xu(ip, 1), yigrid + iy0 * dely, rdy, wiy);
      shape<Order>(xu(ip, 1), yhgrid + hy0 * dely, rdy, why);
      shape<Order>(xu(ip, 2), zigrid + iz0 * delz, rdz, wiz);
      shape<Order>(xu(ip, 2), zhgrid + hz0 * delz, rdz, whz);

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
      ix0 += Lbx - (Order / 2);
      iy0 += Lby - (Order / 2);
      iz0 += Lbz - (Order / 2);
      hx0 += Lbx - (Order / 2);
      hy0 += Lby - (Order / 2);
      hz0 += Lbz - (Order / 2);

      float64 ex = interpolate3d<Order>(uf, iz0, iy0, hx0, 0, wiz, wiy, whx, dt1);
      float64 ey = interpolate3d<Order>(uf, iz0, hy0, ix0, 1, wiz, why, wix, dt1);
      float64 ez = interpolate3d<Order>(uf, hz0, iy0, ix0, 2, whz, wiy, wix, dt1);
      float64 bx = interpolate3d<Order>(uf, hz0, hy0, ix0, 3, whz, why, wix, dt2);
      float64 by = interpolate3d<Order>(uf, hz0, iy0, hx0, 4, whz, wiy, whx, dt2);
      float64 bz = interpolate3d<Order>(uf, iz0, hy0, hx0, 5, wiz, why, whx, dt2);

      // push particle velocity
      push_boris(xu(ip, 3), xu(ip, 4), xu(ip, 5), ex, ey, ez, bx, by, bz);
    }
  }
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  constexpr int size   = Order + 3;
  constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  const float64 rdx   = 1 / delx;
  const float64 rdy   = 1 / dely;
  const float64 rdz   = 1 / delz;
  const float64 dxdt  = delx / delt;
  const float64 dydt  = dely / delt;
  const float64 dzdt  = delz / delt;
  const float64 dx2   = 0.5 * delx;
  const float64 dy2   = 0.5 * dely;
  const float64 dz2   = 0.5 * delz;
  const float64 xmin  = xlim[0] + dx2 * is_odd;
  const float64 ymin  = ylim[0] + dy2 * is_odd;
  const float64 zmin  = zlim[0] + dz2 * is_odd;
  const float64 xgrid = xlim[0] + dx2;
  const float64 ygrid = ylim[0] + dy2;
  const float64 zgrid = zlim[0] + dz2;

  // clear charge/current density
  uj.fill(0);

  for (int is = 0; is < Ns; is++) {
    auto    ps = up[is];
    float64 qs = ps->q;

    // loop over particle
    auto& xu = ps->xu;
    auto& xv = ps->xv;
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 ss[2][3][size]           = {0};
      float64 cur[size][size][size][4] = {0};

      //
      // -*- weights before move -*-
      //
      // grid indices
      int ix0 = digitize(xv(ip, 0), xmin, rdx);
      int iy0 = digitize(xv(ip, 1), ymin, rdy);
      int iz0 = digitize(xv(ip, 2), zmin, rdz);

      // weights
      shape<Order>(xv(ip, 0), xgrid + ix0 * delx, rdx, &ss[0][0][1]);
      shape<Order>(xv(ip, 1), ygrid + iy0 * dely, rdy, &ss[0][1][1]);
      shape<Order>(xv(ip, 2), zgrid + iz0 * delz, rdz, &ss[0][2][1]);

      //
      // -*- weights after move -*-
      //
      // grid indices
      int ix1 = digitize(xu(ip, 0), xmin, rdx);
      int iy1 = digitize(xu(ip, 1), ymin, rdy);
      int iz1 = digitize(xu(ip, 2), zmin, rdz);

      // weights
      shape<Order>(xu(ip, 0), xgrid + ix1 * delx, rdx, &ss[1][0][1 + ix1 - ix0]);
      shape<Order>(xu(ip, 1), ygrid + iy1 * dely, rdy, &ss[1][1][1 + iy1 - iy0]);
      shape<Order>(xu(ip, 2), zgrid + iz1 * delz, rdz, &ss[1][2][1 + iz1 - iz0]);

      //
      // -*- accumulate current via density decomposition -*-
      //
      esirkepov3d<Order>(dxdt, dydt, dzdt, ss, cur);

      // deposit to global array
      ix0 += Lbx - (Order / 2) - 1;
      iy0 += Lby - (Order / 2) - 1;
      iz0 += Lbz - (Order / 2) - 1;
      append_current3d<Order>(uj, iz0, iy0, ix0, cur, qs);
    }
  }
}

DEFINE_MEMBER(void, deposit_moment)()
{
  constexpr int size   = Order + 1;
  constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  const float64 rc    = 1 / cc;
  const float64 rdx   = 1 / delx;
  const float64 rdy   = 1 / dely;
  const float64 rdz   = 1 / delz;
  const float64 dx2   = 0.5 * delx;
  const float64 dy2   = 0.5 * dely;
  const float64 dz2   = 0.5 * delz;
  const float64 xmin  = xlim[0] + dx2 * is_odd;
  const float64 ymin  = ylim[0] + dy2 * is_odd;
  const float64 zmin  = zlim[0] + dz2 * is_odd;
  const float64 xgrid = xlim[0] + dx2;
  const float64 ygrid = ylim[0] + dy2;
  const float64 zgrid = zlim[0] + dz2;

  // clear moment
  um.fill(0);

  for (int is = 0; is < Ns; is++) {
    auto    ps = up[is];
    float64 ms = ps->m;

    // loop over particle
    auto& xu = ps->xu;
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 wx[size]                  = {0};
      float64 wy[size]                  = {0};
      float64 wz[size]                  = {0};
      float64 mom[size][size][size][11] = {0};

      // grid indices
      int ix0 = digitize(xu(ip, 0), xmin, rdx);
      int iy0 = digitize(xu(ip, 1), ymin, rdy);
      int iz0 = digitize(xu(ip, 2), zmin, rdz);

      // weights
      shape<Order>(xu(ip, 0), xgrid + ix0 * delx, rdx, wx);
      shape<Order>(xu(ip, 1), ygrid + iy0 * dely, rdy, wy);
      shape<Order>(xu(ip, 2), zgrid + iz0 * delz, rdz, wz);

      // deposit to local array (this step is not necessary for scalar version)
      for (int jz = 0; jz < size; jz++) {
        for (int jy = 0; jy < size; jy++) {
          for (int jx = 0; jx < size; jx++) {
            float64 ww = ms * wz[jz] * wy[jy] * wx[jx];
            float64 gm = lorentz_factor(xu(ip, 3), xu(ip, 4), xu(ip, 5), rc);

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
            mom[jz][jy][jx][1]  = ww * xu(ip, 3);
            mom[jz][jy][jx][2]  = ww * xu(ip, 4);
            mom[jz][jy][jx][3]  = ww * xu(ip, 5);
            mom[jz][jy][jx][4]  = ww * gm * cc;
            mom[jz][jy][jx][5]  = ww * xu(ip, 3) * xu(ip, 3) / gm;
            mom[jz][jy][jx][6]  = ww * xu(ip, 4) * xu(ip, 4) / gm;
            mom[jz][jy][jx][7]  = ww * xu(ip, 5) * xu(ip, 5) / gm;
            mom[jz][jy][jx][8]  = ww * xu(ip, 3) * xu(ip, 4) / gm;
            mom[jz][jy][jx][9]  = ww * xu(ip, 3) * xu(ip, 5) / gm;
            mom[jz][jy][jx][10] = ww * xu(ip, 4) * xu(ip, 5) / gm;
          }
        }
      }

      // deposit to global array
      ix0 += Lbx - (Order / 2);
      iy0 += Lby - (Order / 2);
      iz0 += Lbz - (Order / 2);
      for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
        for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
          for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
            for (int k = 0; k < 11; k++) {
              um(iz, iy, ix, is, k) += mom[jz][jy][jx][k];
            }
          }
        }
      }
    }
  }
}

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
