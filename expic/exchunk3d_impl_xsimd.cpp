// -*- C++ -*-

//
// Implementation for Vectorized Version with xsimd
//
// This file is to be included from exchunk3d.cpp
//

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Order>                                                                             \
  type ExChunk3D<Order>::name##_impl_xsimd

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
  using simd::simd_f64;
  using simd::simd_i64;
  constexpr int size   = Order + 2;
  constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  const float64  rc     = 1 / cc;
  const float64  rdx    = 1 / delx;
  const float64  rdy    = 1 / dely;
  const float64  rdz    = 1 / delz;
  const float64  ximin  = xlim[0] + 0.5 * delx * is_odd;
  const float64  xhmin  = xlim[0] + 0.5 * delx * is_odd - 0.5 * delx;
  const float64  yimin  = ylim[0] + 0.5 * dely * is_odd;
  const float64  yhmin  = ylim[0] + 0.5 * dely * is_odd - 0.5 * dely;
  const float64  zimin  = zlim[0] + 0.5 * delz * is_odd;
  const float64  zhmin  = zlim[0] + 0.5 * delz * is_odd - 0.5 * delz;
  const float64  xigrid = ximin - 0.5 * delx * is_odd + 0.5 * delx;
  const float64  xhgrid = xhmin - 0.5 * delx * is_odd + 0.5 * delx;
  const float64  yigrid = yimin - 0.5 * dely * is_odd + 0.5 * dely;
  const float64  yhgrid = yhmin - 0.5 * dely * is_odd + 0.5 * dely;
  const float64  zigrid = zimin - 0.5 * delz * is_odd + 0.5 * delz;
  const float64  zhgrid = zhmin - 0.5 * delz * is_odd + 0.5 * delz;
  const simd_i64 index  = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  //
  // loop body for both vectorized and scalar version
  //
  auto LoopBody = [=](auto& uf, auto xu[], auto dt1) {
    using T_float = std::remove_reference_t<decltype(*xu)>;
    using T_int   = xsimd::as_integer_t<T_float>;

    const T_float rc_simd    = rc;
    const T_float rdx_simd   = rdx;
    const T_float rdy_simd   = rdy;
    const T_float rdz_simd   = rdz;
    const T_float ximin_simd = ximin;
    const T_float xhmin_simd = xhmin;
    const T_float yimin_simd = yimin;
    const T_float yhmin_simd = yhmin;
    const T_float zimin_simd = zimin;
    const T_float zhmin_simd = zhmin;

    T_float wix[size] = {0};
    T_float whx[size] = {0};
    T_float wiy[size] = {0};
    T_float why[size] = {0};
    T_float wiz[size] = {0};
    T_float whz[size] = {0};

    auto gam = lorentz_factor(xu[3], xu[4], xu[5], rc_simd);
    auto dt2 = dt1 * rc_simd / gam;

    // grid indices and positions
    auto ix0 = digitize(xu[0], ximin_simd, rdx_simd);
    auto hx0 = digitize(xu[0], xhmin_simd, rdx_simd);
    auto iy0 = digitize(xu[1], yimin_simd, rdy_simd);
    auto hy0 = digitize(xu[1], yhmin_simd, rdy_simd);
    auto iz0 = digitize(xu[2], zimin_simd, rdz_simd);
    auto hz0 = digitize(xu[2], zhmin_simd, rdz_simd);
    auto xig = xigrid + to_float(ix0) * delx;
    auto xhg = xhgrid + to_float(hx0) * delx;
    auto yig = yigrid + to_float(iy0) * dely;
    auto yhg = yhgrid + to_float(hy0) * dely;
    auto zig = zigrid + to_float(iz0) * delz;
    auto zhg = zhgrid + to_float(hz0) * delz;

    // weights
    shape<Order>(xu[0], xig, rdx_simd, wix);
    shape<Order>(xu[0], xhg, rdx_simd, whx);
    shape<Order>(xu[1], yig, rdy_simd, wiy);
    shape<Order>(xu[1], yhg, rdy_simd, why);
    shape<Order>(xu[2], zig, rdz_simd, wiz);
    shape<Order>(xu[2], zhg, rdz_simd, whz);

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

    auto ex = interpolate3d<Order>(uf, iz0, iy0, hx0, 0, wiz, wiy, whx, dt1);
    auto ey = interpolate3d<Order>(uf, iz0, hy0, ix0, 1, wiz, why, wix, dt1);
    auto ez = interpolate3d<Order>(uf, hz0, iy0, ix0, 2, whz, wiy, wix, dt1);
    auto bx = interpolate3d<Order>(uf, hz0, hy0, ix0, 3, whz, why, wix, dt2);
    auto by = interpolate3d<Order>(uf, hz0, iy0, hx0, 4, whz, wiy, whx, dt2);
    auto bz = interpolate3d<Order>(uf, iz0, hy0, hx0, 5, wiz, why, whx, dt2);

    // push particle velocity
    push_boris(xu[3], xu[4], xu[5], ex, ey, ez, bx, by, bz);
  };

  //
  // main computation
  //
  for (int is = 0; is < Ns; is++) {
    auto ps      = up[is];
    int  np_simd = (ps->Np / simd_f64::size) * simd_f64::size;

    //
    // vectorized loop
    //
    for (int ip = 0; ip < np_simd; ip += simd_f64::size) {
      // local SIMD register
      simd_f64 xu[6];
      simd_f64 dt1 = 0.5 * ps->q / ps->m * delt;

      // load particles to SIMD register
      xu[0] = simd_f64::gather(&ps->xu(ip, 0), index);
      xu[1] = simd_f64::gather(&ps->xu(ip, 1), index);
      xu[2] = simd_f64::gather(&ps->xu(ip, 2), index);
      xu[3] = simd_f64::gather(&ps->xu(ip, 3), index);
      xu[4] = simd_f64::gather(&ps->xu(ip, 4), index);
      xu[5] = simd_f64::gather(&ps->xu(ip, 5), index);

      LoopBody(uf, xu, dt1);

      // store particles to memory
      xu[3].scatter(&ps->xu(ip, 3), index);
      xu[4].scatter(&ps->xu(ip, 4), index);
      xu[5].scatter(&ps->xu(ip, 5), index);
    }

    //
    // scalar loop for reminder
    //
    for (int ip = np_simd; ip < ps->Np; ip++) {
      float64* xu  = &ps->xu(ip, 0);
      float64  dt1 = 0.5 * ps->q / ps->m * delt;

      LoopBody(uf, xu, dt1);
    }
  }
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  using simd::simd_f64;
  using simd::simd_i64;
  constexpr int size   = Order + 3;
  constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  const float64  rdx   = 1 / delx;
  const float64  rdy   = 1 / dely;
  const float64  rdz   = 1 / delz;
  const float64  dxdt  = delx / delt;
  const float64  dydt  = dely / delt;
  const float64  dzdt  = delz / delt;
  const float64  xmin  = xlim[0] + 0.5 * delx * is_odd;
  const float64  ymin  = ylim[0] + 0.5 * dely * is_odd;
  const float64  zmin  = zlim[0] + 0.5 * delz * is_odd;
  const float64  xgrid = xlim[0] + 0.5 * delx;
  const float64  ygrid = ylim[0] + 0.5 * dely;
  const float64  zgrid = zlim[0] + 0.5 * delz;
  const simd_i64 index = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  // clear charge/current density
  uj.fill(0);

  //
  // loop body for both vectorized and scalar version
  //
  auto LoopBody = [=](auto& uj, auto xv[], auto xu[], auto ss[2][3][size],
                      auto cur[size][size][size][4], auto qs) {
    using T_float = std::remove_reference_t<decltype(*xv)>;
    using T_int   = xsimd::as_integer_t<T_float>;

    const T_float xmin_simd = xmin;
    const T_float ymin_simd = ymin;
    const T_float zmin_simd = zmin;
    const T_float rdx_simd  = rdx;
    const T_float rdy_simd  = rdy;
    const T_float rdz_simd  = rdz;

    //
    // -*- weights before move -*-
    //
    // grid indices and positions
    auto ix0 = digitize(xv[0], xmin_simd, rdx_simd);
    auto iy0 = digitize(xv[1], ymin_simd, rdy_simd);
    auto iz0 = digitize(xv[2], zmin_simd, rdz_simd);
    auto xg0 = xgrid + to_float(ix0) * delx;
    auto yg0 = ygrid + to_float(iy0) * dely;
    auto zg0 = zgrid + to_float(iz0) * delz;

    // weights
    shape<Order>(xv[0], xg0, rdx_simd, &ss[0][0][1]);
    shape<Order>(xv[1], yg0, rdy_simd, &ss[0][1][1]);
    shape<Order>(xv[2], zg0, rdz_simd, &ss[0][2][1]);

    //
    // -*- weights after move -*-
    //
    // grid indices and positions
    auto ix1 = digitize(xu[0], xmin_simd, rdx_simd);
    auto iy1 = digitize(xu[1], ymin_simd, rdy_simd);
    auto iz1 = digitize(xu[2], zmin_simd, rdz_simd);
    auto xg1 = xgrid + to_float(ix1) * delx;
    auto yg1 = ygrid + to_float(iy1) * dely;
    auto zg1 = zgrid + to_float(iz1) * delz;

    // weights
    shape<Order>(xu[0], xg1, rdx_simd, &ss[1][0][1]);
    shape<Order>(xu[1], yg1, rdy_simd, &ss[1][1][1]);
    shape<Order>(xu[2], zg1, rdz_simd, &ss[1][2][1]);

    // shift weights according to particle movement
    T_int shift[3] = {ix1 - ix0, iy1 - iy0, iz1 - iz0};
    esirkepov3d_shift_weights_after_movement<Order>(shift, ss[1]);

    //
    // -*- accumulate current via density decomposition -*-
    //
    esirkepov3d<Order>(dxdt, dydt, dzdt, ss, cur);

    // deposit to global array
    ix0 += Lbx - (Order / 2) - 1;
    iy0 += Lby - (Order / 2) - 1;
    iz0 += Lbz - (Order / 2) - 1;
    append_current3d<Order>(uj, iz0, iy0, ix0, cur, qs);
  };

  //
  // main computation
  //
  for (int is = 0; is < Ns; is++) {
    auto ps      = up[is];
    int  np_simd = (ps->Np / simd_f64::size) * simd_f64::size;

    //
    // vectorized loop
    //
    for (int ip = 0; ip < np_simd; ip += simd_f64::size) {
      // local SIMD register
      simd_f64 xv[3];
      simd_f64 xu[3];
      simd_f64 ss[2][3][size]           = {0};
      simd_f64 cur[size][size][size][4] = {0};

      // load particles to SIMD register
      xv[0] = simd_f64::gather(&ps->xv(ip, 0), index);
      xv[1] = simd_f64::gather(&ps->xv(ip, 1), index);
      xv[2] = simd_f64::gather(&ps->xv(ip, 2), index);
      xu[0] = simd_f64::gather(&ps->xu(ip, 0), index);
      xu[1] = simd_f64::gather(&ps->xu(ip, 1), index);
      xu[2] = simd_f64::gather(&ps->xu(ip, 2), index);

      LoopBody(uj, xv, xu, ss, cur, ps->q);
    }

    //
    // scalar loop for reminder
    //
    for (int ip = np_simd; ip < ps->Np; ip++) {
      float64* xv                       = &ps->xv(ip, 0);
      float64* xu                       = &ps->xu(ip, 0);
      float64  ss[2][3][size]           = {0};
      float64  cur[size][size][size][4] = {0};

      LoopBody(uj, xv, xu, ss, cur, ps->q);
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
