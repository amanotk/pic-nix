// -*- C++ -*-

//
// Implementation for Vectorized Version with xsimd
//
// This file is to be included from exchunk3d.cpp
//

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Order>                                                                             \
  type ExChunk3D<Order>::name##_impl_xsimd

namespace
{
//
// Loop body for push_velocity
//
template <int Order, typename T_float>
struct PushVelocityXsimd {
  static constexpr int size   = Order + 2;
  static constexpr int is_odd = Order % 2 == 0 ? 0 : 1;
  using simd_f64              = simd::simd_f64;
  using simd_i64              = simd::simd_i64;

  int     lbx;
  int     lby;
  int     lbz;
  float64 dx;
  float64 dy;
  float64 dz;
  float64 xigrid;
  float64 xhgrid;
  float64 yigrid;
  float64 yhgrid;
  float64 zigrid;
  float64 zhgrid;

  T_float  rc;
  T_float  rdx;
  T_float  rdy;
  T_float  rdz;
  T_float  ximin;
  T_float  xhmin;
  T_float  yimin;
  T_float  yhmin;
  T_float  zimin;
  T_float  zhmin;
  simd_i64 index;

  PushVelocityXsimd(float64 delt, float64 delx, float64 dely, float64 delz, float64 xlim[3],
                    float64 ylim[3], float64 zlim[3], int Lbx, int Lby, int Lbz, float64 cc)
      : dx(delx), dy(dely), dz(delz), lbx(Lbx), lby(Lby), lbz(Lbz)
  {
    rc     = 1 / cc;
    rdx    = 1 / dx;
    rdy    = 1 / dy;
    rdz    = 1 / dz;
    ximin  = xlim[0] + 0.5 * delx * is_odd;
    xhmin  = xlim[0] + 0.5 * delx * is_odd - 0.5 * delx;
    yimin  = ylim[0] + 0.5 * dely * is_odd;
    yhmin  = ylim[0] + 0.5 * dely * is_odd - 0.5 * dely;
    zimin  = zlim[0] + 0.5 * delz * is_odd;
    zhmin  = zlim[0] + 0.5 * delz * is_odd - 0.5 * delz;
    xigrid = xlim[0] + 0.5 * delx;
    xhgrid = xlim[0];
    yigrid = ylim[0] + 0.5 * dely;
    yhgrid = ylim[0];
    zigrid = zlim[0] + 0.5 * delz;
    zhgrid = zlim[0];
    index  = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;
  }

  template <typename T_array>
  void operator()(T_array& uf, T_float xu[], T_float dt1)
  {
    using T_int = xsimd::as_integer_t<T_float>;

    T_float wix[size] = {0};
    T_float whx[size] = {0};
    T_float wiy[size] = {0};
    T_float why[size] = {0};
    T_float wiz[size] = {0};
    T_float whz[size] = {0};

    auto gam = lorentz_factor(xu[3], xu[4], xu[5], rc);
    auto dt2 = dt1 * rc / gam;

    // grid indices and positions
    auto ix0 = digitize(xu[0], ximin, rdx);
    auto hx0 = digitize(xu[0], xhmin, rdx);
    auto iy0 = digitize(xu[1], yimin, rdy);
    auto hy0 = digitize(xu[1], yhmin, rdy);
    auto iz0 = digitize(xu[2], zimin, rdz);
    auto hz0 = digitize(xu[2], zhmin, rdz);
    auto xig = xigrid + to_float(ix0) * dx;
    auto xhg = xhgrid + to_float(hx0) * dx;
    auto yig = yigrid + to_float(iy0) * dy;
    auto yhg = yhgrid + to_float(hy0) * dy;
    auto zig = zigrid + to_float(iz0) * dz;
    auto zhg = zhgrid + to_float(hz0) * dz;

    // weights
    shape<Order>(xu[0], xig, rdx, wix);
    shape<Order>(xu[0], xhg, rdx, whx);
    shape<Order>(xu[1], yig, rdy, wiy);
    shape<Order>(xu[1], yhg, rdy, why);
    shape<Order>(xu[2], zig, rdz, wiz);
    shape<Order>(xu[2], zhg, rdz, whz);

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
    ix0 += lbx - (Order / 2);
    iy0 += lby - (Order / 2);
    iz0 += lbz - (Order / 2);
    hx0 += lbx - (Order / 2);
    hy0 += lby - (Order / 2);
    hz0 += lbz - (Order / 2);

    auto ex = interpolate3d<Order>(uf, iz0, iy0, hx0, 0, wiz, wiy, whx, dt1);
    auto ey = interpolate3d<Order>(uf, iz0, hy0, ix0, 1, wiz, why, wix, dt1);
    auto ez = interpolate3d<Order>(uf, hz0, iy0, ix0, 2, whz, wiy, wix, dt1);
    auto bx = interpolate3d<Order>(uf, hz0, hy0, ix0, 3, whz, why, wix, dt2);
    auto by = interpolate3d<Order>(uf, hz0, iy0, hx0, 4, whz, wiy, whx, dt2);
    auto bz = interpolate3d<Order>(uf, iz0, hy0, hx0, 5, wiz, why, whx, dt2);

    // push particle velocity
    push_boris(xu[3], xu[4], xu[5], ex, ey, ez, bx, by, bz);
  };
};

//
// Loop body for deposit_current
//
template <int Order, typename T_float>
struct DepositCurrentXsimd {
  static constexpr int size   = Order + 3;
  static constexpr int is_odd = Order % 2 == 0 ? 0 : 1;
  using simd_f64              = simd::simd_f64;
  using simd_i64              = simd::simd_i64;

  int     lbx;
  int     lby;
  int     lbz;
  float64 dx;
  float64 dy;
  float64 dz;
  float64 dxdt;
  float64 dydt;
  float64 dzdt;
  float64 xgrid;
  float64 ygrid;
  float64 zgrid;

  T_float  rdx;
  T_float  rdy;
  T_float  rdz;
  T_float  xmin;
  T_float  ymin;
  T_float  zmin;
  simd_i64 index;

  DepositCurrentXsimd(float64 delt, float64 delx, float64 dely, float64 delz, float64 xlim[3],
                      float64 ylim[3], float64 zlim[3], int Lbx, int Lby, int Lbz, float64 cc)
      : dx(delx), dy(dely), dz(delz), dxdt(delx / delt), dydt(dely / delt), dzdt(delz / delt),
        lbx(Lbx), lby(Lby), lbz(Lbz)
  {
    rdx   = 1 / dx;
    rdy   = 1 / dy;
    rdz   = 1 / dz;
    xmin  = xlim[0] + 0.5 * delx * is_odd;
    ymin  = ylim[0] + 0.5 * dely * is_odd;
    zmin  = zlim[0] + 0.5 * delz * is_odd;
    xgrid = xlim[0] + 0.5 * delx;
    ygrid = ylim[0] + 0.5 * dely;
    zgrid = zlim[0] + 0.5 * delz;
    index = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;
  }

  template <typename T_array>
  void operator()(T_array& uj, T_float xv[], T_float xu[], T_float (&ss)[2][3][size],
                  T_float (&cur)[size][size][size][4], float64 qs)
  {
    using T_int = xsimd::as_integer_t<T_float>;

    //
    // -*- weights before move -*-
    //
    // grid indices and positions
    auto ix0 = digitize(xv[0], xmin, rdx);
    auto iy0 = digitize(xv[1], ymin, rdy);
    auto iz0 = digitize(xv[2], zmin, rdz);
    auto xg0 = xgrid + to_float(ix0) * dx;
    auto yg0 = ygrid + to_float(iy0) * dy;
    auto zg0 = zgrid + to_float(iz0) * dz;

    // weights
    shape<Order>(xv[0], xg0, rdx, &ss[0][0][1]);
    shape<Order>(xv[1], yg0, rdy, &ss[0][1][1]);
    shape<Order>(xv[2], zg0, rdz, &ss[0][2][1]);

    //
    // -*- weights after move -*-
    //
    // grid indices and positions
    auto ix1 = digitize(xu[0], xmin, rdx);
    auto iy1 = digitize(xu[1], ymin, rdy);
    auto iz1 = digitize(xu[2], zmin, rdz);
    auto xg1 = xgrid + to_float(ix1) * dx;
    auto yg1 = ygrid + to_float(iy1) * dy;
    auto zg1 = zgrid + to_float(iz1) * dz;

    // weights
    shape<Order>(xu[0], xg1, rdx, &ss[1][0][1]);
    shape<Order>(xu[1], yg1, rdy, &ss[1][1][1]);
    shape<Order>(xu[2], zg1, rdz, &ss[1][2][1]);

    // shift weights according to particle movement
    T_int shift[3] = {ix1 - ix0, iy1 - iy0, iz1 - iz0};
    esirkepov3d_shift_weights_after_movement<Order>(shift, ss[1]);

    //
    // -*- accumulate current via density decomposition -*-
    //
    esirkepov3d<Order>(dxdt, dydt, dzdt, ss, cur);

    // deposit to global array
    ix0 += lbx - (Order / 2) - 1;
    iy0 += lby - (Order / 2) - 1;
    iz0 += lbz - (Order / 2) - 1;
    append_current3d<Order>(uj, iz0, iy0, ix0, cur, qs);
  }
};
} // namespace

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
  constexpr int  size   = Order + 2;
  constexpr int  is_odd = Order % 2 == 0 ? 0 : 1;
  const simd_i64 index  = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  PushVelocityXsimd<Order, simd_f64> LoopBodyV(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby,
                                               Lbz, cc);
  PushVelocityXsimd<Order, float64>  LoopBodyS(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby,
                                               Lbz, cc);

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

      LoopBodyV(uf, xu, dt1);

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

      LoopBodyS(uf, xu, dt1);
    }
  }
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  using simd::simd_f64;
  using simd::simd_i64;
  constexpr int  size   = Order + 3;
  constexpr int  is_odd = Order % 2 == 0 ? 0 : 1;
  const simd_i64 index  = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  DepositCurrentXsimd<Order, simd_f64> LoopBodyV(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby,
                                                 Lbz, cc);
  DepositCurrentXsimd<Order, float64>  LoopBodyS(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby,
                                                 Lbz, cc);

  // clear charge/current density
  uj.fill(0);

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

      LoopBodyV(uj, xv, xu, ss, cur, ps->q);
    }

    //
    // scalar loop for reminder
    //
    for (int ip = np_simd; ip < ps->Np; ip++) {
      float64* xv                       = &ps->xv(ip, 0);
      float64* xu                       = &ps->xu(ip, 0);
      float64  ss[2][3][size]           = {0};
      float64  cur[size][size][size][4] = {0};

      LoopBodyS(uj, xv, xu, ss, cur, ps->q);
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
