// -*- C++ -*-
#include "nix/nix.hpp"
#include "nix/particle_primitives.hpp"

namespace exchunk3d_impl
{
//
// Implementation of loop body for push_position
//
template <int Order, typename T_float>
struct Position {
  T_float rc;

  Position(float64 cc) : rc(1 / cc)
  {
  }

  void operator()(T_float xv[], T_float xu[], float64 delt)
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

struct VelocityOption {
  enum Interp {
    InterpMC = 0,
    InterpWT = 1,
  };

  enum Pusher {
    PusherBoris = 0,
  };
};

//
// Implementation of loop body for push_velocity
//
template <int Order, typename T_float, int Interpolation = VelocityOption::InterpMC>
struct Velocity {
  static constexpr int size   = Order + 2;
  static constexpr int is_odd = Order % 2 == 0 ? 0 : 1;
  using simd_f64              = simd::simd_f64;
  using simd_i64              = simd::simd_i64;

  int     lbx;
  int     lby;
  int     lbz;
  float64 dt;
  float64 dx;
  float64 dy;
  float64 dz;
  float64 xigrid;
  float64 xhgrid;
  float64 yigrid;
  float64 yhgrid;
  float64 zigrid;
  float64 zhgrid;

  T_float rc;
  T_float dtx;
  T_float dty;
  T_float dtz;
  T_float rdtx;
  T_float rdty;
  T_float rdtz;
  T_float rdt;
  T_float rdx;
  T_float rdy;
  T_float rdz;
  T_float ximin;
  T_float xhmin;
  T_float yimin;
  T_float yhmin;
  T_float zimin;
  T_float zhmin;

  Velocity(float64 delt, float64 delx, float64 dely, float64 delz, float64 xlim[3], float64 ylim[3],
           float64 zlim[3], int Lbx, int Lby, int Lbz, float64 cc)
      : dt(delt), dx(delx), dy(dely), dz(delz), lbx(Lbx), lby(Lby), lbz(Lbz)
  {
    rc     = 1 / cc;
    dtx    = cc * dt / dx;
    dty    = cc * dt / dx;
    dtz    = cc * dt / dx;
    rdtx   = 1 / dtx;
    rdty   = 1 / dty;
    rdtz   = 1 / dtz;
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
  }

  auto calc_weights(T_float xu[], T_float wix[size], T_float wiy[size], T_float wiz[size],
                    T_float whx[size], T_float why[size], T_float whz[size])
  {
    // grid indices and positions
    auto ix0 = digitize(xu[0], ximin, rdx);
    auto iy0 = digitize(xu[1], yimin, rdy);
    auto iz0 = digitize(xu[2], zimin, rdz);
    auto hx0 = digitize(xu[0], xhmin, rdx);
    auto hy0 = digitize(xu[1], yhmin, rdy);
    auto hz0 = digitize(xu[2], zhmin, rdz);
    auto xig = xigrid + to_float(ix0) * dx;
    auto yig = yigrid + to_float(iy0) * dy;
    auto zig = zigrid + to_float(iz0) * dz;
    auto xhg = xhgrid + to_float(hx0) * dx;
    auto yhg = yhgrid + to_float(hy0) * dy;
    auto zhg = zhgrid + to_float(hz0) * dz;

    // weights
    if constexpr (Interpolation == VelocityOption::InterpMC) {
      // MC weights
      shape<Order>(xu[0], xig, rdx, wix);
      shape<Order>(xu[1], yig, rdy, wiy);
      shape<Order>(xu[2], zig, rdz, wiz);
    } else if constexpr (Interpolation == VelocityOption::InterpWT) {
      // WT weights
      shape_wt<Order>(xu[0], xig, rdx, dtx, rdtx, wix);
      shape_wt<Order>(xu[1], yig, rdy, dty, rdty, wiy);
      shape_wt<Order>(xu[2], zig, rdz, dtz, rdtz, wiz);
    } else {
      static_assert([] { return false; }(), "Invalid interpolation type");
    }
    shape<Order>(xu[0], xhg, rdx, whx);
    shape<Order>(xu[1], yhg, rdy, why);
    shape<Order>(xu[2], zhg, rdz, whz);

    return std::make_tuple(ix0, iy0, iz0, hx0, hy0, hz0);
  }

  template <typename T_array>
  void sorted(T_array& uf, int iz, int iy, int ix, T_float xu[], T_float dt1)
  {
    constexpr int stencil = Order;

    T_float wix[size] = {0};
    T_float wiy[size] = {0};
    T_float wiz[size] = {0};
    T_float whx[size] = {0};
    T_float why[size] = {0};
    T_float whz[size] = {0};

    auto [ix0, iy0, iz0, hx0, hy0, hz0] = calc_weights(xu, wix, wiy, wiz, whx, why, whz);

    // shift weights according to particle position
    interpolate3d_shift_weights<Order>(hx0 - ix0, whx);
    interpolate3d_shift_weights<Order>(hy0 - iy0, why);
    interpolate3d_shift_weights<Order>(hz0 - iz0, whz);

    ix -= ((Order + 1) / 2);
    iy -= ((Order + 1) / 2);
    iz -= ((Order + 1) / 2);

    auto gam = lorentz_factor(xu[3], xu[4], xu[5], rc);
    auto dt2 = dt1 * rc / gam;
    auto ex  = interpolate3d<stencil>(uf, iz, iy, ix, 0, wiz, wiy, whx, dt1);
    auto ey  = interpolate3d<stencil>(uf, iz, iy, ix, 1, wiz, why, wix, dt1);
    auto ez  = interpolate3d<stencil>(uf, iz, iy, ix, 2, whz, wiy, wix, dt1);
    auto bx  = interpolate3d<stencil>(uf, iz, iy, ix, 3, whz, why, wix, dt2);
    auto by  = interpolate3d<stencil>(uf, iz, iy, ix, 4, whz, wiy, whx, dt2);
    auto bz  = interpolate3d<stencil>(uf, iz, iy, ix, 5, wiz, why, whx, dt2);

    // push particle velocity
    push_boris(xu[3], xu[4], xu[5], ex, ey, ez, bx, by, bz);
  }

  template <typename T_array>
  void unsorted(T_array& uf, T_float xu[], T_float dt1)
  {
    constexpr int stencil = Order - 1;

    T_float wix[size] = {0};
    T_float wiy[size] = {0};
    T_float wiz[size] = {0};
    T_float whx[size] = {0};
    T_float why[size] = {0};
    T_float whz[size] = {0};

    auto [ix0, iy0, iz0, hx0, hy0, hz0] = calc_weights(xu, wix, wiy, wiz, whx, why, whz);

    ix0 += lbx - (Order / 2);
    iy0 += lby - (Order / 2);
    iz0 += lbz - (Order / 2);
    hx0 += lbx - (Order / 2);
    hy0 += lby - (Order / 2);
    hz0 += lbz - (Order / 2);

    auto gam = lorentz_factor(xu[3], xu[4], xu[5], rc);
    auto dt2 = dt1 * rc / gam;
    auto ex  = interpolate3d<stencil>(uf, iz0, iy0, hx0, 0, wiz, wiy, whx, dt1);
    auto ey  = interpolate3d<stencil>(uf, iz0, hy0, ix0, 1, wiz, why, wix, dt1);
    auto ez  = interpolate3d<stencil>(uf, hz0, iy0, ix0, 2, whz, wiy, wix, dt1);
    auto bx  = interpolate3d<stencil>(uf, hz0, hy0, ix0, 3, whz, why, wix, dt2);
    auto by  = interpolate3d<stencil>(uf, hz0, iy0, hx0, 4, whz, wiy, whx, dt2);
    auto bz  = interpolate3d<stencil>(uf, iz0, hy0, hx0, 5, wiz, why, whx, dt2);

    // push particle velocity
    push_boris(xu[3], xu[4], xu[5], ex, ey, ez, bx, by, bz);
  }
};

//
// Implementation of loop body for deposit_current
//
template <int Order, typename T_float>
struct Current {
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

  T_float rdx;
  T_float rdy;
  T_float rdz;
  T_float xmin;
  T_float ymin;
  T_float zmin;

  Current(float64 delt, float64 delx, float64 dely, float64 delz, float64 xlim[3], float64 ylim[3],
          float64 zlim[3], int Lbx, int Lby, int Lbz, float64 cc)
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
  }

  auto calc_local_current(T_float xv[], T_float xu[], T_float qs, T_float cur[size][size][size][4])
  {
    using T_int = xsimd::as_integer_t<T_float>;

    T_float ss[2][3][size] = {0};

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
    esirkepov3d_shift_weights<Order>(shift, ss[1]);

    //
    // -*- accumulate current via density decomposition -*-
    //
    esirkepov3d<Order>(dxdt, dydt, dzdt, qs, ss, cur);

    return std::make_tuple(ix0, iy0, iz0);
  }

  template <typename T_array, typename T_int>
  void deposit_global_current(T_array& uj, T_int iz, T_int iy, T_int ix,
                              T_float cur[size][size][size][4])
  {
    ix -= ((Order + 1) / 2) + 1;
    iy -= ((Order + 1) / 2) + 1;
    iz -= ((Order + 1) / 2) + 1;
    append_current3d<Order>(uj, iz, iy, ix, cur);
  }

  template <typename T_array>
  void unsorted(T_array& uj, T_float xv[], T_float xu[], float64 qs)
  {
    T_float cur[size][size][size][4] = {0};

    // calculate local current
    auto [ix0, iy0, iz0] = calc_local_current(xv, xu, qs, cur);

    // deposit to global array
    ix0 += lbx - (Order / 2) - 1;
    iy0 += lby - (Order / 2) - 1;
    iz0 += lbz - (Order / 2) - 1;
    append_current3d<Order>(uj, iz0, iy0, ix0, cur);
  }
};

//
// Implementation of loop body for deposit_moment
//
template <int Order, typename T_float>
struct Moment {
  static constexpr int index_ro = 0;
  static constexpr int index_tx = 1;
  static constexpr int index_ty = 2;
  static constexpr int index_tz = 3;
  static constexpr int index_tt = 4;
  static constexpr int index_xx = 5;
  static constexpr int index_yy = 6;
  static constexpr int index_zz = 7;
  static constexpr int index_xy = 8;
  static constexpr int index_yz = 9;
  static constexpr int index_zx = 10;
  static constexpr int size     = Order + 1;
  static constexpr int is_odd   = Order % 2 == 0 ? 0 : 1;
  using simd_f64                = simd::simd_f64;
  using simd_i64                = simd::simd_i64;

  int     lbx;
  int     lby;
  int     lbz;
  float64 dx;
  float64 dy;
  float64 dz;
  float64 xgrid;
  float64 ygrid;
  float64 zgrid;

  T_float cc;
  T_float rc;
  T_float rdx;
  T_float rdy;
  T_float rdz;
  T_float xmin;
  T_float ymin;
  T_float zmin;

  Moment(float64 delx, float64 dely, float64 delz, float64 xlim[3], float64 ylim[3],
         float64 zlim[3], int Lbx, int Lby, int Lbz, float64 cc)
      : dx(delx), dy(dely), dz(delz), lbx(Lbx), lby(Lby), lbz(Lbz), cc(cc)
  {
    rc    = 1 / cc;
    rdx   = 1 / dx;
    rdy   = 1 / dy;
    rdz   = 1 / dz;
    xmin  = xlim[0] + 0.5 * delx * is_odd;
    ymin  = ylim[0] + 0.5 * dely * is_odd;
    zmin  = zlim[0] + 0.5 * delz * is_odd;
    xgrid = xlim[0] + 0.5 * delx;
    ygrid = ylim[0] + 0.5 * dely;
    zgrid = zlim[0] + 0.5 * delz;
  }

  template <typename T_array, typename T_int>
  static void append_moment3d(T_array& um, T_int iz0, T_int iy0, T_int ix0, int is,
                              T_float moment[size][size][size][11])
  {
    constexpr bool is_scalar = std::is_integral_v<T_int> && std::is_floating_point_v<T_float>;
    constexpr bool is_vector = std::is_integral_v<T_int> && std::is_same_v<T_float, simd_f64>;

    if constexpr (is_scalar == true) {
      // naive scalar implementation
      for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
        for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
          for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
            for (int k = 0; k < 11; k++) {
              um(iz, iy, ix, is, k) += moment[jz][jy][jx][k];
            }
          }
        }
      }
    } else if constexpr (is_vector == true) {
      // all particle contributions are added to the same grid point
      for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
        for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
          for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
            for (int k = 0; k < 11; k++) {
              um(iz, iy, ix, is, k) += xsimd::reduce_add(moment[jz][jy][jx][k]);
            }
          }
        }
      }
    } else {
      static_assert([] { return false; }(), "Invalid combination of types");
    }
  }

  auto calc_local_moment(T_float xu[], T_float ms, T_float mom[size][size][size][11])
  {
    using T_int = xsimd::as_integer_t<T_float>;

    T_float wx[size] = {0};
    T_float wy[size] = {0};
    T_float wz[size] = {0};

    // grid indices and positions
    auto ix = digitize(xu[0], xmin, rdx);
    auto iy = digitize(xu[1], ymin, rdy);
    auto iz = digitize(xu[2], zmin, rdz);
    auto xg = xgrid + to_float(ix) * dx;
    auto yg = ygrid + to_float(iy) * dy;
    auto zg = zgrid + to_float(iz) * dz;

    // weights
    shape<Order>(xu[0], xg, rdx, wx);
    shape<Order>(xu[1], yg, rdy, wy);
    shape<Order>(xu[2], zg, rdz, wz);

    for (int jz = 0; jz < size; jz++) {
      for (int jy = 0; jy < size; jy++) {
        for (int jx = 0; jx < size; jx++) {
          T_float ww = ms * wx[jx] * wy[jy] * wz[jz];
          T_float gm = lorentz_factor(xu[3], xu[4], xu[5], rc);

          mom[jz][jy][jx][index_ro] += ww;
          mom[jz][jy][jx][index_tx] += ww * xu[3];
          mom[jz][jy][jx][index_ty] += ww * xu[4];
          mom[jz][jy][jx][index_tz] += ww * xu[5];
          mom[jz][jy][jx][index_tt] += ww * gm * cc;
          mom[jz][jy][jx][index_xx] += ww * xu[3] * xu[3] / gm;
          mom[jz][jy][jx][index_yy] += ww * xu[4] * xu[4] / gm;
          mom[jz][jy][jx][index_zz] += ww * xu[5] * xu[5] / gm;
          mom[jz][jy][jx][index_xy] += ww * xu[3] * xu[4] / gm;
          mom[jz][jy][jx][index_yz] += ww * xu[4] * xu[5] / gm;
          mom[jz][jy][jx][index_zx] += ww * xu[5] * xu[3] / gm;
        }
      }
    }

    return std::make_tuple(ix, iy, iz);
  }

  template <typename T_array, typename T_int>
  void deposit_global_moment(T_array& um, T_int iz, T_int iy, T_int ix, int is,
                             T_float mom[size][size][size][11])
  {
    ix -= ((Order + 1) / 2) + 1;
    iy -= ((Order + 1) / 2) + 1;
    iz -= ((Order + 1) / 2) + 1;
    append_moment3d(um, iz, iy, ix, is, mom);
  }

  template <typename T_array>
  void unsorted(T_array& um, int is, T_float xu[], float64 ms)
  {
    T_float mom[size][size][size][11] = {0};

    // calculate local current
    auto [ix0, iy0, iz0] = calc_local_moment(xu, ms, mom);

    // deposit to global array
    ix0 += lbx - (Order / 2) - 1;
    iy0 += lby - (Order / 2) - 1;
    iz0 += lbz - (Order / 2) - 1;
    append_moment3d(um, iz0, iy0, ix0, is, mom);
  }
};
} // namespace exchunk3d_impl

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
