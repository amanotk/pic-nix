// -*- C++ -*-
#ifndef _ENGINE_VELOCITY_HPP_
#define _ENGINE_VELOCITY_HPP_

#include "nix/nix.hpp"

#include "nix/interp.hpp"
#include "nix/primitives.hpp"

namespace pic_engine
{
enum PusherType {
  PusherBoris       = 0,
  PusherVay         = 1,
  PusherHigueraCary = 2,
  // number of pushers
  PusherSize,
};

enum InterpType {
  InterpMC = 0,
  InterpWT = 1,
  // number of shapes
  InterpSize,
};

static const char* PusherName[PusherSize] = {"Boris", "Vay", "HigueraCary"};
static const char* InterpName[InterpSize] = {"MC", "WT"};

template <int Dim, int Order, int Pusher, int Interp>
class BaseVelocity
{
public:
  static constexpr int Sx     = Order + 2;
  static constexpr int Sy     = Dim > 1 ? Order + 2 : 1;
  static constexpr int Sz     = Dim > 2 ? Order + 2 : 1;
  static constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  bool    has_xdim;
  bool    has_ydim;
  bool    has_zdim;
  int     ns;
  int     lbx;
  int     lby;
  int     lbz;
  int     ubx;
  int     uby;
  int     ubz;
  float64 cc;
  float64 dt;
  float64 dx;
  float64 dy;
  float64 dz;
  float64 xmin;
  float64 ymin;
  float64 zmin;

  template <typename T_data>
  BaseVelocity(const T_data& data, bool has_dim[3])
  {
    has_xdim = has_dim[2];
    has_ydim = has_dim[1];
    has_zdim = has_dim[0];
    ns       = data.Ns;
    lbx      = data.Lbx;
    lby      = data.Lby;
    lbz      = data.Lbz;
    ubx      = data.Ubx;
    uby      = data.Uby;
    ubz      = data.Ubz;
    cc       = data.cc;
    dx       = data.delx;
    dy       = data.dely;
    dz       = data.delz;
    xmin     = data.xlim[0];
    ymin     = data.ylim[0];
    zmin     = data.zlim[0];
  }

  template <typename T_float>
  void push_impl(T_float xu[], T_float ex, T_float ey, T_float ez, T_float bx, T_float by,
                 T_float bz)
  {
    T_float cc = this->cc;

    if constexpr (Pusher == PusherBoris) {
      push_boris(xu[3], xu[4], xu[5], ex, ey, ez, bx, by, bz, cc);
    }

    if constexpr (Pusher == PusherVay) {
      push_vay(xu[3], xu[4], xu[5], ex, ey, ez, bx, by, bz, cc);
    }

    if constexpr (Pusher == PusherHigueraCary) {
      push_higuera_cary(xu[3], xu[4], xu[5], ex, ey, ez, bx, by, bz, cc);
    }
  }

  template <typename T_particle, typename T_array>
  void call_scalar_impl(T_particle& up, T_array& uf, float64 delt)
  {

    auto push = [&](T_array& uf, float64 xu[], float64 dt) {
      // 1D version
      if constexpr (Dim == 1) {
        push_scalar1d(uf, xu, dt);
        return;
      }

      // 2D version
      if constexpr (Dim == 2) {
        push_scalar2d(uf, xu, dt);
        return;
      }

      // 3D version
      if constexpr (Dim == 3) {
        push_scalar3d(uf, xu, dt);
        return;
      }
    };

    for (int is = 0; is < ns; is++) {
      float64 qmdt = 0.5 * up[is]->q / up[is]->m * delt;

      for (int ip = 0; ip < up[is]->Np; ip++) {
        push(uf, &up[is]->xu(ip, 0), qmdt);
      }
    }
  }

  template <typename T_particle, typename T_array>
  void call_vector_impl(T_particle& up, T_array& uf, float64 delt)
  {
    using namespace nix;
    using namespace nix::primitives;
    const simd_i64 index = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

    const int ubx = has_xdim ? this->ubx + is_odd : this->ubx;
    const int uby = has_ydim ? this->uby + is_odd : this->uby;
    const int ubz = has_zdim ? this->ubz + is_odd : this->ubz;

    auto push = [&](T_array& uf, int iz, int iy, int ix, auto xu[], auto dt) {
      // 1D version
      if constexpr (Dim == 1) {
        push_vector1d(uf, iz, iy, ix, xu, dt);
        return;
      }

      // 2D version
      if constexpr (Dim == 2) {
        push_vector2d(uf, iz, iy, ix, xu, dt);
        return;
      }

      // 3D version
      if constexpr (Dim == 3) {
        push_vector3d(uf, iz, iy, ix, xu, dt);
        return;
      }
    };

    for (int iz = lbz, jz = 0; iz <= ubz; iz++, jz++) {
      for (int iy = lby, jy = 0; iy <= uby; iy++, jy++) {
        for (int ix = lbx, jx = 0; ix <= ubx; ix++, jx++) {
          // process particles in the cell
          for (int is = 0; is < ns; is++) {
            int     ii      = up[is]->flatindex(jz, jy, jx); // 1D grid index
            int     ip_zero = up[is]->pindex(ii);
            int     np_cell = up[is]->pindex(ii + 1) - ip_zero;
            int     np_simd = (np_cell / simd_f64::size) * simd_f64::size;
            float64 qmdt    = 0.5 * up[is]->q / up[is]->m * delt;

            //
            // vectorized loop
            //
            for (int ip = ip_zero; ip < ip_zero + np_simd; ip += simd_f64::size) {
              // local SIMD register
              simd_f64 xu[6];

              // load particles to SIMD register
              xu[0] = simd_f64::gather(&up[is]->xu(ip, 0), index);
              xu[1] = simd_f64::gather(&up[is]->xu(ip, 1), index);
              xu[2] = simd_f64::gather(&up[is]->xu(ip, 2), index);
              xu[3] = simd_f64::gather(&up[is]->xu(ip, 3), index);
              xu[4] = simd_f64::gather(&up[is]->xu(ip, 4), index);
              xu[5] = simd_f64::gather(&up[is]->xu(ip, 5), index);

              push(uf, iz, iy, ix, xu, simd_f64(qmdt));

              // store particles to memory
              xu[3].scatter(&up[is]->xu(ip, 3), index);
              xu[4].scatter(&up[is]->xu(ip, 4), index);
              xu[5].scatter(&up[is]->xu(ip, 5), index);
            }

            //
            // scalar loop for reminder
            //
            for (int ip = ip_zero + np_simd; ip < ip_zero + np_cell; ip++) {
              auto xu = &up[is]->xu(ip, 0);

              push(uf, iz, iy, ix, xu, qmdt);
            }
          }
        }
      }
    }
  }

  template <typename T_float>
  auto weights1d(T_float xu[], T_float wix[Sx], T_float whx[Sx])
  {
    T_float ximin  = xmin + 0.5 * dx * is_odd;
    T_float xhmin  = xmin + 0.5 * dx * is_odd - 0.5 * dx;
    T_float xigrid = xmin + 0.5 * dx;
    T_float xhgrid = xmin;
    T_float rdx    = 1 / dx;
    T_float dtx    = cc * dt / dx;
    T_float rdtx   = 1 / dtx;

    // grid indices and positions
    auto ix0 = digitize(xu[0], ximin, rdx);
    auto hx0 = digitize(xu[0], xhmin, rdx);
    auto xig = xigrid + to_float(ix0) * dx;
    auto xhg = xhgrid + to_float(hx0) * dx;

    // MC weights
    if constexpr (Interp == InterpMC) {
      shape_mc<Order>(xu[0], xig, rdx, wix);
    }

    // WT weights
    if constexpr (Interp == InterpWT) {
      shape_wt<Order>(xu[0], xig, rdx, dtx, rdtx, wix);
    }

    shape_mc<Order>(xu[0], xhg, rdx, whx);

    return std::make_tuple(ix0, hx0);
  }

  template <typename T_float>
  auto weights2d(T_float xu[], T_float wix[Sx], T_float wiy[Sy], T_float whx[Sx], T_float why[Sy])
  {
    T_float ximin  = xmin + 0.5 * dx * is_odd;
    T_float xhmin  = xmin + 0.5 * dx * is_odd - 0.5 * dx;
    T_float yimin  = ymin + 0.5 * dy * is_odd;
    T_float yhmin  = ymin + 0.5 * dy * is_odd - 0.5 * dy;
    T_float xigrid = xmin + 0.5 * dx;
    T_float xhgrid = xmin;
    T_float yigrid = ymin + 0.5 * dy;
    T_float yhgrid = ymin;
    T_float rdx    = 1 / dx;
    T_float rdy    = 1 / dy;
    T_float dtx    = cc * dt / dx;
    T_float dty    = cc * dt / dy;
    T_float rdtx   = 1 / dtx;
    T_float rdty   = 1 / dty;

    // grid indices and positions
    auto ix0 = digitize(xu[0], ximin, rdx);
    auto iy0 = digitize(xu[1], yimin, rdy);
    auto hx0 = digitize(xu[0], xhmin, rdx);
    auto hy0 = digitize(xu[1], yhmin, rdy);
    auto xig = xigrid + to_float(ix0) * dx;
    auto yig = yigrid + to_float(iy0) * dy;
    auto xhg = xhgrid + to_float(hx0) * dx;
    auto yhg = yhgrid + to_float(hy0) * dy;

    // MC weights
    if constexpr (Interp == InterpMC) {
      shape_mc<Order>(xu[0], xig, rdx, wix);
      shape_mc<Order>(xu[1], yig, rdy, wiy);
    }

    // WT weights
    if constexpr (Interp == InterpWT) {
      shape_wt<Order>(xu[0], xig, rdx, dtx, rdtx, wix);
      shape_wt<Order>(xu[1], yig, rdy, dty, rdty, wiy);
    }

    shape_mc<Order>(xu[0], xhg, rdx, whx);
    shape_mc<Order>(xu[1], yhg, rdy, why);

    return std::make_tuple(ix0, iy0, hx0, hy0);
  }

  template <typename T_float>
  auto weights3d(T_float xu[], T_float wix[Sx], T_float wiy[Sy], T_float wiz[Sz], T_float whx[Sx],
                 T_float why[Sy], T_float whz[Sz])
  {
    T_float ximin  = xmin + 0.5 * dx * is_odd;
    T_float xhmin  = xmin + 0.5 * dx * is_odd - 0.5 * dx;
    T_float yimin  = ymin + 0.5 * dy * is_odd;
    T_float yhmin  = ymin + 0.5 * dy * is_odd - 0.5 * dy;
    T_float zimin  = zmin + 0.5 * dz * is_odd;
    T_float zhmin  = zmin + 0.5 * dz * is_odd - 0.5 * dz;
    T_float xigrid = xmin + 0.5 * dx;
    T_float xhgrid = xmin;
    T_float yigrid = ymin + 0.5 * dy;
    T_float yhgrid = ymin;
    T_float zigrid = zmin + 0.5 * dz;
    T_float zhgrid = zmin;
    T_float rdx    = 1 / dx;
    T_float rdy    = 1 / dy;
    T_float rdz    = 1 / dz;
    T_float dtx    = cc * dt / dx;
    T_float dty    = cc * dt / dy;
    T_float dtz    = cc * dt / dz;
    T_float rdtx   = 1 / dtx;
    T_float rdty   = 1 / dty;
    T_float rdtz   = 1 / dtz;

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

    // MC weights
    if constexpr (Interp == InterpMC) {
      shape_mc<Order>(xu[0], xig, rdx, wix);
      shape_mc<Order>(xu[1], yig, rdy, wiy);
      shape_mc<Order>(xu[2], zig, rdz, wiz);
    }

    // WT weights
    if constexpr (Interp == InterpWT) {
      shape_wt<Order>(xu[0], xig, rdx, dtx, rdtx, wix);
      shape_wt<Order>(xu[1], yig, rdy, dty, rdty, wiy);
      shape_wt<Order>(xu[2], zig, rdz, dtz, rdtz, wiz);
    }

    shape_mc<Order>(xu[0], xhg, rdx, whx);
    shape_mc<Order>(xu[1], yhg, rdy, why);
    shape_mc<Order>(xu[2], zhg, rdz, whz);

    return std::make_tuple(ix0, iy0, iz0, hx0, hy0, hz0);
  }

  template <typename T_array>
  void push_scalar1d(T_array& uf, float64 xu[], float64 dt)
  {
    constexpr int Stencil = Order - 1;
    using nix::interp::interp1d;

    // 1D version
    const int iz0 = lbz;
    const int iy0 = lby;

    float64 wix[Sx] = {0};
    float64 whx[Sx] = {0};

    auto [ix0, hx0] = weights1d(xu, wix, whx);

    ix0 += lbx - (Order / 2);
    hx0 += lbx - (Order / 2);

    auto ex = interp1d<Stencil>(uf, iz0, iy0, hx0, 0, whx, dt);
    auto ey = interp1d<Stencil>(uf, iz0, iy0, ix0, 1, wix, dt);
    auto ez = interp1d<Stencil>(uf, iz0, iy0, ix0, 2, wix, dt);
    auto bx = interp1d<Stencil>(uf, iz0, iy0, ix0, 3, wix, dt);
    auto by = interp1d<Stencil>(uf, iz0, iy0, hx0, 4, whx, dt);
    auto bz = interp1d<Stencil>(uf, iz0, iy0, hx0, 5, whx, dt);

    push_impl(xu, ex, ey, ez, bx, by, bz);
  }

  template <typename T_array>
  void push_scalar2d(T_array& uf, float64 xu[], float64 dt)
  {
    constexpr int Stencil = Order - 1;
    using nix::interp::interp2d;

    // 2D version
    const int iz0 = lbz;

    float64 wix[Sx] = {0};
    float64 wiy[Sy] = {0};
    float64 whx[Sx] = {0};
    float64 why[Sy] = {0};

    auto [ix0, iy0, hx0, hy0] = weights2d(xu, wix, wiy, whx, why);

    ix0 += lbx - (Order / 2);
    iy0 += lby - (Order / 2);
    hx0 += lbx - (Order / 2);
    hy0 += lby - (Order / 2);

    auto ex = interp2d<Stencil>(uf, iz0, iy0, hx0, 0, wiy, whx, dt);
    auto ey = interp2d<Stencil>(uf, iz0, hy0, ix0, 1, why, wix, dt);
    auto ez = interp2d<Stencil>(uf, iz0, iy0, ix0, 2, wiy, wix, dt);
    auto bx = interp2d<Stencil>(uf, iz0, hy0, ix0, 3, why, wix, dt);
    auto by = interp2d<Stencil>(uf, iz0, iy0, hx0, 4, wiy, whx, dt);
    auto bz = interp2d<Stencil>(uf, iz0, hy0, hx0, 5, why, whx, dt);

    push_impl(xu, ex, ey, ez, bx, by, bz);
  }

  template <typename T_array>
  void push_scalar3d(T_array& uf, float64 xu[], float64 dt1)
  {
    constexpr int Stencil = Order - 1;
    using nix::interp::interp3d;

    float64 wix[Sx] = {0};
    float64 wiy[Sy] = {0};
    float64 wiz[Sz] = {0};
    float64 whx[Sx] = {0};
    float64 why[Sy] = {0};
    float64 whz[Sz] = {0};

    auto [ix0, iy0, iz0, hx0, hy0, hz0] = weights3d(xu, wix, wiy, wiz, whx, why, whz);

    ix0 += lbx - (Order / 2);
    iy0 += lby - (Order / 2);
    iz0 += lbz - (Order / 2);
    hx0 += lbx - (Order / 2);
    hy0 += lby - (Order / 2);
    hz0 += lbz - (Order / 2);

    auto ex = interp3d<Stencil>(uf, iz0, iy0, hx0, 0, wiz, wiy, whx, dt);
    auto ey = interp3d<Stencil>(uf, iz0, hy0, ix0, 1, wiz, why, wix, dt);
    auto ez = interp3d<Stencil>(uf, hz0, iy0, ix0, 2, whz, wiy, wix, dt);
    auto bx = interp3d<Stencil>(uf, hz0, hy0, ix0, 3, whz, why, wix, dt);
    auto by = interp3d<Stencil>(uf, hz0, iy0, hx0, 4, whz, wiy, whx, dt);
    auto bz = interp3d<Stencil>(uf, iz0, hy0, hx0, 5, wiz, why, whx, dt);

    push_impl(xu, ex, ey, ez, bx, by, bz);
  }

  template <typename T_array, typename T_float>
  void push_vector1d(T_array& uf, int iz, int iy, int ix, T_float xu[], T_float dt)
  {
    constexpr int Stencil = Order;
    using nix::interp::shift_weights;
    using nix::interp::interp1d;

    T_float wix[Sx] = {0};
    T_float whx[Sx] = {0};

    auto [ix0, hx0] = weights1d(xu, wix, whx);

    // shift weights according to particle position
    shift_weights<Order>(hx0 - ix0, whx);

    ix -= ((Order + 1) / 2);

    auto ex = interp1d<Stencil>(uf, iz, iy, ix, 0, whx, dt);
    auto ey = interp1d<Stencil>(uf, iz, iy, ix, 1, wix, dt);
    auto ez = interp1d<Stencil>(uf, iz, iy, ix, 2, wix, dt);
    auto bx = interp1d<Stencil>(uf, iz, iy, ix, 3, wix, dt);
    auto by = interp1d<Stencil>(uf, iz, iy, ix, 4, whx, dt);
    auto bz = interp1d<Stencil>(uf, iz, iy, ix, 5, whx, dt);

    push_impl(xu, ex, ey, ez, bx, by, bz);
  }

  template <typename T_array, typename T_float>
  void push_vector2d(T_array& uf, int iz, int iy, int ix, T_float xu[], T_float dt)
  {
    constexpr int Stencil = Order;
    using nix::interp::shift_weights;
    using nix::interp::interp2d;

    T_float wix[Sx] = {0};
    T_float wiy[Sy] = {0};
    T_float whx[Sx] = {0};
    T_float why[Sy] = {0};

    auto [ix0, iy0, hx0, hy0] = weights2d(xu, wix, wiy, whx, why);

    // shift weights according to particle position
    shift_weights<Order>(hx0 - ix0, whx);
    shift_weights<Order>(hy0 - iy0, why);

    ix -= ((Order + 1) / 2);
    iy -= ((Order + 1) / 2);

    auto ex = interp2d<Stencil>(uf, iz, iy, ix, 0, wiy, whx, dt);
    auto ey = interp2d<Stencil>(uf, iz, iy, ix, 1, why, wix, dt);
    auto ez = interp2d<Stencil>(uf, iz, iy, ix, 2, wiy, wix, dt);
    auto bx = interp2d<Stencil>(uf, iz, iy, ix, 3, why, wix, dt);
    auto by = interp2d<Stencil>(uf, iz, iy, ix, 4, wiy, whx, dt);
    auto bz = interp2d<Stencil>(uf, iz, iy, ix, 5, why, whx, dt);

    push_impl(xu, ex, ey, ez, bx, by, bz);
  }

  template <typename T_array, typename T_float>
  void push_vector3d(T_array& uf, int iz, int iy, int ix, T_float xu[], T_float dt)
  {
    constexpr int Stencil = Order;
    using nix::interp::shift_weights;
    using nix::interp::interp3d;

    T_float wix[Sx] = {0};
    T_float wiy[Sy] = {0};
    T_float wiz[Sz] = {0};
    T_float whx[Sx] = {0};
    T_float why[Sy] = {0};
    T_float whz[Sz] = {0};

    auto [ix0, iy0, iz0, hx0, hy0, hz0] = weights3d(xu, wix, wiy, wiz, whx, why, whz);

    // shift weights according to particle position
    shift_weights<Order>(hx0 - ix0, whx);
    shift_weights<Order>(hy0 - iy0, why);
    shift_weights<Order>(hz0 - iz0, whz);

    ix -= ((Order + 1) / 2);
    iy -= ((Order + 1) / 2);
    iz -= ((Order + 1) / 2);

    auto ex = interp3d<Stencil>(uf, iz, iy, ix, 0, wiz, wiy, whx, dt);
    auto ey = interp3d<Stencil>(uf, iz, iy, ix, 1, wiz, why, wix, dt);
    auto ez = interp3d<Stencil>(uf, iz, iy, ix, 2, whz, wiy, wix, dt);
    auto bx = interp3d<Stencil>(uf, iz, iy, ix, 3, whz, why, wix, dt);
    auto by = interp3d<Stencil>(uf, iz, iy, ix, 4, whz, wiy, whx, dt);
    auto bz = interp3d<Stencil>(uf, iz, iy, ix, 5, wiz, why, whx, dt);

    push_impl(xu, ex, ey, ez, bx, by, bz);
  }
};

template <int Dim, int Order, int Pusher, int Interp>
class ScalarVelocity : public BaseVelocity<Dim, Order, Pusher, Interp>
{
public:
  using BaseVelocity<Dim, Order, Pusher, Interp>::BaseVelocity; // inherit constructor

  template <typename T_particle, typename T_array>
  void operator()(T_particle& up, T_array& uf, float64 delt)
  {
    this->call_scalar_impl(up, uf, delt);
  }
};

template <int Dim, int Order, int Pusher, int Interp>
class VectorVelocity : public BaseVelocity<Dim, Order, Pusher, Interp>
{
public:
  using BaseVelocity<Dim, Order, Pusher, Interp>::BaseVelocity; // inherit constructor

  template <typename T_particle, typename T_array>
  void operator()(T_particle& up, T_array& uf, float64 delt)
  {
    this->call_vector_impl(up, uf, delt);
  }
};

} // namespace pic_engine

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif