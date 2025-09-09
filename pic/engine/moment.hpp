// -*- C++ -*-
#ifndef _ENGINE_MOMENT_HPP_
#define _ENGINE_MOMENT_HPP_

#include "nix/nix.hpp"

#include "nix/primitives.hpp"

namespace pic_engine
{
template <int Dim, int Order>
class BaseMoment
{
public:
  static constexpr int Sx       = Order + 1;
  static constexpr int Sy       = Dim > 1 ? Order + 1 : 1;
  static constexpr int Sz       = Dim > 2 ? Order + 1 : 1;
  static constexpr int is_odd   = Order % 2 == 0 ? 0 : 1;
  static constexpr int index_t  = 0;
  static constexpr int index_x  = 1;
  static constexpr int index_y  = 2;
  static constexpr int index_z  = 3;
  static constexpr int index_tt = 4;
  static constexpr int index_xx = 5;
  static constexpr int index_yy = 6;
  static constexpr int index_zz = 7;
  static constexpr int index_tx = 8;
  static constexpr int index_ty = 9;
  static constexpr int index_tz = 10;
  static constexpr int index_xy = 11;
  static constexpr int index_yz = 12;
  static constexpr int index_zx = 13;

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
  float64 dx;
  float64 dy;
  float64 dz;
  float64 xmin;
  float64 ymin;
  float64 zmin;

  template <typename T_data>
  BaseMoment(const T_data& data, bool has_dim[3])
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

  template <typename T_particle, typename T_array>
  void call_scalar_impl(T_particle& up, T_array& um)
  {
    using namespace nix;
    using namespace nix::primitives;

    auto deposit_moment = [&](T_particle& up, T_array& um, int is, int ip) {
      float64 mom[Sz][Sy][Sx][14] = {0};
      auto    xu                  = &up[is]->xu(ip, 0);

      // 1D version
      if constexpr (Dim == 1) {
        auto [ix0, iy0, iz0] = local1d(xu, up[is]->m, mom);
        append_moment1d<Order>(um, iz0, iy0, ix0, is, mom[0][0]);
      }

      // 2D version
      if constexpr (Dim == 2) {
        auto [ix0, iy0, iz0] = local2d(xu, up[is]->m, mom);
        append_moment2d<Order>(um, iz0, iy0, ix0, is, mom[0]);
      }

      // 3D version
      if constexpr (Dim == 3) {
        auto [ix0, iy0, iz0] = local3d(xu, up[is]->m, mom);
        append_moment3d<Order>(um, iz0, iy0, ix0, is, mom);
      }
    };

    // clear moment
    fill_all(um, 0);

    for (int is = 0; is < ns; is++) {
      for (int ip = 0; ip < up[is]->Np; ip++) {
        deposit_moment(up, um, is, ip);
      }
    }
  }

  template <typename T_particle, typename T_array>
  void call_vector_impl(T_particle& up, T_array& um)
  {
    using namespace nix;
    using namespace nix::primitives;
    const simd_i64 index = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

    const int ubx = has_xdim ? this->ubx + is_odd : this->ubx;
    const int uby = has_ydim ? this->uby + is_odd : this->uby;
    const int ubz = has_zdim ? this->ubz + is_odd : this->ubz;

    auto local = [&](auto xu[], auto ms, auto mom[Sz][Sy][Sx][14]) {
      // 1D version
      if constexpr (Dim == 1) {
        local1d(xu, ms, mom);
        return;
      }

      // 2D version
      if constexpr (Dim == 2) {
        local2d(xu, ms, mom);
        return;
      }

      // 3D version
      if constexpr (Dim == 3) {
        local3d(xu, ms, mom);
        return;
      }
    };

    auto global = [&](T_array& um, int iz, int iy, int ix, int is, auto mom[Sz][Sy][Sx][14]) {
      // 1D version
      if constexpr (Dim == 1) {
        global1d(um, iz, iy, ix, is, mom);
        return;
      }

      // 2D version
      if constexpr (Dim == 2) {
        global2d(um, iz, iy, ix, is, mom);
        return;
      }

      // 3D version
      if constexpr (Dim == 3) {
        global3d(um, iz, iy, ix, is, mom);
        return;
      }
    };

    // clear moment
    fill_all(um, 0);

    for (int iz = lbz, jz = 0; iz <= ubz; iz++, jz++) {
      for (int iy = lby, jy = 0; iy <= uby; iy++, jy++) {
        for (int ix = lbx, jx = 0; ix <= ubx; ix++, jx++) {
          // process particles in the cell
          for (int is = 0; is < ns; is++) {
            // local moment
            float64  mom[Sz][Sy][Sx][14]      = {0}; // scalar
            simd_f64 mom_simd[Sz][Sy][Sx][14] = {0}; // SIMD register

            int ii      = up[is]->flatindex(jz, jy, jx); // 1D grid index
            int ip_zero = up[is]->pindex(ii);
            int np_cell = up[is]->pindex(ii + 1) - ip_zero;
            int np_simd = (np_cell / simd_f64::size) * simd_f64::size;

            //
            // vectorized loop
            //
            for (int ip = ip_zero; ip < ip_zero + np_simd; ip += simd_f64::size) {
              // local SIMD register
              simd_f64 xu[6];
              simd_f64 ms = up[is]->m;

              // load particles to SIMD register
              xu[0] = simd_f64::gather(&up[is]->xu(ip, 0), index);
              xu[1] = simd_f64::gather(&up[is]->xu(ip, 1), index);
              xu[2] = simd_f64::gather(&up[is]->xu(ip, 2), index);
              xu[3] = simd_f64::gather(&up[is]->xu(ip, 3), index);
              xu[4] = simd_f64::gather(&up[is]->xu(ip, 4), index);
              xu[5] = simd_f64::gather(&up[is]->xu(ip, 5), index);

              local(xu, ms, mom_simd);
            }

            //
            // scalar loop for reminder
            //
            for (int ip = ip_zero + np_simd; ip < ip_zero + np_cell; ip++) {
              float64* xu = &up[is]->xu(ip, 0);
              float64  ms = up[is]->m;

              local(xu, ms, mom);
            }

            // deposit to global array
            global(um, iz, iy, ix, is, mom);
            global(um, iz, iy, ix, is, mom_simd);
          }
        }
      }
    }
  }

  template <typename T_float>
  auto local1d(T_float xu[], T_float ms, T_float mom[Sz][Sy][Sx][14])
  {
    using namespace nix;
    using namespace nix::primitives;
    using T_int = xsimd::as_integer_t<T_float>;

    // 1D version
    T_int iy = lby;
    T_int iz = lbz;
    int   jy = 0;
    int   jz = 0;

    T_float wx[Sx] = {0};
    T_float rc     = 1 / cc;
    T_float rdx    = 1 / dx;
    T_float ximin  = xmin + 0.5 * dx * is_odd;
    T_float xgrid  = xmin + 0.5 * dx;

    // grid indices and positions
    auto ix = digitize(xu[0], ximin, rdx);
    auto xg = xgrid + to_float(ix) * dx;

    // weights
    shape_mc<Order>(xu[0], xg, rdx, wx);

    for (int jx = 0; jx < Sx; jx++) {
      T_float ww = ms * wx[jx];
      T_float gm = lorentz_factor(xu[3], xu[4], xu[5], rc);

      mom[jz][jy][jx][index_t] += ww;
      mom[jz][jy][jx][index_x] += ww * xu[3] / gm;
      mom[jz][jy][jx][index_y] += ww * xu[4] / gm;
      mom[jz][jy][jx][index_z] += ww * xu[5] / gm;
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

    // shift indices
    ix += lbx - (Order / 2);

    return std::make_tuple(ix, iy, iz);
  }

  template <typename T_float>
  auto local2d(T_float xu[], T_float ms, T_float mom[Sz][Sy][Sx][14])
  {
    using namespace nix;
    using namespace nix::primitives;
    using T_int = xsimd::as_integer_t<T_float>;

    // 2D version
    T_int iz = lbz;
    int   jz = 0;

    T_float wx[Sx] = {0};
    T_float wy[Sy] = {0};
    T_float rc     = 1 / cc;
    T_float rdx    = 1 / dx;
    T_float rdy    = 1 / dy;
    T_float ximin  = xmin + 0.5 * dx * is_odd;
    T_float yimin  = ymin + 0.5 * dy * is_odd;
    T_float xgrid  = xmin + 0.5 * dx;
    T_float ygrid  = ymin + 0.5 * dy;

    // grid indices and positions
    auto ix = digitize(xu[0], ximin, rdx);
    auto iy = digitize(xu[1], yimin, rdy);
    auto xg = xgrid + to_float(ix) * dx;
    auto yg = ygrid + to_float(iy) * dy;

    // weights
    shape_mc<Order>(xu[0], xg, rdx, wx);
    shape_mc<Order>(xu[1], yg, rdy, wy);

    for (int jy = 0; jy < Sy; jy++) {
      for (int jx = 0; jx < Sx; jx++) {
        T_float ww = ms * wx[jx] * wy[jy];
        T_float gm = lorentz_factor(xu[3], xu[4], xu[5], rc);

        mom[jz][jy][jx][index_t] += ww;
        mom[jz][jy][jx][index_x] += ww * xu[3] / gm;
        mom[jz][jy][jx][index_y] += ww * xu[4] / gm;
        mom[jz][jy][jx][index_z] += ww * xu[5] / gm;
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

    // shift indices
    ix += lbx - (Order / 2);
    iy += lby - (Order / 2);

    return std::make_tuple(ix, iy, iz);
  }

  template <typename T_float>
  auto local3d(T_float xu[], T_float ms, T_float mom[Sz][Sy][Sx][14])
  {
    using namespace nix;
    using namespace nix::primitives;

    T_float wx[Sx] = {0};
    T_float wy[Sy] = {0};
    T_float wz[Sz] = {0};
    T_float rc     = 1 / cc;
    T_float rdx    = 1 / dx;
    T_float rdy    = 1 / dy;
    T_float rdz    = 1 / dz;
    T_float ximin  = xmin + 0.5 * dx * is_odd;
    T_float yimin  = ymin + 0.5 * dy * is_odd;
    T_float zimin  = zmin + 0.5 * dz * is_odd;
    T_float xgrid  = xmin + 0.5 * dx;
    T_float ygrid  = ymin + 0.5 * dy;
    T_float zgrid  = zmin + 0.5 * dz;

    // grid indices and positions
    auto ix = digitize(xu[0], ximin, rdx);
    auto iy = digitize(xu[1], yimin, rdy);
    auto iz = digitize(xu[2], zimin, rdz);
    auto xg = xgrid + to_float(ix) * dx;
    auto yg = ygrid + to_float(iy) * dy;
    auto zg = zgrid + to_float(iz) * dz;

    // weights
    shape_mc<Order>(xu[0], xg, rdx, wx);
    shape_mc<Order>(xu[1], yg, rdy, wy);
    shape_mc<Order>(xu[2], zg, rdz, wz);

    for (int jz = 0; jz < Sz; jz++) {
      for (int jy = 0; jy < Sy; jy++) {
        for (int jx = 0; jx < Sx; jx++) {
          T_float ww = ms * wx[jx] * wy[jy] * wz[jz];
          T_float gm = lorentz_factor(xu[3], xu[4], xu[5], rc);

          mom[jz][jy][jx][index_t] += ww;
          mom[jz][jy][jx][index_x] += ww * xu[3] / gm;
          mom[jz][jy][jx][index_y] += ww * xu[4] / gm;
          mom[jz][jy][jx][index_z] += ww * xu[5] / gm;
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

    // shift indices
    ix += lbx - (Order / 2);
    iy += lby - (Order / 2);
    iz += lbz - (Order / 2);

    return std::make_tuple(ix, iy, iz);
  }

  template <typename T_array, typename T_float>
  void global1d(T_array& um, int iz, int iy, int ix, int is, T_float mom[Sz][Sy][Sx][14])
  {
    ix -= ((Order + 1) / 2);
    append_moment1d<Order>(um, iz, iy, ix, is, mom[0][0]);
  }

  template <typename T_array, typename T_float>
  void global2d(T_array& um, int iz, int iy, int ix, int is, T_float mom[Sz][Sy][Sx][14])
  {
    ix -= ((Order + 1) / 2);
    iy -= ((Order + 1) / 2);
    append_moment2d<Order>(um, iz, iy, ix, is, mom[0]);
  }

  template <typename T_array, typename T_float>
  void global3d(T_array& um, int iz, int iy, int ix, int is, T_float mom[Sz][Sy][Sx][14])
  {
    ix -= ((Order + 1) / 2);
    iy -= ((Order + 1) / 2);
    iz -= ((Order + 1) / 2);
    append_moment3d<Order>(um, iz, iy, ix, is, mom);
  }
};

template <int Dim, int Order>
class ScalarMoment : public BaseMoment<Dim, Order>
{
public:
  using BaseMoment<Dim, Order>::BaseMoment; // inherit constructor

  template <typename T_particle, typename T_array>
  void operator()(T_particle& up, T_array& um, json& option)
  {
    this->call_scalar_impl(up, um);
  }
};

template <int Dim, int Order>
class VectorMoment : public BaseMoment<Dim, Order>
{
public:
  using BaseMoment<Dim, Order>::BaseMoment; // inherit constructor

  template <typename T_particle, typename T_array>
  void operator()(T_particle& up, T_array& um, json& option)
  {
    this->call_vector_impl(up, um);
  }
};

} // namespace pic_engine

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif