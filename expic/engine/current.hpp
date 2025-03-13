// -*- C++ -*-
#ifndef _ENGINE_CURRENT_HPP_
#define _ENGINE_CURRENT_HPP_

#include "nix/nix.hpp"

#include "nix/esirkepov.hpp"
#include "nix/primitives.hpp"

namespace engine
{
template <int Dim, int Order>
class Current
{
public:
  static constexpr int size   = Order + 3;
  static constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  int     ns;
  int     lbx;
  int     lby;
  int     lbz;
  int     ubx;
  int     uby;
  int     ubz;
  int     stride_x;
  int     stride_y;
  int     stride_z;
  float64 dx;
  float64 dy;
  float64 dz;
  float64 xmin;
  float64 ymin;
  float64 zmin;

  template <typename T_data>
  Current(const T_data& data)
  {
    ns       = data.Ns;
    lbx      = data.Lbx;
    lby      = data.Lby;
    lbz      = data.Lbz;
    ubx      = data.Ubx + is_odd;
    uby      = data.Uby + is_odd;
    ubz      = data.Ubz + is_odd;
    stride_x = 1;
    stride_y = stride_x * (data.Ubx - data.Lbx + 2);
    stride_z = stride_y * (data.Uby - data.Lby + 2);
    dx       = data.delx;
    dy       = data.dely;
    dz       = data.delz;
    xmin     = data.xlim[0];
    ymin     = data.ylim[0];
    zmin     = data.zlim[0];
  }

  template <typename T_particle, typename T_array>
  void call_scalar_impl_2d(T_particle& up, T_array& uj, float64 delt)
  {
    using namespace nix;
    using namespace nix::primitives;

    // 2D version
    const int iz0 = lbz;

    // clear charge/current density
    fill_all(uj, 0);

    for (int is = 0; is < ns; is++) {
      float64 qs = up[is]->q;

      for (int ip = 0; ip < up[is]->Np; ip++) {
        float64 cur[size][size][4] = {0};

        // local current
        auto xv         = &up[is]->xv(ip, 0);
        auto xu         = &up[is]->xu(ip, 0);
        auto [ix0, iy0] = local2d(xv, xu, qs, delt, cur);

        // deposit to global array
        ix0 += lbx - (Order / 2) - 1;
        iy0 += lby - (Order / 2) - 1;
        append_current2d<Order>(uj, iz0, iy0, ix0, cur);
      }
    }
  }

  template <typename T_particle, typename T_array>
  void call_scalar_impl_3d(T_particle& up, T_array& uj, float64 delt)
  {
    using namespace nix;
    using namespace nix::primitives;

    // clear charge/current density
    fill_all(uj, 0);

    for (int is = 0; is < ns; is++) {
      float64 qs = up[is]->q;

      for (int ip = 0; ip < up[is]->Np; ip++) {
        float64 cur[size][size][size][4] = {0};

        // local current
        auto xv              = &up[is]->xv(ip, 0);
        auto xu              = &up[is]->xu(ip, 0);
        auto [ix0, iy0, iz0] = local3d(xv, xu, qs, delt, cur);

        // deposit to global array
        ix0 += lbx - (Order / 2) - 1;
        iy0 += lby - (Order / 2) - 1;
        iz0 += lbz - (Order / 2) - 1;
        append_current3d<Order>(uj, iz0, iy0, ix0, cur);
      }
    }
  }

  template <typename T_particle, typename T_array>
  void call_vector_impl_2d(T_particle& up, T_array& uj, float64 delt)
  {
    using namespace nix;
    using namespace nix::primitives;
    const simd_i64 index = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

    // clear charge/current density
    fill_all(uj, 0);

    for (int iz = lbz, jz = 0; iz <= ubz; iz++, jz++) {
      for (int iy = lby, jy = 0; iy <= uby; iy++, jy++) {
        for (int ix = lbx, jx = 0; ix <= ubx; ix++, jx++) {
          int ii = jz * stride_z + jy * stride_y + jx * stride_x; // 1D grid index

          // local current
          float64  cur[size][size][4]      = {0}; // scalar
          simd_f64 cur_simd[size][size][4] = {0}; // SIMD register

          // process particles in the cell
          for (int is = 0; is < ns; is++) {
            int  ip_zero = up[is]->pindex(ii);
            int  np_cell = up[is]->pindex(ii + 1) - ip_zero;
            int  np_simd = (np_cell / simd_f64::size) * simd_f64::size;
            auto qs      = up[is]->q;

            //
            // vectorized loop
            //
            for (int ip = ip_zero; ip < ip_zero + np_simd; ip += simd_f64::size) {
              // local SIMD register
              simd_f64 xv[3];
              simd_f64 xu[3];

              // load particles to SIMD register
              xv[0] = simd_f64::gather(&up[is]->xv(ip, 0), index);
              xv[1] = simd_f64::gather(&up[is]->xv(ip, 1), index);
              xv[2] = simd_f64::gather(&up[is]->xv(ip, 2), index);
              xu[0] = simd_f64::gather(&up[is]->xu(ip, 0), index);
              xu[1] = simd_f64::gather(&up[is]->xu(ip, 1), index);
              xu[2] = simd_f64::gather(&up[is]->xu(ip, 2), index);

              local2d(xv, xu, qs, delt, cur_simd);
            }

            //
            // scalar loop for reminder
            //
            for (int ip = ip_zero + np_simd; ip < ip_zero + np_cell; ip++) {
              auto xv = &up[is]->xv(ip, 0);
              auto xu = &up[is]->xu(ip, 0);

              local2d(xv, xu, qs, delt, cur);
            }
          }

          // deposit to global array
          global2d(uj, iz, iy, ix, cur);
          global2d(uj, iz, iy, ix, cur_simd);
        }
      }
    }
  }

  template <typename T_particle, typename T_array>
  void call_vector_impl_3d(T_particle& up, T_array& uj, float64 delt)
  {
    using namespace nix;
    using namespace nix::primitives;
    const simd_i64 index = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

    // clear charge/current density
    fill_all(uj, 0);

    for (int iz = lbz, jz = 0; iz <= ubz; iz++, jz++) {
      for (int iy = lby, jy = 0; iy <= uby; iy++, jy++) {
        for (int ix = lbx, jx = 0; ix <= ubx; ix++, jx++) {
          int ii = jz * stride_z + jy * stride_y + jx * stride_x; // 1D grid index

          // local current
          float64  cur[size][size][size][4]      = {0}; // scalar
          simd_f64 cur_simd[size][size][size][4] = {0}; // SIMD register

          // process particles in the cell
          for (int is = 0; is < ns; is++) {
            int  ip_zero = up[is]->pindex(ii);
            int  np_cell = up[is]->pindex(ii + 1) - ip_zero;
            int  np_simd = (np_cell / simd_f64::size) * simd_f64::size;
            auto qs      = up[is]->q;

            //
            // vectorized loop
            //
            for (int ip = ip_zero; ip < ip_zero + np_simd; ip += simd_f64::size) {
              // local SIMD register
              simd_f64 xv[3];
              simd_f64 xu[3];

              // load particles to SIMD register
              xv[0] = simd_f64::gather(&up[is]->xv(ip, 0), index);
              xv[1] = simd_f64::gather(&up[is]->xv(ip, 1), index);
              xv[2] = simd_f64::gather(&up[is]->xv(ip, 2), index);
              xu[0] = simd_f64::gather(&up[is]->xu(ip, 0), index);
              xu[1] = simd_f64::gather(&up[is]->xu(ip, 1), index);
              xu[2] = simd_f64::gather(&up[is]->xu(ip, 2), index);

              local3d(xv, xu, qs, delt, cur_simd);
            }

            //
            // scalar loop for reminder
            //
            for (int ip = ip_zero + np_simd; ip < ip_zero + np_cell; ip++) {
              auto xv = &up[is]->xv(ip, 0);
              auto xu = &up[is]->xu(ip, 0);

              local3d(xv, xu, qs, delt, cur);
            }
          }

          // deposit to global array
          global3d(uj, iz, iy, ix, cur);
          global3d(uj, iz, iy, ix, cur_simd);
        }
      }
    }
  }

  template <typename T_float>
  auto local2d(T_float xv[], T_float xu[], float64 qs, float64 delt, T_float cur[size][size][4])
  {
    using nix::esirkepov::shift_weights;
    using nix::esirkepov::deposit2d;
    using T_int = xsimd::as_integer_t<T_float>;

    T_float ss[2][2][size] = {0};

    T_float q     = qs;
    T_float vz    = (xu[2] - xv[2]) / delt;
    T_float rdx   = 1 / dx;
    T_float rdy   = 1 / dy;
    T_float ximin = xmin + 0.5 * dx * is_odd;
    T_float yimin = ymin + 0.5 * dy * is_odd;
    T_float xgrid = xmin + 0.5 * dx;
    T_float ygrid = ymin + 0.5 * dy;
    float64 dxdt  = dx / delt;
    float64 dydt  = dy / delt;

    //
    // -*- weights before move -*-
    //
    // grid indices and positions
    auto ix0 = digitize(xv[0], ximin, rdx);
    auto iy0 = digitize(xv[1], yimin, rdy);
    auto xg0 = xgrid + to_float(ix0) * dx;
    auto yg0 = ygrid + to_float(iy0) * dy;

    // weights
    shape_mc<Order>(xv[0], xg0, rdx, &ss[0][0][1]);
    shape_mc<Order>(xv[1], yg0, rdy, &ss[0][1][1]);

    //
    // -*- weights after move -*-
    //
    // grid indices and positions
    auto ix1 = digitize(xu[0], ximin, rdx);
    auto iy1 = digitize(xu[1], yimin, rdy);
    auto xg1 = xgrid + to_float(ix1) * dx;
    auto yg1 = ygrid + to_float(iy1) * dy;

    // weights
    shape_mc<Order>(xu[0], xg1, rdx, &ss[1][0][1]);
    shape_mc<Order>(xu[1], yg1, rdy, &ss[1][1][1]);

    // shift weights according to particle movement
    T_int shift[2] = {ix1 - ix0, iy1 - iy0};
    shift_weights<2, Order>(shift, ss[1]);

    //
    // -*- accumulate current via density decomposition -*-
    //
    deposit2d<Order>(dxdt, dydt, vz, q, ss, cur);

    return std::make_tuple(ix0, iy0);
  }

  template <typename T_float>
  auto local3d(T_float xv[], T_float xu[], float64 qs, float64 delt,
               T_float cur[size][size][size][4])
  {
    using nix::esirkepov::shift_weights;
    using nix::esirkepov::deposit3d;
    using T_int = xsimd::as_integer_t<T_float>;

    T_float ss[2][3][size] = {0};

    T_float q     = qs;
    T_float rdx   = 1 / dx;
    T_float rdy   = 1 / dy;
    T_float rdz   = 1 / dz;
    T_float ximin = xmin + 0.5 * dx * is_odd;
    T_float yimin = ymin + 0.5 * dy * is_odd;
    T_float zimin = zmin + 0.5 * dz * is_odd;
    T_float xgrid = xmin + 0.5 * dx;
    T_float ygrid = ymin + 0.5 * dy;
    T_float zgrid = zmin + 0.5 * dz;
    float64 dxdt  = dx / delt;
    float64 dydt  = dy / delt;
    float64 dzdt  = dz / delt;

    //
    // -*- weights before move -*-
    //
    // grid indices and positions
    auto ix0 = digitize(xv[0], ximin, rdx);
    auto iy0 = digitize(xv[1], yimin, rdy);
    auto iz0 = digitize(xv[2], zimin, rdz);
    auto xg0 = xgrid + to_float(ix0) * dx;
    auto yg0 = ygrid + to_float(iy0) * dy;
    auto zg0 = zgrid + to_float(iz0) * dz;

    // weights
    shape_mc<Order>(xv[0], xg0, rdx, &ss[0][0][1]);
    shape_mc<Order>(xv[1], yg0, rdy, &ss[0][1][1]);
    shape_mc<Order>(xv[2], zg0, rdz, &ss[0][2][1]);

    //
    // -*- weights after move -*-
    //
    // grid indices and positions
    auto ix1 = digitize(xu[0], ximin, rdx);
    auto iy1 = digitize(xu[1], yimin, rdy);
    auto iz1 = digitize(xu[2], zimin, rdz);
    auto xg1 = xgrid + to_float(ix1) * dx;
    auto yg1 = ygrid + to_float(iy1) * dy;
    auto zg1 = zgrid + to_float(iz1) * dz;

    // weights
    shape_mc<Order>(xu[0], xg1, rdx, &ss[1][0][1]);
    shape_mc<Order>(xu[1], yg1, rdy, &ss[1][1][1]);
    shape_mc<Order>(xu[2], zg1, rdz, &ss[1][2][1]);

    // shift weights according to particle movement
    T_int shift[3] = {ix1 - ix0, iy1 - iy0, iz1 - iz0};
    shift_weights<3, Order>(shift, ss[1]);

    //
    // -*- accumulate current via density decomposition -*-
    //
    deposit3d<Order>(dxdt, dydt, dzdt, q, ss, cur);

    return std::make_tuple(ix0, iy0, iz0);
  }

  template <typename T_array, typename T_float>
  void global2d(T_array& uj, int iz, int iy, int ix, T_float cur[size][size][4])
  {
    ix -= ((Order + 1) / 2) + 1;
    iy -= ((Order + 1) / 2) + 1;
    iz -= ((Order + 1) / 2) + 1;
    append_current2d<Order>(uj, iz, iy, ix, cur);
  }

  template <typename T_array, typename T_float>
  void global3d(T_array& uj, int iz, int iy, int ix, T_float cur[size][size][size][4])
  {
    ix -= ((Order + 1) / 2) + 1;
    iy -= ((Order + 1) / 2) + 1;
    iz -= ((Order + 1) / 2) + 1;
    append_current3d<Order>(uj, iz, iy, ix, cur);
  }
};

template <int Dim, int Order>
class ScalarCurrent : public Current<Dim, Order>
{
public:
  template <typename T_data>
  ScalarCurrent(const T_data& data) : Current<Dim, Order>(data)
  {
  }

  template <typename T_particle, typename T_array>
  void operator()(T_particle& up, T_array& uj, float64 delt)
  {
    if constexpr (Dim == 1) {
      //
      // 1D version
      //
      static_assert([] { return false; }(), "1D is not yet supported");
    }

    if constexpr (Dim == 2) {
      //
      // 2D version
      //
      this->call_scalar_impl_2d(up, uj, delt);
    }

    if constexpr (Dim == 3) {
      //
      // 3D version
      //
      this->call_scalar_impl_3d(up, uj, delt);
    }
  }
};

template <int Dim, int Order>
class VectorCurrent : public Current<Dim, Order>
{
public:
  template <typename T_data>
  VectorCurrent(const T_data& data) : Current<Dim, Order>(data)
  {
  }

  template <typename T_particle, typename T_array>
  void operator()(T_particle& up, T_array& uj, float64 delt)
  {
    if constexpr (Dim == 1) {
      //
      // 1D version
      //
      static_assert([] { return false; }(), "1D is not yet supported");
    }

    if constexpr (Dim == 2) {
      //
      // 2D version
      //
      this->call_vector_impl_2d(up, uj, delt);
    }

    if constexpr (Dim == 3) {
      //
      // 3D version
      //
      this->call_vector_impl_3d(up, uj, delt);
    }
  }
};

} // namespace engine

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif