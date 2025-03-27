// -*- C++ -*-
#ifndef _INTERP_HPP_
#define _INTERP_HPP_

#include "nix.hpp"
#include "xsimd/xsimd.hpp"

NIX_NAMESPACE_BEGIN

namespace interp
{
namespace
{
template <int Order, typename T_array, typename T_float>
static auto interp1d_impl_sorted(T_array& eb, int iz0, int iy0, int ix0, int ik,
                                 T_float wx[Order + 2], T_float dt)
{
  int     iz       = iz0;
  int     iy       = iy0;
  T_float result_x = 0;
  for (int jx = 0, ix = ix0; jx < Order + 2; jx++, ix++) {
    result_x += eb(iz, iy, ix, ik) * wx[jx];
  }

  return result_x * dt;
}

template <int Order, typename T_array, typename T_int, typename T_float>
static auto interp1d_impl_unsorted(T_array& eb, int iz0, int iy0, T_int ix0, int ik,
                                   T_float wx[Order + 2], T_float dt)
{
  auto stride_z = get_stride(eb, 0);
  auto stride_y = get_stride(eb, 1);
  auto stride_x = get_stride(eb, 2);
  auto stride_c = get_stride(eb, 3);
  auto pointer  = get_data_pointer(eb);

  T_int   index_z  = iz0 * stride_z;
  T_int   index_y  = iy0 * stride_y + index_z;
  T_float result_x = 0;
  for (int jx = 0; jx < Order + 2; jx++) {
    T_int index_x = (ix0 + jx) * stride_x + index_y;
    T_int index   = index_x + ik * stride_c;
    result_x += simd_f64::gather(pointer, index) * wx[jx];
  }

  return result_x * dt;
}

template <int Order, typename T_array, typename T_float>
static auto interp2d_impl_sorted(T_array& eb, int iz0, int iy0, int ix0, int ik,
                                 T_float wy[Order + 2], T_float wx[Order + 2], T_float dt)
{
  int     iz       = iz0;
  T_float result_y = 0;
  for (int jy = 0, iy = iy0; jy < Order + 2; jy++, iy++) {
    T_float result_x = 0;
    for (int jx = 0, ix = ix0; jx < Order + 2; jx++, ix++) {
      result_x += eb(iz, iy, ix, ik) * wx[jx];
    }
    result_y += result_x * wy[jy];
  }

  return result_y * dt;
}

template <int Order, typename T_array, typename T_int, typename T_float>
static auto interp2d_impl_unsorted(T_array& eb, int iz0, T_int iy0, T_int ix0, int ik,
                                   T_float wy[Order + 2], T_float wx[Order + 2], T_float dt)
{
  auto stride_z = get_stride(eb, 0);
  auto stride_y = get_stride(eb, 1);
  auto stride_x = get_stride(eb, 2);
  auto stride_c = get_stride(eb, 3);
  auto pointer  = get_data_pointer(eb);

  T_int   index_z  = iz0 * stride_z;
  T_float result_y = 0;
  for (int jy = 0; jy < Order + 2; jy++) {
    T_int   index_y  = (iy0 + jy) * stride_y + index_z;
    T_float result_x = 0;
    for (int jx = 0; jx < Order + 2; jx++) {
      T_int index_x = (ix0 + jx) * stride_x + index_y;
      T_int index   = index_x + ik * stride_c;
      result_x += simd_f64::gather(pointer, index) * wx[jx];
    }
    result_y += result_x * wy[jy];
  }

  return result_y * dt;
}

/// implementations of interpolate3d for sorted index
template <int Order, typename T_array, typename T_float>
static auto interp3d_impl_sorted(T_array& eb, int iz0, int iy0, int ix0, int ik,
                                 T_float wz[Order + 2], T_float wy[Order + 2],
                                 T_float wx[Order + 2], T_float dt)
{
  T_float result_z = 0;
  for (int jz = 0, iz = iz0; jz < Order + 2; jz++, iz++) {
    T_float result_y = 0;
    for (int jy = 0, iy = iy0; jy < Order + 2; jy++, iy++) {
      T_float result_x = 0;
      for (int jx = 0, ix = ix0; jx < Order + 2; jx++, ix++) {
        result_x += eb(iz, iy, ix, ik) * wx[jx];
      }
      result_y += result_x * wy[jy];
    }
    result_z += result_y * wz[jz];
  }

  return result_z * dt;
}

/// implementation of interpolate3d for unsorted index
template <int Order, typename T_array, typename T_int, typename T_float>
static auto interp3d_impl_unsorted(T_array& eb, T_int iz0, T_int iy0, T_int ix0, int ik,
                                   T_float wz[Order + 2], T_float wy[Order + 2],
                                   T_float wx[Order + 2], T_float dt)
{
  auto stride_z = get_stride(eb, 0);
  auto stride_y = get_stride(eb, 1);
  auto stride_x = get_stride(eb, 2);
  auto stride_c = get_stride(eb, 3);
  auto pointer  = get_data_pointer(eb);

  T_float result_z = 0;
  for (int jz = 0; jz < Order + 2; jz++) {
    T_int   index_z  = (iz0 + jz) * stride_z;
    T_float result_y = 0;
    for (int jy = 0; jy < Order + 2; jy++) {
      T_int   index_y  = (iy0 + jy) * stride_y + index_z;
      T_float result_x = 0;
      for (int jx = 0; jx < Order + 2; jx++) {
        T_int index_x = (ix0 + jx) * stride_x + index_y;
        T_int index   = index_x + ik * stride_c;
        result_x += simd_f64::gather(pointer, index) * wx[jx];
      }
      result_y += result_x * wy[jy];
    }
    result_z += result_y * wz[jz];
  }

  return result_z * dt;
}
} // namespace

template <int Order, typename T_int, typename T_float>
static void shift_weights(T_int shift, T_float ww[Order + 2])
{
  constexpr bool is_scalar = std::is_integral_v<T_int> && std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    if (shift > 0) {
      for (int ii = Order + 1; ii > 0; ii--) {
        ww[ii] = ww[ii - 1];
      }
      ww[0] = 0;
    }
  } else if constexpr (is_vector == true) {
    using value_type = typename T_float::value_type;

    auto cond = xsimd::batch_bool_cast<value_type>(shift > 0);
    for (int ii = Order + 1; ii > 0; ii--) {
      ww[ii] = xsimd::select(cond, ww[ii - 1], ww[ii]);
    }
    ww[0] = xsimd::select(cond, T_float(0), ww[0]);
  } else {
    static_assert([] { return false; }(), "Invalid combination of types");
  }
}

template <int Order, typename T_array, typename T_int, typename T_float>
static auto interp1d(T_array& eb, int iz0, int iy0, T_int ix0, int ik, T_float wx[Order + 2],
                     T_float dt)
{
  constexpr bool is_sorted = std::is_integral_v<T_int>;

  if constexpr (is_sorted == true) {
    // both scalar and vector implementation for scalar index
    return interp1d_impl_sorted<Order>(eb, iz0, iy0, ix0, ik, wx, dt);
  } else {
    // vector implementation for vector index
    return interp1d_impl_unsorted<Order>(eb, iz0, iy0, ix0, ik, wx, dt);
  }
}

template <int Order, typename T_array, typename T_int, typename T_float>
static auto interp2d(T_array& eb, int iz0, T_int iy0, T_int ix0, int ik, T_float wy[Order + 2],
                     T_float wx[Order + 2], T_float dt)
{
  constexpr bool is_sorted = std::is_integral_v<T_int>;

  if constexpr (is_sorted == true) {
    // both scalar and vector implementation for scalar index
    return interp2d_impl_sorted<Order>(eb, iz0, iy0, ix0, ik, wy, wx, dt);
  } else {
    // vector implementation for vector index
    return interp2d_impl_unsorted<Order>(eb, iz0, iy0, ix0, ik, wy, wx, dt);
  }
}

///
/// @brief calculate electromagnetic field at particle position by interpolation
///
/// @param[in] eb  electromagnetic field (4D array)
/// @param[in] iz0 first z-index of eb
/// @param[in] iy0 first y-index of eb
/// @param[in] ix0 first x-index of eb
/// @param[in] ik  index for electromagnetic field component
/// @param[in] wz  weight in z direction
/// @param[in] wy  weight in y direction
/// @param[in] wx  weight in x direction
/// @param[in] dt  time step (multiplied to the returned electromagnetic field)
///
template <int Order, typename T_array, typename T_int, typename T_float>
static auto interp3d(T_array& eb, T_int iz0, T_int iy0, T_int ix0, int ik, T_float wz[Order + 2],
                     T_float wy[Order + 2], T_float wx[Order + 2], T_float dt)
{
  constexpr bool is_sorted = std::is_integral_v<T_int>;

  if constexpr (is_sorted == true) {
    // both scalar and vector implementation for scalar index
    return interp3d_impl_sorted<Order>(eb, iz0, iy0, ix0, ik, wz, wy, wx, dt);
  } else {
    // vector implementation for vector index
    return interp3d_impl_unsorted<Order>(eb, iz0, iy0, ix0, ik, wz, wy, wx, dt);
  }
}

} // namespace interp

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
