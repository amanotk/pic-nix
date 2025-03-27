// -*- C++ -*-
#ifndef _PARTICLE_PRIMITIVES_HPP_
#define _PARTICLE_PRIMITIVES_HPP_

#include "nix.hpp"
#include "xsimd/xsimd.hpp"

NIX_NAMESPACE_BEGIN

namespace primitives
{
/// convert to integer (scalar or vector)
template <typename T_float>
static auto to_int(T_float x)
{
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    return static_cast<int>(x);
  } else if constexpr (is_vector == true) {
    return xsimd::to_int(x);
  } else {
    static_assert([] { return false; }(), "Only scalar or vector of floating point is allowed");
  }
}

/// convert to floating point (scalar or vector)
template <typename T_int>
static auto to_float(T_int x)
{
  constexpr bool is_scalar = std::is_integral_v<T_int>;
  constexpr bool is_vector = std::is_same_v<T_int, simd_i32> || std::is_same_v<T_int, simd_i64>;

  if constexpr (is_scalar == true) {
    return static_cast<float64>(x);
  } else if constexpr (is_vector == true) {
    return xsimd::to_float(x);
  } else {
    static_assert([] { return false; }(), "Only scalar or vector of floating point is allowed");
  }
}

/// digitize for calculating grid index for particles
template <typename T_float>
static auto digitize(T_float x, T_float xmin, T_float rdx)
{
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    return static_cast<int>(floor((x - xmin) * rdx));
  } else if constexpr (is_vector == true) {
    return xsimd::to_int(xsimd::floor((x - xmin) * rdx));
  } else {
    static_assert([] { return false; }(), "Only scalar or vector of floating point is allowed");
  }
}

/// return sign of argument
template <typename T_float>
static auto sign(T_float x)
{
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    return copysign(1.0, x);
  } else if constexpr (is_vector == true) {
    return xsimd::copysign(T_float(1.0), x);
  } else {
    static_assert([] { return false; }(), "Only scalar or vector of floating point is allowed");
  }
}

/// return minimum of two arguments
template <typename T_float>
static auto min(T_float x, T_float y)
{
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    return std::min(x, y);
  } else if constexpr (is_vector == true) {
    return xsimd::min(x, y);
  } else {
    static_assert([] { return false; }(), "Only scalar or vector of floating point is allowed");
  }
}

/// return maximum of two arguments
template <typename T_float>
static auto max(T_float x, T_float y)
{
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    return std::max(x, y);
  } else if constexpr (is_vector == true) {
    return xsimd::max(x, y);
  } else {
    static_assert([] { return false; }(), "Only scalar or vector of floating point is allowed");
  }
}

/// return absolute value of argument
template <typename T_float>
static auto abs(T_float x)
{
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    return std::abs(x);
  } else if constexpr (is_vector == true) {
    return xsimd::abs(x);
  } else {
    static_assert([] { return false; }(), "Only scalar or vector of floating point is allowed");
  }
}

/// ternary operator
template <typename T_float, typename T_bool>
static auto ifthenelse(T_bool cond, T_float x, T_float y)
{
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    return cond ? x : y;
  } else if constexpr (is_vector == true) {
    return xsimd::select(cond, x, y);
  } else {
    static_assert([] { return false; }(), "Only scalar or vector of floating point is allowed");
  }
}

/// return square root
template <typename T_float>
static auto sqrt(T_float x)
{
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    return std::sqrt(x);
  } else if constexpr (is_vector == true) {
    return xsimd::sqrt(x);
  } else {
    static_assert([] { return false; }(), "Only scalar or vector of floating point is allowed");
  }
}

/// return Lorentz factor for given four velocity and rc = 1/c
template <typename T_float>
static auto lorentz_factor(T_float ux, T_float uy, T_float uz, T_float rc)
{
  return sqrt(1 + (ux * ux + uy * uy + uz * uz) * rc * rc);
}

/// Boris pusher for equation of motion
template <typename T_float>
static void push_boris(T_float& ux, T_float& uy, T_float& uz, T_float ex, T_float ey, T_float ez,
                       T_float bx, T_float by, T_float bz, T_float cc)
{
  T_float gm, bb, vx, vy, vz;

  ux += ex;
  uy += ey;
  uz += ez;

  // Lorentz factor for rotation
  gm = 1 / sqrt(cc * cc + ux * ux + uy * uy + uz * uz);

  bx *= gm;
  by *= gm;
  bz *= gm;
  bb = 2.0 / (1.0 + bx * bx + by * by + bz * bz);

  vx = ux + (uy * bz - uz * by);
  vy = uy + (uz * bx - ux * bz);
  vz = uz + (ux * by - uy * bx);

  ux += (vy * bz - vz * by) * bb + ex;
  uy += (vz * bx - vx * bz) * bb + ey;
  uz += (vx * by - vy * bx) * bb + ez;
}

/// Vay (2008) pusher for equation of motion
template <typename T_float>
static void push_vay(T_float& ux, T_float& uy, T_float& uz, T_float ex, T_float ey, T_float ez,
                     T_float bx, T_float by, T_float bz, T_float cc)
{
  T_float gm, bb, bu, xx, yy, vx, vy, vz;

  gm = 1 / sqrt(cc * cc + ux * ux + uy * uy + uz * uz);
  vx = ux + 2 * ex + gm * (uy * bz - uz * by);
  vy = uy + 2 * ey + gm * (uz * bx - ux * bz);
  vz = uz + 2 * ez + gm * (ux * by - uy * bx);

  // use Lorentz factor defiend by Eq.(11) of Vay (2008)
  gm = (cc * cc + vx * vx + vy * vy + vz * vz);
  bb = bx * bx + by * by + bz * bz;
  bu = bx * vx + by * vy + bz * vz;
  xx = gm - bb;
  yy = bb + bu * bu;
  gm = 1 / sqrt(0.5 * (xx + sqrt(xx * xx + 4 * yy)));

  bx *= gm;
  by *= gm;
  bz *= gm;
  bu = bx * vx + by * vy + bz * vz;
  bb = 1.0 / (1.0 + bx * bx + by * by + bz * bz);

  ux = (vx + bu * bx + (vy * bz - vz * by)) * bb;
  uy = (vy + bu * by + (vz * bx - vx * bz)) * bb;
  uz = (vz + bu * bz + (vx * by - vy * bx)) * bb;
}

/// Higuera-Cary (2017) pusher for equation of motion
template <typename T_float>
static void push_higuera_cary(T_float& ux, T_float& uy, T_float& uz, T_float ex, T_float ey,
                              T_float ez, T_float bx, T_float by, T_float bz, T_float cc)
{
  T_float gm, bb, bu, xx, yy, vx, vy, vz;

  ux += ex;
  uy += ey;
  uz += ez;

  // use Lorentz factor defiend by Eq.(20) of Higuera-Cary (2017)
  gm = cc * cc + ux * ux + uy * uy + uz * uz;
  bb = bx * bx + by * by + bz * bz;
  bu = bx * ux + by * uy + bz * uz;
  xx = gm - bb;
  yy = bb + bu * bu;
  gm = 1 / sqrt(0.5 * (xx + sqrt(xx * xx + 4 * yy)));

  bx *= gm;
  by *= gm;
  bz *= gm;
  bb = 2.0 / (1.0 + bx * bx + by * by + bz * bz);

  vx = ux + (uy * bz - uz * by);
  vy = uy + (uz * bx - ux * bz);
  vz = uz + (ux * by - uy * bx);

  ux += (vy * bz - vz * by) * bb + ex;
  uy += (vz * bx - vx * bz) * bb + ey;
  uz += (vx * by - vy * bx) * bb + ez;
}

/// implementation of first-order particle shape function
template <typename T_float>
static void shape_mc1(T_float x, T_float X, T_float rdx, T_float s[2])
{
  T_float delta = (x - X) * rdx;

  s[0] = 1 - delta;
  s[1] = delta;
}

/// implementation of second-order particle shape function
template <typename T_float>
static void shape_mc2(T_float x, T_float X, T_float rdx, T_float s[3])
{
  T_float delta = (x - X) * rdx;

  T_float w0 = delta;
  T_float w1 = 0.5 - w0;
  T_float w2 = 0.5 + w0;

  s[0] = 0.50 * w1 * w1;
  s[1] = 0.75 - w0 * w0;
  s[2] = 0.50 * w2 * w2;
}

/// implementation of third-order particle shape function
template <typename T_float>
static void shape_mc3(T_float x, T_float X, T_float rdx, T_float s[4])
{
  const T_float a     = 1 / 6.0;
  T_float       delta = (x - X) * rdx;

  T_float w1      = delta;
  T_float w2      = 1 - delta;
  T_float w1_pow2 = w1 * w1;
  T_float w2_pow2 = w2 * w2;
  T_float w1_pow3 = w1_pow2 * w1;
  T_float w2_pow3 = w2_pow2 * w2;

  s[0] = a * w2_pow3;
  s[1] = a * (4 - 6 * w1_pow2 + 3 * w1_pow3);
  s[2] = a * (4 - 6 * w2_pow2 + 3 * w2_pow3);
  s[3] = a * w1_pow3;
}

/// implementation of fourth-order particle shape function
template <typename T_float>
static void shape_mc4(T_float x, T_float X, T_float rdx, T_float s[5])
{
  const T_float a     = 1 / 384.0;
  const T_float b     = 1 / 96.0;
  const T_float c     = 115 / 192.0;
  const T_float d     = 1 / 8.0;
  T_float       delta = (x - X) * rdx;

  T_float w1      = 1 + delta;
  T_float w2      = 1 - delta;
  T_float w3      = 1 + delta * 2;
  T_float w4      = 1 - delta * 2;
  T_float w0_pow2 = delta * delta;
  T_float w1_pow2 = w1 * w1;
  T_float w2_pow2 = w2 * w2;
  T_float w1_pow3 = w1_pow2 * w1;
  T_float w2_pow3 = w2_pow2 * w2;
  T_float w1_pow4 = w1_pow3 * w1;
  T_float w2_pow4 = w2_pow3 * w2;
  T_float w3_pow4 = w3 * w3 * w3 * w3;
  T_float w4_pow4 = w4 * w4 * w4 * w4;

  s[0] = a * w4_pow4;
  s[1] = b * (55 + 20 * w1 - 120 * w1_pow2 + 80 * w1_pow3 - 16 * w1_pow4);
  s[2] = c + d * w0_pow2 * (2 * w0_pow2 - 5);
  s[3] = b * (55 + 20 * w2 - 120 * w2_pow2 + 80 * w2_pow3 - 16 * w2_pow4);
  s[4] = a * w3_pow4;
}

/// implementation of first-order particle shape function for WT scheme
template <typename T_float>
static void shape_wt1(T_float x, T_float X, T_float rdx, T_float dt, T_float rdt, T_float s[2])
{
  const T_float zero  = 0.0;
  const T_float one   = 1.0;
  T_float       delta = (x - X) * rdx;

  T_float ss = min(one, max(zero, 0.25 * rdt * (1 + 2 * dt - 2 * delta)));

  s[0] = ss;
  s[1] = 1 - ss;
}

/// implementation of second-order particle shape function for WT scheme
template <typename T_float>
static void shape_wt2(T_float x, T_float X, T_float rdx, T_float dt, T_float rdt, T_float s[3])
{
  const T_float zero  = 0.0;
  const T_float one   = 1.0;
  T_float       delta = (x - X) * rdx;

  T_float t1 = ifthenelse(delta < -dt, one, zero);
  T_float t2 = 1 - t1;
  T_float t3 = ifthenelse(delta < +dt, one, zero);
  T_float t4 = 1 - t3;
  T_float w0 = abs(delta);
  T_float w1 = dt - delta;
  T_float w2 = dt + delta;

  T_float s0_1 = w0;
  T_float s1_1 = 1 - w0;
  T_float s2_1 = 0;
  T_float s0_2 = 0.25 * rdt * w1 * w1;
  T_float s1_2 = 0.50 * rdt * (dt * (2 - dt) - w0 * w0);
  T_float s2_2 = 0.25 * rdt * w2 * w2;
  T_float s0_3 = s2_1;
  T_float s1_3 = s1_1;
  T_float s2_3 = s0_1;

  s[0] = s0_1 * t1 + s0_2 * t2 * t3 + s0_3 * t4;
  s[1] = s1_1 * t1 + s1_2 * t2 * t3 + s1_3 * t4;
  s[2] = s2_1 * t1 + s2_2 * t2 * t3 + s2_3 * t4;
}

/// implementation of third-order particle shape function for WT scheme
template <typename T_float>
static void shape_wt3(T_float x, T_float X, T_float rdx, T_float dt, T_float rdt, T_float s[4])
{
  const T_float zero  = 0.0;
  const T_float one   = 1.0;
  const T_float a     = 1 / 96.0;
  const T_float b     = 1 / 24.0;
  const T_float c     = 1 / 12.0;
  const T_float adt   = a * rdt;
  T_float       delta = (x - X) * rdx;

  T_float t1        = ifthenelse(delta < 0.5 - dt, one, zero);
  T_float t2        = 1 - t1;
  T_float t3        = ifthenelse(delta < 0.5 + dt, one, zero);
  T_float t4        = 1 - t3;
  T_float w0        = delta;
  T_float w1        = 1 - delta;
  T_float w2        = 1 + delta;
  T_float w3        = 1 - 2 * delta;
  T_float w4        = 1 + 2 * delta;
  T_float w5        = 2 * dt + w3;
  T_float w6        = 2 * dt - w3;
  T_float w7        = 3 - 2 * delta;
  T_float w0_pow2   = w0 * w0;
  T_float w1_pow2   = w1 * w1;
  T_float w3_pow2   = w3 * w3;
  T_float w3_pow3   = w3_pow2 * w3;
  T_float w4_pow2   = w4 * w4;
  T_float w5_pow3   = w5 * w5 * w5;
  T_float w6_pow3   = w6 * w6 * w6;
  T_float w7_pow2   = w7 * w7;
  T_float dt_pow2   = dt * dt;
  T_float dt_pow3   = dt_pow2 * dt;
  T_float dt_pow2_4 = 4 * dt_pow2;
  T_float s_2_odd   = adt * (-8 * dt_pow3 - 6 * dt * w3_pow2);
  T_float s_2_even  = adt * (-36 * dt_pow2 * w3 - 3 * w3_pow3);

  T_float s0_1 = b * (dt_pow2_4 + 3 * w3_pow2);
  T_float s1_1 = c * (9 - dt_pow2_4 - 12 * w0_pow2);
  T_float s2_1 = b * (dt_pow2_4 + 3 * w4_pow2);
  T_float s3_1 = 0;
  T_float s0_2 = adt * w5_pow3;
  T_float s1_2 = s_2_odd + s_2_even + w1;
  T_float s2_2 = s_2_odd - s_2_even + w0;
  T_float s3_2 = adt * w6_pow3;
  T_float s0_3 = 0;
  T_float s1_3 = b * (dt_pow2_4 + 3 * w7_pow2);
  T_float s2_3 = c * (9 - dt_pow2_4 - 12 * w1_pow2);
  T_float s3_3 = b * (dt_pow2_4 + 3 * w3_pow2);

  s[0] = s0_1 * t1 + s0_2 * t2 * t3 + s0_3 * t4;
  s[1] = s1_1 * t1 + s1_2 * t2 * t3 + s1_3 * t4;
  s[2] = s2_1 * t1 + s2_2 * t2 * t3 + s2_3 * t4;
  s[3] = s3_1 * t1 + s3_2 * t2 * t3 + s3_3 * t4;
}

/// implementation of fourth-order particle shape function for WT scheme
template <typename T_float>
static void shape_wt4(T_float x, T_float X, T_float rdx, T_float dt, T_float rdt, T_float s[5])
{
  const T_float zero  = 0.0;
  const T_float one   = 1.0;
  const T_float a     = 1 / 48.0;
  const T_float b     = 1 / 24.0;
  const T_float c     = 1 / 12.0;
  const T_float d     = 1 / 6.0;
  const T_float adt   = a * rdt;
  const T_float bdt   = b * rdt;
  const T_float cdt   = c * rdt;
  T_float       delta = (x - X) * rdx;

  T_float t1      = ifthenelse(delta < -dt, one, zero);
  T_float t2      = 1 - t1;
  T_float t3      = ifthenelse(delta < +dt, one, zero);
  T_float t4      = 1 - t3;
  T_float w0      = abs(delta);
  T_float w1      = 1 - w0;
  T_float w2      = 1 - delta;
  T_float w3      = 1 + delta;
  T_float w4      = dt - delta;
  T_float w5      = dt + delta;
  T_float w0_pow2 = w0 * w0;
  T_float w0_pow3 = w0_pow2 * w0;
  T_float w0_pow4 = w0_pow3 * w0;
  T_float w1_pow2 = w1 * w1;
  T_float w1_pow3 = w1_pow2 * w1;
  T_float w2_pow3 = w2 * w2 * w2;
  T_float w3_pow3 = w3 * w3 * w3;
  T_float w4_pow4 = w4 * w4 * w4 * w4;
  T_float w5_pow4 = w5 * w5 * w5 * w5;
  T_float dt_pow2 = dt * dt;
  T_float dt_pow3 = dt_pow2 * dt;
  T_float dt_pow4 = dt_pow3 * dt;
  T_float ss1     = -dt_pow4 - 6 * w0_pow2 * dt_pow2 - w0_pow4;
  T_float ss2 =
      3 * dt_pow4 - 8 * dt_pow3 + 18 * w0_pow2 * dt_pow2 + (16 - 24 * w0_pow2) * dt + 3 * w0_pow4;

  T_float s0_1 = d * w0 * (w0_pow2 + dt_pow2);
  T_float s1_1 = d * (4 - 6 * w1_pow2 + 3 * w1_pow3 + (1 - 3 * w0) * dt_pow2);
  T_float s2_1 = d * (4 - 6 * w0_pow2 + 3 * w0_pow3 - (2 - 3 * w0) * dt_pow2);
  T_float s3_1 = d * w1 * (w1_pow2 + dt_pow2);
  T_float s4_1 = 0;
  T_float s0_2 = adt * w4_pow4;
  T_float s1_2 = cdt * (ss1 + 2 * dt_pow3 * w3 + 2 * dt * (-6 * delta + w3_pow3));
  T_float s2_2 = bdt * ss2;
  T_float s3_2 = cdt * (ss1 + 2 * dt_pow3 * w2 + 2 * dt * (+6 * delta + w2_pow3));
  T_float s4_2 = adt * w5_pow4;
  T_float s0_3 = s4_1;
  T_float s1_3 = s3_1;
  T_float s2_3 = s2_1;
  T_float s3_3 = s1_1;
  T_float s4_3 = s0_1;

  s[0] = s0_1 * t1 + s0_2 * t2 * t3 + s0_3 * t4;
  s[1] = s1_1 * t1 + s1_2 * t2 * t3 + s1_3 * t4;
  s[2] = s2_1 * t1 + s2_2 * t2 * t3 + s2_3 * t4;
  s[3] = s3_1 * t1 + s3_2 * t2 * t3 + s3_3 * t4;
  s[4] = s4_1 * t1 + s4_2 * t2 * t3 + s4_3 * t4;
}

///
/// @brief Generic particle shape function
///
/// This function calculate particle assignment weights at grid points using a given order of shape
/// function. For an odd order shape function, the particle position is assume to be
///     X <= x < X + dx.
/// On the other hand, for an even order shape function, the particle position is assume to be
///     X - dx/2 <= x < X + dx/2.
///
/// The weights at the following positions
///     - first-order  : (X, X + dx)
///     - second-order : (X - dx, X, X + dx)
///     - third-order  : (X - dx, X, X + dx, X + 2dx)
///     - fourth-order : (X - 2dx, X - dx, X, X + dx, X + 2dx)
/// will be assigned to s.
///
/// @param[in]  x   particle position
/// @param[in]  X   grid position
/// @param[in]  rdx 1/dx
/// @param[out] s   weights at grid points
///
template <int Order, typename T_float>
static void shape_mc(T_float x, T_float X, T_float rdx, T_float s[Order + 1])
{
  static_assert(Order >= 1 && Order <= 4, "Order must be 1, 2, 3, or 4");

  if constexpr (Order == 1) {
    shape_mc1(x, X, rdx, s);
  } else if constexpr (Order == 2) {
    shape_mc2(x, X, rdx, s);
  } else if constexpr (Order == 3) {
    shape_mc3(x, X, rdx, s);
  } else if constexpr (Order == 4) {
    shape_mc4(x, X, rdx, s);
  }
}

///
/// @brief Generic particle shape function for WT scheme
///
/// This function provides a particle shape function used for the WT scheme, which defines the
/// assignment weights dependent on the time step.
/// Otherwise, the function is the same as the `shape`.
///
/// Reference:
/// - Y. Lu, et al., Journal of Computational Physics 413, 109388 (2020).
///
/// @param[in]  x   particle position
/// @param[in]  X   grid position
/// @param[in]  rdx 1/dx
/// @param[in]  dt  c*dt/dx
/// @param[in]  rdt 1/dt
/// @param[out] s   weights at grid points
///
template <int Order, typename T_float>
static void shape_wt(T_float x, T_float X, T_float rdx, T_float dt, T_float rdt,
                     T_float s[Order + 1])
{
  static_assert(Order >= 1 && Order <= 4, "Order must be 1, 2, 3, or 4");

  if constexpr (Order == 1) {
    shape_wt1(x, X, rdx, dt, rdt, s);
  } else if constexpr (Order == 2) {
    shape_wt2(x, X, rdx, dt, rdt, s);
  } else if constexpr (Order == 3) {
    shape_wt3(x, X, rdx, dt, rdt, s);
  } else if constexpr (Order == 4) {
    shape_wt4(x, X, rdx, dt, rdt, s);
  }
}

/// implementations of append_current3d for different vector sizes
template <typename T_int, typename T_float>
static void append_current_impl(float64* ptr, T_int offset[1], T_float cur[4])
{
  if constexpr (T_float::size == 1) {
    // implementation of vector size 1
    xsimd::store_aligned(ptr + offset[0] + 0, cur[0]);
    xsimd::store_aligned(ptr + offset[0] + 1, cur[1]);
    xsimd::store_aligned(ptr + offset[0] + 2, cur[2]);
    xsimd::store_aligned(ptr + offset[0] + 3, cur[3]);
  } else if constexpr (T_float::size == 2) {
    // implementation of vector size 2
    T_float data[4];

    // transpose
    data[0] = xsimd::zip_lo(cur[0], cur[1]);
    data[1] = xsimd::zip_lo(cur[2], cur[3]);
    data[2] = xsimd::zip_hi(cur[0], cur[1]);
    data[3] = xsimd::zip_hi(cur[2], cur[3]);

    // particle 0
    data[0] += xsimd::load_aligned(ptr + offset[0] + 0);
    xsimd::store_aligned(ptr + offset[0] + 0, data[0]);
    data[1] += xsimd::load_aligned(ptr + offset[0] + 2);
    xsimd::store_aligned(ptr + offset[0] + 2, data[1]);
    // particle 1
    data[2] += xsimd::load_aligned(ptr + offset[1] + 0);
    xsimd::store_aligned(ptr + offset[1] + 0, data[2]);
    data[3] += xsimd::load_aligned(ptr + offset[1] + 2);
    xsimd::store_aligned(ptr + offset[1] + 2, data[3]);
  } else if constexpr (T_float::size == 4) {
    // implementation of vector size 4
    T_float data[4];

    // transpose
    auto p0 = xsimd::zip_lo(cur[0], cur[2]);
    auto p1 = xsimd::zip_hi(cur[0], cur[2]);
    auto p2 = xsimd::zip_lo(cur[1], cur[3]);
    auto p3 = xsimd::zip_hi(cur[1], cur[3]);
    data[0] = xsimd::zip_lo(p0, p2);
    data[1] = xsimd::zip_hi(p0, p2);
    data[2] = xsimd::zip_lo(p1, p3);
    data[3] = xsimd::zip_hi(p1, p3);

    // particle 0
    data[0] += xsimd::load_aligned(ptr + offset[0]);
    xsimd::store_aligned(ptr + offset[0], data[0]);
    // particle 1
    data[1] += xsimd::load_aligned(ptr + offset[1]);
    xsimd::store_aligned(ptr + offset[1], data[1]);
    // particle 2
    data[2] += xsimd::load_aligned(ptr + offset[2]);
    xsimd::store_aligned(ptr + offset[2], data[2]);
    // particle 3
    data[3] += xsimd::load_aligned(ptr + offset[3]);
    xsimd::store_aligned(ptr + offset[3], data[3]);
  } else if constexpr (T_float::size == 8) {
    // implementation of vector size 8
    const T_float zero = 0;
    const auto    mask = (xsimd::detail::make_sequence_as_batch<T_float>() < 4);

    T_float data[8];

    // transpose
    auto p0 = xsimd::zip_lo(cur[0], cur[2]);
    auto p1 = xsimd::zip_hi(cur[0], cur[2]);
    auto p2 = xsimd::zip_lo(cur[1], cur[3]);
    auto p3 = xsimd::zip_hi(cur[1], cur[3]);
    auto q0 = xsimd::zip_lo(p0, p2);
    auto q1 = xsimd::zip_hi(p0, p2);
    auto q2 = xsimd::zip_lo(p1, p3);
    auto q3 = xsimd::zip_hi(p1, p3);
    data[0] = xsimd::select(mask, q0, zero);
    data[1] = xsimd::select(mask, xsimd::rotate_right<4>(q0), zero);
    data[2] = xsimd::select(mask, q1, zero);
    data[3] = xsimd::select(mask, xsimd::rotate_right<4>(q1), zero);
    data[4] = xsimd::select(mask, q2, zero);
    data[5] = xsimd::select(mask, xsimd::rotate_right<4>(q2), zero);
    data[6] = xsimd::select(mask, q3, zero);
    data[7] = xsimd::select(mask, xsimd::rotate_right<4>(q3), zero);

    // particle 0
    data[0] += xsimd::load_aligned(ptr + offset[0]);
    xsimd::store_aligned(ptr + offset[0], data[0]);
    // particle 1
    data[1] += xsimd::load_aligned(ptr + offset[1]);
    xsimd::store_aligned(ptr + offset[1], data[1]);
    // particle 2
    data[2] += xsimd::load_aligned(ptr + offset[2]);
    xsimd::store_aligned(ptr + offset[2], data[2]);
    // particle 3
    data[3] += xsimd::load_aligned(ptr + offset[3]);
    xsimd::store_aligned(ptr + offset[3], data[3]);
    // particle 4
    data[4] += xsimd::load_aligned(ptr + offset[4]);
    xsimd::store_aligned(ptr + offset[4], data[4]);
    // particle 5
    data[5] += xsimd::load_aligned(ptr + offset[5]);
    xsimd::store_aligned(ptr + offset[5], data[5]);
    // particle 6
    data[6] += xsimd::load_aligned(ptr + offset[6]);
    xsimd::store_aligned(ptr + offset[6], data[6]);
    // particle 7
    data[7] += xsimd::load_aligned(ptr + offset[7]);
    xsimd::store_aligned(ptr + offset[7], data[7]);
  } else {
    static_assert([] { return false; }(), "Invalid vector size");
  }
}

/// append local curent contribution to global current array
template <int Order, typename T_array, typename T_int, typename T_float>
static void append_current1d(T_array& uj, int iz, int iy, T_int ix0, T_float current[Order + 3][4])
{
  constexpr int  size      = Order + 3;
  constexpr bool is_scalar = std::is_integral_v<T_int> && std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  constexpr bool is_sorted = std::is_integral_v<T_int>;

  if constexpr (is_scalar == true) {
    // naive scalar implementation
    for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
      uj(iz, iy, ix, 0) += current[jx][0];
      uj(iz, iy, ix, 1) += current[jx][1];
      uj(iz, iy, ix, 2) += current[jx][2];
      uj(iz, iy, ix, 3) += current[jx][3];
    }
  } else if constexpr (is_vector == true && is_sorted == true) {
    // all particle contributions are added to the same grid point
    for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
      uj(iz, iy, ix, 0) += xsimd::reduce_add(current[jx][0]);
      uj(iz, iy, ix, 1) += xsimd::reduce_add(current[jx][1]);
      uj(iz, iy, ix, 2) += xsimd::reduce_add(current[jx][2]);
      uj(iz, iy, ix, 3) += xsimd::reduce_add(current[jx][3]);
    }
  } else if constexpr (is_vector == true && is_sorted == false) {
    // particle contributions are added to different grid points
    auto stride_z = get_stride(uj, 0);
    auto stride_y = get_stride(uj, 1);
    auto stride_x = get_stride(uj, 2);
    auto stride_c = get_stride(uj, 3);
    auto pointer  = get_data_pointer(uj);

    alignas(64) int64 offset[T_float::size];

    for (int jx = 0; jx < size; jx++) {
      T_int ix = ix0 + jx;
      xsimd::store_aligned(offset, iz * stride_z + iy * stride_y + ix * stride_x);
      append_current_impl(pointer, offset, current[jx]);
    }
  } else {
    static_assert([] { return false; }(), "Invalid combination of types");
  }
}

/// append local curent contribution to global current array
template <int Order, typename T_array, typename T_int, typename T_float>
static void append_current2d(T_array& uj, int iz, T_int iy0, T_int ix0,
                             T_float current[Order + 3][Order + 3][4])
{
  constexpr int  size      = Order + 3;
  constexpr bool is_scalar = std::is_integral_v<T_int> && std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  constexpr bool is_sorted = std::is_integral_v<T_int>;

  if constexpr (is_scalar == true) {
    // naive scalar implementation
    for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
      for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
        uj(iz, iy, ix, 0) += current[jy][jx][0];
        uj(iz, iy, ix, 1) += current[jy][jx][1];
        uj(iz, iy, ix, 2) += current[jy][jx][2];
        uj(iz, iy, ix, 3) += current[jy][jx][3];
      }
    }
  } else if constexpr (is_vector == true && is_sorted == true) {
    // all particle contributions are added to the same grid point
    for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
      for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
        uj(iz, iy, ix, 0) += xsimd::reduce_add(current[jy][jx][0]);
        uj(iz, iy, ix, 1) += xsimd::reduce_add(current[jy][jx][1]);
        uj(iz, iy, ix, 2) += xsimd::reduce_add(current[jy][jx][2]);
        uj(iz, iy, ix, 3) += xsimd::reduce_add(current[jy][jx][3]);
      }
    }
  } else if constexpr (is_vector == true && is_sorted == false) {
    // particle contributions are added to different grid points
    auto stride_z = get_stride(uj, 0);
    auto stride_y = get_stride(uj, 1);
    auto stride_x = get_stride(uj, 2);
    auto stride_c = get_stride(uj, 3);
    auto pointer  = get_data_pointer(uj);

    alignas(64) int64 offset[T_float::size];

    for (int jy = 0; jy < size; jy++) {
      T_int iy = iy0 + jy;
      for (int jx = 0; jx < size; jx++) {
        T_int ix = ix0 + jx;
        xsimd::store_aligned(offset, iz * stride_z + iy * stride_y + ix * stride_x);
        append_current_impl(pointer, offset, current[jy][jx]);
      }
    }
  } else {
    static_assert([] { return false; }(), "Invalid combination of types");
  }
}

/// append local curent contribution to global current array
template <int Order, typename T_array, typename T_int, typename T_float>
static void append_current3d(T_array& uj, T_int iz0, T_int iy0, T_int ix0,
                             T_float current[Order + 3][Order + 3][Order + 3][4])
{
  constexpr int  size      = Order + 3;
  constexpr bool is_scalar = std::is_integral_v<T_int> && std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  constexpr bool is_sorted = std::is_integral_v<T_int>;

  if constexpr (is_scalar == true) {
    // naive scalar implementation
    for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
      for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
        for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
          uj(iz, iy, ix, 0) += current[jz][jy][jx][0];
          uj(iz, iy, ix, 1) += current[jz][jy][jx][1];
          uj(iz, iy, ix, 2) += current[jz][jy][jx][2];
          uj(iz, iy, ix, 3) += current[jz][jy][jx][3];
        }
      }
    }
  } else if constexpr (is_vector == true && is_sorted == true) {
    // all particle contributions are added to the same grid point
    for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
      for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
        for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
          uj(iz, iy, ix, 0) += xsimd::reduce_add(current[jz][jy][jx][0]);
          uj(iz, iy, ix, 1) += xsimd::reduce_add(current[jz][jy][jx][1]);
          uj(iz, iy, ix, 2) += xsimd::reduce_add(current[jz][jy][jx][2]);
          uj(iz, iy, ix, 3) += xsimd::reduce_add(current[jz][jy][jx][3]);
        }
      }
    }
  } else if constexpr (is_vector == true && is_sorted == false) {
    // particle contributions are added to different grid points
    auto stride_z = get_stride(uj, 0);
    auto stride_y = get_stride(uj, 1);
    auto stride_x = get_stride(uj, 2);
    auto stride_c = get_stride(uj, 3);
    auto pointer  = get_data_pointer(uj);

    alignas(64) int64 offset[T_float::size];

    for (int jz = 0; jz < size; jz++) {
      T_int iz = iz0 + jz;
      for (int jy = 0; jy < size; jy++) {
        T_int iy = iy0 + jy;
        for (int jx = 0; jx < size; jx++) {
          T_int ix = ix0 + jx;
          xsimd::store_aligned(offset, iz * stride_z + iy * stride_y + ix * stride_x);
          append_current_impl(pointer, offset, current[jz][jy][jx]);
        }
      }
    }
  } else {
    static_assert([] { return false; }(), "Invalid combination of types");
  }
}

template <int Order, typename T_array, typename T_int, typename T_float>
static void append_moment1d(T_array& um, int iz, int iy, T_int ix0, int is,
                            T_float moment[Order + 1][14])
{
  constexpr int  size        = Order + 1;
  constexpr int  num_moments = 14;
  constexpr bool is_scalar   = std::is_integral_v<T_int> && std::is_floating_point_v<T_float>;
  constexpr bool is_vector   = std::is_integral_v<T_int> && std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    // naive scalar implementation
    for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
      for (int k = 0; k < num_moments; k++) {
        um(iz, iy, ix, is, k) += moment[jx][k];
      }
    }
  } else if constexpr (is_vector == true) {
    // all particle contributions are added to the same grid point
    for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
      for (int k = 0; k < num_moments; k++) {
        um(iz, iy, ix, is, k) += xsimd::reduce_add(moment[jx][k]);
      }
    }
  } else {
    static_assert([] { return false; }(), "Invalid combination of types");
  }
}

template <int Order, typename T_array, typename T_int, typename T_float>
static void append_moment2d(T_array& um, int iz, T_int iy0, T_int ix0, int is,
                            T_float moment[Order + 1][Order + 1][14])
{
  constexpr int  size        = Order + 1;
  constexpr int  num_moments = 14;
  constexpr bool is_scalar   = std::is_integral_v<T_int> && std::is_floating_point_v<T_float>;
  constexpr bool is_vector   = std::is_integral_v<T_int> && std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    // naive scalar implementation
    for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
      for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
        for (int k = 0; k < num_moments; k++) {
          um(iz, iy, ix, is, k) += moment[jy][jx][k];
        }
      }
    }
  } else if constexpr (is_vector == true) {
    // all particle contributions are added to the same grid point
    for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
      for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
        for (int k = 0; k < num_moments; k++) {
          um(iz, iy, ix, is, k) += xsimd::reduce_add(moment[jy][jx][k]);
        }
      }
    }
  } else {
    static_assert([] { return false; }(), "Invalid combination of types");
  }
}

template <int Order, typename T_array, typename T_int, typename T_float>
static void append_moment3d(T_array& um, T_int iz0, T_int iy0, T_int ix0, int is,
                            T_float moment[Order + 1][Order + 1][Order + 1][14])
{
  constexpr int  size        = Order + 1;
  constexpr int  num_moments = 14;
  constexpr bool is_scalar   = std::is_integral_v<T_int> && std::is_floating_point_v<T_float>;
  constexpr bool is_vector   = std::is_integral_v<T_int> && std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    // naive scalar implementation
    for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
      for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
        for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
          for (int k = 0; k < num_moments; k++) {
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
          for (int k = 0; k < num_moments; k++) {
            um(iz, iy, ix, is, k) += xsimd::reduce_add(moment[jz][jy][jx][k]);
          }
        }
      }
    }
  } else {
    static_assert([] { return false; }(), "Invalid combination of types");
  }
}

} // namespace primitives

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
