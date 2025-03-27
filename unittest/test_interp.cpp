// -*- C++ -*-

#include "interp.hpp"
#include "nix.hpp"

#include <experimental/mdspan>
#include <iostream>

#include "catch.hpp"

using namespace nix::typedefs;
using namespace nix::interp;

namespace stdex    = std::experimental;
using Array2D      = stdex::mdspan<float64, stdex::dextents<size_t, 2>>;
using Array4D      = stdex::mdspan<float64, stdex::dextents<size_t, 4>>;
using uniform_rand = std::uniform_real_distribution<float64>;
template <typename T>
using aligned_vector = std::vector<T, xsimd::aligned_allocator<T, 64>>;

const float64 epsilon = 1.0e-14;

//
// forward declarations of helper functions
//

template <int Order, typename T_int>
bool test_shift_weights(T_int shift, float64 ww[Order + 2], float64 epsilon);

template <int Order, typename T_array>
bool test_interp1d_scalar(T_array eb, int iz, int iy, int ix0, float64 delt, float64 epsilon);

template <int Order, typename T_array, typename T_int>
bool test_interp1d_xsimd(T_array eb, int iz, int iy, T_int ix0, float64 delt, float64 epsilon);

template <int Order, typename T_array>
bool test_interp2d_scalar(T_array eb, int iz, int iy0, int ix0, float64 delt, float64 epsilon);

template <int Order, typename T_array, typename T_int>
bool test_interp2d_xsimd(T_array eb, int iz, T_int iy0, T_int ix0, float64 delt, float64 epsilon);

template <int Order, typename T_array>
bool test_interp3d_scalar(T_array eb, int iz0, int iy0, int ix0, float64 delt, float64 epsilon);

template <int Order, typename T_array, typename T_int>
bool test_interp3d_xsimd(T_array eb, T_int iz0, T_int iy0, T_int ix0, float64 delt,
                         float64 epsilon);

//
// test cases
//

TEST_CASE("Interpolation shift weights")
{
  std::random_device              rd;
  std::mt19937                    gen(rd());
  std::uniform_int_distribution<> rand(0, 2);

  // initialize shift for vector version
  int64    shift[simd_i64::size];
  simd_i64 shift_simd;

  for (int i = 0; i < simd_i64::size; i++) {
    shift[i] = rand(gen) - 1;
  }
  shift_simd = xsimd::load_unaligned(shift);

  SECTION("First-order")
  {
    const int size     = 3;
    float64   ww[size] = {0.5, 0.5, 0.0};

    // scalar
    REQUIRE(test_shift_weights<1>(0, ww, epsilon) == true);
    REQUIRE(test_shift_weights<1>(1, ww, epsilon) == true);

    // vector
    REQUIRE(test_shift_weights<1>(shift_simd, ww, epsilon) == true);
  }
  SECTION("Second-order")
  {
    const int size     = 4;
    float64   ww[size] = {0.2, 0.6, 0.2, 0.0};

    // scalar
    REQUIRE(test_shift_weights<2>(0, ww, epsilon) == true);
    REQUIRE(test_shift_weights<2>(1, ww, epsilon) == true);

    // vector
    REQUIRE(test_shift_weights<2>(shift_simd, ww, epsilon) == true);
  }
  SECTION("Third-order")
  {
    const int size     = 5;
    float64   ww[size] = {0.1, 0.4, 0.4, 0.1, 0.0};

    // scalar
    REQUIRE(test_shift_weights<3>(0, ww, epsilon) == true);
    REQUIRE(test_shift_weights<3>(1, ww, epsilon) == true);

    // vector
    REQUIRE(test_shift_weights<3>(shift_simd, ww, epsilon) == true);
  }
  SECTION("Fourth-order")
  {
    const int size     = 6;
    float64   ww[size] = {0.1, 0.2, 0.4, 0.2, 0.1, 0.0};

    // scalar
    REQUIRE(test_shift_weights<4>(0, ww, epsilon) == true);
    REQUIRE(test_shift_weights<4>(1, ww, epsilon) == true);

    // vector
    REQUIRE(test_shift_weights<4>(shift_simd, ww, epsilon) == true);
  }
}

TEST_CASE("Interpolation 1D")
{
  const int     Nz   = 1;
  const int     Ny   = 16;
  const int     Nx   = 16;
  const float64 delt = 0.5;

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  // field array (with one z-level)
  aligned_vector<float64> eb_data(Nz * Ny * Nx * 6);
  auto                    eb = stdex::mdspan(eb_data.data(), Nz, Ny, Nx, 6);
  std::transform(eb_data.begin(), eb_data.end(), eb_data.begin(),
                 [&](float64 x) { return rand(engine); });

  // vector index for x (no y, z component needed)
  aligned_vector<int64> ix0_data = {5, 4, 3, 2, 5, 4, 3, 2};

  //
  // first order
  //
  SECTION("First-order interpolation : scalar")
  {
    // Fixed z index = 0 for 2D tests
    REQUIRE(test_interp1d_scalar<1>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp1d_scalar<1>(eb, 0, 2, 8, delt, epsilon) == true);
  }
  SECTION("First-order interpolation : xsimd")
  {
    // scalar indices
    REQUIRE(test_interp1d_xsimd<1>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp1d_xsimd<1>(eb, 0, 2, 8, delt, epsilon) == true);

    // vector indices
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp1d_xsimd<1>(eb, 0, 0, ix0, delt, epsilon) == true);
  }
  //
  // second order
  //
  SECTION("Second-order interpolation : scalar")
  {
    REQUIRE(test_interp1d_scalar<2>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp1d_scalar<2>(eb, 0, 2, 8, delt, epsilon) == true);
  }
  SECTION("Second-order interpolation : xsimd")
  {
    // scalar indices
    REQUIRE(test_interp1d_xsimd<2>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp1d_xsimd<2>(eb, 0, 2, 8, delt, epsilon) == true);

    // vector indices
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp1d_xsimd<2>(eb, 0, 0, ix0, delt, epsilon) == true);
  }
  //
  // third order
  //
  SECTION("Third-order interpolation : scalar")
  {
    REQUIRE(test_interp1d_scalar<3>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp1d_scalar<3>(eb, 0, 2, 8, delt, epsilon) == true);
  }
  SECTION("Third-order interpolation : xsimd")
  {
    // scalar indices
    REQUIRE(test_interp1d_xsimd<3>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp1d_xsimd<3>(eb, 0, 2, 8, delt, epsilon) == true);

    // vector indices
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp1d_xsimd<3>(eb, 0, 0, ix0, delt, epsilon) == true);
  }
  //
  // fourth order
  //
  SECTION("Fourth-order interpolation : scalar")
  {
    REQUIRE(test_interp1d_scalar<4>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp1d_scalar<4>(eb, 0, 2, 8, delt, epsilon) == true);
  }
  SECTION("Fourth-order interpolation : xsimd")
  {
    // scalar indices
    REQUIRE(test_interp1d_xsimd<4>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp1d_xsimd<4>(eb, 0, 2, 8, delt, epsilon) == true);

    // vector indices
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp1d_xsimd<4>(eb, 0, 0, ix0, delt, epsilon) == true);
  }
}

TEST_CASE("Interpolation 2D")
{
  const int     Nz   = 1;
  const int     Ny   = 16;
  const int     Nx   = 16;
  const float64 delt = 0.5;

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  // field array (with one z-level)
  aligned_vector<float64> eb_data(Nz * Ny * Nx * 6);
  auto                    eb = stdex::mdspan(eb_data.data(), Nz, Ny, Nx, 6);
  std::transform(eb_data.begin(), eb_data.end(), eb_data.begin(),
                 [&](float64 x) { return rand(engine); });

  // vector index for y and x (no z component needed)
  aligned_vector<int64> iy0_data = {2, 2, 2, 2, 2, 2, 2, 2};
  aligned_vector<int64> ix0_data = {5, 4, 3, 2, 5, 4, 3, 2};

  //
  // first order
  //
  SECTION("First-order interpolation : scalar")
  {
    // Fixed z index = 0 for 2D tests
    REQUIRE(test_interp2d_scalar<1>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp2d_scalar<1>(eb, 0, 2, 8, delt, epsilon) == true);
  }
  SECTION("First-order interpolation : xsimd")
  {
    // scalar indices
    REQUIRE(test_interp2d_xsimd<1>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp2d_xsimd<1>(eb, 0, 2, 8, delt, epsilon) == true);

    // vector indices
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp2d_xsimd<1>(eb, 0, iy0, ix0, delt, epsilon) == true);
  }
  //
  // second order
  //
  SECTION("Second-order interpolation : scalar")
  {
    REQUIRE(test_interp2d_scalar<2>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp2d_scalar<2>(eb, 0, 2, 8, delt, epsilon) == true);
  }
  SECTION("Second-order interpolation : xsimd")
  {
    // scalar indices
    REQUIRE(test_interp2d_xsimd<2>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp2d_xsimd<2>(eb, 0, 2, 8, delt, epsilon) == true);

    // vector indices
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp2d_xsimd<2>(eb, 0, iy0, ix0, delt, epsilon) == true);
  }
  //
  // third order
  //
  SECTION("Third-order interpolation : scalar")
  {
    REQUIRE(test_interp2d_scalar<3>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp2d_scalar<3>(eb, 0, 2, 8, delt, epsilon) == true);
  }
  SECTION("Third-order interpolation : xsimd")
  {
    // scalar indices
    REQUIRE(test_interp2d_xsimd<3>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp2d_xsimd<3>(eb, 0, 2, 8, delt, epsilon) == true);

    // vector indices
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp2d_xsimd<3>(eb, 0, iy0, ix0, delt, epsilon) == true);
  }
  //
  // fourth order
  //
  SECTION("Fourth-order interpolation : scalar")
  {
    REQUIRE(test_interp2d_scalar<4>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp2d_scalar<4>(eb, 0, 2, 8, delt, epsilon) == true);
  }
  SECTION("Fourth-order interpolation : xsimd")
  {
    // scalar indices
    REQUIRE(test_interp2d_xsimd<4>(eb, 0, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp2d_xsimd<4>(eb, 0, 2, 8, delt, epsilon) == true);

    // vector indices
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp2d_xsimd<4>(eb, 0, iy0, ix0, delt, epsilon) == true);
  }
}

TEST_CASE("Interpolation 3D")
{
  const int     Nz   = 16;
  const int     Ny   = 16;
  const int     Nx   = 16;
  const float64 delt = 0.5;

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  // field array
  aligned_vector<float64> eb_data(Nz * Ny * Nx * 6);
  auto                    eb = stdex::mdspan(eb_data.data(), Nz, Ny, Nx, 6);
  std::transform(eb_data.begin(), eb_data.end(), eb_data.begin(),
                 [&](float64 x) { return rand(engine); });

  // vector index
  aligned_vector<int64> iz0_data = {2, 3, 4, 5, 2, 3, 4, 5};
  aligned_vector<int64> iy0_data = {2, 2, 2, 2, 2, 2, 2, 2};
  aligned_vector<int64> ix0_data = {5, 4, 3, 2, 5, 4, 3, 2};

  //
  // first order
  //
  SECTION("First-order interpolation : scalar")
  {
    REQUIRE(test_interp3d_scalar<1>(eb, 2, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp3d_scalar<1>(eb, 2, 4, 8, delt, epsilon) == true);
  }
  SECTION("First-order interpolation : xsimd")
  {
    // scalar index
    REQUIRE(test_interp3d_xsimd<1>(eb, 2, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp3d_xsimd<1>(eb, 2, 4, 8, delt, epsilon) == true);

    // vector index
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp3d_xsimd<1>(eb, iz0, iy0, ix0, delt, epsilon) == true);
  }
  //
  // second order
  //
  SECTION("Second-order interpolation : scalar")
  {
    REQUIRE(test_interp3d_scalar<2>(eb, 2, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp3d_scalar<2>(eb, 2, 4, 8, delt, epsilon) == true);
  }
  SECTION("Second-order interpolation : xsimd")
  {
    // scalar index
    REQUIRE(test_interp3d_xsimd<2>(eb, 2, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp3d_xsimd<2>(eb, 2, 4, 8, delt, epsilon) == true);

    // vector index
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp3d_xsimd<2>(eb, iz0, iy0, ix0, delt, epsilon) == true);
  }
  //
  // third order
  //
  SECTION("Third-order interpolation : scalar")
  {
    REQUIRE(test_interp3d_scalar<3>(eb, 2, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp3d_scalar<3>(eb, 2, 4, 8, delt, epsilon) == true);
  }
  SECTION("Third-order interpolation : xsimd")
  {
    // scalar index
    REQUIRE(test_interp3d_xsimd<3>(eb, 2, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp3d_xsimd<3>(eb, 2, 4, 8, delt, epsilon) == true);

    // vector index
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp3d_xsimd<3>(eb, iz0, iy0, ix0, delt, epsilon) == true);
  }
  //
  // fourth order
  //
  SECTION("Fourth-order interpolation : scalar")
  {
    REQUIRE(test_interp3d_scalar<4>(eb, 2, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp3d_scalar<4>(eb, 2, 4, 8, delt, epsilon) == true);
  }
  SECTION("Fourth-order interpolation : xsimd")
  {
    // scalar index
    REQUIRE(test_interp3d_xsimd<4>(eb, 2, 2, 2, delt, epsilon) == true);
    REQUIRE(test_interp3d_xsimd<4>(eb, 2, 4, 8, delt, epsilon) == true);

    // vector index
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interp3d_xsimd<4>(eb, iz0, iy0, ix0, delt, epsilon) == true);
  }
}

//
// implementation of helper functions
//

template <int Order, typename T_int>
bool test_shift_weights(T_int shift, float64 ww[Order + 2], float64 epsilon)
{
  constexpr bool is_scalar = std::is_integral_v<T_int>;

  if constexpr (is_scalar == true) {
    //
    // scalar version
    //
    float64 vv[Order + 2];
    for (int i = 0; i < Order + 2; i++) {
      vv[i] = ww[i];
    }

    shift_weights<Order>(shift, vv);

    // check
    bool status = true;

    if (shift > 0) {
      status = status & (std::abs(vv[0]) < epsilon);
      for (int i = 1; i < Order + 2; i++) {
        status = status & (std::abs(vv[i] - ww[i - 1]) < epsilon);
      }
    } else {
      for (int i = 0; i < Order + 2; i++) {
        status = status & (std::abs(vv[i] - ww[i]) < epsilon);
      }
    }

    return status;
  } else {
    //
    // vector version
    //
    simd_f64 vv[Order + 2];
    for (int i = 0; i < Order + 2; i++) {
      vv[i] = ww[i];
    }

    shift_weights<Order>(shift, vv);

    // check
    bool status = true;

    for (int j = 0; j < simd_f64::size; j++) {
      if (shift.get(j) > 0) {
        status = status & (std::abs(vv[0].get(j)) < epsilon);
        for (int i = 1; i < Order + 2; i++) {
          status = status & (std::abs(vv[i].get(j) - ww[i - 1]) < epsilon);
        }
      } else {
        for (int i = 0; i < Order + 2; i++) {
          status = status & (std::abs(vv[i].get(j) - ww[i]) < epsilon);
        }
      }
    }

    return status;
  }
}

template <int Order, typename T_array>
bool test_interp1d_scalar(T_array eb, int iz, int iy, int ix0, float64 delt, float64 epsilon)
{
  constexpr int size = Order + 2;

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  aligned_vector<float64> result1(6, 0.0);
  aligned_vector<float64> result2(6, 0.0);
  aligned_vector<float64> wx_data(size);

  std::transform(wx_data.begin(), wx_data.end(), wx_data.begin(),
                 [&](float64) { return rand(engine); });

  // scalar version
  {
    float64* wx = wx_data.data();
    for (int ik = 0; ik < 6; ik++) {
      result1[ik] = interp1d<Order>(eb, iz, iy, ix0, ik, wx, delt);
    }
  }

  // naive calculation
  {
    float64* wx = wx_data.data();
    for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
      for (int ik = 0; ik < 6; ik++) {
        result2[ik] += eb(iz, iy, ix, ik) * wx[jx] * delt;
      }
    }
  }

  // compare results
  bool status = true;
  for (int i = 0; i < result1.size(); i++) {
    auto error = std::abs(result1[i] - result2[i]) / std::abs(result1[1]);
    status     = status && (error < epsilon);
  }
  return status;
}

template <int Order, typename T_array, typename T_int>
bool test_interp1d_xsimd(T_array eb, int iz, int iy, T_int ix0, float64 delt, float64 epsilon)
{
  constexpr int size = Order + 2;
  static_assert(std::is_integral_v<T_int> || std::is_same_v<T_int, simd_i64>,
                "T_int must be either int or an appropriate SIMD type");

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  aligned_vector<float64> result1(simd_f64::size * 6, 0.0);
  aligned_vector<float64> result2(simd_f64::size * 6, 0.0);
  aligned_vector<float64> wx_data(size * simd_f64::size);
  int                     ix_scalar[simd_f64::size];

  // index
  if constexpr (std::is_integral_v<T_int>) {
    // scalar
    for (int i = 0; i < simd_f64::size; i++) {
      ix_scalar[i] = ix0;
    }
  } else if constexpr (std::is_same_v<T_int, simd_i64>) {
    // vector
    ix0.store_unaligned(ix_scalar);
  }

  for (int i = 0; i < simd_f64::size; i++) {
    for (int j = 0; j < size; j++) {
      wx_data[i + j * simd_f64::size] = rand(engine);
    }
  }

  // SIMD version
  {
    simd_f64 dt_simd = delt;
    simd_f64 wx_simd[size];
    for (int i = 0; i < size; i++) {
      wx_simd[i] = xsimd::load_unaligned(wx_data.data() + i * simd_f64::size);
    }
    for (int ik = 0; ik < 6; ik++) {
      // interpolate and store
      simd_f64 val = interp1d<Order>(eb, iz, iy, ix0, ik, wx_simd, dt_simd);
      val.store_unaligned(result1.data() + ik * simd_f64::size);
    }
  }

  // scalar version
  {
    float64 wx[size];
    for (int ik = 0; ik < 6; ik++) {
      for (int ip = 0; ip < simd_f64::size; ip++) {
        for (int i = 0; i < size; i++) {
          wx[i] = wx_data[i * simd_f64::size + ip];
        }
        // interpolate and store
        result2[ik * simd_f64::size + ip] =
            interp1d<Order>(eb, iz, iy, ix_scalar[ip], ik, wx, delt);
      }
    }
  }

  // compare SIMD and scalar results
  bool status = true;
  for (int i = 0; i < result1.size(); i++) {
    auto error = std::abs(result1[i] - result2[i]) / std::abs(result1[1]);
    status     = status && (error < epsilon);
  }
  return status;
}

template <int Order, typename T_array>
bool test_interp2d_scalar(T_array eb, int iz, int iy0, int ix0, float64 delt, float64 epsilon)
{
  constexpr int size = Order + 2;

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  aligned_vector<float64> result1(6, 0.0);
  aligned_vector<float64> result2(6, 0.0);
  aligned_vector<float64> wx_data(size);
  aligned_vector<float64> wy_data(size);

  std::transform(wx_data.begin(), wx_data.end(), wx_data.begin(),
                 [&](float64) { return rand(engine); });
  std::transform(wy_data.begin(), wy_data.end(), wy_data.begin(),
                 [&](float64) { return rand(engine); });

  // scalar version
  {
    float64* wx = wx_data.data();
    float64* wy = wy_data.data();
    for (int ik = 0; ik < 6; ik++) {
      result1[ik] = interp2d<Order>(eb, iz, iy0, ix0, ik, wy, wx, delt);
    }
  }

  // naive calculation
  {
    float64* wx = wx_data.data();
    float64* wy = wy_data.data();
    for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
      for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
        for (int ik = 0; ik < 6; ik++) {
          result2[ik] += eb(iz, iy, ix, ik) * wy[jy] * wx[jx] * delt;
        }
      }
    }
  }

  // compare results
  bool status = true;
  for (int i = 0; i < result1.size(); i++) {
    auto error = std::abs(result1[i] - result2[i]) / std::abs(result1[1]);
    status     = status && (error < epsilon);
  }
  return status;
}

template <int Order, typename T_array, typename T_int>
bool test_interp2d_xsimd(T_array eb, int iz, T_int iy0, T_int ix0, float64 delt, float64 epsilon)
{
  constexpr int size = Order + 2;
  static_assert(std::is_integral_v<T_int> || std::is_same_v<T_int, simd_i64>,
                "T_int must be either int or an appropriate SIMD type");

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  aligned_vector<float64> result1(simd_f64::size * 6, 0.0);
  aligned_vector<float64> result2(simd_f64::size * 6, 0.0);
  aligned_vector<float64> wx_data(size * simd_f64::size);
  aligned_vector<float64> wy_data(size * simd_f64::size);
  int                     iy_scalar[simd_f64::size];
  int                     ix_scalar[simd_f64::size];

  // index
  if constexpr (std::is_integral_v<T_int>) {
    // scalar
    for (int i = 0; i < simd_f64::size; i++) {
      iy_scalar[i] = iy0;
      ix_scalar[i] = ix0;
    }
  } else if constexpr (std::is_same_v<T_int, simd_i64>) {
    // vector
    iy0.store_unaligned(iy_scalar);
    ix0.store_unaligned(ix_scalar);
  }

  for (int i = 0; i < simd_f64::size; i++) {
    for (int j = 0; j < size; j++) {
      wx_data[i + j * simd_f64::size] = rand(engine);
      wy_data[i + j * simd_f64::size] = rand(engine);
    }
  }

  // SIMD version
  {
    simd_f64 dt_simd = delt;
    simd_f64 wx_simd[size];
    simd_f64 wy_simd[size];
    for (int i = 0; i < size; i++) {
      wx_simd[i] = xsimd::load_unaligned(wx_data.data() + i * simd_f64::size);
      wy_simd[i] = xsimd::load_unaligned(wy_data.data() + i * simd_f64::size);
    }
    for (int ik = 0; ik < 6; ik++) {
      // interpolate and store
      simd_f64 val = interp2d<Order>(eb, iz, iy0, ix0, ik, wy_simd, wx_simd, dt_simd);
      val.store_unaligned(result1.data() + ik * simd_f64::size);
    }
  }

  // scalar version
  {
    float64 wx[size];
    float64 wy[size];
    for (int ik = 0; ik < 6; ik++) {
      for (int ip = 0; ip < simd_f64::size; ip++) {
        for (int i = 0; i < size; i++) {
          wx[i] = wx_data[i * simd_f64::size + ip];
          wy[i] = wy_data[i * simd_f64::size + ip];
        }
        // interpolate and store
        result2[ik * simd_f64::size + ip] =
            interp2d<Order>(eb, iz, iy_scalar[ip], ix_scalar[ip], ik, wy, wx, delt);
      }
    }
  }

  // compare SIMD and scalar results
  bool status = true;
  for (int i = 0; i < result1.size(); i++) {
    auto error = std::abs(result1[i] - result2[i]) / std::abs(result1[1]);
    status     = status && (error < epsilon);
  }
  return status;
}

template <int Order, typename T_array>
bool test_interp3d_scalar(T_array eb, int iz0, int iy0, int ix0, float64 delt, float64 epsilon)
{
  constexpr int size = Order + 2;

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  aligned_vector<float64> result1(6, 0.0);
  aligned_vector<float64> result2(6, 0.0);
  aligned_vector<float64> wx_data(size);
  aligned_vector<float64> wy_data(size);
  aligned_vector<float64> wz_data(size);

  std::transform(wx_data.begin(), wx_data.end(), wx_data.begin(),
                 [&](float64) { return rand(engine); });
  std::transform(wy_data.begin(), wy_data.end(), wy_data.begin(),
                 [&](float64) { return rand(engine); });
  std::transform(wz_data.begin(), wz_data.end(), wz_data.begin(),
                 [&](float64) { return rand(engine); });

  // scalar version
  {
    float64* wx = wx_data.data();
    float64* wy = wy_data.data();
    float64* wz = wz_data.data();

    for (int ik = 0; ik < 6; ik++) {
      result1[ik] = interp3d<Order>(eb, iz0, iy0, ix0, ik, wz, wy, wx, delt);
    }
  }

  // naive calculation
  {
    float64* wx = wx_data.data();
    float64* wy = wy_data.data();
    float64* wz = wz_data.data();

    for (int jz = 0, iz = iz0; jz < Order + 2; jz++, iz++) {
      for (int jy = 0, iy = iy0; jy < Order + 2; jy++, iy++) {
        for (int jx = 0, ix = ix0; jx < Order + 2; jx++, ix++) {
          for (int ik = 0; ik < 6; ik++) {
            result2[ik] += eb(iz, iy, ix, ik) * wz[jz] * wy[jy] * wx[jx] * delt;
          }
        }
      }
    }
  }

  // compare results
  bool status = true;
  for (int i = 0; i < result1.size(); i++) {
    auto error = std::abs(result1[i] - result2[i]) / std::abs(result1[1]);
    status     = status && (error < epsilon);
  }

  return status;
}

template <int Order, typename T_array, typename T_int>
bool test_interp3d_xsimd(T_array eb, T_int iz0, T_int iy0, T_int ix0, float64 delt, float64 epsilon)
{
  constexpr int size = Order + 2;
  static_assert(std::is_integral_v<T_int> || std::is_same_v<T_int, simd_i64>,
                "T_int must be either int or an appropriate SIMD type");

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  aligned_vector<float64> result1(simd_f64::size * 6, 0.0);
  aligned_vector<float64> result2(simd_f64::size * 6, 0.0);
  aligned_vector<float64> wx_data(size * simd_f64::size);
  aligned_vector<float64> wy_data(size * simd_f64::size);
  aligned_vector<float64> wz_data(size * simd_f64::size);
  int                     iz_scalar[simd_f64::size];
  int                     iy_scalar[simd_f64::size];
  int                     ix_scalar[simd_f64::size];

  // index
  if constexpr (std::is_integral_v<T_int>) {
    // scalar
    for (int i = 0; i < simd_f64::size; i++) {
      iz_scalar[i] = iz0;
      iy_scalar[i] = iy0;
      ix_scalar[i] = ix0;
    }
  } else if constexpr (std::is_same_v<T_int, simd_i64>) {
    // vector
    iz0.store_unaligned(iz_scalar);
    iy0.store_unaligned(iy_scalar);
    ix0.store_unaligned(ix_scalar);
  }

  for (int i = 0; i < simd_f64::size; i++) {
    for (int j = 0; j < Order + 1; j++) {
      wx_data[i + j * simd_f64::size] = rand(engine);
      wy_data[i + j * simd_f64::size] = rand(engine);
      wz_data[i + j * simd_f64::size] = rand(engine);
    }
    // Note: trick for compatibility for scalar and sorted vector versions
    wx_data[i + (Order + 1) * simd_f64::size] = 0;
    wy_data[i + (Order + 1) * simd_f64::size] = 0;
    wz_data[i + (Order + 1) * simd_f64::size] = 0;
  }

  // SIMD version
  {
    simd_f64 dt_simd = delt;
    simd_f64 wx_simd[size];
    simd_f64 wy_simd[size];
    simd_f64 wz_simd[size];

    // load weights
    for (int i = 0; i < size; i++) {
      wx_simd[i] = xsimd::load_unaligned(wx_data.data() + i * simd_f64::size);
      wy_simd[i] = xsimd::load_unaligned(wy_data.data() + i * simd_f64::size);
      wz_simd[i] = xsimd::load_unaligned(wz_data.data() + i * simd_f64::size);
    }

    for (int ik = 0; ik < 6; ik++) {
      // interpolate and store
      simd_f64 val = interp3d<Order>(eb, iz0, iy0, ix0, ik, wz_simd, wy_simd, wx_simd, dt_simd);
      val.store_unaligned(result1.data() + ik * simd_f64::size);
    }
  }

  // scalar version
  {
    float64 wx[size];
    float64 wy[size];
    float64 wz[size];

    for (int ik = 0; ik < 6; ik++) {
      for (int ip = 0; ip < simd_f64::size; ip++) {
        for (int i = 0; i < size; i++) {
          wx[i] = wx_data[i * simd_f64::size + ip];
          wy[i] = wy_data[i * simd_f64::size + ip];
          wz[i] = wz_data[i * simd_f64::size + ip];
        }
        // interpolate and store
        result2[ik * simd_f64::size + ip] =
            interp3d<Order>(eb, iz_scalar[ip], iy_scalar[ip], ix_scalar[ip], ik, wz, wy, wx, delt);
      }
    }
  }

  // compare SIMD and scalar results
  bool status = true;
  for (int i = 0; i < result1.size(); i++) {
    auto error = std::abs(result1[i] - result2[i]) / std::abs(result1[1]);
    status     = status && (error < epsilon);
  }

  return status;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
