// -*- C++ -*-

#include "esirkepov.hpp"
#include "nix.hpp"
#include "primitives.hpp"

#include <experimental/mdspan>
#include <iostream>

#include "catch.hpp"

using namespace nix::typedefs;
using namespace nix::esirkepov;
using nix::primitives::digitize;
using nix::primitives::shape_mc;

namespace stdex    = std::experimental;
using Array2D      = stdex::mdspan<float64, stdex::dextents<size_t, 2>>;
using uniform_rand = std::uniform_real_distribution<float64>;

const double epsilon = 1.0e-13;

//
// forward declarations of helper functions
//

void set_random_particle(Array2D& xu, float64 delh, float64 delv);

template <int Dim, int Order, typename T_int>
bool test_shift_weights(T_int shift[], float64 ww[Order + 3]);

template <int N>
bool test_conservation1d(const float64 delt, const float64 delh, const float64 rho[N],
                         const float64 cur[N][4], const float64 epsilon);

template <int Order>
bool deposit1d_scalar(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                      float64 rho[Order + 3], float64 cur[Order + 3][4], const float64 epsilon);

template <int Order, typename T_float>
bool deposit1d_xsimd(const float64 delt, const float64 delh, T_float xu[7], T_float xv[7],
                     T_float rho[Order + 3], T_float cur[Order + 3][4], const float64 epsilon);

template <int Order, typename T_array>
bool test_deposit1d_scalar(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                           const float64 epsilon);

template <int Order, typename T_array>
bool test_deposit1d_xsimd(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                          const float64 epsilon);

template <int N>
bool test_conservation2d(const float64 delt, const float64 delh, const float64 rho[N][N],
                         const float64 cur[N][N][4], const float64 epsilon);

template <int Order>
bool deposit2d_scalar(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                      float64 rho[Order + 3][Order + 3], float64 cur[Order + 3][Order + 3][3],
                      const float64 epsilon);

template <int Order, typename T_float>
bool deposit2d_xsimd(const float64 delt, const float64 delh, T_float xu[7], T_float xv[7],
                     T_float rho[Order + 3][Order + 3], T_float cur[Order + 3][Order + 3][4],
                     const float64 epsilon);

template <int Order, typename T_array>
bool test_deposit2d_scalar(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                           const float64 epsilon);

template <int Order, typename T_array>
bool test_deposit2d_xsimd(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                          const float64 epsilon);

template <int N>
bool test_conservation3d(const float64 delt, const float64 delh, const float64 rho[N][N][N],
                         const float64 cur[N][N][N][4], const float64 epsilon);

template <int Order>
bool deposit3d_scalar(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                      float64 rho[Order + 3][Order + 3][Order + 3],
                      float64 cur[Order + 3][Order + 3][Order + 3][4], const float64 epsilon);

template <int Order, typename T_float>
bool deposit3d_xsimd(const float64 delt, const float64 delh, T_float xu[7], T_float xv[7],
                     T_float rho[Order + 3][Order + 3][Order + 3],
                     T_float cur[Order + 3][Order + 3][Order + 3][4], const float64 epsilon);

template <int Order, typename T_array>
bool test_deposit3d_scalar(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                           const float64 epsilon);

template <int Order, typename T_array>
bool test_deposit3d_xsimd(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                          const float64 epsilon);

//
// test cases
//

TEST_CASE("Esirkepov shift_weights")
{
  std::random_device              rd;
  std::mt19937                    gen(rd());
  std::uniform_int_distribution<> rand(0, 2);

  // initialize shift for vector version
  int64    shift[simd_i64::size][3];
  simd_i64 shift_simd2d[2];
  simd_i64 shift_simd3d[3];

  for (int i = 0; i < simd_i64::size; i++) {
    shift[i][0] = rand(gen) - 1;
    shift[i][1] = rand(gen) - 1;
    shift[i][2] = rand(gen) - 1;
  }
  shift_simd2d[0] = xsimd::load_unaligned(shift[0]);
  shift_simd2d[1] = xsimd::load_unaligned(shift[1]);
  shift_simd3d[0] = xsimd::load_unaligned(shift[0]);
  shift_simd3d[1] = xsimd::load_unaligned(shift[1]);
  shift_simd3d[2] = xsimd::load_unaligned(shift[2]);

  SECTION("First-order")
  {
    const int size     = 4;
    float64   ww[size] = {0.0, 0.5, 0.5, 0.0};

    // scalar 2D
    REQUIRE(test_shift_weights<2, 1>(std::array<int, 3>{+1, 0}.data(), ww) == true);
    REQUIRE(test_shift_weights<2, 1>(std::array<int, 3>{0, -1}.data(), ww) == true);

    // vector 2D
    REQUIRE(test_shift_weights<2, 1>(shift_simd2d, ww) == true);

    // scalar 3D
    REQUIRE(test_shift_weights<3, 1>(std::array<int, 3>{+1, -1, 0}.data(), ww) == true);
    REQUIRE(test_shift_weights<3, 1>(std::array<int, 3>{0, +1, +1}.data(), ww) == true);
    REQUIRE(test_shift_weights<3, 1>(std::array<int, 3>{+1, 0, -1}.data(), ww) == true);

    // vector 3D
    REQUIRE(test_shift_weights<3, 1>(shift_simd3d, ww) == true);
  }
  SECTION("Second-order")
  {
    const int size     = 5;
    float64   ww[size] = {0.0, 0.2, 0.6, 0.2, 0.0};

    // scalar 2D
    REQUIRE(test_shift_weights<2, 2>(std::array<int, 3>{+1, 0}.data(), ww) == true);
    REQUIRE(test_shift_weights<2, 2>(std::array<int, 3>{0, -1}.data(), ww) == true);

    // vector 2D
    REQUIRE(test_shift_weights<2, 2>(shift_simd2d, ww) == true);

    // scalar 3D
    REQUIRE(test_shift_weights<3, 2>(std::array<int, 3>{+1, -1, 0}.data(), ww) == true);
    REQUIRE(test_shift_weights<3, 2>(std::array<int, 3>{0, +1, +1}.data(), ww) == true);
    REQUIRE(test_shift_weights<3, 2>(std::array<int, 3>{+1, 0, -1}.data(), ww) == true);

    // vector 3D
    REQUIRE(test_shift_weights<3, 2>(shift_simd3d, ww) == true);
  }
  SECTION("Third-order")
  {
    const int size     = 6;
    float64   ww[size] = {0.0, 0.1, 0.4, 0.4, 0.1, 0.0};

    // scalar 2D
    REQUIRE(test_shift_weights<2, 3>(std::array<int, 3>{+1, 0}.data(), ww) == true);
    REQUIRE(test_shift_weights<2, 3>(std::array<int, 3>{0, -1}.data(), ww) == true);

    // vector 2D
    REQUIRE(test_shift_weights<2, 3>(shift_simd2d, ww) == true);

    // scalar 3D
    REQUIRE(test_shift_weights<3, 3>(std::array<int, 3>{+1, -1, 0}.data(), ww) == true);
    REQUIRE(test_shift_weights<3, 3>(std::array<int, 3>{0, +1, +1}.data(), ww) == true);
    REQUIRE(test_shift_weights<3, 3>(std::array<int, 3>{+1, 0, -1}.data(), ww) == true);

    // vector 3D
    REQUIRE(test_shift_weights<3, 3>(shift_simd3d, ww) == true);
  }
  SECTION("Fourth-order")
  {
    const int size     = 7;
    float64   ww[size] = {0.0, 0.1, 0.2, 0.4, 0.2, 0.1, 0.0};

    // scalar 2D
    REQUIRE(test_shift_weights<2, 4>(std::array<int, 3>{+1, 0}.data(), ww) == true);
    REQUIRE(test_shift_weights<2, 4>(std::array<int, 3>{0, -1}.data(), ww) == true);

    // vector 2D
    REQUIRE(test_shift_weights<2, 4>(shift_simd2d, ww) == true);

    // scalar 3D
    REQUIRE(test_shift_weights<3, 4>(std::array<int, 3>{+1, -1, 0}.data(), ww) == true);
    REQUIRE(test_shift_weights<3, 4>(std::array<int, 3>{0, +1, +1}.data(), ww) == true);
    REQUIRE(test_shift_weights<3, 4>(std::array<int, 3>{+1, 0, -1}.data(), ww) == true);

    // vector 3D
    REQUIRE(test_shift_weights<3, 4>(shift_simd3d, ww) == true);
  }
}

TEST_CASE("Esirkepov scheme in 1D")
{
  const int     Np   = 1024;
  const float64 delt = GENERATE(0.5, 1.0, 2.0);
  const float64 delh = GENERATE(0.5, 1.0, 2.0);
  const float64 delv = delh / delt;

  float64 xv_data[Np * 7];
  float64 xu_data[Np * 7];
  auto    xv = stdex::mdspan(xv_data, Np, 7);
  auto    xu = stdex::mdspan(xu_data, Np, 7);

  set_random_particle(xu, delh, delv);

  //
  // first order
  //
  SECTION("First-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit1d_scalar<1>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("First-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit1d_xsimd<1>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  //
  // second order
  //
  SECTION("Second-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit1d_scalar<2>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("Second-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit1d_xsimd<2>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  //
  // third order
  //
  SECTION("Third-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit1d_scalar<3>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("Third-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit1d_xsimd<3>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  //
  // forth order
  //
  SECTION("Fourth-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit1d_scalar<4>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("Fourth-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit1d_xsimd<4>(xu, xv, Np, delt, delh, epsilon) == true);
  }
}

TEST_CASE("Esirkepov scheme in 2D")
{
  const int     Np   = 1024;
  const float64 delt = GENERATE(0.5, 1.0, 2.0);
  const float64 delh = GENERATE(0.5, 1.0, 2.0);
  const float64 delv = delh / delt;

  float64 xv_data[Np * 7];
  float64 xu_data[Np * 7];
  auto    xv = stdex::mdspan(xv_data, Np, 7);
  auto    xu = stdex::mdspan(xu_data, Np, 7);

  set_random_particle(xu, delh, delv);

  //
  // first order
  //
  SECTION("First-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit2d_scalar<1>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("First-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit2d_xsimd<1>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  //
  // second order
  //
  SECTION("Second-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit2d_scalar<2>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("Second-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit2d_xsimd<2>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  //
  // third order
  //
  SECTION("Third-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit2d_scalar<3>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("Third-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit2d_xsimd<3>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  //
  // forth order
  //
  SECTION("Fourth-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit2d_scalar<4>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("Fourth-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit2d_xsimd<4>(xu, xv, Np, delt, delh, epsilon) == true);
  }
}

TEST_CASE("Esirkepov scheme in 3D")
{
  const int     Np   = 1024;
  const float64 delt = GENERATE(0.5, 1.0, 2.0);
  const float64 delh = GENERATE(0.5, 1.0, 2.0);
  const float64 delv = delh / delt;

  float64 xv_data[Np * 7];
  float64 xu_data[Np * 7];
  auto    xv = stdex::mdspan(xv_data, Np, 7);
  auto    xu = stdex::mdspan(xu_data, Np, 7);

  set_random_particle(xu, delh, delv);

  //
  // first order
  //
  SECTION("First-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit3d_scalar<1>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("First-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit3d_xsimd<1>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  //
  // second order
  //
  SECTION("Second-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit3d_scalar<2>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("Second-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit3d_xsimd<2>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  //
  // third order
  //
  SECTION("Third-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit3d_scalar<3>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("Third-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit3d_xsimd<3>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  //
  // forth order
  //
  SECTION("Fourth-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_deposit3d_scalar<4>(xu, xv, Np, delt, delh, epsilon) == true);
  }
  SECTION("Fourth-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_deposit3d_xsimd<4>(xu, xv, Np, delt, delh, epsilon) == true);
  }
}

//
// implementation of helper functions
//

void set_random_particle(Array2D& xu, float64 delh, float64 delv)
{
  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0, 1);

  for (int ip = 0; ip < xu.extent(0); ip++) {
    xu(ip, 0) = rand(engine) * delh;
    xu(ip, 1) = rand(engine) * delh;
    xu(ip, 2) = rand(engine) * delh;
    xu(ip, 3) = rand(engine) * delv * 2 - delv;
    xu(ip, 4) = rand(engine) * delv * 2 - delv;
    xu(ip, 5) = rand(engine) * delv * 2 - delv;
  }
}

template <int Dim, int Order, typename T_int>
bool test_shift_weights(T_int shift[Dim], float64 ww[Order + 3])
{
  constexpr bool is_scalar = std::is_integral_v<T_int>;
  const int      size      = Order + 3;

  if constexpr (is_scalar) {
    // scalar version
    float64 ss[Dim][size];
    for (int dir = 0; dir < Dim; dir++) {
      for (int i = 0; i < size; i++) {
        ss[dir][i] = ww[i];
      }
    }

    // test
    shift_weights<Dim, Order>(shift, ss);

    // check results
    bool status = true;
    for (int dir = 0; dir < Dim; dir++) {
      if (shift[dir] == 0) {
        for (int i = 0; i < size; i++) {
          status = status && (ss[dir][i] == ww[i]);
        }
      } else if (shift[dir] < 0) {
        for (int i = 0; i < size - 1; i++) {
          status = status && (ss[dir][i] == ww[i + 1]);
        }
      } else if (shift[dir] > 0) {
        for (int i = 1; i < size; i++) {
          status = status && (ss[dir][i] == ww[i - 1]);
        }
      }
    }

    return status;
  } else {
    // vector version
    simd_f64 ss[Dim][size];
    for (int dir = 0; dir < Dim; dir++) {
      for (int i = 0; i < size; i++) {
        ss[dir][i] = ww[i];
      }
    }

    // test
    shift_weights<Dim, Order>(shift, ss);

    // check results
    bool status = true;
    for (int j = 0; j < simd_f64::size; j++) {
      for (int dir = 0; dir < Dim; dir++) {
        if (shift[dir].get(j) == 0) {
          for (int i = 0; i < size; i++) {
            status = status && (ss[dir][i].get(j) == ww[i]);
          }
        } else if (shift[dir].get(j) < 0) {
          for (int i = 0; i < size - 1; i++) {
            status = status && (ss[dir][i].get(j) == ww[i + 1]);
          }
        } else if (shift[dir].get(j) > 0) {
          for (int i = 1; i < size; i++) {
            status = status && (ss[dir][i].get(j) == ww[i - 1]);
          }
        }
      }
    }

    return status;
  }
}

//
// for 1D version
//
template <int N>
bool test_conservation1d(const float64 delt, const float64 delh, const float64 rho[N],
                         const float64 cur[N][4], const float64 epsilon)
{
  bool    status      = true;
  float64 errsum      = 0.0;
  float64 errnrm      = 0.0;
  float64 J[N + 1][2] = {0};

  for (int jx = 0; jx < N; jx++) {
    J[jx][0] = cur[jx][0];
    J[jx][1] = cur[jx][1];
  }

  for (int jx = 0; jx < N; jx++) {
    errnrm += std::abs(J[jx][0]);
    errsum += std::abs((J[jx][0] - rho[jx]) + (delt / delh) * (J[jx + 1][1] - J[jx][1]));
  }

  status = status && (errsum < epsilon * errnrm);
  return status;
}

template <int Order>
bool deposit1d_scalar(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                      float64 rho[Order + 3], float64 cur[Order + 3][4], const float64 epsilon)
{
  const float64 rdh  = 1 / delh;
  const float64 dhdt = delh / delt;
  const float64 qs   = 1;

  bool    status  = true;
  float64 rhosum0 = 0;
  float64 rhosum1 = 0;
  float64 rhosum2 = 0;
  float64 vy      = 0;
  float64 vz      = 0;

  float64 ss[2][1][Order + 3] = {0};

  xv[0] = xu[0];
  xv[1] = xu[1];
  xv[2] = xu[2];
  xu[0] = xu[0] + xu[3] * delt;
  xu[1] = xu[1] + xu[4] * delt;
  xu[2] = xu[2] + xu[5] * delt;
  vy    = xu[4];
  vz    = xu[5];

  //
  // before move
  //
  int ix0 = digitize(xv[0], 0.0, rdh);

  shape_mc<Order>(xv[0], ix0 * delh, rdh, &ss[0][0][1]);

  // check charge density
  for (int jx = 0; jx < Order + 3; jx++) {
    float64 r = ss[0][0][jx];
    rhosum0 += cur[jx][0];
    rhosum1 += r;
    rho[jx] += r;
  }

  //
  // after move
  //
  int ix1 = digitize(xu[0], 0.0, rdh);

  shape_mc<Order>(xu[0], ix1 * delh, rdh, &ss[1][0][1 + ix1 - ix0]);

  // calculate charge and current density
  deposit1d<Order>(dhdt, vy, vz, qs, ss, cur);

  // check charge density
  for (int jx = 0; jx < Order + 3; jx++) {
    rhosum2 += cur[jx][0];
  }

  // contribution to charge density is normalized to unity
  status = status && (std::abs(rhosum1 - 1) < epsilon);

  // charge density increases exactly by one
  status = status && (std::abs(rhosum2 - (rhosum0 + 1)) < epsilon * std::abs(rhosum2));

  return status;
}

template <int Order, typename T_float>
bool deposit1d_xsimd(const float64 delt, const float64 delh, T_float xu[7], T_float xv[7],
                     T_float rho[Order + 3], T_float cur[Order + 3][4], const float64 epsilon)
{
  const T_float zero = 0;
  const T_float rdh  = 1 / delh;
  const float64 dhdt = delh / delt;
  const T_float qs   = 1;

  bool    status  = true;
  T_float rhosum0 = 0;
  T_float rhosum1 = 0;
  T_float rhosum2 = 0;
  T_float vy      = 0;
  T_float vz      = 0;

  T_float ss[2][1][Order + 3] = {0};

  xv[0] = xu[0];
  xv[1] = xu[1];
  xv[2] = xu[2];
  xu[0] = xu[0] + xu[3] * delt;
  xu[1] = xu[1] + xu[4] * delt;
  xu[2] = xu[2] + xu[5] * delt;
  vy    = xu[4];
  vz    = xu[5];

  //
  // before move
  //
  auto ix0 = digitize(xv[0], zero, rdh);

  shape_mc<Order>(xv[0], xsimd::to_float(ix0) * delh, rdh, &ss[0][0][1]);

  // check charge density
  for (int jx = 0; jx < Order + 3; jx++) {
    T_float r = ss[0][0][jx];
    rhosum0 += cur[jx][0];
    rhosum1 += r;
    rho[jx] += r;
  }

  //
  // after move
  //
  auto ix1 = digitize(xu[0], zero, rdh);

  shape_mc<Order>(xu[0], xsimd::to_float(ix1) * delh, rdh, &ss[1][0][1]);

  //
  // in-place shift of ss[1] according to particle movement
  //
  xsimd::batch<int64_t> shift[1] = {ix1 - ix0};
  shift_weights<1, Order>(shift, ss[1]);

  // calculate charge and current density
  deposit1d<Order>(dhdt, vy, vz, qs, ss, cur);

  // check charge density
  for (int jx = 0; jx < Order + 3; jx++) {
    rhosum2 += cur[jx][0];
  }

  // error check: normalize the accumulated sums by the SIMD width
  {
    float64 rho0 = xsimd::reduce_add(rhosum0) / T_float::size;
    float64 rho1 = xsimd::reduce_add(rhosum1) / T_float::size;
    float64 rho2 = xsimd::reduce_add(rhosum2) / T_float::size;

    // contribution to charge density is normalized to unity
    status = status && (std::abs(rho1 - 1) < epsilon);

    // charge density increases exactly by one
    status = status && (std::abs(rho2 - (rho0 + 1)) < epsilon * std::abs(rho2));
  }

  return status;
}

template <int Order, typename T_array>
bool test_deposit1d_scalar(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                           const float64 epsilon)
{
  const int size    = Order + 3;
  bool      status1 = true;
  bool      status2 = true;

  float64 cur[size][4] = {0};
  float64 rho[size]    = {0};

  for (int ip = 0; ip < Np; ip++) {
    float64* xu_ptr = &xu(ip, 0);
    float64* xv_ptr = &xv(ip, 0);

    status1 = status1 && deposit1d_scalar<Order>(delt, delh, xu_ptr, xv_ptr, rho, cur, epsilon);
  }
  status2 = status2 && test_conservation1d<Order + 3>(delt, delh, rho, cur, epsilon);

  return status1 && status2;
}

template <int Order, typename T_array>
bool test_deposit1d_xsimd(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                          const float64 epsilon)
{
  const int size    = Order + 3;
  bool      status1 = true;
  bool      status2 = true;

  float64  rho[size]         = {0};
  float64  cur[size][4]      = {0};
  simd_f64 rho_simd[size]    = {0};
  simd_f64 cur_simd[size][4] = {0};
  simd_f64 xu_simd[7];
  simd_f64 xv_simd[7];
  simd_i64 index_simd = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  // SIMD version
  for (int ip = 0; ip < Np - Np % simd_f64::size; ip += simd_f64::size) {
    // Load data
    for (int k = 0; k < 7; k++) {
      xu_simd[k] = simd_f64::gather(&xu(ip, k), index_simd);
      xv_simd[k] = simd_f64::gather(&xv(ip, k), index_simd);
    }

    status1 = status1 &&
              deposit1d_xsimd<Order>(delt, delh, xu_simd, xv_simd, rho_simd, cur_simd, epsilon);
  }

  // scalar version
  for (int ip = 0; ip < Np - Np % simd_f64::size; ip++) {
    float64* xu_ptr = &xu(ip, 0);
    float64* xv_ptr = &xv(ip, 0);

    status1 = status1 && deposit1d_scalar<Order>(delt, delh, xu_ptr, xv_ptr, rho, cur, epsilon);
  }

  // compare scalar and SIMD results
  for (int i = 0; i < size; i++) {
    for (int l = 0; l < 4; l++) {
      float64 cur_sum = xsimd::reduce_add(cur_simd[i][l]);
      float64 cur_err = std::abs(cur[i][l] - cur_sum);
      status2 = status2 && ((cur_err <= epsilon) || (cur_err <= std::abs(cur_sum) * epsilon));
    }
    float64 rho_sum = xsimd::reduce_add(rho_simd[i]);
    float64 rho_err = std::abs(rho[i] - rho_sum);
    status2         = status2 && ((rho_err <= epsilon) || (rho_err <= std::abs(rho_sum) * epsilon));
  }

  return status1 && status2;
}

//
// for 2D version
//

template <int N>
bool test_conservation2d(const float64 delt, const float64 delh, const float64 rho[N][N],
                         const float64 cur[N][N][4], const float64 epsilon)
{
  bool    status = true;
  float64 errsum = 0.0;
  float64 errnrm = 0.0;

  float64 J[N + 1][N + 1][4] = {0};

  for (int jy = 0; jy < N; jy++) {
    for (int jx = 0; jx < N; jx++) {
      J[jy][jx][0] = cur[jy][jx][0];
      J[jy][jx][1] = cur[jy][jx][1];
      J[jy][jx][2] = cur[jy][jx][2];
      J[jy][jx][3] = cur[jy][jx][3];
    }
  }

  for (int jy = 0; jy < N; jy++) {
    for (int jx = 0; jx < N; jx++) {
      errnrm += std::abs(J[jy][jx][0]);
      errsum += std::abs((J[jy][jx][0] - rho[jy][jx]) +
                         (delt / delh) * (J[jy][jx + 1][1] - J[jy][jx][1]) +
                         (delt / delh) * (J[jy + 1][jx][2] - J[jy][jx][2]));
    }
  }

  status = status && (errsum < epsilon * errnrm);
  return status;
}

template <int Order>
bool deposit2d_scalar(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                      float64 rho[Order + 3][Order + 3], float64 cur[Order + 3][Order + 3][4],
                      const float64 epsilon)
{
  const float64 rdh  = 1 / delh;
  const float64 dhdt = delh / delt;
  const float64 qs   = 1;

  bool    status  = true;
  float64 rhosum0 = 0;
  float64 rhosum1 = 0;
  float64 rhosum2 = 0;
  float64 vz      = 0;

  float64 ss[2][2][Order + 3] = {0};

  xv[0] = xu[0];
  xv[1] = xu[1];
  xv[2] = xu[2];
  xu[0] = xu[0] + xu[3] * delt;
  xu[1] = xu[1] + xu[4] * delt;
  xu[2] = xu[2] + xu[5] * delt;
  vz    = xu[5];

  //
  // before move
  //
  int ix0 = digitize(xv[0], 0.0, rdh);
  int iy0 = digitize(xv[1], 0.0, rdh);

  shape_mc<Order>(xv[0], ix0 * delh, rdh, &ss[0][0][1]);
  shape_mc<Order>(xv[1], iy0 * delh, rdh, &ss[0][1][1]);

  // check charge density
  for (int jy = 0; jy < Order + 3; jy++) {
    for (int jx = 0; jx < Order + 3; jx++) {
      float64 r = ss[0][0][jx] * ss[0][1][jy];
      rhosum0 += cur[jy][jx][0];
      rhosum1 += r;
      rho[jy][jx] += r;
    }
  }

  //
  // after move
  //
  int ix1 = digitize(xu[0], 0.0, rdh);
  int iy1 = digitize(xu[1], 0.0, rdh);

  shape_mc<Order>(xu[0], ix1 * delh, rdh, &ss[1][0][1 + ix1 - ix0]);
  shape_mc<Order>(xu[1], iy1 * delh, rdh, &ss[1][1][1 + iy1 - iy0]);

  // calculate charge and current density
  deposit2d<Order>(dhdt, dhdt, vz, qs, ss, cur);

  // check charge density
  for (int jy = 0; jy < Order + 3; jy++) {
    for (int jx = 0; jx < Order + 3; jx++) {
      rhosum2 += cur[jy][jx][0];
    }
  }

  // contribution to charge density is normalized to unity
  status = status && (std::abs(rhosum1 - 1) < epsilon);

  // charge density increases exactly by one
  status = status && (std::abs(rhosum2 - (rhosum0 + 1)) < epsilon * std::abs(rhosum2));

  return status;
}

template <int Order, typename T_float>
bool deposit2d_xsimd(const float64 delt, const float64 delh, T_float xu[7], T_float xv[7],
                     T_float rho[Order + 3][Order + 3], T_float cur[Order + 3][Order + 3][4],
                     const float64 epsilon)
{
  const T_float zero = 0;
  const T_float rdh  = 1 / delh;
  const float64 dhdt = delh / delt;
  const T_float qs   = 1;

  bool    status  = true;
  T_float rhosum0 = 0;
  T_float rhosum1 = 0;
  T_float rhosum2 = 0;
  T_float vz      = 0;

  T_float ss[2][2][Order + 3] = {0};

  xv[0] = xu[0];
  xv[1] = xu[1];
  xv[2] = xu[2];
  xu[0] = xu[0] + xu[3] * delt;
  xu[1] = xu[1] + xu[4] * delt;
  xu[2] = xu[2] + xu[5] * delt;
  vz    = xu[5];

  //
  // before move
  //
  auto ix0 = digitize(xv[0], zero, rdh);
  auto iy0 = digitize(xv[1], zero, rdh);

  shape_mc<Order>(xv[0], xsimd::to_float(ix0) * delh, rdh, &ss[0][0][1]);
  shape_mc<Order>(xv[1], xsimd::to_float(iy0) * delh, rdh, &ss[0][1][1]);

  // check charge density
  for (int jy = 0; jy < Order + 3; jy++) {
    for (int jx = 0; jx < Order + 3; jx++) {
      T_float r = ss[0][0][jx] * ss[0][1][jy];
      rhosum0 += cur[jy][jx][0];
      rhosum1 += r;
      rho[jy][jx] += r;
    }
  }

  //
  // after move
  //
  auto ix1 = digitize(xu[0], zero, rdh);
  auto iy1 = digitize(xu[1], zero, rdh);

  shape_mc<Order>(xu[0], xsimd::to_float(ix1) * delh, rdh, &ss[1][0][1]);
  shape_mc<Order>(xu[1], xsimd::to_float(iy1) * delh, rdh, &ss[1][1][1]);

  //
  // in-place shift of ss[1] according to particle movement
  //
  xsimd::batch<int64_t> shift[2] = {ix1 - ix0, iy1 - iy0};
  shift_weights<2, Order>(shift, ss[1]);

  // calculate charge and current density
  deposit2d<Order>(dhdt, dhdt, vz, qs, ss, cur);

  // check charge density
  for (int jy = 0; jy < Order + 3; jy++) {
    for (int jx = 0; jx < Order + 3; jx++) {
      rhosum2 += cur[jy][jx][0];
    }
  }

  // error check: normalize the accumulated sums by the SIMD width
  {
    float64 rho0 = xsimd::reduce_add(rhosum0) / T_float::size;
    float64 rho1 = xsimd::reduce_add(rhosum1) / T_float::size;
    float64 rho2 = xsimd::reduce_add(rhosum2) / T_float::size;

    // contribution to charge density is normalized to unity
    status = status && (std::abs(rho1 - 1) < epsilon);

    // charge density increases exactly by one
    status = status && (std::abs(rho2 - (rho0 + 1)) < epsilon * std::abs(rho2));
  }

  return status;
}

template <int Order, typename T_array>
bool test_deposit2d_scalar(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                           const float64 epsilon)
{
  const int size    = Order + 3;
  bool      status1 = true;
  bool      status2 = true;

  float64 cur[size][size][4] = {0};
  float64 rho[size][size]    = {0};

  for (int ip = 0; ip < Np; ip++) {
    float64* xu_ptr = &xu(ip, 0);
    float64* xv_ptr = &xv(ip, 0);

    status1 = status1 && deposit2d_scalar<Order>(delt, delh, xu_ptr, xv_ptr, rho, cur, epsilon);
  }
  status2 = status2 && test_conservation2d(delt, delh, rho, cur, epsilon);

  return status1 && status2;
}

template <int Order, typename T_array>
bool test_deposit2d_xsimd(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                          const float64 epsilon)
{
  const int size    = Order + 3;
  bool      status1 = true;
  bool      status2 = true;

  float64  rho[size][size]         = {0};
  float64  cur[size][size][4]      = {0};
  simd_f64 rho_simd[size][size]    = {0};
  simd_f64 cur_simd[size][size][4] = {0};
  simd_f64 xu_simd[7];
  simd_f64 xv_simd[7];
  simd_i64 index_simd = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  // SIMD version
  for (int ip = 0; ip < Np - Np % simd_f64::size; ip += simd_f64::size) {
    // Load data
    for (int k = 0; k < 7; k++) {
      xu_simd[k] = simd_f64::gather(&xu(ip, k), index_simd);
      xv_simd[k] = simd_f64::gather(&xv(ip, k), index_simd);
    }

    status1 = status1 &&
              deposit2d_xsimd<Order>(delt, delh, xu_simd, xv_simd, rho_simd, cur_simd, epsilon);
  }

  // scalar version
  for (int ip = 0; ip < Np - Np % simd_f64::size; ip++) {
    float64* xu_ptr = &xu(ip, 0);
    float64* xv_ptr = &xv(ip, 0);

    status1 = status1 && deposit2d_scalar<Order>(delt, delh, xu_ptr, xv_ptr, rho, cur, epsilon);
  }

  // compare scalar and SIMD results
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      for (int l = 0; l < 4; l++) {
        float64 cur_sum = xsimd::reduce_add(cur_simd[i][j][l]);
        float64 cur_err = std::abs(cur[i][j][l] - cur_sum);
        status2 = status2 && ((cur_err <= epsilon) || (cur_err <= std::abs(cur_sum) * epsilon));
      }
      float64 rho_sum = xsimd::reduce_add(rho_simd[i][j]);
      float64 rho_err = std::abs(rho[i][j] - rho_sum);
      status2 = status2 && ((rho_err <= epsilon) || (rho_err <= std::abs(rho_sum) * epsilon));
    }
  }

  return status1 && status2;
}

//
// for 3D version
//

template <int N>
bool test_conservation3d(const float64 delt, const float64 delh, const float64 rho[N][N][N],
                         const float64 cur[N][N][N][4], const float64 epsilon)
{
  bool    status = true;
  float64 errsum = 0.0;
  float64 errnrm = 0.0;

  float64 J[N + 1][N + 1][N + 1][4] = {0};

  for (int jz = 0; jz < N; jz++) {
    for (int jy = 0; jy < N; jy++) {
      for (int jx = 0; jx < N; jx++) {
        J[jz][jy][jx][0] = cur[jz][jy][jx][0];
        J[jz][jy][jx][1] = cur[jz][jy][jx][1];
        J[jz][jy][jx][2] = cur[jz][jy][jx][2];
        J[jz][jy][jx][3] = cur[jz][jy][jx][3];
      }
    }
  }

  for (int jz = 0; jz < N; jz++) {
    for (int jy = 0; jy < N; jy++) {
      for (int jx = 0; jx < N; jx++) {
        errnrm += std::abs(J[jz][jy][jx][0]);
        errsum += std::abs((J[jz][jy][jx][0] - rho[jz][jy][jx]) +
                           delt / delh * (J[jz][jy][jx + 1][1] - J[jz][jy][jx][1]) +
                           delt / delh * (J[jz][jy + 1][jx][2] - J[jz][jy][jx][2]) +
                           delt / delh * (J[jz + 1][jy][jx][3] - J[jz][jy][jx][3]));
      }
    }
  }

  status = status && (errsum < epsilon * errnrm);
  return status;
}

template <int Order>
bool deposit3d_scalar(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                      float64 rho[Order + 3][Order + 3][Order + 3],
                      float64 cur[Order + 3][Order + 3][Order + 3][4], const float64 epsilon)
{
  const float64 rdh  = 1 / delh;
  const float64 dhdt = delh / delt;
  const float64 qs   = 1;

  bool    status  = true;
  float64 rhosum0 = 0;
  float64 rhosum1 = 0;
  float64 rhosum2 = 0;

  float64 ss[2][3][Order + 3] = {0};

  xv[0] = xu[0];
  xv[1] = xu[1];
  xv[2] = xu[2];
  xu[0] = xu[0] + xu[3] * delt;
  xu[1] = xu[1] + xu[4] * delt;
  xu[2] = xu[2] + xu[5] * delt;

  //
  // before move
  //
  int ix0 = digitize(xv[0], 0.0, rdh);
  int iy0 = digitize(xv[1], 0.0, rdh);
  int iz0 = digitize(xv[2], 0.0, rdh);

  shape_mc<Order>(xv[0], ix0 * delh, rdh, &ss[0][0][1]);
  shape_mc<Order>(xv[1], iy0 * delh, rdh, &ss[0][1][1]);
  shape_mc<Order>(xv[2], iz0 * delh, rdh, &ss[0][2][1]);

  // check charge density
  for (int jz = 0; jz < Order + 3; jz++) {
    for (int jy = 0; jy < Order + 3; jy++) {
      for (int jx = 0; jx < Order + 3; jx++) {
        float64 r = ss[0][0][jx] * ss[0][1][jy] * ss[0][2][jz];
        rhosum0 += cur[jz][jy][jx][0];
        rhosum1 += r;
        rho[jz][jy][jx] += r;
      }
    }
  }

  //
  // after move
  //
  int ix1 = digitize(xu[0], 0.0, rdh);
  int iy1 = digitize(xu[1], 0.0, rdh);
  int iz1 = digitize(xu[2], 0.0, rdh);

  shape_mc<Order>(xu[0], ix1 * delh, rdh, &ss[1][0][1 + ix1 - ix0]);
  shape_mc<Order>(xu[1], iy1 * delh, rdh, &ss[1][1][1 + iy1 - iy0]);
  shape_mc<Order>(xu[2], iz1 * delh, rdh, &ss[1][2][1 + iz1 - iz0]);

  // calculate charge and current density
  deposit3d<Order>(dhdt, dhdt, dhdt, qs, ss, cur);

  // check charge density
  for (int jz = 0; jz < Order + 3; jz++) {
    for (int jy = 0; jy < Order + 3; jy++) {
      for (int jx = 0; jx < Order + 3; jx++) {
        rhosum2 += cur[jz][jy][jx][0];
      }
    }
  }

  // contribution to charge density is normalized to unity
  status = status && (std::abs(rhosum1 - 1) < epsilon);

  // charge density increases exactly by one
  status = status && (std::abs(rhosum2 - (rhosum0 + 1)) < epsilon * std::abs(rhosum2));

  return status;
}

template <int Order, typename T_float>
bool deposit3d_xsimd(const float64 delt, const float64 delh, T_float xu[7], T_float xv[7],
                     T_float rho[Order + 3][Order + 3][Order + 3],
                     T_float cur[Order + 3][Order + 3][Order + 3][4], const float64 epsilon)
{
  const T_float zero = 0;
  const T_float rdh  = 1 / delh;
  const float64 dhdt = delh / delt;
  const T_float qs   = 1;

  bool    status  = true;
  T_float rhosum0 = 0;
  T_float rhosum1 = 0;
  T_float rhosum2 = 0;

  T_float ss[2][3][Order + 3] = {0};

  xv[0] = xu[0];
  xv[1] = xu[1];
  xv[2] = xu[2];
  xu[0] = xu[0] + xu[3] * delt;
  xu[1] = xu[1] + xu[4] * delt;
  xu[2] = xu[2] + xu[5] * delt;

  //
  // before move
  //
  auto ix0 = digitize(xv[0], zero, rdh);
  auto iy0 = digitize(xv[1], zero, rdh);
  auto iz0 = digitize(xv[2], zero, rdh);

  shape_mc<Order>(xv[0], xsimd::to_float(ix0) * delh, rdh, &ss[0][0][1]);
  shape_mc<Order>(xv[1], xsimd::to_float(iy0) * delh, rdh, &ss[0][1][1]);
  shape_mc<Order>(xv[2], xsimd::to_float(iz0) * delh, rdh, &ss[0][2][1]);

  // check charge density
  for (int jz = 0; jz < Order + 3; jz++) {
    for (int jy = 0; jy < Order + 3; jy++) {
      for (int jx = 0; jx < Order + 3; jx++) {
        T_float r = ss[0][0][jx] * ss[0][1][jy] * ss[0][2][jz];
        rhosum0 += cur[jz][jy][jx][0];
        rhosum1 += r;
        rho[jz][jy][jx] += r;
      }
    }
  }

  //
  // after move
  //
  auto ix1 = digitize(xu[0], zero, rdh);
  auto iy1 = digitize(xu[1], zero, rdh);
  auto iz1 = digitize(xu[2], zero, rdh);

  shape_mc<Order>(xu[0], xsimd::to_float(ix1) * delh, rdh, &ss[1][0][1]);
  shape_mc<Order>(xu[1], xsimd::to_float(iy1) * delh, rdh, &ss[1][1][1]);
  shape_mc<Order>(xu[2], xsimd::to_float(iz1) * delh, rdh, &ss[1][2][1]);

  //
  // in-place shift of ss[1] according to particle movement
  //
  xsimd::batch<int64_t> shift[3] = {ix1 - ix0, iy1 - iy0, iz1 - iz0};
  shift_weights<3, Order>(shift, ss[1]);

  // calculate charge and current density
  deposit3d<Order>(dhdt, dhdt, dhdt, qs, ss, cur);

  // check charge density
  for (int jz = 0; jz < Order + 3; jz++) {
    for (int jy = 0; jy < Order + 3; jy++) {
      for (int jx = 0; jx < Order + 3; jx++) {
        rhosum2 += cur[jz][jy][jx][0];
      }
    }
  }

  // error check
  {
    float64 rho0 = xsimd::reduce_add(rhosum0) / T_float::size;
    float64 rho1 = xsimd::reduce_add(rhosum1) / T_float::size;
    float64 rho2 = xsimd::reduce_add(rhosum2) / T_float::size;

    // contribution to charge density is normalized to unity
    status = status && (std::abs(rho1 - 1) < epsilon);

    // charge density increases exactly by one
    status = status && (std::abs(rho2 - (rho0 + 1)) < epsilon * std::abs(rho2));
  }

  return status;
}

template <int Order, typename T_array>
bool test_deposit3d_scalar(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                           const float64 epsilon)
{
  const int size    = Order + 3;
  bool      status1 = true;
  bool      status2 = true;

  float64 cur[size][size][size][4] = {0};
  float64 rho[size][size][size]    = {0};

  for (int ip = 0; ip < Np; ip++) {
    float64* xu_ptr = &xu(ip, 0);
    float64* xv_ptr = &xv(ip, 0);

    status1 = status1 && deposit3d_scalar<Order>(delt, delh, xu_ptr, xv_ptr, rho, cur, epsilon);
  }
  status2 = status2 && test_conservation3d(delt, delh, rho, cur, epsilon);

  return status1 && status2;
}

template <int Order, typename T_array>
bool test_deposit3d_xsimd(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                          const float64 epsilon)
{
  const int size    = Order + 3;
  bool      status1 = true;
  bool      status2 = true;

  float64  cur[size][size][size][4]      = {0};
  float64  rho[size][size][size]         = {0};
  simd_f64 cur_simd[size][size][size][4] = {0};
  simd_f64 rho_simd[size][size][size]    = {0};
  simd_f64 xu_simd[7];
  simd_f64 xv_simd[7];
  simd_i64 index_simd = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  // SIMD version
  for (int ip = 0; ip < Np - Np % simd_f64::size; ip += simd_f64::size) {
    //  load data
    for (int k = 0; k < 7; k++) {
      xu_simd[k] = simd_f64::gather(&xu(ip, k), index_simd);
      xv_simd[k] = simd_f64::gather(&xv(ip, k), index_simd);
    }

    status1 = status1 &&
              deposit3d_xsimd<Order>(delt, delh, xu_simd, xv_simd, rho_simd, cur_simd, epsilon);
  }

  // scalar version
  for (int ip = 0; ip < Np - Np % simd_f64::size; ip++) {
    float64* xu_ptr = &xu(ip, 0);
    float64* xv_ptr = &xv(ip, 0);

    status1 = status1 && deposit3d_scalar<Order>(delt, delh, xu_ptr, xv_ptr, rho, cur, epsilon);
  }

  // compare scalar and SIMD results
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        for (int l = 0; l < 4; l++) {
          float64 cur_sum = xsimd::reduce_add(cur_simd[i][j][k][l]);
          float64 cur_err = std::abs(cur[i][j][k][l] - cur_sum);
          status2 = status2 && ((cur_err <= epsilon) || (cur_err <= std::abs(cur_sum) * epsilon));
        }
        float64 rho_sum = xsimd::reduce_add(rho_simd[i][j][k]);
        float64 rho_err = std::abs(rho[i][j][k] - rho_sum);
        status2 = status2 && ((rho_err <= epsilon) || (rho_err <= std::abs(rho_sum) * epsilon));
      }
    }
  }

  return status1 && status2;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
