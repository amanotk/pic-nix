// -*- C++-*-
#include "thirdparty/catch.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <petscdmda.h>

#include "petsc_poisson.hpp"
#include "test_parallel_common.hpp"

using namespace nix::typedefs;
using nix::Dims3D;

constexpr float64 tolerance = 1.0e-10;
constexpr float64 ksp_rtol  = 1.0e-15;

class PetscPoisson1DTest : public elliptic::PetscPoisson1D
{
private:
  float64 kx;

public:
  PetscPoisson1DTest(const int n) : elliptic::PetscPoisson1D({1, 1, n}, 1.0 / n)
  {
    kx = 3 * nix::math::pi2;

    initialize_rhs();
    KSPSetTolerances(this->ksp_obj, ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  }

  float64 get_source(float64 x)
  {
    return std::sin(kx * x);
  }

  float64 get_solution(float64 x)
  {
    float64 kappa_x = std::sin(0.5 * kx * delx) / (0.5 * delx);
    float64 denom   = kappa_x * kappa_x + 1.0e-32;
    return std::sin(kx * x) / denom;
  }

  void initialize_rhs()
  {
    VecSet(this->vector_src_g, 0.0);
    VecSet(this->vector_sol_g, 0.0);

    DMDALocalInfo info;
    DMDAGetLocalInfo(this->dm_obj, &info);

    PetscScalar* data = nullptr;
    VecGetArray(this->vector_src_g, &data);

    PetscInt offset = 0;
    for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
      float64 x    = static_cast<float64>(ix) * delx;
      data[offset] = get_source(x);
      offset++;
    }

    VecRestoreArray(this->vector_src_g, &data);
  }

  float64 get_error_norm()
  {
    PetscScalar* sol      = nullptr;
    PetscScalar* err      = nullptr;
    PetscReal    err_norm = 0.0;
    PetscReal    sol_norm = 0.0;
    Vec          vector_err_g;

    VecDuplicate(vector_sol_g, &vector_err_g);
    VecGetArray(vector_sol_g, &sol);
    VecGetArray(vector_err_g, &err);

    DMDALocalInfo info;
    DMDAGetLocalInfo(this->dm_obj, &info);

    PetscInt offset = 0;
    for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
      float64 x   = static_cast<float64>(ix) * delx;
      err[offset] = get_solution(x) - sol[offset];
      offset++;
    }

    VecRestoreArray(vector_err_g, &err);
    VecRestoreArray(vector_sol_g, &sol);

    VecNorm(vector_err_g, NORM_2, &err_norm);
    VecNorm(vector_sol_g, NORM_2, &sol_norm);

    VecDestroy(&vector_err_g);

    return static_cast<float64>(err_norm / (sol_norm + 1.0e-32));
  }
};

class PetscPoisson2DTest : public elliptic::PetscPoisson2D
{
private:
  float64 kx;
  float64 ky;

public:
  PetscPoisson2DTest(const int n) : elliptic::PetscPoisson2D({1, n, n}, 1.0 / n)
  {
    kx = 3 * nix::math::pi2;
    ky = 4 * nix::math::pi2;

    initialize_rhs();
    KSPSetTolerances(this->ksp_obj, ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  }

  float64 get_source(float64 x, float64 y)
  {
    return std::sin(kx * x) * std::sin(ky * y);
  }

  float64 get_solution(float64 x, float64 y)
  {
    const float64 kappa_x = std::sin(0.5 * kx * delx) / (0.5 * delx);
    const float64 kappa_y = std::sin(0.5 * ky * dely) / (0.5 * dely);
    const float64 denom   = kappa_x * kappa_x + kappa_y * kappa_y + 1.0e-32;
    return std::sin(kx * x) * std::sin(ky * y) / denom;
  }

  void initialize_rhs()
  {
    VecSet(vector_src_g, 0.0);
    VecSet(vector_sol_g, 0.0);

    DMDALocalInfo info;
    DMDAGetLocalInfo(dm_obj, &info);

    PetscScalar* data = nullptr;
    VecGetArray(vector_src_g, &data);

    PetscInt offset = 0;
    for (int iy = info.ys; iy < info.ys + info.ym; ++iy) {
      const float64 y = static_cast<float64>(iy) * dely;
      for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
        const float64 x = static_cast<float64>(ix) * delx;
        data[offset]    = get_source(x, y);
        offset++;
      }
    }

    VecRestoreArray(vector_src_g, &data);
  }

  float64 get_error_norm()
  {
    Vec          vector_err_g;
    PetscScalar* sol      = nullptr;
    PetscScalar* err      = nullptr;
    PetscReal    err_norm = 0.0;
    PetscReal    sol_norm = 0.0;

    VecDuplicate(vector_sol_g, &vector_err_g);
    VecGetArray(vector_sol_g, &sol);
    VecGetArray(vector_err_g, &err);

    DMDALocalInfo info;
    DMDAGetLocalInfo(dm_obj, &info);

    PetscInt offset = 0;
    for (int iy = info.ys; iy < info.ys + info.ym; ++iy) {
      const float64 y = static_cast<float64>(iy) * dely;
      for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
        const float64 x = static_cast<float64>(ix) * delx;
        err[offset]     = get_solution(x, y) - sol[offset];
        offset++;
      }
    }

    VecRestoreArray(vector_err_g, &err);
    VecRestoreArray(vector_sol_g, &sol);

    VecNorm(vector_err_g, NORM_2, &err_norm);
    VecNorm(vector_sol_g, NORM_2, &sol_norm);

    VecDestroy(&vector_err_g);

    return static_cast<float64>(err_norm / (sol_norm + 1.0e-32));
  }
};

class PetscPoisson3DTest : public elliptic::PetscPoisson3D
{
private:
  float64 kx;
  float64 ky;
  float64 kz;

public:
  PetscPoisson3DTest(const int n) : elliptic::PetscPoisson3D({n, n, n}, 1.0 / n)
  {
    kx = 3 * nix::math::pi2;
    ky = 4 * nix::math::pi2;
    kz = 5 * nix::math::pi2;

    initialize_rhs();
    KSPSetTolerances(this->ksp_obj, ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  }

  float64 get_source(float64 x, float64 y, float64 z)
  {
    return std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z);
  }

  float64 get_solution(float64 x, float64 y, float64 z)
  {
    const float64 kappa_x = std::sin(0.5 * kx * delx) / (0.5 * delx);
    const float64 kappa_y = std::sin(0.5 * ky * dely) / (0.5 * dely);
    const float64 kappa_z = std::sin(0.5 * kz * delz) / (0.5 * delz);
    const float64 denom   = kappa_x * kappa_x + kappa_y * kappa_y + kappa_z * kappa_z + 1.0e-32;
    return std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z) / denom;
  }

  void initialize_rhs()
  {
    VecSet(vector_src_g, 0.0);
    VecSet(vector_sol_g, 0.0);

    DMDALocalInfo info;
    DMDAGetLocalInfo(dm_obj, &info);

    PetscScalar* data = nullptr;
    VecGetArray(vector_src_g, &data);

    PetscInt offset = 0;
    for (int iz = info.zs; iz < info.zs + info.zm; ++iz) {
      const float64 z = static_cast<float64>(iz) * delz;
      for (int iy = info.ys; iy < info.ys + info.ym; ++iy) {
        const float64 y = static_cast<float64>(iy) * dely;
        for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
          const float64 x = static_cast<float64>(ix) * delx;
          data[offset++]  = get_source(x, y, z);
        }
      }
    }

    VecRestoreArray(vector_src_g, &data);
  }

  float64 get_error_norm()
  {
    Vec          vector_err_g;
    PetscScalar* sol      = nullptr;
    PetscScalar* err      = nullptr;
    PetscReal    err_norm = 0.0;
    PetscReal    sol_norm = 0.0;

    VecDuplicate(vector_sol_g, &vector_err_g);
    VecGetArray(vector_sol_g, &sol);
    VecGetArray(vector_err_g, &err);

    DMDALocalInfo info;
    DMDAGetLocalInfo(dm_obj, &info);

    PetscInt offset = 0;
    for (int iz = info.zs; iz < info.zs + info.zm; ++iz) {
      const float64 z = static_cast<float64>(iz) * delz;
      for (int iy = info.ys; iy < info.ys + info.ym; ++iy) {
        const float64 y = static_cast<float64>(iy) * dely;
        for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
          const float64 x = static_cast<float64>(ix) * delx;
          err[offset]     = get_solution(x, y, z) - sol[offset];
          ++offset;
        }
      }
    }

    VecRestoreArray(vector_err_g, &err);
    VecRestoreArray(vector_sol_g, &sol);

    VecNorm(vector_err_g, NORM_2, &err_norm);
    VecNorm(vector_sol_g, NORM_2, &sol_norm);

    VecDestroy(&vector_err_g);

    return static_cast<float64>(err_norm / (sol_norm + 1.0e-32));
  }
};

TEST_CASE("PetscPoisson1D solver", "[np=1]")
{
  PetscPoisson1DTest solver(64);
  MockChunkAccessor  accessor;
  solver.solve(accessor);

  const auto err_norm = solver.get_error_norm();
  const auto res_norm = solver.get_residual_norm();

  REQUIRE(err_norm < tolerance);
  REQUIRE(res_norm < tolerance);
}

TEST_CASE("PetscPoisson2D solver", "[np=1]")
{
  PetscPoisson2DTest solver(32);
  MockChunkAccessor  accessor;
  solver.solve(accessor);

  const auto err_norm = solver.get_error_norm();
  const auto res_norm = solver.get_residual_norm();

  REQUIRE(err_norm < tolerance);
  REQUIRE(res_norm < tolerance);
}

TEST_CASE("PetscPoisson3D solver", "[np=1]")
{
  PetscPoisson3DTest solver(16);
  MockChunkAccessor  accessor;
  solver.solve(accessor);

  const auto err_norm = solver.get_error_norm();
  const auto res_norm = solver.get_residual_norm();

  REQUIRE(err_norm < tolerance);
  REQUIRE(res_norm < tolerance);
}
