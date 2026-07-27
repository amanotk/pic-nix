// -*- C++ -*-
#ifndef _HYBRID_ENGINE_SSOR2_HPP_
#define _HYBRID_ENGINE_SSOR2_HPP_

#include "engine/ohm_solver.hpp"
#include "nix/array_types.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace hybrid::engine
{
struct Ssor2Stats {
  int          iterations = 0;
  nix::float64 residual   = 0;
  bool         converged  = false;
};

struct Ssor2Config {
  int          max_iterations = 100;
  nix::float64 tolerance      = 1.0e-5;
};

struct Ssor2Workspace {
  nix::Array4D<nix::float64>& field;
  nix::Array4D<nix::float64>& source;
  nix::Array4D<nix::float64>& resistive;
  int                         Lbx;
  int                         Ubx;
  int                         Lby;
  int                         Uby;
  int                         Lbz;
  int                         Ubz;
};

inline void ssor2_forward_sweep(Ssor2Workspace& workspace, const OhmSolverCoefficients& coeff)
{
  auto& eb  = workspace.field;
  auto& src = workspace.source;
  for (int iz = workspace.Lbz; iz <= workspace.Ubz; ++iz) {
    for (int iy = workspace.Lby; iy <= workspace.Uby; ++iy) {
      for (int ix = workspace.Lbx; ix <= workspace.Ubx; ++ix) {
        const nix::float64 coeff_denom = src(iz, iy, ix, 0) + coeff.diagonal_minus_2_laplacian;
        eb(iz, iy, ix, 0)              = (src(iz, iy, ix, 1) +
                             coeff.laplacian_x * (eb(iz, iy, ix - 1, 0) + eb(iz, iy, ix + 1, 0)) +
                             coeff.laplacian_y * (eb(iz, iy - 1, ix, 0) + eb(iz, iy + 1, ix, 0)) +
                             coeff.laplacian_z * (eb(iz - 1, iy, ix, 0) + eb(iz + 1, iy, ix, 0))) /
                            coeff_denom;
        eb(iz, iy, ix, 1) = (src(iz, iy, ix, 2) +
                             coeff.laplacian_x * (eb(iz, iy, ix - 1, 1) + eb(iz, iy, ix + 1, 1)) +
                             coeff.laplacian_y * (eb(iz, iy - 1, ix, 1) + eb(iz, iy + 1, ix, 1)) +
                             coeff.laplacian_z * (eb(iz - 1, iy, ix, 1) + eb(iz + 1, iy, ix, 1))) /
                            coeff_denom;
        eb(iz, iy, ix, 2) = (src(iz, iy, ix, 3) +
                             coeff.laplacian_x * (eb(iz, iy, ix - 1, 2) + eb(iz, iy, ix + 1, 2)) +
                             coeff.laplacian_y * (eb(iz, iy - 1, ix, 2) + eb(iz, iy + 1, ix, 2)) +
                             coeff.laplacian_z * (eb(iz - 1, iy, ix, 2) + eb(iz + 1, iy, ix, 2))) /
                            coeff_denom;
      }
    }
  }
}

inline void ssor2_backward_sweep(Ssor2Workspace& workspace, const OhmSolverCoefficients& coeff)
{
  auto& eb  = workspace.field;
  auto& src = workspace.source;
  for (int iz = workspace.Ubz; iz >= workspace.Lbz; --iz) {
    for (int iy = workspace.Uby; iy >= workspace.Lby; --iy) {
      for (int ix = workspace.Ubx; ix >= workspace.Lbx; --ix) {
        const nix::float64 coeff_denom = src(iz, iy, ix, 0) + coeff.diagonal_minus_2_laplacian;
        eb(iz, iy, ix, 0)              = (src(iz, iy, ix, 1) +
                             coeff.laplacian_x * (eb(iz, iy, ix - 1, 0) + eb(iz, iy, ix + 1, 0)) +
                             coeff.laplacian_y * (eb(iz, iy - 1, ix, 0) + eb(iz, iy + 1, ix, 0)) +
                             coeff.laplacian_z * (eb(iz - 1, iy, ix, 0) + eb(iz + 1, iy, ix, 0))) /
                            coeff_denom;
        eb(iz, iy, ix, 1) = (src(iz, iy, ix, 2) +
                             coeff.laplacian_x * (eb(iz, iy, ix - 1, 1) + eb(iz, iy, ix + 1, 1)) +
                             coeff.laplacian_y * (eb(iz, iy - 1, ix, 1) + eb(iz, iy + 1, ix, 1)) +
                             coeff.laplacian_z * (eb(iz - 1, iy, ix, 1) + eb(iz + 1, iy, ix, 1))) /
                            coeff_denom;
        eb(iz, iy, ix, 2) = (src(iz, iy, ix, 3) +
                             coeff.laplacian_x * (eb(iz, iy, ix - 1, 2) + eb(iz, iy, ix + 1, 2)) +
                             coeff.laplacian_y * (eb(iz, iy - 1, ix, 2) + eb(iz, iy + 1, ix, 2)) +
                             coeff.laplacian_z * (eb(iz - 1, iy, ix, 2) + eb(iz + 1, iy, ix, 2))) /
                            coeff_denom;
      }
    }
  }
}

inline std::pair<nix::float64, nix::float64>
ssor2_local_residual(const Ssor2Workspace& workspace, const OhmSolverCoefficients& coeff)
{
  auto&        eb        = workspace.field;
  const auto&  src       = workspace.source;
  nix::float64 error_sum = 0;
  nix::float64 norm_sum  = 0;
  for (int iz = workspace.Lbz; iz <= workspace.Ubz; ++iz) {
    for (int iy = workspace.Lby; iy <= workspace.Uby; ++iy) {
      for (int ix = workspace.Lbx; ix <= workspace.Ubx; ++ix) {
        const nix::float64 coefficient = src(iz, iy, ix, 0) + coeff.diagonal_minus_2_laplacian;
        const nix::float64 laplacian_e0 =
            coeff.laplacian_x * (eb(iz, iy, ix - 1, 0) + eb(iz, iy, ix + 1, 0)) +
            coeff.laplacian_y * (eb(iz, iy - 1, ix, 0) + eb(iz, iy + 1, ix, 0)) +
            coeff.laplacian_z * (eb(iz - 1, iy, ix, 0) + eb(iz + 1, iy, ix, 0));
        const nix::float64 laplacian_e1 =
            coeff.laplacian_x * (eb(iz, iy, ix - 1, 1) + eb(iz, iy, ix + 1, 1)) +
            coeff.laplacian_y * (eb(iz, iy - 1, ix, 1) + eb(iz, iy + 1, ix, 1)) +
            coeff.laplacian_z * (eb(iz - 1, iy, ix, 1) + eb(iz + 1, iy, ix, 1));
        const nix::float64 laplacian_e2 =
            coeff.laplacian_x * (eb(iz, iy, ix - 1, 2) + eb(iz, iy, ix + 1, 2)) +
            coeff.laplacian_y * (eb(iz, iy - 1, ix, 2) + eb(iz, iy + 1, ix, 2)) +
            coeff.laplacian_z * (eb(iz - 1, iy, ix, 2) + eb(iz + 1, iy, ix, 2));

        const nix::float64 ex =
            eb(iz, iy, ix, 0) * coefficient - (src(iz, iy, ix, 1) + laplacian_e0);
        const nix::float64 ey =
            eb(iz, iy, ix, 1) * coefficient - (src(iz, iy, ix, 2) + laplacian_e1);
        const nix::float64 ez =
            eb(iz, iy, ix, 2) * coefficient - (src(iz, iy, ix, 3) + laplacian_e2);

        error_sum += ex * ex + ey * ey + ez * ez;
        norm_sum += src(iz, iy, ix, 1) * src(iz, iy, ix, 1) +
                    src(iz, iy, ix, 2) * src(iz, iy, ix, 2) +
                    src(iz, iy, ix, 3) * src(iz, iy, ix, 3);
      }
    }
  }
  return {error_sum, norm_sum};
}

inline Ssor2Stats solve_ssor2(Ssor2Workspace& workspace, const OhmSolverCoefficients& coeff,
                              const Ssor2Config& config)
{
  Ssor2Stats stats = {};

  for (stats.iterations = 1; stats.iterations <= config.max_iterations; ++stats.iterations) {
    ssor2_forward_sweep(workspace, coeff);
    ssor2_backward_sweep(workspace, coeff);

    auto [error_sum, norm_sum] = ssor2_local_residual(workspace, coeff);
    stats.residual             = std::sqrt(error_sum) / (std::sqrt(norm_sum) + 1.0e-32);
    if (stats.residual < config.tolerance) {
      stats.converged = true;
      break;
    }
  }
  return stats;
}
} // namespace hybrid::engine

#endif
