// -*- C++ -*-
#ifndef _HYBRID_ENGINE_SSOR2_HPP_
#define _HYBRID_ENGINE_SSOR2_HPP_

#include "engine/ohm_solver.hpp"
#include "nix/array_types.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace hybrid::engine
{
struct Ssor2Config {
  int          max_iterations = 100;
  nix::float64 tolerance      = 1.0e-5;
};

struct Ssor2Workspace {
  nix::Array4D<nix::float64>& field;
  nix::Array4D<nix::float64>& source;
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

class LegacySsor2 final : public OhmSolver
{
public:
  LegacySsor2(OhmSolverCoefficients coefficients, Ssor2Config config)
      : coefficients_(coefficients), config_(config)
  {
  }

  OhmSolveStats solve(OhmSolveContext& context) override
  {
    OhmSolveStats stats = {};
    for (int iteration = 1; iteration <= config_.max_iterations; ++iteration) {
      context.for_each_system([&](OhmSystemView& system) {
        Ssor2Workspace workspace = {system.electric_field,
                                    system.source,
                                    system.Lbx,
                                    system.Ubx,
                                    system.Lby,
                                    system.Uby,
                                    system.Lbz,
                                    system.Ubz};
        ssor2_forward_sweep(workspace, coefficients_);
      });
      context.exchange_electric();
      context.for_each_system([&](OhmSystemView& system) {
        Ssor2Workspace workspace = {system.electric_field,
                                    system.source,
                                    system.Lbx,
                                    system.Ubx,
                                    system.Lby,
                                    system.Uby,
                                    system.Lbz,
                                    system.Ubz};
        ssor2_backward_sweep(workspace, coefficients_);
      });
      context.exchange_electric();

      nix::float64 local_error_sum = 0;
      nix::float64 local_norm_sum  = 0;
      context.for_each_system([&](OhmSystemView& system) {
        Ssor2Workspace workspace         = {system.electric_field,
                                            system.source,
                                            system.Lbx,
                                            system.Ubx,
                                            system.Lby,
                                            system.Uby,
                                            system.Lbz,
                                            system.Ubz};
        const auto [error_sum, norm_sum] = ssor2_local_residual(workspace, coefficients_);
        local_error_sum += error_sum;
        local_norm_sum += norm_sum;
      });
      const auto [error_sum, norm_sum] = context.global_reduce(local_error_sum, local_norm_sum);
      stats.iterations                 = iteration;
      stats.residual_norm              = std::sqrt(error_sum);
      stats.source_norm                = std::sqrt(norm_sum);
      stats.relative_residual          = stats.residual_norm / (stats.source_norm + 1.0e-32);
      if (!std::isfinite(stats.relative_residual)) {
        throw std::runtime_error("Hybrid SSOR2 produced a non-finite residual");
      }
      if (stats.relative_residual < config_.tolerance) {
        stats.converged = true;
      }
      context.record_iteration(iteration, stats);
      if (stats.converged) {
        return stats;
      }
    }
    return stats;
  }

private:
  OhmSolverCoefficients coefficients_;
  Ssor2Config           config_;
};

inline OhmSolveStats solve_ssor2(Ssor2Workspace& workspace, const OhmSolverCoefficients& coeff,
                                 const Ssor2Config& config)
{
  OhmSolveContext context = {
      [&](const OhmSystemOperation& operation) {
        OhmSystemView system = {workspace.field, workspace.source, workspace.Lbx, workspace.Ubx,
                                workspace.Lby,   workspace.Uby,    workspace.Lbz, workspace.Ubz};
        operation(system);
      },
      []() {},
      [](nix::float64 error_sum, nix::float64 norm_sum) {
        return std::pair{error_sum, norm_sum};
      },
      [](int, const OhmSolveStats&) {},
  };
  LegacySsor2 solver(coeff, config);
  return solver.solve(context);
}
} // namespace hybrid::engine

#endif
