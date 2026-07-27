// -*- C++ -*-
#ifndef _HYBRID_ENGINE_OHM_SOLVER_HPP_
#define _HYBRID_ENGINE_OHM_SOLVER_HPP_

#include "engine/ohm_source.hpp"

namespace hybrid::engine
{
struct OhmSolverCoefficients {
  nix::float64 laplacian_x;
  nix::float64 laplacian_y;
  nix::float64 laplacian_z;
  nix::float64 diagonal_minus_2_laplacian;
};

inline OhmSolverCoefficients compute_ssor2_coefficients(nix::float64 light_speed,
                                                        nix::float64 spacing_x,
                                                        nix::float64 spacing_y,
                                                        nix::float64 spacing_z)
{
  const nix::float64 cdx = light_speed / spacing_x;
  const nix::float64 cdy = light_speed / spacing_y;
  const nix::float64 cdz = light_speed / spacing_z;
  const nix::float64 cx  = cdx * cdx;
  const nix::float64 cy  = cdy * cdy;
  const nix::float64 cz  = cdz * cdz;
  return {cx, cy, cz, 2 * (cx + cy + cz)};
}

inline nix::float64 ssor2_update(const OhmSource& src, const nix::float64 left,
                                 const nix::float64 right, const nix::float64 back,
                                 const nix::float64 front, const nix::float64 down,
                                 const nix::float64 up, const OhmSolverCoefficients& coeff,
                                 int component)
{
  return (src[component] + coeff.laplacian_x * (left + right) + coeff.laplacian_y * (back + front) +
          coeff.laplacian_z * (down + up)) /
         (src[current_component::charge] + coeff.diagonal_minus_2_laplacian);
}

inline nix::float64 ssor2_residual(const OhmSource& src, const nix::float64 eb,
                                   const nix::float64 left, const nix::float64 right,
                                   const nix::float64 back, const nix::float64 front,
                                   const nix::float64 down, const nix::float64 up,
                                   const OhmSolverCoefficients& coeff, int component)
{
  const nix::float64 coefficient =
      src[current_component::charge] + coeff.diagonal_minus_2_laplacian;
  return eb * coefficient - (src[component] + coeff.laplacian_x * (left + right) +
                             coeff.laplacian_y * (back + front) + coeff.laplacian_z * (down + up));
}

inline bool ssor2_converged(nix::float64 error_sum, nix::float64 norm_sum, nix::float64 tolerance,
                            int iteration, int max_iterations)
{
  if (iteration >= max_iterations) {
    return true;
  }
  const nix::float64 relative_error = std::sqrt(error_sum) / (std::sqrt(norm_sum) + 1.0e-32);
  return relative_error < tolerance;
}
} // namespace hybrid::engine

#endif
