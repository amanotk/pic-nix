// -*- C++ -*-
#include "engine/ssor2.hpp"
#include "test_hybrid_context.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace
{
std::array<nix::float64, 3> exact_electric(int gz, int gy, int gx, const nix::Dims3D& dims)
{
  const nix::float64 x = 2 * nix::math::pi * gx / dims[2];
  const nix::float64 y = 2 * nix::math::pi * gy / dims[1];
  const nix::float64 z = 2 * nix::math::pi * gz / dims[0];
  return {0.7 + 0.2 * std::sin(x) + 0.1 * std::cos(y),
          -0.4 + 0.15 * std::cos(x + z) - 0.05 * std::sin(y),
          0.2 + 0.12 * std::sin(x - y) + 0.08 * std::cos(z)};
}

void exchange_rank4(hybrid::HybridChunk& chunk, nix::Array4D<nix::float64>& array,
                    hybrid::BoundaryMode mode)
{
  chunk.boundary_pack(array, mode);
  chunk.boundary_begin(array, mode);
  chunk.boundary_end(array, mode);
  chunk.boundary_unpack(array, mode);
}
} // namespace

TEST_CASE("global SSOR2 solves an asymmetric manufactured MPI system")
{
  const int      mpi_size = get_mpi_size();
  HybridTestGrid grid{{4, 4, 8}, {true, true, true}, {1, 1, mpi_size}, mpi_size};
  auto           context = build_hybrid_exchange_context(grid, false);
  auto&          chunk   = *context.chunk;
  auto           data    = chunk.get_internal_data();
  const auto     offset  = chunk.get_offset();

  data.work_field_cell.fill(0);
  data.ohm_source.fill(0);
  const auto coefficients = hybrid::engine::compute_ssor2_coefficients(1.0, 1.0, 1.0, 1.0);
  for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        const int  gz     = offset[0] + iz - data.Lbz;
        const int  gy     = offset[1] + iy - data.Lby;
        const int  gx     = offset[2] + ix - data.Lbx;
        const auto center = exact_electric(gz, gy, gx, grid.gdims);
        const auto xm =
            exact_electric(gz, gy, (gx + grid.gdims[2] - 1) % grid.gdims[2], grid.gdims);
        const auto xp = exact_electric(gz, gy, (gx + 1) % grid.gdims[2], grid.gdims);
        const auto ym =
            exact_electric(gz, (gy + grid.gdims[1] - 1) % grid.gdims[1], gx, grid.gdims);
        const auto yp = exact_electric(gz, (gy + 1) % grid.gdims[1], gx, grid.gdims);
        const auto zm =
            exact_electric((gz + grid.gdims[0] - 1) % grid.gdims[0], gy, gx, grid.gdims);
        const auto         zp      = exact_electric((gz + 1) % grid.gdims[0], gy, gx, grid.gdims);
        const nix::float64 density = 2.0 + 0.03 * gx + 0.02 * gy + 0.01 * gz;
        data.ohm_source(iz, iy, ix, 0) = density;
        for (int component = 0; component < 3; ++component) {
          data.ohm_source(iz, iy, ix, component + 1) =
              (density + coefficients.diagonal_minus_2_laplacian) * center[component] -
              coefficients.laplacian_x * (xm[component] + xp[component]) -
              coefficients.laplacian_y * (ym[component] + yp[component]) -
              coefficients.laplacian_z * (zm[component] + zp[component]);
        }
      }
    }
  }

  hybrid::engine::OhmSolveContext solve_context = {
      [&](const hybrid::engine::OhmSystemOperation& operation) {
        hybrid::engine::OhmSystemView system = {data.work_field_cell,
                                                data.ohm_source,
                                                data.Lbx,
                                                data.Ubx,
                                                data.Lby,
                                                data.Uby,
                                                data.Lbz,
                                                data.Ubz};
        operation(system);
      },
      [&]() { exchange_rank4(chunk, data.work_field_cell, hybrid::BoundaryCopy6); },
      [](nix::float64 error_sum, nix::float64 norm_sum) {
        const nix::float64 local[2]  = {error_sum, norm_sum};
        nix::float64       global[2] = {};
        MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return std::pair{global[0], global[1]};
      },
      [](int, const hybrid::engine::OhmSolveStats&) {},
  };
  hybrid::engine::LegacySsor2 solver(coefficients, {500, 1.0e-10});
  const auto                  stats = solver.solve(solve_context);

  int min_iterations = 0;
  int max_iterations = 0;
  MPI_Allreduce(&stats.iterations, &min_iterations, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&stats.iterations, &max_iterations, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  nix::float64 local_error = 0;
  for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        const int  gz       = offset[0] + iz - data.Lbz;
        const int  gy       = offset[1] + iy - data.Lby;
        const int  gx       = offset[2] + ix - data.Lbx;
        const auto expected = exact_electric(gz, gy, gx, grid.gdims);
        for (int component = 0; component < 3; ++component) {
          local_error = std::max(local_error, std::abs(data.work_field_cell(iz, iy, ix, component) -
                                                       expected[component]));
        }
      }
    }
  }
  nix::float64 global_error = 0;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  REQUIRE(stats.converged);
  REQUIRE(stats.relative_residual < 1.0e-10);
  REQUIRE(min_iterations == max_iterations);
  REQUIRE(global_error < 1.0e-9);
}

TEST_CASE("Ohm moment copy halos preserve constant stress derivatives")
{
  const int      mpi_size = get_mpi_size();
  HybridTestGrid grid{{4, 4, 8}, {true, true, true}, {1, 1, mpi_size}, mpi_size};
  auto           context = build_hybrid_exchange_context(grid, false);
  auto&          chunk   = *context.chunk;
  auto           data    = chunk.get_internal_data();

  data.ohm_moment.fill(0);
  for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        for (int component = hybrid::moment_component::stress_xx;
             component < hybrid::num_moment_components; ++component) {
          data.ohm_moment(iz, iy, ix, component) = 1.0 + component;
        }
      }
    }
  }
  exchange_rank4(chunk, data.ohm_moment, hybrid::BoundaryCopy10);

  nix::float64 local_derivative = 0;
  for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        local_derivative = std::max(
            local_derivative,
            std::abs(data.ohm_moment(iz, iy, ix + 1, hybrid::moment_component::stress_xx) -
                     data.ohm_moment(iz, iy, ix - 1, hybrid::moment_component::stress_xx)));
        local_derivative = std::max(
            local_derivative,
            std::abs(data.ohm_moment(iz, iy + 1, ix, hybrid::moment_component::stress_yy) -
                     data.ohm_moment(iz, iy - 1, ix, hybrid::moment_component::stress_yy)));
        local_derivative = std::max(
            local_derivative,
            std::abs(data.ohm_moment(iz + 1, iy, ix, hybrid::moment_component::stress_zz) -
                     data.ohm_moment(iz - 1, iy, ix, hybrid::moment_component::stress_zz)));
      }
    }
  }
  nix::float64 global_derivative = 0;
  MPI_Allreduce(&local_derivative, &global_derivative, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(global_derivative == 0.0);
}
