// -*- C++ -*-

#include "pic_poisson.hpp"

#include <catch2/catch_test_macros.hpp>
#include <elliptic/chunk_accessor.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <petscvec.h>

#include "test_parallel.hpp"

namespace
{
float64 analytic_solution(float64 kz, float64 ky, float64 kx, float64 kappa2_sum, float64 z,
                          float64 y, float64 x);
float64 analytic_source(float64 kz, float64 ky, float64 kx, float64 z, float64 y, float64 x);
float64 compute_kappa_component(float64 k, float64 h);

template <typename ChunkT>
std::unique_ptr<ChunkT> make_chunk_impl(const nix::Dims3D& dims, const nix::Bool3D& has_dim,
                                        const nix::Dims3D& global_dims, const nix::Dims3D& offset,
                                        int id, json config)
{
  auto chunk = std::make_unique<ChunkT>(dims, has_dim, id);
  int  o[3]  = {offset[0], offset[1], offset[2]};
  int  g[3]  = {global_dims[0], global_dims[1], global_dims[2]};

  chunk->set_global_context(o, g);
  chunk->set_coordinate(1.0, 1.0, 1.0);
  chunk->setup(config);
  chunk->allocate();

  auto data = chunk->get_internal_data();
  data.cc   = 1.0;

  return chunk;
}

class TestPicPoisson : public PicPoisson
{
public:
  using PicPoisson::PicPoisson;

  virtual int copy_chunk_to_src(elliptic::ChunkAccessor& accessor) override
  {
    int count = PicPoisson::copy_chunk_to_src(accessor);
    scatter_forward_begin();
    scatter_forward_end();
    return count;
  }

  virtual int copy_sol_to_chunk(elliptic::ChunkAccessor& accessor) override
  {
    scatter_reverse_begin();
    scatter_reverse_end();
    int count = PicPoisson::copy_sol_to_chunk(accessor);
    return count;
  }

  void copy_src_to_sol()
  {
    VecCopy(vector_src_g, vector_sol_g);
  }
};

class TestPicChunk : public PicChunk
{
public:
  using PicChunk::PicChunk;

  std::array<float64, 3> get_coordinates(int iz, int iy, int ix) const
  {
    auto data = const_cast<TestPicChunk*>(this)->get_internal_data();
    auto gz   = offset[0] + (iz - data.Lbz);
    auto gy   = offset[1] + (iy - data.Lby);
    auto gx   = offset[2] + (ix - data.Lbx);
    return {static_cast<float64>(gz) * data.delz, static_cast<float64>(gy) * data.dely,
            static_cast<float64>(gx) * data.delx};
  }

  int get_cell_count() const
  {
    auto      data = const_cast<TestPicChunk*>(this)->get_internal_data();
    const int nz   = data.Ubz - data.Lbz + 1;
    const int ny   = data.Uby - data.Lby + 1;
    const int nx   = data.Ubx - data.Lbx + 1;
    return nz * ny * nx;
  }

  void populate_source(int mz, int my, int mx, const nix::Dims3D& global_dims)
  {
    auto          data = const_cast<TestPicChunk*>(this)->get_internal_data();
    const float64 lx   = static_cast<float64>(global_dims[2]) * data.delx;
    const float64 ly   = static_cast<float64>(global_dims[1]) * data.dely;
    const float64 lz   = static_cast<float64>(global_dims[0]) * data.delz;
    const float64 kx   = static_cast<float64>(mx) * nix::math::pi2 / lx;
    const float64 ky   = static_cast<float64>(my) * nix::math::pi2 / ly;
    const float64 kz   = static_cast<float64>(mz) * nix::math::pi2 / lz;

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          auto          coords   = get_coordinates(iz, iy, ix);
          const float64 z        = coords[0];
          const float64 y        = coords[1];
          const float64 x        = coords[2];
          data.uj(iz, iy, ix, 0) = analytic_source(kz, ky, kx, z, y, x);
        }
      }
    }
  }

  float64 compute_solution_error(int mz, int my, int mx, const nix::Dims3D& global_dims) const
  {
    auto          data   = const_cast<TestPicChunk*>(this)->get_internal_data();
    float64       sum    = 0.0;
    int           count  = 0;
    const float64 lx     = static_cast<float64>(global_dims[2]) * data.delx;
    const float64 ly     = static_cast<float64>(global_dims[1]) * data.dely;
    const float64 lz     = static_cast<float64>(global_dims[0]) * data.delz;
    const float64 kx     = static_cast<float64>(mx) * nix::math::pi2 / lx;
    const float64 ky     = static_cast<float64>(my) * nix::math::pi2 / ly;
    const float64 kz     = static_cast<float64>(mz) * nix::math::pi2 / lz;
    const float64 kappax = compute_kappa_component(kx, data.delx);
    const float64 kappay = compute_kappa_component(ky, data.dely);
    const float64 kappaz = compute_kappa_component(kz, data.delz);
    const float64 kappa2_sum =
        kappax * kappax + kappay * kappay + kappaz * kappaz + static_cast<float64>(1.0e-32);

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          auto          coords   = get_coordinates(iz, iy, ix);
          const float64 z        = coords[0];
          const float64 y        = coords[1];
          const float64 x        = coords[2];
          const float64 expected = analytic_solution(kz, ky, kx, kappa2_sum, z, y, x);
          const float64 diff     = data.phi(iz, iy, ix) - expected;
          sum += diff * diff;
          ++count;
        }
      }
    }
    return sum / static_cast<float64>(count);
  }
};

json make_default_config()
{
  json config;
  config["option"]                  = json::object();
  config["option"]["vectorization"] = "scalar";
  config["option"]["order"]         = 2;
  config["option"]["pusher"]        = "Boris";
  config["option"]["interpolation"] = "MC";
  config["option"]["seed_type"]     = "fixed";
  config["option"]["friedman"]      = 0.0;
  config["option"]["cell_load"]     = 1.0;
  config["option"]["buffer_ratio"]  = 0.2;
  return config;
}

json make_default_option()
{
  json option;
  option["petsc"]             = json::object();
  option["petsc"]["ksp_type"] = "cg";
  option["petsc"]["pc_type"]  = "amg";
  option["petsc"]["ksp_rtol"] = 1.0e-14;
  return option;
}

float64 analytic_solution(float64 kz, float64 ky, float64 kx, float64 kappa2_sum, float64 z,
                          float64 y, float64 x)
{
  return std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z) / kappa2_sum;
}

float64 analytic_source(float64 kz, float64 ky, float64 kx, float64 z, float64 y, float64 x)
{
  return std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z);
}

inline float64 compute_kappa_component(float64 k, float64 h)
{
  return (k == 0.0) ? 0.0 : std::sin(0.5 * k * h) / (0.5 * h);
}

nix::Dims3D make_chunk_grid_dims(const nix::Dims3D& global_dims, const nix::Dims3D& chunk_dims)
{
  return {global_dims[0] / chunk_dims[0], global_dims[1] / chunk_dims[1],
          global_dims[2] / chunk_dims[2]};
}

std::array<int, 3> make_rank_coords(int rank, const std::array<int, 3>& proc_dims)
{
  const int p0 = rank % proc_dims[0];
  const int p1 = (rank / proc_dims[0]) % proc_dims[1];
  const int p2 = rank / (proc_dims[0] * proc_dims[1]);
  return {p0, p1, p2};
}

int flatten_chunk_index(int cz, int cy, int cx, const nix::Dims3D& chunk_grid_dims)
{
  return elliptic::ChunkAccessor::flatten_index(cz, cy, cx, chunk_grid_dims);
}

std::unique_ptr<TestPicChunk> make_test_chunk(const nix::Dims3D& dims, const nix::Bool3D& has_dim,
                                              const nix::Dims3D& global_dims,
                                              const nix::Dims3D& offset, int id, json config)
{
  auto chunk = make_chunk_impl<TestPicChunk>(dims, has_dim, global_dims, offset, id, config);
  chunk->allocate_mpi_buffers();
  return chunk;
}

void initialize_chunkvec(std::vector<std::unique_ptr<PicChunk>>& chunkvec, int rank,
                         const std::array<int, 3>& proc_dims, const nix::Dims3D& global_dims,
                         const nix::Dims3D& chunk_dims, const nix::Bool3D& has_dim, int mz, int my,
                         int mx, json config)
{
  const nix::Dims3D chunk_grid_dims = make_chunk_grid_dims(global_dims, chunk_dims);
  const auto        rank_coords     = make_rank_coords(rank, proc_dims);
  const nix::Dims3D block_chunks    = {chunk_grid_dims[0] / proc_dims[0],
                                       chunk_grid_dims[1] / proc_dims[1],
                                       chunk_grid_dims[2] / proc_dims[2]};

  for (int lz = 0; lz < block_chunks[0]; ++lz) {
    for (int ly = 0; ly < block_chunks[1]; ++ly) {
      for (int lx = 0; lx < block_chunks[2]; ++lx) {
        const int         cz       = rank_coords[0] * block_chunks[0] + lz;
        const int         cy       = rank_coords[1] * block_chunks[1] + ly;
        const int         cx       = rank_coords[2] * block_chunks[2] + lx;
        const nix::Dims3D offset   = {cz * chunk_dims[0], cy * chunk_dims[1], cx * chunk_dims[2]};
        const int         chunk_id = flatten_chunk_index(cz, cy, cx, chunk_grid_dims);
        auto  chunk = make_test_chunk(chunk_dims, has_dim, global_dims, offset, chunk_id, config);
        auto* chunk_ptr = chunk.get();
        chunkvec.push_back(std::move(chunk));
        chunk_ptr->populate_source(mz, my, mx, global_dims);
      }
    }
  }
}

void require_src_equals_sol(const std::vector<std::unique_ptr<PicChunk>>& chunks, float64 tol)
{
  for (const auto& chunk : chunks) {
    auto data = chunk->get_internal_data();
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          const float64 diff = std::abs(data.phi(iz, iy, ix) - data.uj(iz, iy, ix, 0));
          REQUIRE(diff < tol);
        }
      }
    }
  }
}

void require_rms_error_below(const std::vector<std::unique_ptr<PicChunk>>& chunks, int mz, int my,
                             int mx, const nix::Dims3D& global_dims, float64 tol)
{
  float64 local_sum = 0.0;
  int     local_cnt = 0;
  for (const auto& base_chunk : chunks) {
    auto* chunk = dynamic_cast<TestPicChunk*>(base_chunk.get());
    REQUIRE(chunk != nullptr);
    const int     cnt = chunk->get_cell_count();
    const float64 mse = chunk->compute_solution_error(mz, my, mx, global_dims);
    local_sum += mse * static_cast<float64>(cnt);
    local_cnt += cnt;
  }

  float64 global_sum = 0.0;
  int     global_cnt = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_cnt, &global_cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  const float64 err = std::sqrt(global_sum / static_cast<float64>(global_cnt));
  REQUIRE(err < tol);
}

} // namespace

TEST_CASE("PicPoisson solves 1D periodic Poisson", "[np=8]")
{
  if (!require_mpi_size(8)) {
    return;
  }

  const float64     tol             = 1.0e-12;
  const int         mz              = 0;
  const int         my              = 0;
  const int         mx              = 3;
  const nix::Dims3D global_dims     = {1, 1, 64};
  const nix::Dims3D chunk_dims      = {1, 1, 8};
  const nix::Bool3D has_dim         = {false, false, true};
  const auto        proc_dims       = std::array<int, 3>{2, 2, 2};
  const auto        chunk_grid_dims = make_chunk_grid_dims(global_dims, chunk_dims);
  const json        config          = make_default_config();
  const json        option          = make_default_option();

  REQUIRE(chunk_grid_dims[0] % proc_dims[0] == 0);
  REQUIRE(chunk_grid_dims[1] % proc_dims[1] == 0);
  REQUIRE(chunk_grid_dims[2] % proc_dims[2] == 0);

  // create chunks
  std::vector<std::unique_ptr<PicChunk>> chunkvec;
  initialize_chunkvec(chunkvec, get_mpi_rank(), proc_dims, global_dims, chunk_dims, has_dim, mz, my,
                      mx, config);

  TestPicPoisson poisson(global_dims, 1.0);
  poisson.bind_chunks(chunkvec);
  auto accessor = poisson.get_accessor();

  poisson.set_option(option);
  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  REQUIRE(poisson.solve(accessor) == 0);
  poisson.copy_sol_to_chunk(accessor);

  require_rms_error_below(chunkvec, mz, my, mx, global_dims, tol);
}

TEST_CASE("PicPoisson solves 2D periodic Poisson", "[np=8]")
{
  if (!require_mpi_size(8)) {
    return;
  }

  const float64     tol             = 1.0e-12;
  const int         mz              = 0;
  const int         my              = 2;
  const int         mx              = 3;
  const nix::Dims3D global_dims     = {1, 32, 24};
  const nix::Dims3D chunk_dims      = {1, 8, 12};
  const nix::Bool3D has_dim         = {false, true, true};
  const auto        proc_dims       = std::array<int, 3>{2, 2, 2};
  const auto        chunk_grid_dims = make_chunk_grid_dims(global_dims, chunk_dims);
  const json        config          = make_default_config();
  const json        option          = make_default_option();

  REQUIRE(chunk_grid_dims[0] % proc_dims[0] == 0);
  REQUIRE(chunk_grid_dims[1] % proc_dims[1] == 0);
  REQUIRE(chunk_grid_dims[2] % proc_dims[2] == 0);

  // create chunks
  std::vector<std::unique_ptr<PicChunk>> chunkvec;
  initialize_chunkvec(chunkvec, get_mpi_rank(), proc_dims, global_dims, chunk_dims, has_dim, mz, my,
                      mx, config);

  TestPicPoisson poisson(global_dims, 1.0);
  poisson.bind_chunks(chunkvec);
  auto accessor = poisson.get_accessor();

  poisson.set_option(option);
  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  REQUIRE(poisson.solve(accessor) == 0);
  poisson.copy_sol_to_chunk(accessor);

  require_rms_error_below(chunkvec, mz, my, mx, global_dims, tol);
}

TEST_CASE("PicPoisson solves 3D periodic Poisson", "[np=8]")
{
  if (!require_mpi_size(8)) {
    return;
  }

  const float64     tol             = 1.0e-12;
  const int         rank            = get_mpi_rank();
  const int         mz              = 2;
  const int         my              = 3;
  const int         mx              = 4;
  const nix::Dims3D global_dims     = {32, 32, 32};
  const nix::Dims3D chunk_dims      = {8, 8, 8};
  const nix::Bool3D has_dim         = {true, true, true};
  const auto        proc_dims       = std::array<int, 3>{2, 2, 2};
  const auto        chunk_grid_dims = make_chunk_grid_dims(global_dims, chunk_dims);
  const json        config          = make_default_config();
  const json        option          = make_default_option();

  REQUIRE(chunk_grid_dims[0] % proc_dims[0] == 0);
  REQUIRE(chunk_grid_dims[1] % proc_dims[1] == 0);
  REQUIRE(chunk_grid_dims[2] % proc_dims[2] == 0);

  // create chunks
  std::vector<std::unique_ptr<PicChunk>> chunkvec;
  initialize_chunkvec(chunkvec, rank, proc_dims, global_dims, chunk_dims, has_dim, mz, my, mx,
                      config);

  TestPicPoisson poisson(global_dims, 1.0);
  poisson.bind_chunks(chunkvec);
  auto accessor = poisson.get_accessor();

  poisson.set_option(option);
  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  REQUIRE(poisson.solve(accessor) == 0);
  poisson.copy_sol_to_chunk(accessor);

  require_rms_error_below(chunkvec, mz, my, mx, global_dims, tol);
}

TEST_CASE("PicPoisson scatter", "[np=2]")
{
  if (!require_mpi_size(2)) {
    return;
  }

  const float64     tol             = 1.0e-15;
  const int         rank            = get_mpi_rank();
  const int         mz              = 1;
  const int         my              = 1;
  const int         mx              = 1;
  const nix::Dims3D global_dims     = {16, 16, 16};
  const nix::Dims3D chunk_dims      = {4, 4, 4};
  const nix::Bool3D has_dim         = {true, true, true};
  const auto        proc_dims       = std::array<int, 3>{1, 1, 2};
  const auto        chunk_grid_dims = make_chunk_grid_dims(global_dims, chunk_dims);
  const json        config          = make_default_config();

  REQUIRE(chunk_grid_dims[0] % proc_dims[0] == 0);
  REQUIRE(chunk_grid_dims[1] % proc_dims[1] == 0);
  REQUIRE(chunk_grid_dims[2] % proc_dims[2] == 0);

  // create chunks
  std::vector<std::unique_ptr<PicChunk>> chunkvec;
  initialize_chunkvec(chunkvec, rank, proc_dims, global_dims, chunk_dims, has_dim, mz, my, mx,
                      config);

  TestPicPoisson poisson(global_dims, 1.0);
  poisson.bind_chunks(chunkvec);
  auto accessor = poisson.get_accessor();

  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  poisson.copy_src_to_sol();
  poisson.copy_sol_to_chunk(accessor);

  require_src_equals_sol(chunkvec, tol);
}

TEST_CASE("PicPoisson phi ghost exchange", "[np=2]")
{
  if (!require_mpi_size(2)) {
    return;
  }

  const float64     tol             = 1.0e-12;
  const int         rank            = get_mpi_rank();
  const int         mz              = 0;
  const int         my              = 0;
  const int         mx              = 2;
  const nix::Dims3D global_dims     = {1, 1, 16};
  const nix::Dims3D chunk_dims      = {1, 1, 4};
  const nix::Bool3D has_dim         = {false, false, true};
  const auto        proc_dims       = std::array<int, 3>{1, 1, 2};
  const auto        chunk_grid_dims = make_chunk_grid_dims(global_dims, chunk_dims);
  const json        config          = make_default_config();
  const json        option          = make_default_option();

  REQUIRE(chunk_grid_dims[0] % proc_dims[0] == 0);
  REQUIRE(chunk_grid_dims[1] % proc_dims[1] == 0);
  REQUIRE(chunk_grid_dims[2] % proc_dims[2] == 0);

  // create chunks
  std::vector<std::unique_ptr<PicChunk>> chunkvec;
  initialize_chunkvec(chunkvec, rank, proc_dims, global_dims, chunk_dims, has_dim, mz, my, mx,
                      config);

  TestPicPoisson poisson(global_dims, 1.0);
  poisson.bind_chunks(chunkvec);
  auto accessor = poisson.get_accessor();

  poisson.set_option(option);
  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  REQUIRE(poisson.solve(accessor) == 0);
  poisson.copy_sol_to_chunk(accessor);

  for (auto& chunk : chunkvec) {
    chunk->set_boundary_pack(BoundaryPhi);
    chunk->set_boundary_begin(BoundaryPhi);
  }
  for (auto& chunk : chunkvec) {
    chunk->set_boundary_end(BoundaryPhi);
    chunk->set_boundary_unpack(BoundaryPhi);
  }

  for (const auto& base_chunk : chunkvec) {
    auto* chunk = dynamic_cast<TestPicChunk*>(base_chunk.get());
    REQUIRE(chunk != nullptr);
    auto          data   = chunk->get_internal_data();
    const float64 lx     = static_cast<float64>(global_dims[2]) * data.delx;
    const float64 ly     = static_cast<float64>(global_dims[1]) * data.dely;
    const float64 lz     = static_cast<float64>(global_dims[0]) * data.delz;
    const float64 kx     = static_cast<float64>(mx) * nix::math::pi2 / lx;
    const float64 ky     = static_cast<float64>(my) * nix::math::pi2 / ly;
    const float64 kz     = static_cast<float64>(mz) * nix::math::pi2 / lz;
    const float64 kappax = compute_kappa_component(kx, data.delx);
    const float64 kappay = compute_kappa_component(ky, data.dely);
    const float64 kappaz = compute_kappa_component(kz, data.delz);
    const float64 kappa2_sum =
        kappax * kappax + kappay * kappay + kappaz * kappaz + static_cast<float64>(1.0e-32);

    for (int ix = 0; ix < data.Lbx; ++ix) {
      auto          coords   = chunk->get_coordinates(0, 0, ix);
      const float64 x        = coords[2];
      const float64 y        = coords[1];
      const float64 z        = coords[0];
      const float64 expected = analytic_solution(kz, ky, kx, kappa2_sum, z, y, x);
      const float64 diff     = std::abs(data.phi(0, 0, ix) - expected);
      REQUIRE(diff < tol);
    }
    for (int ix = data.Ubx + 1; ix <= data.Ubx + data.boundary_margin; ++ix) {
      auto          coords   = chunk->get_coordinates(0, 0, ix);
      const float64 x        = coords[2];
      const float64 y        = coords[1];
      const float64 z        = coords[0];
      const float64 expected = analytic_solution(kz, ky, kx, kappa2_sum, z, y, x);
      const float64 diff     = std::abs(data.phi(0, 0, ix) - expected);
      REQUIRE(diff < tol);
    }
  }
}
