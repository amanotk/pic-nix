// -*- C++ -*-

#include "pic_poisson.hpp"

#include <catch2/catch_test_macros.hpp>
#include <elliptic/chunk_accessor.hpp>
#include <mpi.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include <petscvec.h>

#include "test_parallel.hpp"

namespace
{
json make_config()
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

std::unique_ptr<PicChunk> make_chunk(const nix::Dims3D& dims, const nix::Bool3D& has_dim,
                                     const nix::Dims3D& global_dims, const nix::Dims3D& offset,
                                     int id)
{
  auto chunk = std::make_unique<PicChunk>(dims, has_dim, id);
  int  o[3]  = {offset[0], offset[1], offset[2]};
  int  g[3]  = {global_dims[0], global_dims[1], global_dims[2]};

  chunk->set_global_context(o, g);
  chunk->set_coordinate(1.0, 1.0, 1.0);
  auto config = make_config();
  chunk->setup(config);
  chunk->allocate();

  auto data = chunk->get_internal_data();
  data.cc   = 1.0;

  return chunk;
}

int flatten_global(int gz, int gy, int gx, const nix::Dims3D& dims)
{
  return elliptic::ChunkAccessor::flatten_index(gz, gy, gx, dims);
}

void fill_rho_sequence(PicChunk& chunk, const nix::Dims3D& global_dims)
{
  auto offset = chunk.get_offset();
  auto data   = chunk.get_internal_data();

  for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        const int gz           = offset[0] + (iz - data.Lbz);
        const int gy           = offset[1] + (iy - data.Lby);
        const int gx           = offset[2] + (ix - data.Lbx);
        data.uj(iz, iy, ix, 0) = static_cast<float64>(flatten_global(gz, gy, gx, global_dims) + 1);
      }
    }
  }
}

std::vector<float64> make_reference_potential(const nix::Dims3D& dims)
{
  const int            nglobal = dims[0] * dims[1] * dims[2];
  std::vector<float64> phi_ref(nglobal, 0.0);

  for (int gz = 0; gz < dims[0]; ++gz) {
    for (int gy = 0; gy < dims[1]; ++gy) {
      for (int gx = 0; gx < dims[2]; ++gx) {
        const int     idx = flatten_global(gz, gy, gx, dims);
        const float64 argx =
            static_cast<float64>(gx) / static_cast<float64>(dims[2]) * nix::math::pi2;
        const float64 argy =
            static_cast<float64>(gy) / static_cast<float64>(dims[1]) * nix::math::pi2;
        const float64 argz =
            static_cast<float64>(gz) / static_cast<float64>(dims[0]) * nix::math::pi2;
        phi_ref[idx] = std::cos(argx) + 0.25 * std::cos(argy) + 0.125 * std::cos(argz);
      }
    }
  }

  return phi_ref;
}

std::vector<float64> make_rhs_from_potential(const std::vector<float64>& phi_ref,
                                             const nix::Dims3D&          dims)
{
  const int nglobal = dims[0] * dims[1] * dims[2];
  auto      index   = [dims](int gz, int gy, int gx) {
    const int z = (gz + dims[0]) % dims[0];
    const int y = (gy + dims[1]) % dims[1];
    const int x = (gx + dims[2]) % dims[2];
    return flatten_global(z, y, x, dims);
  };

  std::vector<float64> rhs(nglobal, 0.0);
  for (int gz = 0; gz < dims[0]; ++gz) {
    for (int gy = 0; gy < dims[1]; ++gy) {
      for (int gx = 0; gx < dims[2]; ++gx) {
        const int     idx = flatten_global(gz, gy, gx, dims);
        const float64 lapx =
            phi_ref[index(gz, gy, gx + 1)] + phi_ref[index(gz, gy, gx - 1)] - 2.0 * phi_ref[idx];
        const float64 lapy =
            phi_ref[index(gz, gy + 1, gx)] + phi_ref[index(gz, gy - 1, gx)] - 2.0 * phi_ref[idx];
        const float64 lapz =
            phi_ref[index(gz + 1, gy, gx)] + phi_ref[index(gz - 1, gy, gx)] - 2.0 * phi_ref[idx];

        rhs[idx] = -(lapx + lapy + lapz);
      }
    }
  }

  return rhs;
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

  void copy_rhs_to_solution()
  {
    VecCopy(vector_src_g, vector_sol_g);
  }
};

void scatter_rhs_to_chunks(const std::vector<std::unique_ptr<PicChunk>>& storage,
                           const std::vector<float64>& rhs, const nix::Dims3D& dims)
{
  for (const auto& chunk_ptr : storage) {
    auto offset = chunk_ptr->get_offset();
    auto data   = chunk_ptr->get_internal_data();

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          const int gz           = offset[0] + (iz - data.Lbz);
          const int gy           = offset[1] + (iy - data.Lby);
          const int gx           = offset[2] + (ix - data.Lbx);
          const int idx          = elliptic::ChunkAccessor::flatten_index(gz, gy, gx, dims);
          data.uj(iz, iy, ix, 0) = rhs[idx];
        }
      }
    }
  }
}

std::vector<float64> gather_phi(const std::vector<std::unique_ptr<PicChunk>>& storage,
                                const nix::Dims3D&                            dims)
{
  const int            nglobal = dims[0] * dims[1] * dims[2];
  std::vector<float64> local(nglobal, 0.0);

  for (const auto& chunk_ptr : storage) {
    auto offset = chunk_ptr->get_offset();
    auto data   = chunk_ptr->get_internal_data();

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          const int gz  = offset[0] + (iz - data.Lbz);
          const int gy  = offset[1] + (iy - data.Lby);
          const int gx  = offset[2] + (ix - data.Lbx);
          const int idx = elliptic::ChunkAccessor::flatten_index(gz, gy, gx, dims);
          local[idx]    = data.phi(iz, iy, ix);
        }
      }
    }
  }

  std::vector<float64> global(local.size(), 0.0);
  MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()), MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  return global;
}
} // namespace

TEST_CASE("PicPoisson gather/scatter copies rho to phi", "[np=2]")
{
  if (!require_mpi_size(2)) {
    return;
  }

  const nix::Dims3D                      global_dims = {2, 2, 4};
  const nix::Dims3D                      chunk_dims  = {2, 2, 2};
  const nix::Bool3D                      has_dim     = {true, true, true};
  std::vector<std::unique_ptr<PicChunk>> storage;

  const int         rank   = get_mpi_rank();
  const nix::Dims3D offset = {0, 0, chunk_dims[2] * rank};
  storage.push_back(make_chunk(chunk_dims, has_dim, global_dims, offset, rank));

  fill_rho_sequence(*storage[0], global_dims);

  TestPicPoisson               poisson(global_dims, 1.0);
  PicPoisson::PicChunkAccessor accessor(storage);
  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  poisson.copy_rhs_to_solution();
  poisson.copy_sol_to_chunk(accessor);

  auto data = storage[0]->get_internal_data();
  for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        REQUIRE(data.phi(iz, iy, ix) == data.uj(iz, iy, ix, 0));
      }
    }
  }
}

TEST_CASE("PicPoisson solves periodic Poisson", "[np=2]")
{
  if (!require_mpi_size(2)) {
    return;
  }

  const nix::Dims3D                      global_dims = {16, 16, 16};
  const nix::Dims3D                      chunk_dims  = {16, 16, 8};
  const nix::Bool3D                      has_dim     = {true, true, true};
  std::vector<std::unique_ptr<PicChunk>> storage;

  const int         rank   = get_mpi_rank();
  const nix::Dims3D offset = {0, 0, chunk_dims[2] * rank};
  storage.push_back(make_chunk(chunk_dims, has_dim, global_dims, offset, rank));

  auto phi_ref = make_reference_potential(global_dims);
  auto rhs     = make_rhs_from_potential(phi_ref, global_dims);
  scatter_rhs_to_chunks(storage, rhs, global_dims);

  TestPicPoisson poisson(global_dims, 1.0);
  json           opts;
  opts["petsc"] = {{"ksp_type", "cg"}, {"pc_type", "none"}, {"ksp_rtol", 1.0e-12}};
  poisson.set_option(opts);

  PicPoisson::PicChunkAccessor accessor(storage);
  poisson.update_mapping(accessor);
  poisson.copy_chunk_to_src(accessor);
  REQUIRE(poisson.solve(accessor) == 0);
  poisson.copy_sol_to_chunk(accessor);

  auto phi_sol = gather_phi(storage, global_dims);

  const int     nglobal = static_cast<int>(phi_ref.size());
  const float64 mean_ref =
      std::accumulate(phi_ref.begin(), phi_ref.end(), 0.0) / static_cast<float64>(nglobal);
  const float64 mean_sol =
      std::accumulate(phi_sol.begin(), phi_sol.end(), 0.0) / static_cast<float64>(nglobal);

  float64 diff2 = 0.0;
  float64 ref2  = 0.0;
  for (int i = 0; i < nglobal; ++i) {
    const float64 sol = phi_sol[i] - mean_sol;
    const float64 ref = phi_ref[i] - mean_ref;
    const float64 d   = sol - ref;
    diff2 += d * d;
    ref2 += ref * ref;
  }

  REQUIRE(diff2 > 0.0);
  REQUIRE(ref2 > 0.0);
  const float64 rel_err = std::sqrt(diff2) / (std::sqrt(ref2) + 1.0e-14);
  REQUIRE(rel_err < 1.0e-10);
}
