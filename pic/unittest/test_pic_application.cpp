// -*- C++ -*-

#include "argparser.hpp"
#include "nix/random.hpp"
#include "pic_application.hpp"
#include "pic_chunk.hpp"
#include "pic_diag.hpp"
#include "test_parallel.hpp"

#include <catch2/catch_test_macros.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

float64 compute_kappa_component(float64 k, float64 h);
float64 analytic_source(float64 kz, float64 ky, float64 kx, float64 z, float64 y, float64 x);
float64 analytic_solution(float64 kz, float64 ky, float64 kx, float64 kappa2_sum, float64 z,
                          float64 y, float64 x);

struct GridConfig {
  int nx;
  int ny;
  int nz;
  int cx;
  int cy;
  int cz;
};

struct CliArgs {
  std::vector<std::string> args;
  std::vector<const char*> argv;

  int argc() const
  {
    return static_cast<int>(argv.size());
  }

  char** cargv()
  {
    return const_cast<char**>(argv.data());
  }
};

class TestApplication : public PicApplication
{
public:
  using PicApplication::PicApplication;

  void initialize_for_test(int argc, char** argv)
  {
    PicApplication::initialize(argc, argv);
    PicApplication::setup_chunks();
  }

  void solve_poisson_for_test(int mz, int my, int mx)
  {
    auto              app_data    = PicApplication::get_internal_data();
    const nix::Dims3D global_dims = {app_data.ndims[0], app_data.ndims[1], app_data.ndims[2]};
    for (auto& chunk : app_data.chunkvec) {
      auto* pic_chunk = dynamic_cast<PicChunk*>(chunk.get());
      REQUIRE(pic_chunk != nullptr);
      populate_chunk_source(*pic_chunk, global_dims, mz, my, mx);
    }
    PicApplication::solve_poisson();
  }

  void push_particles_for_test()
  {
    PicApplication::push();
  }

  void solve_poisson_after_push()
  {
    PicApplication::solve_poisson();
  }

  void require_rms_error_below(int mz, int my, int mx, float64 tol)
  {
    const float64 rms_err = compute_global_rms_error(mz, my, mx);
    REQUIRE(rms_err < tol);
  }

  void finalize_for_test()
  {
    PicApplication::finalize();
  }

protected:
  float64 get_available_etime() override
  {
    if (curstep >= 1) {
      return -1.0;
    }
    return std::numeric_limits<float64>::max();
  }

private:
  void populate_chunk_source(PicChunk& chunk, const nix::Dims3D& global_dims, int mz, int my,
                             int mx)
  {
    auto data   = chunk.get_internal_data();
    auto offset = chunk.get_offset();

    const float64 lx = static_cast<float64>(global_dims[2]) * data.delx;
    const float64 ly = static_cast<float64>(global_dims[1]) * data.dely;
    const float64 lz = static_cast<float64>(global_dims[0]) * data.delz;
    const float64 kz = static_cast<float64>(mz) * nix::math::pi2 / lz;
    const float64 ky = static_cast<float64>(my) * nix::math::pi2 / ly;
    const float64 kx = static_cast<float64>(mx) * nix::math::pi2 / lx;

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          const int     gz       = offset[0] + (iz - data.Lbz);
          const int     gy       = offset[1] + (iy - data.Lby);
          const int     gx       = offset[2] + (ix - data.Lbx);
          const float64 z        = static_cast<float64>(gz) * data.delz;
          const float64 y        = static_cast<float64>(gy) * data.dely;
          const float64 x        = static_cast<float64>(gx) * data.delx;
          data.uj(iz, iy, ix, 0) = analytic_source(kz, ky, kx, z, y, x);
          data.phi(iz, iy, ix)   = 0.0;
        }
      }
    }
  }

  float64 compute_global_rms_error(int mz, int my, int mx)
  {
    auto              app_data    = PicApplication::get_internal_data();
    const nix::Dims3D global_dims = {app_data.ndims[0], app_data.ndims[1], app_data.ndims[2]};

    float64 local_sum = 0.0;
    int     local_cnt = 0;

    for (auto& base_chunk : app_data.chunkvec) {
      auto* chunk = dynamic_cast<PicChunk*>(base_chunk.get());
      REQUIRE(chunk != nullptr);
      auto data   = chunk->get_internal_data();
      auto offset = chunk->get_offset();

      const float64 lx     = static_cast<float64>(global_dims[2]) * data.delx;
      const float64 ly     = static_cast<float64>(global_dims[1]) * data.dely;
      const float64 lz     = static_cast<float64>(global_dims[0]) * data.delz;
      const float64 kz     = static_cast<float64>(mz) * nix::math::pi2 / lz;
      const float64 ky     = static_cast<float64>(my) * nix::math::pi2 / ly;
      const float64 kx     = static_cast<float64>(mx) * nix::math::pi2 / lx;
      const float64 kappax = compute_kappa_component(kx, data.delx);
      const float64 kappay = compute_kappa_component(ky, data.dely);
      const float64 kappaz = compute_kappa_component(kz, data.delz);
      const float64 kappa2_sum =
          kappax * kappax + kappay * kappay + kappaz * kappaz + static_cast<float64>(1.0e-32);

      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
            const int     gz       = offset[0] + (iz - data.Lbz);
            const int     gy       = offset[1] + (iy - data.Lby);
            const int     gx       = offset[2] + (ix - data.Lbx);
            const float64 z        = static_cast<float64>(gz) * data.delz;
            const float64 y        = static_cast<float64>(gy) * data.dely;
            const float64 x        = static_cast<float64>(gx) * data.delx;
            const float64 expected = analytic_solution(kz, ky, kx, kappa2_sum, z, y, x);
            const float64 diff     = data.phi(iz, iy, ix) - expected;
            local_sum += diff * diff;
            ++local_cnt;
          }
        }
      }
    }

    float64 global_sum = 0.0;
    int     global_cnt = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_cnt, &global_cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return std::sqrt(global_sum / static_cast<float64>(global_cnt));
  }
};

namespace
{
void exchange_phi_boundaries(TestApplication& app)
{
  const auto& chunkvec = app.get_internal_data().chunkvec;
  for (auto& chunk_ptr : chunkvec) {
    auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
    REQUIRE(chunk != nullptr);
    chunk->set_boundary_pack(BoundaryPhi);
    chunk->set_boundary_begin(BoundaryPhi);
  }
  for (auto& chunk_ptr : chunkvec) {
    auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
    REQUIRE(chunk != nullptr);
    chunk->set_boundary_end(BoundaryPhi);
    chunk->set_boundary_unpack(BoundaryPhi);
  }
}

template <typename ChunkVec>
float64 compute_global_charge_mean(const ChunkVec& chunkvec)
{
  float64 local_sum   = 0.0;
  int     local_count = 0;
  for (auto& chunk_ptr : chunkvec) {
    auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
    REQUIRE(chunk != nullptr);
    auto data = chunk->get_internal_data();
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          local_sum += data.uj(iz, iy, ix, 0);
          ++local_count;
        }
      }
    }
  }

  float64 global_sum   = 0.0;
  int     global_count = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  return global_count == 0 ? 0.0 : global_sum / static_cast<float64>(global_count);
}

template <typename ChunkVec>
float64 compute_divergence_rms(const ChunkVec& chunkvec, float64 rho_mean)
{
  float64 local_error = 0.0;
  int     local_count = 0;

  for (auto& chunk_ptr : chunkvec) {
    auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
    REQUIRE(chunk != nullptr);
    auto          data    = chunk->get_internal_data();
    const float64 inv_dx2 = 1.0 / (data.delx * data.delx);
    const float64 inv_dy2 = 1.0 / (data.dely * data.dely);
    const float64 inv_dz2 = 1.0 / (data.delz * data.delz);

    for (int iz = data.Lbz + 1; iz <= data.Ubz - 1; ++iz) {
      for (int iy = data.Lby + 1; iy <= data.Uby - 1; ++iy) {
        for (int ix = data.Lbx + 1; ix <= data.Ubx - 1; ++ix) {
          const float64 rho    = data.uj(iz, iy, ix, 0) - rho_mean;
          const float64 center = data.phi(iz, iy, ix);
          const float64 lap_x =
              (data.phi(iz, iy, ix + 1) - 2.0 * center + data.phi(iz, iy, ix - 1)) * inv_dx2;
          const float64 lap_y =
              (data.phi(iz, iy + 1, ix) - 2.0 * center + data.phi(iz, iy - 1, ix)) * inv_dy2;
          const float64 lap_z =
              (data.phi(iz + 1, iy, ix) - 2.0 * center + data.phi(iz - 1, iy, ix)) * inv_dz2;
          const float64 div_e = -(lap_x + lap_y + lap_z);
          const float64 diff  = div_e - rho;
          local_error += diff * diff;
          ++local_count;
        }
      }
    }
  }

  float64 global_error = 0.0;
  int     global_count = 0;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  REQUIRE(global_count > 0);
  return std::sqrt(global_error / static_cast<float64>(global_count));
}
} // namespace

class TestChunk : public PicChunk
{
public:
  using PicChunk::PicChunk;

  void setup(json& config) override
  {
    PicChunk::setup(config);

    float64 delt;
    float64 delh;

    Ns   = config["Ns"].get<int>();
    cc   = config["cc"].get<float64>();
    delt = config["delt"].get<float64>();
    delh = config["delh"].get<float64>();

    // set grid size and coordinate
    set_coordinate(delh, delh, delh);

    // initialize field
    {
      float64 Ex = config["Ex"].get<float64>();
      float64 Ey = config["Ey"].get<float64>();
      float64 Ez = config["Ez"].get<float64>();
      float64 Bx = config["Bx"].get<float64>();
      float64 By = config["By"].get<float64>();
      float64 Bz = config["Bz"].get<float64>();

      // memory allocation
      allocate();

      for (int iz = Lbz; iz <= Ubz; iz++) {
        for (int iy = Lby; iy <= Uby; iy++) {
          for (int ix = Lbx; ix <= Ubx; ix++) {
            uf(iz, iy, ix, 0) = Ex;
            uf(iz, iy, ix, 1) = Ey;
            uf(iz, iy, ix, 2) = Ez;
            uf(iz, iy, ix, 3) = Bx;
            uf(iz, iy, ix, 4) = By;
            uf(iz, iy, ix, 5) = Bz;
          }
        }
      }

      this->allocate_mpi_buffers();

      // setup for Friedman filter
      this->init_friedman();
    }

    // initialize particles
    {
      float64           target      = 1 + this->get_buffer_ratio();
      int               random_seed = option["random_seed"].get<int>();
      std::mt19937_64   mtp(random_seed);
      std::mt19937_64   mtv(random_seed);
      nix::rand_uniform uniform(0.0, 1.0);
      nix::rand_normal  normal(0.0, 1.0);

      json particle = config["particle"];

      up.resize(Ns);
      for (int is = 0; is < Ns; is++) {
        int     np = particle[is]["np"].get<int>();
        int     mp = np * dims[0] * dims[1] * dims[2];
        int64   id = mp;
        float64 ro = particle[is]["ro"].get<float64>();
        float64 qm = particle[is]["qm"].get<float64>();
        float64 vt = particle[is]["vt"].get<float64>();

        id *= this->myid;

        up[is]     = std::make_shared<ParticleType>(mp * target, *this);
        up[is]->m  = ro / np;
        up[is]->q  = qm * up[is]->m;
        up[is]->Np = mp;

        mtp.seed(random_seed + is * 2 + 1);
        mtv.seed(random_seed + is * 3 + 2);
        for (int ip = 0; ip < up[is]->Np; ip++) {
          float64* ptcl = &up[is]->xu(ip, 0);
          int64*   id64 = reinterpret_cast<int64*>(ptcl);

          ptcl[0] = uniform(mtp) * xlim[2] + xlim[0];
          ptcl[1] = uniform(mtp) * ylim[2] + ylim[0];
          ptcl[2] = uniform(mtp) * zlim[2] + zlim[0];
          ptcl[3] = normal(mtv) * vt;
          ptcl[4] = normal(mtv) * vt;
          ptcl[5] = normal(mtv) * vt;
          id64[6] = id + ip;
        }
      }

      // initial sort
      this->sort_particle(up);
    }
  }
};

class TestInterface : public PicApplicationInterface
{
public:
  PtrChunk create_chunk(nix::Dims3D dims, nix::Bool3D has_dim, int id) override
  {
    return std::make_unique<TestChunk>(dims, has_dim, id);
  }
};

std::string replace_all(std::string text, const std::string& needle, const std::string& value)
{
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.replace(pos, needle.size(), value);
    pos += value.size();
  }
  return text;
}

std::filesystem::path write_config_for_grid(const GridConfig& cfg, int rank)
{
  const char*           tmpdir_env = std::getenv("PICNIX_TMPDIR");
  std::filesystem::path base       = tmpdir_env != nullptr ? tmpdir_env : ".";
  std::error_code       ec;
  std::filesystem::create_directories(base, ec);

  std::filesystem::path config_path = base / "test_pic_application.toml";

  const char* config_template = R"TOML(
[application]
  [application.option]
    seed_type = 'fixed'
    random_seed = 0
  [application.petsc]
    ksp_type = 'cg'
    pc_type = 'jacobi'
    ksp_rtol = 1.0e-14

[parameter]
  Nx = @NX@
  Ny = @NY@
  Nz = @NZ@
  Cx = @CX@
  Cy = @CY@
  Cz = @CZ@
  Ex = 0.0
  Ey = 0.0
  Ez = 0.0
  Bx = 0.0
  By = 0.0
  Bz = 0.0
  Ns = 2
  cc = 1.0
  delt = 0.1
  delh = 1.0

[[parameter.particle]]
  np = 1
  qm = -1.0
  ro = 1.0
  vt = 0.0

[[parameter.particle]]
  np = 1
  qm = 1.0
  ro = 1.0
  vt = 0.0

[[diagnostic]]
  name = 'history'
  begin = 1
  interval = 1000000
)TOML";

  if (rank == 0) {
    std::string config = config_template;
    config             = replace_all(config, "@NX@", std::to_string(cfg.nx));
    config             = replace_all(config, "@NY@", std::to_string(cfg.ny));
    config             = replace_all(config, "@NZ@", std::to_string(cfg.nz));
    config             = replace_all(config, "@CX@", std::to_string(cfg.cx));
    config             = replace_all(config, "@CY@", std::to_string(cfg.cy));
    config             = replace_all(config, "@CZ@", std::to_string(cfg.cz));

    std::ofstream ofs(config_path);
    ofs << config;
  }

  return config_path;
}

void cleanup_tmpdir(int rank)
{
  const char* tmpdir_env = std::getenv("PICNIX_TMPDIR");
  if (tmpdir_env == nullptr) {
    return;
  }

  if (rank == 0) {
    std::error_code ec;
    std::filesystem::remove_all(tmpdir_env, ec);
  }
}

void cleanup_config_and_tmpdir(const std::filesystem::path& config_path, int rank)
{
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::filesystem::remove(config_path);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  cleanup_tmpdir(rank);
}

CliArgs make_cli_args(const std::filesystem::path& config_path)
{
  CliArgs cli;
  cli.args = {
      "./test_pic_application",
      "-c",
      config_path.string(),
  };
  cli.argv = nix::ArgParser::convert_to_clargs(cli.args);
  return cli;
}

float64 compute_kappa_component(float64 k, float64 h)
{
  return (k == 0.0) ? 0.0 : std::sin(0.5 * k * h) / (0.5 * h);
}

float64 analytic_source(float64 kz, float64 ky, float64 kx, float64 z, float64 y, float64 x)
{
  return std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z);
}

float64 analytic_solution(float64 kz, float64 ky, float64 kx, float64 kappa2_sum, float64 z,
                          float64 y, float64 x)
{
  return analytic_source(kz, ky, kx, z, y, x) / kappa2_sum;
}

TEST_CASE("pic_application_interface_smoke", "[np=1][np=8]")
{
  int nprocess = get_mpi_size();
  int rank     = get_mpi_rank();

  if (nprocess != 1 && nprocess != 8) {
    SUCCEED("Skipping test: only np=1 or np=8 are supported.");
    return;
  }

  const GridConfig      grid_config = GridConfig{8, 8, 8, 2, 2, 2};
  std::filesystem::path config_path = write_config_for_grid(grid_config, rank);

  MPI_Barrier(MPI_COMM_WORLD);

  CliArgs         cli       = make_cli_args(config_path);
  auto            interface = std::make_shared<TestInterface>();
  TestApplication app(cli.argc(), cli.cargv(), interface);

  REQUIRE(app.main() == 0);

  cleanup_config_and_tmpdir(config_path, rank);
}

TEST_CASE("PicApplication solve_poisson analytic periodic", "[np=8]")
{
  if (!require_mpi_size(8)) {
    return;
  }

  const int     rank = get_mpi_rank();
  const int     mz   = 1;
  const int     my   = 2;
  const int     mx   = 3;
  const float64 tol  = 1.0e-12;

  const GridConfig      grid_config = {32, 32, 32, 4, 4, 4};
  std::filesystem::path config_path = write_config_for_grid(grid_config, rank);

  MPI_Barrier(MPI_COMM_WORLD);

  CliArgs         cli       = make_cli_args(config_path);
  auto            interface = std::make_shared<TestInterface>();
  TestApplication app(cli.argc(), cli.cargv(), interface);

  app.initialize_for_test(cli.argc(), cli.cargv());
  app.solve_poisson_for_test(mz, my, mx);
  app.require_rms_error_below(mz, my, mx, tol);
  app.finalize_for_test();

  cleanup_config_and_tmpdir(config_path, rank);
}

TEST_CASE("PicApplication preserves Gauss law after particle push", "[np=8]")
{
  if (!require_mpi_size(8)) {
    return;
  }

  const int             rank        = get_mpi_rank();
  const GridConfig      grid_config = GridConfig{32, 32, 32, 4, 4, 4};
  std::filesystem::path config_path = write_config_for_grid(grid_config, rank);

  MPI_Barrier(MPI_COMM_WORLD);

  CliArgs         cli       = make_cli_args(config_path);
  auto            interface = std::make_shared<TestInterface>();
  TestApplication app(cli.argc(), cli.cargv(), interface);

  app.initialize_for_test(cli.argc(), cli.cargv());
  app.push_particles_for_test();
  app.solve_poisson_after_push();

  exchange_phi_boundaries(app);
  const auto&   chunkvec = app.get_internal_data().chunkvec;
  const float64 rho_mean = compute_global_charge_mean(chunkvec);
  const float64 rms_diff = compute_divergence_rms(chunkvec, rho_mean);

  const float64 tol = 1.0e-7;
  REQUIRE(rms_diff < tol);

  app.finalize_for_test();
  cleanup_config_and_tmpdir(config_path, rank);
}
