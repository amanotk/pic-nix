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

  void update_poisson_efield_from_analytic(int mz, int my, int mx)
  {
    auto              app_data    = PicApplication::get_internal_data();
    const nix::Dims3D global_dims = {app_data.ndims[0], app_data.ndims[1], app_data.ndims[2]};
    for (auto& chunk : app_data.chunkvec) {
      auto* pic_chunk = dynamic_cast<PicChunk*>(chunk.get());
      REQUIRE(pic_chunk != nullptr);
      populate_chunk_source(*pic_chunk, global_dims, mz, my, mx);
    }
    PicApplication::update_poisson_efield();
  }

  void update_poisson_efield_from_particle()
  {
    PicApplication::calculate_moment();
    populate_rho_from_moment();
    PicApplication::update_poisson_efield();
  }

  void push()
  {
    PicApplication::push();
  }

  void require_poisson_error_below(int mz, int my, int mx, float64 tol)
  {
    const float64 rms_err = compute_poisson_error(mz, my, mx);
    REQUIRE(rms_err < tol);
  }

  void require_divergence_error_below(float64 tol)
  {
    const float64 err = compute_divergence_error();
    std::cout << "Divergence error (RMS of div(E)): " << err << std::endl;
    REQUIRE(err < tol);
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
  void exchange_phi_boundaries()
  {
    const auto& chunkvec = this->get_internal_data().chunkvec;
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

  void exchange_emf_boundaries()
  {
    const auto& chunkvec = this->get_internal_data().chunkvec;
    for (auto& chunk_ptr : chunkvec) {
      auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
      REQUIRE(chunk != nullptr);
      chunk->set_boundary_pack(BoundaryEmf);
      chunk->set_boundary_begin(BoundaryEmf);
    }
    for (auto& chunk_ptr : chunkvec) {
      auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
      REQUIRE(chunk != nullptr);
      chunk->set_boundary_end(BoundaryEmf);
      chunk->set_boundary_unpack(BoundaryEmf);
    }
  }

  float64 compute_divergence_error()
  {
    const auto& chunkvec  = this->get_internal_data().chunkvec;
    float64     local_efd = 0.0;
    float64     local_bfd = 0.0;
    int         local_cnt = 0;

    for (auto& chunk_ptr : chunkvec) {
      auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
      REQUIRE(chunk != nullptr);
      auto data = chunk->get_internal_data();

      float64 efd = 0.0;
      float64 bfd = 0.0;
      chunk->get_diverror(efd, bfd);
      local_efd += efd;
      local_bfd += bfd;

      // count interior cells
      auto dims = chunk->get_dims();
      local_cnt += dims[0] * dims[1] * dims[2];
    }

    float64 global_efd = 0.0;
    float64 global_bfd = 0.0;
    int     global_cnt = 0;
    MPI_Allreduce(&local_efd, &global_efd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_bfd, &global_bfd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_cnt, &global_cnt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    float64 rms_efd = std::sqrt(global_efd / static_cast<float64>(global_cnt));
    float64 rms_bfd = std::sqrt(global_bfd / static_cast<float64>(global_cnt));

    // check only div(E) error here
    REQUIRE(global_cnt > 0);
    return rms_efd;
  }

  void compute_efield_from_potential()
  {
    const auto& chunkvec = this->get_internal_data().chunkvec;

    // exchange phi boundaries to ensure proper stencil
    exchange_phi_boundaries();

    for (auto& chunk_ptr : chunkvec) {
      auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
      REQUIRE(chunk != nullptr);
      auto data = chunk->get_internal_data();

      const float64 rdx = 1.0 / data.delx;
      const float64 rdy = 1.0 / data.dely;
      const float64 rdz = 1.0 / data.delz;

      // Calculate E = -grad(phi)
      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
            data.uf(iz, iy, ix, 0) = -(data.phi(iz, iy, ix) - data.phi(iz, iy, ix - 1)) * rdx;
            data.uf(iz, iy, ix, 1) = -(data.phi(iz, iy, ix) - data.phi(iz, iy - 1, ix)) * rdy;
            data.uf(iz, iy, ix, 2) = -(data.phi(iz, iy, ix) - data.phi(iz - 1, iy, ix)) * rdz;
          }
        }
      }
    }

    // exchange E field boundaries
    exchange_emf_boundaries();
  }

  void populate_rho_from_moment()
  {
    auto app_data = this->get_internal_data();
    for (auto& chunk_ptr : app_data.chunkvec) {
      auto* chunk = dynamic_cast<PicChunk*>(chunk_ptr.get());
      REQUIRE(chunk != nullptr);
      auto data = chunk->get_internal_data();

      data.uj.fill(0.0);
      for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
        for (int iy = data.Lby; iy <= data.Uby; ++iy) {
          for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
            float64 rho = 0.0;
            for (int is = 0; is < data.Ns; ++is) {
              const float64 q_over_m = data.up[is]->q / data.up[is]->m;
              rho += data.um(iz, iy, ix, is, 0) * q_over_m;
            }
            data.uj(iz, iy, ix, 0) = rho;
          }
        }
      }
    }
  }

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

  float64 compute_poisson_error(int mz, int my, int mx)
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

        mtp.seed(random_seed + is);
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

#if PICNIX_ENABLE_PETSC
  const char* config_template = R"TOML(
 [application]
   [application.option]
      seed_type = 'fixed'
   [application.poisson_petsc]
    ksp_type = 'cg'
    pc_type = 'gamg'
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
  Bx = 0.05
  By = 0.10
  Bz = 0.15
  Ns = 2
  cc = 1.0
  delt = 0.1
  delh = 1.0

[[parameter.particle]]
  np = 5
  qm = 1.0
  ro = 1.0
  vt = 1.0

[[parameter.particle]]
  np = 5
  qm = -1.0
  ro = 1.0
  vt = 1.0

[[diagnostic]]
  name = 'history'
  begin = 1
  interval = 1000000
)TOML";
#else
  const char* config_template = R"TOML(
 [application]
   [application.option]
     seed_type = 'fixed'
   [application.poisson_basic]
     max_iter = 2000
     tol = 1.0e-12

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
  Bx = 0.05
  By = 0.10
  Bz = 0.15
  Ns = 2
  cc = 1.0
  delt = 0.1
  delh = 1.0

[[parameter.particle]]
  np = 5
  qm = 1.0
  ro = 1.0
  vt = 1.0

[[parameter.particle]]
  np = 5
  qm = -1.0
  ro = 1.0
  vt = 1.0

[[diagnostic]]
  name = 'history'
  begin = 1
  interval = 1000000
)TOML";
#endif

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

CliArgs make_cli_args(const std::filesystem::path&    config_path,
                      const std::vector<std::string>& extra_args = {})
{
  CliArgs cli;
  cli.args = {
      "./test_pic_application",
      "-c",
      config_path.string(),
  };
  cli.args.insert(cli.args.end(), extra_args.begin(), extra_args.end());
  cli.argv = nix::ArgParser::convert_to_clargs(cli.args);
  return cli;
}

void cleanup_checkpoint(const std::filesystem::path& prefix, int rank)
{
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::filesystem::remove(prefix.string() + ".msgpack");
    std::filesystem::remove(prefix.string() + ".status.json");
    std::filesystem::remove(prefix.string() + ".status.json.tmp");
    std::filesystem::remove_all(prefix);
  }
  MPI_Barrier(MPI_COMM_WORLD);
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

TEST_CASE("pic_application_writes_complete_checkpoint_status", "[np=1][np=8]")
{
  int nprocess = get_mpi_size();
  int rank     = get_mpi_rank();

  if (nprocess != 1 && nprocess != 8) {
    SUCCEED("Skipping test: only np=1 or np=8 are supported.");
    return;
  }

  const GridConfig      grid_config = GridConfig{8, 8, 8, 2, 2, 2};
  std::filesystem::path config_path = write_config_for_grid(grid_config, rank);
  std::filesystem::path checkpoint  = config_path.parent_path() / "test_pic_checkpoint";

  cleanup_checkpoint(checkpoint, rank);
  MPI_Barrier(MPI_COMM_WORLD);

  CliArgs         cli       = make_cli_args(config_path, {"-s", checkpoint.string()});
  auto            interface = std::make_shared<TestInterface>();
  TestApplication app(cli.argc(), cli.cargv(), interface);

  REQUIRE(app.main() == 0);

  if (rank == 0) {
    std::ifstream status_file(checkpoint.string() + ".status.json");
    json          status = json::parse(status_file);

    REQUIRE(status["status"] == "complete");
    REQUIRE(status["prefix"] == checkpoint.string());
    REQUIRE(status["nprocess"] == nprocess);
    REQUIRE(status.contains("curstep") == true);
    REQUIRE(status.contains("curtime") == true);
    REQUIRE(status.contains("timestamp") == true);
  }

  cleanup_checkpoint(checkpoint, rank);
  cleanup_config_and_tmpdir(config_path, rank);
}

TEST_CASE("PicApplication solve_poisson analytic periodic", "[np=8]")
{
  if (!require_mpi_size(8)) {
    return;
  }

  const float64 tol  = 1.0e-12;
  const int     rank = get_mpi_rank();
  const int     mz   = 1;
  const int     my   = 2;
  const int     mx   = 3;

  const GridConfig      grid_config = {32, 32, 32, 4, 4, 4};
  std::filesystem::path config_path = write_config_for_grid(grid_config, rank);

  MPI_Barrier(MPI_COMM_WORLD);

  CliArgs         cli       = make_cli_args(config_path);
  auto            interface = std::make_shared<TestInterface>();
  TestApplication app(cli.argc(), cli.cargv(), interface);

  app.initialize_for_test(cli.argc(), cli.cargv());
  app.update_poisson_efield_from_analytic(mz, my, mx);
  app.require_poisson_error_below(mz, my, mx, tol);
  app.finalize_for_test();

  cleanup_config_and_tmpdir(config_path, rank);
}

TEST_CASE("PicApplication preserves Gauss's law", "[np=8]")
{
  if (!require_mpi_size(8)) {
    return;
  }

  const float64         tol         = 1.0e-12;
  const int             rank        = get_mpi_rank();
  const GridConfig      grid_config = GridConfig{32, 32, 32, 4, 4, 4};
  std::filesystem::path config_path = write_config_for_grid(grid_config, rank);

  MPI_Barrier(MPI_COMM_WORLD);

  CliArgs         cli       = make_cli_args(config_path);
  auto            interface = std::make_shared<TestInterface>();
  TestApplication app(cli.argc(), cli.cargv(), interface);

  app.initialize_for_test(cli.argc(), cli.cargv());

  // check Gauss's law before particle push
  app.update_poisson_efield_from_particle();
  app.require_divergence_error_below(tol);

  // check Gauss's law after particle push
  app.push();
  app.require_divergence_error_below(tol);

  app.finalize_for_test();

  cleanup_config_and_tmpdir(config_path, rank);
}
