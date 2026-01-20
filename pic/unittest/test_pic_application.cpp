// -*- C++ -*-

#include "argparser.hpp"
#include "diag.hpp"
#include "mpistream.hpp"
#include "nix/random.hpp"
#include "pic_application.hpp"
#include "pic_chunk.hpp"
#include "pic_diag.hpp"
#include "test_parallel.hpp"

#include <catch2/catch_test_macros.hpp>

#include <mpi.h>

#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

class TestApplication : public PicApplication
{
public:
  using PicApplication::PicApplication;

protected:
  void initialize(int argc, char** argv) override
  {
    curstep = 0;
    curtime = 0.0;

    argparser = create_argparser();
    argparser->parse_check(argc, argv);

    cfgparser = create_cfgparser();
    cfgparser->parse_file(argparser->get_config());

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    REQUIRE(mpi_initialized != 0);

    nthread = nix::get_max_threads();

    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
    MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);

    wclock = nix::wall_clock();
    MPI_Bcast(&wclock, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    initialize_base_directory();

    json        config            = cfgparser->get_application();
    std::string path              = "";
    int         max_files_per_dir = 1000;

    if (config.contains("mpistream") == false) {
      MpiStream::initialize(path, max_files_per_dir);
    } else if (config["mpistream"].is_object() == true) {
      namespace fs = std::filesystem;
      config            = config["mpistream"];
      path              = fs::path(get_basedir()) / config.value("path", path);
      max_files_per_dir = config.value("max_files_per_dir", max_files_per_dir);
      MpiStream::initialize(path, max_files_per_dir);
    } else if (config["mpistream"] == false) {
    } else {
    }

    statehandler = create_statehandler();
    balancer     = create_balancer();
    logger       = create_logger();
    chunkmap     = create_chunkmap();

    initialize_debugprinting();
    initialize_dimensions();
    initialize_domain();
    initialize_diagnostic();

    Ns = cfgparser->get_parameter()["Ns"];
    for (int mode = 0; mode < NumBoundaryMode; mode++) {
      for (int iz = 0; iz < 3; iz++) {
        for (int iy = 0; iy < 3; iy++) {
          for (int ix = 0; ix < 3; ix++) {
            MPI_Comm_dup(MPI_COMM_WORLD, &mpicommvec(mode, iz, iy, ix));
          }
        }
      }
    }
  }

  void finalize() override
  {
    for (int mode = 0; mode < NumBoundaryMode; mode++) {
      for (int iz = 0; iz < 3; iz++) {
        for (int iy = 0; iy < 3; iy++) {
          for (int ix = 0; ix < 3; ix++) {
            MPI_Comm_free(&mpicommvec(mode, iz, iy, ix));
          }
        }
      }
    }

    logger->flush();

    if (argparser->get_save() != "") {
      statehandler->save(get_interface(), argparser->get_save());
    }

    MpiStream::finalize();
    nix::Diag::finalize();
  }

  float64 get_available_etime() override
  {
    if (curstep >= 1) {
      return -1.0;
    }
    return std::numeric_limits<float64>::max();
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

      // allocate MPI buffer for field
      this->set_mpi_buffer(mpibufvec[BoundaryEmf], 0, 0, sizeof(float64) * 6);
      this->set_mpi_buffer(mpibufvec[BoundaryCur], 0, 0, sizeof(float64) * 4);
      this->set_mpi_buffer(mpibufvec[BoundaryMom], 0, 0, sizeof(float64) * Ns * 14);

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

        mtp.seed(random_seed); // for charge neutrality
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

std::filesystem::path write_config_for_size(int nprocess, int rank)
{
  const char* tmpdir_env = std::getenv("PICNIX_TMPDIR");
  std::filesystem::path base = tmpdir_env != nullptr ? tmpdir_env : ".";

  std::filesystem::path config_path = base / "test_pic_application.toml";

  const char* config_template = R"TOML(
[application]
  [application.option]
    seed_type = 'fixed'

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
  Ns = 1
  cc = 1.0
  delt = 0.1
  delh = 1.0

[[parameter.particle]]
  np = 1
  qm = -1.0
  ro = 1.0
  vt = 0.0

[[diagnostic]]
  name = 'history'
  begin = 1
  interval = 1000000
)TOML";

  int nx = 4;
  int ny = 4;
  int nz = 4;
  int cx = 1;
  int cy = 1;
  int cz = 1;
  if (nprocess == 8) {
    nx = 8;
    ny = 8;
    nz = 8;
    cx = 2;
    cy = 2;
    cz = 2;
  }

  if (rank == 0) {
    std::string config = config_template;
    config = replace_all(config, "@NX@", std::to_string(nx));
    config = replace_all(config, "@NY@", std::to_string(ny));
    config = replace_all(config, "@NZ@", std::to_string(nz));
    config = replace_all(config, "@CX@", std::to_string(cx));
    config = replace_all(config, "@CY@", std::to_string(cy));
    config = replace_all(config, "@CZ@", std::to_string(cz));

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

TEST_CASE("pic_application_interface_smoke", "[np=1][np=8]")
{
  int nprocess = get_mpi_size();
  int rank     = get_mpi_rank();

  if (nprocess != 1 && nprocess != 8) {
    SUCCEED("Skipping test: only np=1 or np=8 are supported.");
    return;
  }

  std::filesystem::path config_path = write_config_for_size(nprocess, rank);
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<std::string> args = {
    "./test_pic_application",
    "-c",
    config_path.string(),
  };
  std::vector<const char*> argv = nix::ArgParser::convert_to_clargs(args);

  int             argc      = static_cast<int>(argv.size());
  char**          cargv     = const_cast<char**>(&argv[0]);
  auto            interface = std::make_shared<TestInterface>();
  TestApplication app(argc, cargv, interface);

  REQUIRE(app.main() == 0);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::filesystem::remove(config_path);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  cleanup_tmpdir(rank);
}
