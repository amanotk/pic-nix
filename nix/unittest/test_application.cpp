// -*- C++ -*-

#include "application.hpp"
#include "argparser.hpp"
#include "chunk.hpp"
#include "chunkmap.hpp"
#include "diag.hpp"
#include "diag/io_handler.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

using namespace nix;

const std::string config_filename = "config.json";
const std::string config_content  = R"(
{
  "application": {
    "basedir": ".",
    "log": {
      "prefix": "log",
      "path": ".",
      "interval": 100
    },
    "rebalance": {
      "loglevel": 1,
      "interval": 100
    }
  },
  "diagnostic": [
    {
      "name": "foo",
      "prefix": "foo",
      "path": ".",
      "interval": 100
    },
    {
      "name": "bar",
      "prefix": "bar",
      "path": ".",
      "interval": 100
    }
  ],
  "parameter": {
    "Nx": 16,
    "Ny": 16,
    "Nz": 16,
    "Cx": 2,
    "Cy": 2,
    "Cz": 2,
    "delt": 1.0,
    "delh": 1.0
  }
}
)";

class TestApplication : public Application
{
public:
  using Interface    = Application::Interface;
  using PtrInterface = Application::PtrInterface;

  TestApplication(int argc, char** argv, PtrInterface interface)
      : Application(argc, argv, interface)
  {
    std::ofstream ofs(config_filename);
    ofs << config_content;
  }

  ~TestApplication()
  {
    std::filesystem::remove(config_filename);
  }

  void set_test_configuration(json configuration)
  {
    cfgparser = create_cfgparser();
    cfgparser->overwrite(configuration);
  }

  void write_test_configuration(const json& configuration)
  {
    std::ofstream ofs(config_filename);
    ofs << configuration.dump(2);
  }

  int get_test_curstep() const
  {
    return curstep;
  }

  std::string normalize_test_checkpoint_prefix(json configuration, const std::string& prefix)
  {
    set_test_configuration(configuration);
    return normalize_checkpoint_prefix(prefix);
  }

  std::unique_ptr<ChunkMap> create_test_chunkmap()
  {
    return create_chunkmap();
  }

  MpiThreadMode select_test_mpi_thread_mode(json configuration, int provided)
  {
    set_test_configuration(configuration);
    mpi_thread_provided = provided;
    initialize_mpi_thread_mode();
    return mpi_thread_mode;
  }

  int get_test_mpi_thread_requested(json configuration)
  {
    set_test_configuration(configuration);
    return get_mpi_thread_requested();
  }

  void prepare_test_state(json configuration)
  {
    set_test_configuration(configuration);
    chunkmap = create_chunkmap();

    thisrank = 0;
    nprocess = 1;
    nthread  = 1;
    curstep  = 0;
    curtime  = 0;
    wclock   = 0;
    for (int i = 0; i < 4; i++) {
      ndims[i] = 1;
      cdims[i] = 1;
    }
  }
};

void cleanup_periodic_checkpoint(const std::string& prefix)
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::filesystem::remove(prefix + ".msgpack");
    std::filesystem::remove(prefix + ".status.json");
    std::filesystem::remove(prefix + ".status.json.tmp");
    std::filesystem::remove_all(prefix);

    for (int slot = 0; slot < 2; slot++) {
      const std::string concrete_prefix = prefix + "." + std::to_string(slot);
      std::filesystem::remove(concrete_prefix + ".msgpack");
      std::filesystem::remove(concrete_prefix + ".status.json");
      std::filesystem::remove(concrete_prefix + ".status.json.tmp");
      std::filesystem::remove_all(concrete_prefix);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

class ShutdownDiag : public Diag
{
public:
  static inline bool shutdown_before_mpi  = false;
  static inline bool destroyed_before_mpi = false;

  ShutdownDiag() : Diag("foo")
  {
  }

  ~ShutdownDiag() override
  {
    int finalized = 0;
    MPI_Finalized(&finalized);
    destroyed_before_mpi = finalized == 0;
  }

  void shutdown() override
  {
    int finalized = 0;
    MPI_Finalized(&finalized);
    shutdown_before_mpi = finalized == 0;
  }
};

class HandlerTestDiag : public Diag
{
public:
  HandlerTestDiag() : Diag("handler-test")
  {
  }

  std::shared_ptr<info_type> get_info()
  {
    return info;
  }
};

class ShutdownTestApplication : public TestApplication
{
public:
  using TestApplication::TestApplication;

protected:
  void initialize_diagnostic() override
  {
    Application::initialize_diagnostic();
    diagvec.push_back(std::make_unique<ShutdownDiag>());
  }
};

TEST_CASE("diagnostic handlers complete writes before returning")
{
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    SUCCEED("Skipping test because it requires one MPI rank");
    return;
  }

  const std::filesystem::path basedir = "diag_io_handler_sync";
  std::filesystem::remove_all(basedir);

  for (const std::string mode : {"mpiio", "posix"}) {
    Diag::initialize(basedir.string(), mode, "");
    {
      HandlerTestDiag                diag;
      std::unique_ptr<DiagIoHandler> handler;
      if (mode == "mpiio") {
        handler = std::make_unique<MpiioDiagIoHandler>(diag.get_info());
      } else {
        handler = std::make_unique<PosixDiagIoHandler>(diag.get_info());
      }

      const std::vector<uint8_t> expected = {1, 2, 3, 4};
      Buffer                     buffer(expected.size());
      std::copy(expected.begin(), expected.end(), buffer.get());

      const auto filename = basedir / (mode + ".data");
      size_t     disp     = 0;
      handler->open_file(filename.string(), &disp, "w");
      REQUIRE(handler->write(buffer, disp) == expected.size());
      std::fill(buffer.get(), buffer.get() + buffer.size, 0);
      handler->close_file();

      std::vector<uint8_t> actual(expected.size());
      std::ifstream        input(filename, std::ios::binary);
      input.read(reinterpret_cast<char*>(actual.data()), actual.size());
      REQUIRE(actual == expected);
    }
    Diag::finalize();
  }

  std::filesystem::remove_all(basedir);
}

TEST_CASE("test_main")
{
  std::vector<std::string> args = {"./test_application", "-c", config_filename, "--emax", "1"};
  std::vector<const char*> argv = ArgParser::convert_to_clargs(args);

  int    argc      = static_cast<int>(argv.size());
  char** cargv     = const_cast<char**>(argv.data());
  auto   interface = std::make_shared<TestApplication::Interface>();

  TestApplication app(argc, cargv, interface);

  REQUIRE(app.main() == 0);

  std::filesystem::remove("profile.msgpack");
  std::filesystem::remove("log.msgpack");
}

TEST_CASE("checkpoint prefixes are normalized lexically")
{
  const std::filesystem::path basedir = "checkpoint_prefix_normalization";
  std::filesystem::remove_all(basedir);
  std::filesystem::create_directories(basedir / "target");
  std::filesystem::create_directory_symlink("target", basedir / "alias");

  json configuration                      = json::parse(config_content);
  configuration["application"]["basedir"] = basedir.string();

  std::vector<std::string> args = {"./test_application", "-c", config_filename};
  std::vector<const char*> argv = ArgParser::convert_to_clargs(args);

  auto            interface = std::make_shared<TestApplication::Interface>();
  TestApplication app(static_cast<int>(argv.size()), const_cast<char**>(argv.data()), interface);

  const std::filesystem::path prefix =
      std::filesystem::path("alias") / "subdir" / ".." / "checkpoint";
  const std::string expected = std::filesystem::absolute(basedir / "alias" / "checkpoint").string();

  REQUIRE(app.normalize_test_checkpoint_prefix(configuration, prefix.string()) == expected);

  std::filesystem::remove_all(basedir);
}

TEST_CASE("periodic checkpointing rotates two slots and loads the latest")
{
  const std::string checkpoint_prefix = "periodic_checkpoint";
  cleanup_periodic_checkpoint(checkpoint_prefix);

  json configuration                         = json::parse(config_content);
  configuration["application"]["checkpoint"] = {
      {"interval", 1.0e-9},
      {"prefix", checkpoint_prefix},
  };

  std::vector<std::string> args = {"./test_application", "-c", config_filename, "-t", "3"};
  std::vector<const char*> argv = ArgParser::convert_to_clargs(args);

  auto            interface = std::make_shared<TestApplication::Interface>();
  TestApplication app(static_cast<int>(argv.size()), const_cast<char**>(argv.data()), interface);
  app.write_test_configuration(configuration);

  REQUIRE(app.main() == 0);

  int steps[2]     = {-1, -1};
  int status_valid = 1;
  int latest_step  = -1;
  int rank         = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    for (int slot = 0; slot < 2; slot++) {
      const std::string concrete_prefix = checkpoint_prefix + "." + std::to_string(slot);
      std::ifstream     ifs(concrete_prefix + ".status.json");
      json              status = json::parse(ifs, nullptr, false);

      status_valid &= ifs.is_open() && status.is_object() && status.contains("status") &&
                      status["status"] == "complete" && status.contains("curstep") &&
                      status["curstep"].is_number_integer();
      if (status_valid != 0) {
        steps[slot] = status["curstep"].get<int>();
        latest_step = std::max(latest_step, steps[slot]);
      }
    }

    status_valid &= steps[0] >= 0 && steps[1] >= 0 && steps[0] != steps[1];
    status_valid &= std::filesystem::exists(checkpoint_prefix + ".latest") == false;
  }
  MPI_Bcast(&status_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&latest_step, 1, MPI_INT, 0, MPI_COMM_WORLD);

  REQUIRE(status_valid == 1);

  int restart_step = latest_step;
  if (rank == 0) {
    const int         newest_slot = steps[0] > steps[1] ? 0 : 1;
    const std::string status_filename =
        checkpoint_prefix + "." + std::to_string(newest_slot) + ".status.json";
    json status;
    {
      std::ifstream ifs(status_filename);
      status = json::parse(ifs, nullptr, false);
    }
    status["prefix"] = "not-a-checkpoint-prefix";

    std::ofstream ofs(status_filename);
    ofs << status.dump(2);
    restart_step = std::min(steps[0], steps[1]);
  }
  MPI_Bcast(&restart_step, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<std::string> restart_args = {
      "./test_application", "-c", config_filename, "-l", checkpoint_prefix, "-t", "2"};
  std::vector<const char*> restart_argv = ArgParser::convert_to_clargs(restart_args);

  auto            restart_interface = std::make_shared<TestApplication::Interface>();
  TestApplication restart(static_cast<int>(restart_argv.size()),
                          const_cast<char**>(restart_argv.data()), restart_interface);
  restart.write_test_configuration(configuration);

  REQUIRE(restart.main() == 0);
  REQUIRE(restart.get_test_curstep() == restart_step);

  int stale_status_valid = 1;
  if (rank == 0) {
    const int         stale_slot = steps[0] > steps[1] ? 0 : 1;
    const std::string status_filename =
        checkpoint_prefix + "." + std::to_string(stale_slot) + ".status.json";
    std::ifstream ifs(status_filename);
    json          status = json::parse(ifs, nullptr, false);
    stale_status_valid   = ifs.is_open() && status.is_object() && status.contains("status") &&
                         status["status"] == "in_progress";
  }
  MPI_Bcast(&stale_status_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  REQUIRE(stale_status_valid == 1);

  cleanup_periodic_checkpoint(checkpoint_prefix);
  std::filesystem::remove("profile.msgpack");
  std::filesystem::remove("log.msgpack");
}

TEST_CASE("logical periodic checkpoint load falls back to an exact checkpoint")
{
  const std::string checkpoint_prefix = "legacy_checkpoint";
  cleanup_periodic_checkpoint(checkpoint_prefix);

  json configuration                         = json::parse(config_content);
  configuration["application"]["checkpoint"] = {
      {"interval", 0.0},
      {"prefix", checkpoint_prefix},
  };

  std::vector<std::string> args = {"./test_application", "-c", config_filename, "-t", "0", "-s",
                                   checkpoint_prefix};
  std::vector<const char*> argv = ArgParser::convert_to_clargs(args);

  auto            interface = std::make_shared<TestApplication::Interface>();
  TestApplication app(static_cast<int>(argv.size()), const_cast<char**>(argv.data()), interface);
  app.write_test_configuration(configuration);

  REQUIRE(app.main() == 0);

  std::vector<std::string> restart_args = {
      "./test_application", "-c", config_filename, "-l", checkpoint_prefix, "-t", "0"};
  std::vector<const char*> restart_argv = ArgParser::convert_to_clargs(restart_args);

  auto            restart_interface = std::make_shared<TestApplication::Interface>();
  TestApplication restart(static_cast<int>(restart_argv.size()),
                          const_cast<char**>(restart_argv.data()), restart_interface);
  restart.write_test_configuration(configuration);

  REQUIRE(restart.main() == 0);
  REQUIRE(restart.get_test_curstep() == 1);

  cleanup_periodic_checkpoint(checkpoint_prefix);
  std::filesystem::remove("profile.msgpack");
  std::filesystem::remove("log.msgpack");
}

TEST_CASE("periodic checkpointing selects the newest completion timestamp")
{
  const std::string checkpoint_prefix = "timestamp_checkpoint";
  cleanup_periodic_checkpoint(checkpoint_prefix);

  json configuration                         = json::parse(config_content);
  configuration["application"]["checkpoint"] = {
      {"interval", 1.0e-9},
      {"prefix", checkpoint_prefix},
  };

  std::vector<std::string> args = {"./test_application", "-c", config_filename, "-t", "3"};
  std::vector<const char*> argv = ArgParser::convert_to_clargs(args);

  auto            interface = std::make_shared<TestApplication::Interface>();
  TestApplication app(static_cast<int>(argv.size()), const_cast<char**>(argv.data()), interface);
  app.write_test_configuration(configuration);

  REQUIRE(app.main() == 0);

  int steps[2]     = {-1, -1};
  int status_valid = 1;
  int restart_step = -1;
  int rank         = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    json statuses[2];
    for (int slot = 0; slot < 2; slot++) {
      const std::string status_filename =
          checkpoint_prefix + "." + std::to_string(slot) + ".status.json";
      std::ifstream ifs(status_filename);
      statuses[slot] = json::parse(ifs, nullptr, false);

      status_valid &=
          ifs.is_open() && statuses[slot].is_object() && statuses[slot]["status"] == "complete" &&
          statuses[slot]["curstep"].is_number_integer() && statuses[slot]["timestamp"].is_number();
      if (status_valid != 0) {
        steps[slot] = statuses[slot]["curstep"].get<int>();
      }
    }

    if (status_valid != 0) {
      const int older_slot              = steps[0] < steps[1] ? 0 : 1;
      const int newer_slot              = 1 - older_slot;
      statuses[older_slot]["timestamp"] = statuses[newer_slot]["timestamp"].get<float64>() + 1.0;

      const std::string status_filename =
          checkpoint_prefix + "." + std::to_string(older_slot) + ".status.json";
      std::ofstream ofs(status_filename);
      ofs << statuses[older_slot].dump(2);
      restart_step = steps[older_slot];
    }
  }
  MPI_Bcast(&status_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&restart_step, 1, MPI_INT, 0, MPI_COMM_WORLD);

  REQUIRE(status_valid == 1);

  std::vector<std::string> restart_args = {
      "./test_application", "-c", config_filename, "-l", checkpoint_prefix, "-t", "2"};
  std::vector<const char*> restart_argv = ArgParser::convert_to_clargs(restart_args);

  auto            restart_interface = std::make_shared<TestApplication::Interface>();
  TestApplication restart(static_cast<int>(restart_argv.size()),
                          const_cast<char**>(restart_argv.data()), restart_interface);
  restart.write_test_configuration(configuration);

  REQUIRE(restart.main() == 0);
  REQUIRE(restart.get_test_curstep() == restart_step);

  cleanup_periodic_checkpoint(checkpoint_prefix);
  std::filesystem::remove("profile.msgpack");
  std::filesystem::remove("log.msgpack");
}

TEST_CASE("parsed configuration is forwarded by value")
{
  auto interface = std::make_shared<TestApplication::Interface>();

  TestApplication app(0, nullptr, interface);
  json            configuration = json::parse(config_content);
  app.set_test_configuration(configuration);

  json app_copy                     = app.get_configuration();
  json interface_copy               = interface->get_configuration();
  app_copy["parameter"]["Nx"]       = 32;
  interface_copy["parameter"]["Nx"] = 64;

  REQUIRE(app.get_configuration() == configuration);
  REQUIRE(interface->get_configuration() == configuration);
}

TEST_CASE("SFC first-axis configuration creates the requested chunk map")
{
  auto            interface = std::make_shared<TestApplication::Interface>();
  TestApplication app(0, nullptr, interface);
  json            configuration = json::parse(config_content);

  SECTION("Gilbert by default")
  {
    app.set_test_configuration(configuration);
    auto chunkmap = app.create_test_chunkmap();
    REQUIRE(chunkmap->to_json()["sfc_first_axis"].is_null());
  }

  SECTION("axis first")
  {
    configuration["application"]["option"]["sfc_first_axis"] = "z";
    app.set_test_configuration(configuration);
    auto chunkmap = app.create_test_chunkmap();
    REQUIRE(chunkmap->to_json()["sfc_first_axis"] == "z");
  }
}

TEST_CASE("MPI thread mode selects a compatible execution strategy")
{
  auto            interface = std::make_shared<TestApplication::Interface>();
  TestApplication app(0, nullptr, interface);
  json            configuration = json::parse(config_content);

  SECTION("auto selects funneled for serialized MPI")
  {
    REQUIRE(app.select_test_mpi_thread_mode(configuration, MPI_THREAD_SERIALIZED) ==
            MpiThreadMode::Funneled);
  }

  SECTION("auto selects funneled when multiple is available")
  {
    REQUIRE(app.select_test_mpi_thread_mode(configuration, MPI_THREAD_MULTIPLE) ==
            MpiThreadMode::Funneled);
  }

  SECTION("funneled can be forced with multiple support")
  {
    configuration["application"]["option"]["mpi_thread_mode"] = "funneled";
    REQUIRE(app.select_test_mpi_thread_mode(configuration, MPI_THREAD_MULTIPLE) ==
            MpiThreadMode::Funneled);
  }

  SECTION("multiple can be selected when available")
  {
    configuration["application"]["option"]["mpi_thread_mode"] = "multiple";
    REQUIRE(app.select_test_mpi_thread_mode(configuration, MPI_THREAD_MULTIPLE) ==
            MpiThreadMode::Multiple);
  }
}

TEST_CASE("MPI thread mode requests the configured support level")
{
  auto            interface = std::make_shared<TestApplication::Interface>();
  TestApplication app(0, nullptr, interface);
  json            configuration = json::parse(config_content);

  SECTION("auto preserves the build default")
  {
    REQUIRE(app.get_test_mpi_thread_requested(configuration) == NIX_MPI_THREAD_LEVEL);
  }

  SECTION("funneled requests only funneled support")
  {
    configuration["application"]["option"]["mpi_thread_mode"] = "funneled";
    REQUIRE(app.get_test_mpi_thread_requested(configuration) == MPI_THREAD_FUNNELED);
  }

  SECTION("multiple requests multiple support")
  {
    configuration["application"]["option"]["mpi_thread_mode"] = "multiple";
    REQUIRE(app.get_test_mpi_thread_requested(configuration) == MPI_THREAD_MULTIPLE);
  }
}

TEST_CASE("restart rejects an SFC mismatch")
{
  json gilbert_configuration                                    = json::parse(config_content);
  json axis_configuration                                       = gilbert_configuration;
  axis_configuration["application"]["option"]["sfc_first_axis"] = "x";

  auto            gilbert_interface = std::make_shared<TestApplication::Interface>();
  TestApplication gilbert(0, nullptr, gilbert_interface);
  gilbert.prepare_test_state(gilbert_configuration);
  json state = gilbert.to_json();

  SECTION("matching old Gilbert checkpoint")
  {
    state["chunkmap"].erase("sfc_first_axis");
    REQUIRE(gilbert.from_json(state));
  }

  SECTION("different SFC")
  {
    auto            axis_interface = std::make_shared<TestApplication::Interface>();
    TestApplication axis(0, nullptr, axis_interface);
    axis.prepare_test_state(axis_configuration);
    REQUIRE_FALSE(axis.from_json(state));
  }

  SECTION("matching axis-first SFC")
  {
    auto            axis_interface = std::make_shared<TestApplication::Interface>();
    TestApplication axis(0, nullptr, axis_interface);
    axis.prepare_test_state(axis_configuration);
    json axis_state = axis.to_json();
    REQUIRE(axis.from_json(axis_state));
  }

  SECTION("different first axes")
  {
    auto            x_interface = std::make_shared<TestApplication::Interface>();
    TestApplication x_axis(0, nullptr, x_interface);
    x_axis.prepare_test_state(axis_configuration);
    json axis_state = x_axis.to_json();

    auto            y_interface = std::make_shared<TestApplication::Interface>();
    TestApplication y_axis(0, nullptr, y_interface);
    axis_configuration["application"]["option"]["sfc_first_axis"] = "y";
    y_axis.prepare_test_state(axis_configuration);
    REQUIRE_FALSE(y_axis.from_json(axis_state));
  }
}

TEST_CASE("diagnostics shut down before MPI finalization")
{
  ShutdownDiag::shutdown_before_mpi  = false;
  ShutdownDiag::destroyed_before_mpi = false;

  std::vector<std::string> args = {"./test_application", "-c", config_filename, "--emax", "1"};
  std::vector<const char*> argv = ArgParser::convert_to_clargs(args);

  int    argc      = static_cast<int>(argv.size());
  char** cargv     = const_cast<char**>(argv.data());
  auto   interface = std::make_shared<ShutdownTestApplication::Interface>();

  {
    ShutdownTestApplication app(argc, cargv, interface);
    REQUIRE(app.main() == 0);
  }

  REQUIRE(ShutdownDiag::shutdown_before_mpi);
  REQUIRE(ShutdownDiag::destroyed_before_mpi);

  std::filesystem::remove("profile.msgpack");
  std::filesystem::remove("log.msgpack");
}
