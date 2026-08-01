// -*- C++ -*-

#include "application.hpp"
#include "argparser.hpp"
#include "chunk.hpp"
#include "chunkmap.hpp"
#include "diag.hpp"

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
};

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
