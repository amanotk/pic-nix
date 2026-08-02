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

  std::unique_ptr<ChunkMap> create_test_chunkmap()
  {
    return create_chunkmap();
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
