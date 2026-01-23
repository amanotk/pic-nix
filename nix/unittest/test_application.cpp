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
