// -*- C++ -*-

#include "argparser.hpp"

#include "catch.hpp"

using namespace nix;

TEST_CASE("Basic")
{
  ArgParser parser;

  SECTION("default")
  {
    std::string args = "./a.out -c config.json";

    parser.parse_check(args);

    REQUIRE(parser.get_config() == "config.json");
    REQUIRE(parser.get_load() == "");
    REQUIRE(parser.get_save() == "");
  }

  SECTION("time")
  {
    std::string args = "./a.out -c config.json --tmax 100 --emax 200";

    parser.parse_check(args);

    REQUIRE(parser.get_physical_time_max() == 100);
    REQUIRE(parser.get_elapsed_time_max() == 200);
  }

  SECTION("verbosity")
  {
    std::string args = "./a.out -c config.json --verbose 1";

    parser.parse_check(args);

    REQUIRE(parser.get_verbosity() == 1);
  }

  SECTION("C-style command line arguments")
  {
    std::vector<std::string> args = {"./a.out", "-c", "config.json", "--tmax", "100"};
    std::vector<const char*> argv = ArgParser::convert_to_clargs(args);

    // NOTE: parse_check gives compilation error!
    parser.parse(argv.size(), &argv[0]);

    REQUIRE(parser.get_config() == "config.json");
    REQUIRE(parser.get_physical_time_max() == 100);
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
