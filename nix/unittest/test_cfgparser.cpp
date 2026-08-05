// -*- C++ -*-

#include "cfgparser.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace nix;

TEST_CASE("Basic")
{
  CfgParser parser;
}

TEST_CASE("get_root returns an isolated copy from a const parser")
{
  CfgParser parser;
  json      configuration = {
      {"application", {{"option", json::object()}}},
      {"diagnostic", json::array()},
      {"parameter",
            {{"Nx", 16},
             {"Ny", 16},
             {"Nz", 16},
             {"Cx", 4},
             {"Cy", 4},
             {"Cz", 4},
             {"delt", 1.0},
             {"delh", 1.0}}},
  };
  parser.overwrite(configuration);

  const CfgParser& const_parser = parser;
  json             root_copy    = const_parser.get_root();
  root_copy["parameter"]["Nx"]  = 32;

  REQUIRE(const_parser.get_root() == configuration);
  REQUIRE(const_parser.get_root()["parameter"]["Nx"] == 16);
}

TEST_CASE("check_mandatory_sections")
{
  CfgParser parser;

  SECTION("successful")
  {
    json root = {{"application", 0}, {"diagnostic", 0}, {"parameter", 0}};

    REQUIRE(parser.check_mandatory_sections(root) == true);
  }

  SECTION("application is missing")
  {
    json root = {{"diagnostic", 0}, {"parameter", 0}};

    REQUIRE(parser.check_mandatory_sections(root) == false);
  }

  SECTION("diagnostic is missing")
  {
    json root = {{"application", 0}, {"parameter", 0}};

    REQUIRE(parser.check_mandatory_sections(root) == false);
  }

  SECTION("parameter is missing")
  {
    json root = {{"application", 0}, {"diagnostic", 0}};

    REQUIRE(parser.check_mandatory_sections(root) == false);
  }
}

TEST_CASE("check_mandatory_parameters")
{
  CfgParser parser;

  json parameter = {
      {"Nx", 16}, {"Ny", 16}, {"Nz", 16},    {"Cx", 4},
      {"Cy", 4},  {"Cz", 4},  {"delt", 1.0}, {"delh", 1.0},
  };

  SECTION("successful")
  {
    REQUIRE(parser.check_mandatory_parameters(parameter) == true);
  }

  SECTION("missing something")
  {
    auto check_parameter_with_removed_item([&](const std::string key) {
      auto tmp = parameter;
      tmp.erase(key);
      return parser.check_mandatory_parameters(tmp);
    });

    REQUIRE(check_parameter_with_removed_item("Nx") == false);
    REQUIRE(check_parameter_with_removed_item("Ny") == false);
    REQUIRE(check_parameter_with_removed_item("Nz") == false);
    REQUIRE(check_parameter_with_removed_item("Cx") == false);
    REQUIRE(check_parameter_with_removed_item("Cy") == false);
    REQUIRE(check_parameter_with_removed_item("Cz") == false);
    REQUIRE(check_parameter_with_removed_item("delt") == false);
    REQUIRE(check_parameter_with_removed_item("delh") == false);
  }
}

TEST_CASE("check_dimensions")
{
  CfgParser parser;

  SECTION("successful")
  {
    json parameter = json::array({
        // 0
        {
            {"Nx", 2},
            {"Ny", 2},
            {"Nz", 2},
            {"Cx", 1},
            {"Cy", 1},
            {"Cz", 1},
        },
        // 1
        {
            {"Nx", 8},
            {"Ny", 8},
            {"Nz", 8},
            {"Cx", 2},
            {"Cy", 2},
            {"Cz", 2},
        },
        // 2
        {
            {"Nx", 8},
            {"Ny", 8},
            {"Nz", 8},
            {"Cx", 1},
            {"Cy", 1},
            {"Cz", 8},
        },
    });

    REQUIRE(parser.check_dimensions(parameter[0]) == true);
    REQUIRE(parser.check_dimensions(parameter[1]) == true);
    REQUIRE(parser.check_dimensions(parameter[2]) == true);
  }

  SECTION("failure")
  {
    json parameter = json::array({
        // 0
        {
            {"Nx", 8},
            {"Ny", 8},
            {"Nz", 8},
            {"Cx", 3},
            {"Cy", 1},
            {"Cz", 1},
        },
        // 1
        {
            {"Nx", 8},
            {"Ny", 8},
            {"Nz", 8},
            {"Cx", 1},
            {"Cy", 3},
            {"Cz", 1},
        },
        // 2
        {
            {"Nx", 8},
            {"Ny", 8},
            {"Nz", 8},
            {"Cx", 1},
            {"Cy", 3},
            {"Cz", 1},
        },
    });

    REQUIRE(parser.check_dimensions(parameter[0]) == false);
    REQUIRE(parser.check_dimensions(parameter[1]) == false);
    REQUIRE(parser.check_dimensions(parameter[2]) == false);
  }
}

TEST_CASE("check_application_options")
{
  CfgParser parser;

  SECTION("Gilbert is the default")
  {
    json option = json::object();
    REQUIRE(parser.check_application_options(option));
  }

  SECTION("valid first axes")
  {
    for (const char* axis : {"x", "y", "z"}) {
      json option = {{"sfc_first_axis", axis}};
      REQUIRE(parser.check_application_options(option));
    }
  }

  SECTION("invalid first axis")
  {
    json option = {{"sfc_first_axis", "time"}};
    REQUIRE_FALSE(parser.check_application_options(option));
  }

  SECTION("valid MPI thread modes")
  {
    for (const char* mode : {"auto", "multiple", "funneled"}) {
      json option = {{"mpi_thread_mode", mode}};
      REQUIRE(parser.check_application_options(option));
    }
  }

  SECTION("invalid MPI thread mode")
  {
    json option = {{"mpi_thread_mode", "serialized"}};
    REQUIRE_FALSE(parser.check_application_options(option));
  }

  SECTION("MPI thread mode must be a string")
  {
    json option = {{"mpi_thread_mode", 0}};
    REQUIRE_FALSE(parser.check_application_options(option));
  }

  SECTION("first axis must be a string")
  {
    json option = {{"sfc_first_axis", 0}};
    REQUIRE_FALSE(parser.check_application_options(option));
  }

  SECTION("option section must be a table")
  {
    json scalar = 0;
    json array  = json::array();
    REQUIRE_FALSE(parser.check_application_options(scalar));
    REQUIRE_FALSE(parser.check_application_options(array));
  }

  SECTION("full configuration rejects an invalid axis")
  {
    json configuration = {
        {"application", {{"option", {{"sfc_first_axis", "time"}}}}},
        {"diagnostic", json::array()},
        {"parameter",
         {{"Nx", 16},
          {"Ny", 16},
          {"Nz", 16},
          {"Cx", 4},
          {"Cy", 4},
          {"Cz", 4},
          {"delt", 1.0},
          {"delh", 1.0}}},
    };
    REQUIRE_FALSE(parser.validate(configuration));
  }
}

TEST_CASE("parse_file")
{
  CfgParser   parser;
  std::string filename;
  std::string content;

  SECTION("json")
  {
    filename = "test_cfgparser.json";
    content  = R"(
    {
      "application": {
        "rebalance": {},
        "log": {}
      },
      "diagnostic": [
        {},
        {},
        {}
      ],
      "parameter": {
          "Nx": 16,
          "Ny": 16,
          "Nz": 16,
          "Cx": 4,
          "Cy": 4,
          "Cz": 4,
          "delt": 1.0,
          "delh": 1.0
      }
    }
    )";
  }
  SECTION("toml")
  {
    filename = "test_cfgparser.toml";
    content  = R"(
    [application]
    [application.rebalance]
    [application.log]

    [[diagnostic]]
    [[diagnostic]]
    [[diagnostic]]

    [parameter]
    Nx = 16
    Ny = 16
    Nz = 16
    Cx = 4
    Cy = 4
    Cz = 4
    delt = 1.0
    delh = 1.0
    )";
  }

  // save
  {
    std::ofstream ofs(filename);
    ofs << content;
  }

  REQUIRE(parser.parse_file(filename) == true);

  // cleanup
  std::filesystem::remove(filename);
}
