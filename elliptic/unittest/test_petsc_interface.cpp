// -*- C++-*-
#include "thirdparty/catch.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <type_traits>

#include <petscao.h>
#include <petscdmda.h>
#include <petscsys.h>
#include <petscvec.h>

#include "petsc_interface.hpp"
#include "test_parallel_common.hpp"

using namespace nix::typedefs;
using elliptic::PetscInterface;

struct PetscInterfaceTest final : public PetscInterface {
  using PetscInterface::apply_petsc_option;
  using PetscInterface::make_petsc_option;

  PetscInterfaceTest() : PetscInterface({1, 1, 2})
  {
  }

  int set_matrix() override
  {
    return 0;
  }

  int solve() override
  {
    return 0;
  }
};

template <typename T>
bool is_option_valid(const PetscInterface::OptionVec& option, std::string key, T val)
{
  constexpr bool is_string =
      std::is_same_v<T, std::string> || std::is_same_v<T, const char*> || std::is_same_v<T, char*>;
  constexpr bool is_bool    = std::is_same_v<T, bool>;
  constexpr bool is_integer = std::is_integral_v<T>;
  constexpr bool is_float   = std::is_floating_point_v<T>;

  for (const auto& opt : option) {
    if constexpr (is_bool) {
      if (opt.first == key && ((val && opt.second == "true") || (!val && opt.second == "false"))) {
        return true;
      }
    }

    if constexpr (is_integer) {
      if (opt.first == key && std::stoi(opt.second) == val) {
        return true;
      }
    }

    if constexpr (is_float) {
      if (opt.first == key && std::abs(std::stod(opt.second) - val) < 1e-12) {
        return true;
      }
    }

    if constexpr (is_string) {
      if (opt.first == key && opt.second == val) {
        return true;
      }
    }
  }

  return false;
}

TEST_CASE("PetscInterface::toml_options", "[np=1]")
{
  std::string toml_string = R"(
  ksp_monitor = true
  ksp_type = "gmres"
  ksp_max_it = 100
  ksp_rtol = 1.0e-8
  )";

  auto config = toml::parse_str(toml_string);
  auto option = PetscInterfaceTest::make_petsc_option(config);

  REQUIRE(is_option_valid(option, "ksp_monitor", true));
  REQUIRE(is_option_valid(option, "ksp_type", "gmres"));
  REQUIRE(is_option_valid(option, "ksp_max_it", 100));
  REQUIRE(is_option_valid(option, "ksp_rtol", 1e-8));

  REQUIRE(PetscOptionsClear(nullptr) == 0);
  REQUIRE(PetscInterfaceTest::apply_petsc_option(option) == 0);
}

TEST_CASE("PetscInterface::json_options", "[np=1]")
{
  const std::string json_string = R"({
    "ksp_monitor": true,
    "ksp_type": "gmres",
    "ksp_max_it": 100,
    "ksp_rtol": 1.0e-8
  })";

  const auto config = nlohmann::json::parse(json_string);
  const auto option = PetscInterfaceTest::make_petsc_option(config);

  REQUIRE(is_option_valid(option, "ksp_monitor", true));
  REQUIRE(is_option_valid(option, "ksp_type", "gmres"));
  REQUIRE(is_option_valid(option, "ksp_max_it", 100));
  REQUIRE(is_option_valid(option, "ksp_rtol", 1e-8));

  REQUIRE(PetscOptionsClear(nullptr) == 0);
  REQUIRE(PetscInterfaceTest::apply_petsc_option(option) == 0);
}