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

#include "petsc_poisson.hpp"
#include "test_parallel_common.hpp"

using namespace nix::typedefs;
using elliptic::PetscPoisson3D;

TEST_CASE("PetscPoisson::compile", "[np=1]")
{
  elliptic::PetscPoisson3D poisson3d({10, 10, 10});
}
