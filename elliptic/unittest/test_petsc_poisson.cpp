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

TEST_CASE("PetscPoisson1D::compile", "[np=1]")
{
  elliptic::PetscPoisson1D solver({1, 1, 10}, 1.0);
  solver.solve();

  std::cout << std::scientific << "error : " << solver.get_residual_norm() << std::endl;
}

TEST_CASE("PetscPoisson2D::compile", "[np=1]")
{
  elliptic::PetscPoisson2D solver({1, 10, 10}, 1.0);
  solver.solve();

  std::cout << std::scientific << "error : " << solver.get_residual_norm() << std::endl;
}

TEST_CASE("PetscPoisson3D::compile", "[np=1]")
{
  elliptic::PetscPoisson3D solver({10, 10, 10}, 1.0);
  solver.solve();

  std::cout << std::scientific << "error : " << solver.get_residual_norm() << std::endl;
}
