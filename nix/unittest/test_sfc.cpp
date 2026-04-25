// -*- C++ -*-

#include "sfc.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

//
// 2D
//
TEST_CASE("SFC2D")
{
  SECTION("even")
  {
    size_t Nx = GENERATE(1, 4, 20, 100);
    size_t Ny = GENERATE(1, 4, 20, 100);

    std::vector<int> index(Ny * Nx, 0);
    std::vector<int> coord(Ny * Nx * 2, 0);

    sfc::get_map2d(Ny, Nx, 2, index, coord);
    REQUIRE(sfc::check_locality2d(coord, Ny * Nx));
    REQUIRE(sfc::check_index(index));
  }
  SECTION("odd-x")
  {
    const int distmax2 = 2;
    size_t    Nx       = GENERATE(3, 7, 9);
    size_t    Ny       = GENERATE(4, 8, 16);

    std::vector<int> index(Ny * Nx, 0);
    std::vector<int> coord(Ny * Nx * 2, 0);

    sfc::get_map2d(Ny, Nx, 2, index, coord);
    REQUIRE(sfc::check_locality2d(coord, Ny * Nx, distmax2));
    REQUIRE(sfc::check_index(index));
  }
  SECTION("odd-y")
  {
    const int distmax2 = 2;
    size_t    Nx       = GENERATE(4, 8, 16);
    size_t    Ny       = GENERATE(3, 7, 9);

    std::vector<int> index(Ny * Nx, 0);
    std::vector<int> coord(Ny * Nx * 2, 0);

    sfc::get_map2d(Ny, Nx, 2, index, coord);
    REQUIRE(sfc::check_locality2d(coord, Ny * Nx, distmax2));
    REQUIRE(sfc::check_index(index));
  }
}

//
// 3D
//
TEST_CASE("SFC3D")
{
  SECTION("even")
  {
    size_t Nx = GENERATE(1, 4, 20, 100);
    size_t Ny = GENERATE(1, 4, 20, 100);
    size_t Nz = GENERATE(1, 4, 20, 100);

    std::vector<int> index(Nz * Ny * Nx, 0);
    std::vector<int> coord(Nz * Ny * Nx * 3, 0);

    sfc::get_map3d(Nz, Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality3d(coord, Nz * Ny * Nx));
    REQUIRE(sfc::check_index(index));
  }
  SECTION("odd-x")
  {
    const int distmax2 = 2;
    size_t    Nx       = GENERATE(3, 7, 9);
    size_t    Ny       = GENERATE(4, 8, 16);
    size_t    Nz       = GENERATE(4, 8, 16);

    std::vector<int> index(Nz * Ny * Nx, 0);
    std::vector<int> coord(Nz * Ny * Nx * 3, 0);

    sfc::get_map3d(Nz, Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality3d(coord, Nz * Ny * Nx, distmax2));
    REQUIRE(sfc::check_index(index));
  }
  SECTION("odd-y")
  {
    const int distmax2 = 2;
    size_t    Nx       = GENERATE(4, 8, 16);
    size_t    Ny       = GENERATE(3, 7, 9);
    size_t    Nz       = GENERATE(4, 8, 16);

    std::vector<int> index(Nz * Ny * Nx, 0);
    std::vector<int> coord(Nz * Ny * Nx * 3, 0);

    sfc::get_map3d(Nz, Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality3d(coord, Nz * Ny * Nx, distmax2));
    REQUIRE(sfc::check_index(index));
  }
  SECTION("odd-z")
  {
    const int distmax2 = 2;
    size_t    Nx       = GENERATE(4, 8, 16);
    size_t    Ny       = GENERATE(4, 8, 16);
    size_t    Nz       = GENERATE(3, 7, 9);

    std::vector<int> index(Nz * Ny * Nx, 0);
    std::vector<int> coord(Nz * Ny * Nx * 3, 0);

    sfc::get_map3d(Nz, Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality3d(coord, Nz * Ny * Nx, distmax2));
    REQUIRE(sfc::check_index(index));
  }
}
