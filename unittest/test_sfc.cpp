// -*- C++ -*-

#include "sfc.hpp"
#include "xtensorall.hpp"

#include "catch.hpp"

//
// 2D
//
TEST_CASE("SFC2D")
{
  SECTION("even")
  {
    size_t Nx = GENERATE(1, 4, 20, 100);
    size_t Ny = GENERATE(1, 4, 20, 100);

    xt::xtensor<int, 2> index({Ny, Nx});
    xt::xtensor<int, 2> coord({Ny * Nx, 2});

    sfc::get_map2d(Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality2d(coord));
    REQUIRE(sfc::check_index(index));
  }
  SECTION("odd-x")
  {
    const int distmax2 = 2;
    size_t    Nx       = GENERATE(3, 7, 9);
    size_t    Ny       = GENERATE(4, 8, 16);

    xt::xtensor<int, 2> index({Ny, Nx});
    xt::xtensor<int, 2> coord({Ny * Nx, 2});

    sfc::get_map2d(Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality2d(coord, distmax2));
    REQUIRE(sfc::check_index(index));
  }
  SECTION("odd-y")
  {
    const int distmax2 = 2;
    size_t    Nx       = GENERATE(4, 8, 16);
    size_t    Ny       = GENERATE(3, 7, 9);

    xt::xtensor<int, 2> index({Ny, Nx});
    xt::xtensor<int, 2> coord({Ny * Nx, 2});

    sfc::get_map2d(Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality2d(coord, distmax2));
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

    xt::xtensor<int, 3> index({Nz, Ny, Nx});
    xt::xtensor<int, 2> coord({Nz * Ny * Nx, 3});

    sfc::get_map3d(Nz, Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality3d(coord));
    REQUIRE(sfc::check_index(index));
  }
  SECTION("odd-x")
  {
    const int distmax2 = 2;
    size_t    Nx       = GENERATE(3, 7, 9);
    size_t    Ny       = GENERATE(4, 8, 16);
    size_t    Nz       = GENERATE(4, 8, 16);

    xt::xtensor<int, 3> index({Nz, Ny, Nx});
    xt::xtensor<int, 2> coord({Nz * Ny * Nx, 3});

    sfc::get_map3d(Nz, Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality3d(coord, distmax2));
    REQUIRE(sfc::check_index(index));
  }
  SECTION("odd-y")
  {
    const int distmax2 = 2;
    size_t    Nx       = GENERATE(4, 8, 16);
    size_t    Ny       = GENERATE(3, 7, 9);
    size_t    Nz       = GENERATE(4, 8, 16);

    xt::xtensor<int, 3> index({Nz, Ny, Nx});
    xt::xtensor<int, 2> coord({Nz * Ny * Nx, 3});

    sfc::get_map3d(Nz, Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality3d(coord, distmax2));
    REQUIRE(sfc::check_index(index));
  }
  SECTION("odd-z")
  {
    const int distmax2 = 2;
    size_t    Nx       = GENERATE(4, 8, 16);
    size_t    Ny       = GENERATE(4, 8, 16);
    size_t    Nz       = GENERATE(3, 7, 9);

    xt::xtensor<int, 3> index({Nz, Ny, Nx});
    xt::xtensor<int, 2> coord({Nz * Ny * Nx, 3});

    sfc::get_map3d(Nz, Ny, Nx, index, coord);
    REQUIRE(sfc::check_locality3d(coord, distmax2));
    REQUIRE(sfc::check_index(index));
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
