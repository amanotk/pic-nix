// -*- C++ -*-

#include "sfc.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace
{
void check_axis_first(size_t Nz, size_t Ny, size_t Nx, sfc::SfcAxis axis)
{
  size_t line_size;
  int    axis_column;

  switch (axis) {
  case sfc::SfcAxis::X:
    line_size   = Nx;
    axis_column = 0;
    break;
  case sfc::SfcAxis::Y:
    line_size   = Ny;
    axis_column = 1;
    break;
  case sfc::SfcAxis::Z:
    line_size   = Nz;
    axis_column = 2;
    break;
  }

  size_t           size = Nz * Ny * Nx;
  std::vector<int> index(size, 0);
  std::vector<int> coord(size * 3, 0);
  sfc::get_map3d_axis_first(Nz, Ny, Nx, axis, index, coord);

  REQUIRE(sfc::check_index(index));
  REQUIRE(sfc::check_locality3d(coord, size));

  for (size_t line = 0; line < size / line_size; line++) {
    int expected = line % 2 == 0 ? 0 : static_cast<int>(line_size - 1);
    int delta    = line % 2 == 0 ? 1 : -1;

    for (size_t offset = 0; offset < line_size; offset++) {
      size_t id = line * line_size + offset;
      REQUIRE(coord[id * 3 + axis_column] == expected);
      expected += delta;
    }
  }
}
} // namespace

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

TEST_CASE("SFC3D axis first")
{
  SECTION("axis names")
  {
    REQUIRE(sfc::parse_axis("x") == sfc::SfcAxis::X);
    REQUIRE(sfc::parse_axis("y") == sfc::SfcAxis::Y);
    REQUIRE(sfc::parse_axis("z") == sfc::SfcAxis::Z);
    REQUIRE_FALSE(sfc::parse_axis("invalid").has_value());
    REQUIRE(std::string(sfc::axis_name(sfc::SfcAxis::X)) == "x");
    REQUIRE(std::string(sfc::axis_name(sfc::SfcAxis::Y)) == "y");
    REQUIRE(std::string(sfc::axis_name(sfc::SfcAxis::Z)) == "z");
  }

  SECTION("three dimensional")
  {
    check_axis_first(4, 6, 8, sfc::SfcAxis::X);
    check_axis_first(4, 6, 8, sfc::SfcAxis::Y);
    check_axis_first(4, 6, 8, sfc::SfcAxis::Z);
  }

  SECTION("odd dimensions remain connected")
  {
    check_axis_first(3, 5, 7, sfc::SfcAxis::X);
    check_axis_first(3, 5, 7, sfc::SfcAxis::Y);
    check_axis_first(3, 5, 7, sfc::SfcAxis::Z);
  }

  SECTION("production shock dimensions")
  {
    check_axis_first(12, 48, 480, sfc::SfcAxis::X);
  }

  SECTION("degenerate dimensions")
  {
    check_axis_first(1, 5, 7, sfc::SfcAxis::X);
    check_axis_first(5, 1, 7, sfc::SfcAxis::Y);
    check_axis_first(5, 7, 1, sfc::SfcAxis::Z);
    check_axis_first(1, 1, 7, sfc::SfcAxis::X);
    check_axis_first(1, 1, 1, sfc::SfcAxis::Z);
  }
}
