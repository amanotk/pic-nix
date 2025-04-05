// -*- C++ -*-

#include "xtensor_particle.hpp"
#include <iostream>

#include "catch.hpp"

using namespace nix::typedefs;
using namespace nix::primitives;
using Particle     = nix::XtensorParticle;
using uniform_rand = std::uniform_real_distribution<float64>;

class MockChunk
{
private:
  int     boundary_margin;
  int     Nx;
  int     Ny;
  int     Nz;
  float64 delh;

public:
  MockChunk(int Nz, int Ny, int Nx) : boundary_margin(1), Nz(Nz), Ny(Ny), Nx(Nx), delh(1.0)
  {
  }

  int get_boundary_margin() const
  {
    return boundary_margin;
  }

  bool has_xdim() const
  {
    return Nx > 1;
  }

  bool has_ydim() const
  {
    return Ny > 1;
  }

  bool has_zdim() const
  {
    return Nz > 1;
  }

  float64 get_delx() const
  {
    return delh;
  }

  float64 get_dely() const
  {
    return delh;
  }

  float64 get_delz() const
  {
    return delh;
  }

  auto get_xbound() const
  {
    return std::pair(boundary_margin, Nx - 1 + boundary_margin);
  }

  auto get_ybound() const
  {
    return std::pair(boundary_margin, Ny - 1 + boundary_margin);
  }

  auto get_zbound() const
  {
    return std::pair(boundary_margin, Nz - 1 + boundary_margin);
  }

  auto get_xrange() const
  {
    float64 xmin = -std::numeric_limits<float64>::max();
    float64 xmax = +std::numeric_limits<float64>::max();

    if (has_xdim()) {
      xmin = 0.0;
      xmax = delh * Nx;
    }

    return std::pair(xmin, xmax);
  }

  auto get_yrange() const
  {
    float64 ymin = -std::numeric_limits<float64>::max();
    float64 ymax = +std::numeric_limits<float64>::max();

    if (has_ydim()) {
      ymin = 0.0;
      ymax = delh * Ny;
    }

    return std::pair(ymin, ymax);
  }

  auto get_zrange() const
  {
    float64 zmin = -std::numeric_limits<float64>::max();
    float64 zmax = +std::numeric_limits<float64>::max();

    if (has_zdim()) {
      zmin = 0.0;
      zmax = delh * Nz;
    }

    return std::pair(zmin, zmax);
  }

  auto get_xrange_global() const
  {
    return std::pair(0.0, delh * Nx);
  }

  auto get_yrange_global() const
  {
    return std::pair(0.0, delh * Ny);
  }

  auto get_zrange_global() const
  {
    return std::pair(0.0, delh * Nz);
  }
};

// set random particle position
void set_random_particle(Particle& particle, int k, float64 rmin, float64 rmax)
{
  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(rmin, rmax);

  for (int ip = 0; ip < particle.Np; ip++) {
    particle.xu.at(ip, k) = rand(engine);
  }
}

// check particle sort for 1D mesh
bool check_sort1d(Particle& particle, const int Nx)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  bool status = true;

  for (int ii = 0; ii < particle.Ng; ii++) {
    for (int ip = particle.pindex.at(ii); ip < particle.pindex.at(ii + 1); ip++) {
      int jx = digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
      int jj = jx;

      status = status & (ii == jj);
    }
  }

  return status;
}

// check particle sort for 2D mesh
bool check_sort2d(Particle& particle, const int Nx, const int Ny)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  bool status = true;

  for (int ii = 0; ii < particle.Ng; ii++) {
    for (int ip = particle.pindex.at(ii); ip < particle.pindex.at(ii + 1); ip++) {
      int jx = digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
      int jy = digitize(particle.xu.at(ip, 1), xmin, 1 / delh);
      int jj = jx + jy * Nx;

      status = status & (ii == jj);
    }
  }

  return status;
}

// check particle sort for 3D mesh
bool check_sort3d(Particle& particle, const int Nx, const int Ny, const int Nz)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  bool status = true;

  for (int ii = 0; ii < particle.Ng; ii++) {
    for (int ip = particle.pindex.at(ii); ip < particle.pindex.at(ii + 1); ip++) {
      int jx = digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
      int jy = digitize(particle.xu.at(ip, 1), xmin, 1 / delh);
      int jz = digitize(particle.xu.at(ip, 2), xmin, 1 / delh);
      int jj = jx + jy * Nx + jz * Nx * Ny;

      status = status & (ii == jj);
    }
  }

  return status;
}

//
// create particle
//
TEST_CASE("CreateParticle")
{
  const int Np = 1000;
  const int Nx = 8;
  const int Ny = 8;
  const int Nz = 8;
  const int Ng = (Nx + 2) * (Ny + 2) * (Nz + 2);

  MockChunk chunk(Nz, Ny, Nx);
  Particle  particle(Np, chunk);
  particle.Np = Np;

  // check array size
  REQUIRE(particle.xu.size() == Np * Particle::Nc);
  REQUIRE(particle.xv.size() == Np * Particle::Nc);
  REQUIRE(particle.gindex.size() == Np);
  REQUIRE(particle.pindex.size() == Ng + 1);
  REQUIRE(particle.pcount.size() == (Ng + 1) * nix::simd_width);

  // check particle data
  REQUIRE(xt::allclose(particle.xu, 0.0));
  REQUIRE(xt::allclose(particle.xv, 0.0));
}

//
// swap particle
//
TEST_CASE("SwapParticle")
{
  const int Np = 1000;
  const int Nx = 8;
  const int Ny = 8;
  const int Nz = 8;

  MockChunk chunk(Nz, Ny, Nx);
  Particle  particle(Np, chunk);
  particle.Np = Np;

  float64* ptr1 = particle.xu.data();
  float64* ptr2 = particle.xv.data();

  // set random number
  {
    std::random_device seed;
    std::mt19937_64    engine(seed());
    uniform_rand       rand(0.0, 1.0);

    for (int i = 0; i < Np * Particle::Nc; i++) {
      ptr1[i] = rand(engine);
      ptr2[i] = rand(engine);
    }
  }

  // before swap
  REQUIRE(particle.xu.data() == ptr1);
  REQUIRE(std::equal(particle.xu.begin(), particle.xu.end(), ptr1));
  REQUIRE(particle.xv.data() == ptr2);
  REQUIRE(std::equal(particle.xv.begin(), particle.xv.end(), ptr2));

  // swap xu and xv
  particle.swap();

  // after swap
  REQUIRE(particle.xu.data() == ptr2);
  REQUIRE(std::equal(particle.xu.begin(), particle.xu.end(), ptr2));
  REQUIRE(particle.xv.data() == ptr1);
  REQUIRE(std::equal(particle.xv.begin(), particle.xv.end(), ptr1));
}

//
// sort particle 1D
//
TEST_CASE("SortParticle1D")
{
  const int Np = GENERATE(100, 1000, 10000);
  const int Nx = GENERATE(8, 64, 256);
  const int Ny = 1;
  const int Nz = 1;

  MockChunk chunk(Nz, Ny, Nx);
  Particle  particle(Np, chunk);

  auto [xmin, xmax] = chunk.get_xrange();
  auto delx         = chunk.get_delx();

  // set random particle position
  set_random_particle(particle, 0, xmin - delx, xmax + delx);

  // count and sort
  particle.count(0, particle.Np, true, 1);
  particle.sort();

  // check result
  REQUIRE(check_sort1d(particle, Nx) == true);
}

//
// sort particle 2D
//
TEST_CASE("SortParticle2D")
{
  const int Np = GENERATE(100, 1000, 10000);
  const int Nx = GENERATE(8, 16);
  const int Ny = GENERATE(8, 16);
  const int Nz = 1;

  MockChunk chunk(Nz, Ny, Nx);
  Particle  particle(Np, chunk);

  auto [xmin, xmax] = chunk.get_xrange();
  auto [ymin, ymax] = chunk.get_yrange();
  auto delx         = chunk.get_delx();
  auto dely         = chunk.get_dely();

  // set random particle position
  set_random_particle(particle, 0, xmin - delx, xmax + delx);
  set_random_particle(particle, 1, ymin - dely, ymax + dely);

  // count and sort
  particle.count(0, particle.Np, true, 1);
  particle.sort();

  // check result
  REQUIRE(check_sort2d(particle, Nx, Ny) == true);
}

//
// sort particle 3D
//
TEST_CASE("SortParticle3D")
{
  const int Np = GENERATE(100, 1000, 10000);
  const int Nx = GENERATE(8, 16);
  const int Ny = GENERATE(8, 16);
  const int Nz = GENERATE(8, 16);

  MockChunk chunk(Nz, Ny, Nx);
  Particle  particle(Np, chunk);

  auto [xmin, xmax] = chunk.get_xrange();
  auto [ymin, ymax] = chunk.get_yrange();
  auto [zmin, zmax] = chunk.get_zrange();
  auto delx         = chunk.get_delx();
  auto dely         = chunk.get_dely();
  auto delz         = chunk.get_delz();

  // set random particle position
  set_random_particle(particle, 0, xmin - delx, xmax + delx);
  set_random_particle(particle, 1, ymin - dely, ymax + dely);
  set_random_particle(particle, 2, zmin - delz, zmax + delz);

  // count and sort
  particle.count(0, particle.Np, true, 1);
  particle.sort();

  // check result
  REQUIRE(check_sort3d(particle, Nx, Ny, Nz) == true);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
