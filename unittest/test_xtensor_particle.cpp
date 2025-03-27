// -*- C++ -*-

#include "xtensor_particle.hpp"
#include <iostream>

#include "catch.hpp"

using namespace nix::typedefs;
using namespace nix::primitives;
using Particle     = nix::XtensorParticle<7>;
using uniform_rand = std::uniform_real_distribution<float64>;

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

// prepare particle sort for 1D mesh
void prepare_sort1d(Particle& particle, const int Nx)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  // set random particle position
  set_random_particle(particle, 0, xmin - delh, xmin + delh * (Nx + 1));

  // prepare for sort
  int last = particle.Ng;
  particle.gindex.fill(0);
  particle.pcount.fill(0);
  for (int ip = 0; ip < particle.Np; ip++) {
    int ix = digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
    int ii = ix;

    // take care out-of-bounds particles
    ii = (ix < 0 || ix >= Nx) ? last : ii;

    particle.increment(ip, ii);
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

// prepare particle sort for 2D mesh
void prepare_sort2d(Particle& particle, const int Nx, const int Ny)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  // set random particle position
  set_random_particle(particle, 0, xmin - delh, xmin + delh * (Nx + 1));
  set_random_particle(particle, 1, xmin - delh, xmin + delh * (Ny + 1));

  // prepare for sort
  int last = particle.Ng;
  particle.gindex.fill(0);
  particle.pcount.fill(0);
  for (int ip = 0; ip < particle.Np; ip++) {
    int ix = digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
    int iy = digitize(particle.xu.at(ip, 1), xmin, 1 / delh);
    int ii = ix + iy * Nx;

    // take care out-of-bounds particles
    ii = (ix < 0 || ix >= Nx) ? last : ii;
    ii = (iy < 0 || iy >= Ny) ? last : ii;

    particle.increment(ip, ii);
  }
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

// prepare particle sort for 3D mesh
void prepare_sort3d(Particle& particle, const int Nx, const int Ny, const int Nz)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  // set random particle position
  set_random_particle(particle, 0, xmin - delh, xmin + delh * (Nx + 1));
  set_random_particle(particle, 1, xmin - delh, xmin + delh * (Ny + 1));
  set_random_particle(particle, 2, xmin - delh, xmin + delh * (Nz + 1));

  // prepare for sort
  int last = particle.Ng;
  particle.gindex.fill(0);
  particle.pcount.fill(0);
  for (int ip = 0; ip < particle.Np; ip++) {
    int ix = digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
    int iy = digitize(particle.xu.at(ip, 1), xmin, 1 / delh);
    int iz = digitize(particle.xu.at(ip, 2), xmin, 1 / delh);
    int ii = ix + iy * Nx + iz * Nx * Ny;

    // take care out-of-bounds particles
    ii = (ix < 0 || ix >= Nx) ? last : ii;
    ii = (iy < 0 || iy >= Ny) ? last : ii;
    ii = (iz < 0 || iz >= Nz) ? last : ii;

    particle.increment(ip, ii);
  }
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
  const int Ng = 100;

  Particle particle(Np, Ng);
  particle.Np = Np;

  // check array size
  REQUIRE(particle.xu.size() == Np * Particle::Nc);
  REQUIRE(particle.xv.size() == Np * Particle::Nc);
  REQUIRE(particle.gindex.size() == Np);
  REQUIRE(particle.pindex.size() == Ng + 1);
  REQUIRE(particle.pcount.size() == (Ng + 1) * Particle::simd_width);

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
  const int Ng = 100;

  Particle particle(Np, Ng);
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

  Particle particle(Np, Nx);
  particle.Np = Np;

  // sort
  prepare_sort1d(particle, Nx);
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

  Particle particle(Np, Nx * Ny);
  particle.Np = Np;

  // sort
  prepare_sort2d(particle, Nx, Ny);
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

  Particle particle(Np, Nx * Ny * Nz);
  particle.Np = Np;

  // sort
  prepare_sort3d(particle, Nx, Ny, Nz);
  particle.sort();

  // check result
  REQUIRE(check_sort3d(particle, Nx, Ny, Nz) == true);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
