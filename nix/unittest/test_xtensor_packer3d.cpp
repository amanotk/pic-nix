// -*- C++ -*-

#include "xtensor/xtensor_packer3d.hpp"
#include "xtensor/xtensor_particle.hpp"

#include "xtensor/field_layout.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace nix;

struct MockData {
  int Lbx;
  int Ubx;
  int Lby;
  int Uby;
  int Lbz;
  int Ubz;
};

template <typename T>
std::vector<T> gather_expected_5d(const xt::xtensor<T, 5>& x, const MockData& data)
{
  std::vector<T> expected;
  for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        for (size_t c1 = 0; c1 < x.shape(3); ++c1) {
          for (size_t c2 = 0; c2 < x.shape(4); ++c2) {
            expected.push_back(x(iz, iy, ix, c1, c2));
          }
        }
      }
    }
  }
  return expected;
}

template <typename T>
std::vector<T> gather_expected_4d(const xt::xtensor<T, 4>& x, size_t size_z, size_t size_y,
                                  size_t size_x, size_t size_c)
{
  std::vector<T> expected;
  for (size_t iz = 0; iz < size_z; ++iz) {
    for (size_t iy = 0; iy < size_y; ++iy) {
      for (size_t ix = 0; ix < size_x; ++ix) {
        for (size_t ic = 0; ic < size_c; ++ic) {
          expected.push_back(x(iz, iy, ix, ic));
        }
      }
    }
  }
  return expected;
}

void set_particle_id(XtensorParticle& particle, int ip, int64 id64)
{
  std::memcpy(&particle.xu(ip, 6), &id64, sizeof(int64));
}

TEST_CASE("XtensorPacker3D pack_coordinate packs slice and reports size")
{
  XtensorPacker3D         packer;
  xt::xtensor<float64, 1> coord = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  int                     Lb    = 2;
  int                     Ub    = 6;
  int                     addr  = 5;

  const size_t expected_count = addr + sizeof(float64) * (Ub - Lb + 1);
  REQUIRE(packer.pack_coordinate(Lb, Ub, coord, nullptr, addr) == expected_count);

  std::vector<uint8_t> buffer(expected_count, 0);
  REQUIRE(packer.pack_coordinate(Lb, Ub, coord, buffer.data(), addr) == expected_count);

  auto* out = reinterpret_cast<float64*>(buffer.data() + addr);
  for (int i = 0; i <= Ub - Lb; ++i) {
    REQUIRE(out[i] == coord(Lb + i));
  }
}

TEST_CASE("XtensorPacker3D pack_array_raw packs strided region")
{
  XtensorPacker3D         packer;
  xt::xtensor<float64, 5> x({2, 3, 4, 2, 2});
  for (size_t iz = 0; iz < x.shape(0); ++iz) {
    for (size_t iy = 0; iy < x.shape(1); ++iy) {
      for (size_t ix = 0; ix < x.shape(2); ++ix) {
        for (size_t c1 = 0; c1 < x.shape(3); ++c1) {
          for (size_t c2 = 0; c2 < x.shape(4); ++c2) {
            x(iz, iy, ix, c1, c2) =
                static_cast<float64>(iz * 10000 + iy * 1000 + ix * 100 + c1 * 10 + c2);
          }
        }
      }
    }
  }

  MockData     data{1, 3, 1, 2, 0, 1};
  const size_t expected_count = sizeof(float64) * (2 * 2 * 3 * 2 * 2);

  REQUIRE(packer.pack_array_raw(x, data, nullptr, 0) == expected_count);

  std::vector<uint8_t> buffer(expected_count, 0);
  REQUIRE(packer.pack_array_raw(x, data, buffer.data(), 0) == expected_count);

  auto  expected = gather_expected_5d(x, data);
  auto* out      = reinterpret_cast<float64*>(buffer.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(out[i] == expected[i]);
  }
}

TEST_CASE("XtensorPacker3D pack_field colocates 1D field components")
{
  XtensorPacker3D         packer;
  xt::xtensor<float64, 4> x({1, 1, 4, 6});
  for (size_t ix = 0; ix < x.shape(2); ++ix) {
    for (size_t c = 0; c < x.shape(3); ++c) {
      x(0, 0, ix, c) = static_cast<float64>(ix * 10 + c);
    }
  }

  MockData     data{0, 2, 0, 0, 0, 0};
  const size_t expected_count = sizeof(float64) * 1 * 1 * 3 * 6;

  std::vector<uint8_t> buffer(expected_count, 0);
  REQUIRE(packer.pack_field(x, data, 1, buffer.data(), 0) == expected_count);

  xt::xtensor<float64, 4> colocated({1, 1, 3, 6});
  for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
    colocated(0, 0, ix, 0) = 0.5 * (x(0, 0, ix, 0) + x(0, 0, ix + 1, 0));
    colocated(0, 0, ix, 1) = x(0, 0, ix, 1);
    colocated(0, 0, ix, 2) = x(0, 0, ix, 2);
    colocated(0, 0, ix, 3) = x(0, 0, ix, 3);
    colocated(0, 0, ix, 4) = 0.5 * (x(0, 0, ix, 4) + x(0, 0, ix + 1, 4));
    colocated(0, 0, ix, 5) = 0.5 * (x(0, 0, ix, 5) + x(0, 0, ix + 1, 5));
  }

  auto  expected = gather_expected_4d(colocated, 1, 1, 3, 6);
  auto* out      = reinterpret_cast<float64*>(buffer.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(out[i] == Catch::Approx(expected[i]));
  }
}

TEST_CASE("XtensorPacker3D pack_field colocates 2D field components")
{
  XtensorPacker3D         packer;
  xt::xtensor<float64, 4> x({1, 3, 3, 6});
  for (size_t iy = 0; iy < x.shape(1); ++iy) {
    for (size_t ix = 0; ix < x.shape(2); ++ix) {
      for (size_t c = 0; c < x.shape(3); ++c) {
        x(0, iy, ix, c) = static_cast<float64>(iy * 100 + ix * 10 + c);
      }
    }
  }

  MockData     data{0, 1, 0, 1, 0, 0};
  const size_t expected_count = sizeof(float64) * 1 * 2 * 2 * 6;

  std::vector<uint8_t> buffer(expected_count, 0);
  REQUIRE(packer.pack_field(x, data, 1, buffer.data(), 0) == expected_count);

  xt::xtensor<float64, 4> colocated({1, 2, 2, 6});
  for (int iy = data.Lby; iy <= data.Uby; ++iy) {
    for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
      colocated(0, iy, ix, 0) = 0.5 * (x(0, iy, ix, 0) + x(0, iy, ix + 1, 0));
      colocated(0, iy, ix, 1) = 0.5 * (x(0, iy, ix, 1) + x(0, iy + 1, ix, 1));
      colocated(0, iy, ix, 2) = x(0, iy, ix, 2);
      colocated(0, iy, ix, 3) = 0.5 * (x(0, iy, ix, 3) + x(0, iy + 1, ix, 3));
      colocated(0, iy, ix, 4) = 0.5 * (x(0, iy, ix, 4) + x(0, iy, ix + 1, 4));
      colocated(0, iy, ix, 5) = 0.25 * (x(0, iy, ix, 5) + x(0, iy + 1, ix + 1, 5) +
                                        x(0, iy, ix + 1, 5) + x(0, iy + 1, ix, 5));
    }
  }

  auto  expected = gather_expected_4d(colocated, 1, 2, 2, 6);
  auto* out      = reinterpret_cast<float64*>(buffer.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(out[i] == Catch::Approx(expected[i]));
  }
}

TEST_CASE("XtensorPacker3D pack_field colocates 3D field components")
{
  XtensorPacker3D         packer;
  xt::xtensor<float64, 4> x({3, 3, 3, 6});
  for (size_t iz = 0; iz < x.shape(0); ++iz) {
    for (size_t iy = 0; iy < x.shape(1); ++iy) {
      for (size_t ix = 0; ix < x.shape(2); ++ix) {
        for (size_t c = 0; c < x.shape(3); ++c) {
          x(iz, iy, ix, c) = static_cast<float64>(iz * 1000 + iy * 100 + ix * 10 + c);
        }
      }
    }
  }

  MockData     data{0, 1, 0, 1, 0, 1};
  const size_t expected_count = sizeof(float64) * 2 * 2 * 2 * 6;

  std::vector<uint8_t> buffer(expected_count, 0);
  REQUIRE(packer.pack_field(x, data, 1, buffer.data(), 0) == expected_count);

  xt::xtensor<float64, 4> colocated({2, 2, 2, 6});
  for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
    for (int iy = data.Lby; iy <= data.Uby; ++iy) {
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        colocated(iz, iy, ix, 0) = 0.5 * (x(iz, iy, ix, 0) + x(iz, iy, ix + 1, 0));
        colocated(iz, iy, ix, 1) = 0.5 * (x(iz, iy, ix, 1) + x(iz, iy + 1, ix, 1));
        colocated(iz, iy, ix, 2) = 0.5 * (x(iz, iy, ix, 2) + x(iz + 1, iy, ix, 2));
        colocated(iz, iy, ix, 3) = 0.25 * (x(iz, iy, ix, 3) + x(iz + 1, iy + 1, ix, 3) +
                                           x(iz, iy + 1, ix, 3) + x(iz + 1, iy, ix, 3));
        colocated(iz, iy, ix, 4) = 0.25 * (x(iz, iy, ix, 4) + x(iz + 1, iy, ix + 1, 4) +
                                           x(iz + 1, iy, ix, 4) + x(iz, iy, ix + 1, 4));
        colocated(iz, iy, ix, 5) = 0.25 * (x(iz, iy, ix, 5) + x(iz, iy + 1, ix + 1, 5) +
                                           x(iz, iy, ix + 1, 5) + x(iz, iy + 1, ix, 5));
      }
    }
  }

  auto  expected = gather_expected_4d(colocated, 2, 2, 2, 6);
  auto* out      = reinterpret_cast<float64*>(buffer.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(out[i] == Catch::Approx(expected[i]));
  }
}

TEST_CASE("XtensorPacker3D compact colocation excludes ghost cells")
{
  XtensorPacker3D         packer;
  xt::xtensor<float64, 4> x({5, 6, 7, 6});
  for (size_t iz = 0; iz < x.shape(0); ++iz) {
    for (size_t iy = 0; iy < x.shape(1); ++iy) {
      for (size_t ix = 0; ix < x.shape(2); ++ix) {
        for (size_t c = 0; c < x.shape(3); ++c) {
          x(iz, iy, ix, c) = static_cast<float64>(iz * 1000 + iy * 100 + ix * 10 + c);
        }
      }
    }
  }

  MockData data{1, 5, 1, 4, 1, 3};
  auto     y = packer.colocate_field(x, data);

  REQUIRE(y.shape() == std::array<std::size_t, 4>{3, 4, 5, 6});
  REQUIRE(y(0, 0, 0, 0) == Catch::Approx(0.5 * (x(1, 1, 1, 0) + x(1, 1, 2, 0))));
  REQUIRE(y(2, 3, 4, 5) ==
          Catch::Approx(0.25 * (x(3, 4, 5, 5) + x(3, 5, 6, 5) + x(3, 4, 6, 5) + x(3, 5, 5, 5))));
}

TEST_CASE("XtensorPacker3D compact colocation handles current in 1D 2D and 3D")
{
  XtensorPacker3D packer;

  SECTION("1D")
  {
    xt::xtensor<float64, 4> x({3, 3, 6, 4});
    for (size_t ix = 0; ix < x.shape(2); ++ix) {
      x(1, 1, ix, 0) = 10.0 + ix;
      x(1, 1, ix, 1) = 20.0 + ix;
      x(1, 1, ix, 2) = 30.0 + ix;
      x(1, 1, ix, 3) = 40.0 + ix;
    }

    auto y = packer.colocate_current(x, MockData{1, 4, 1, 1, 1, 1});
    REQUIRE(y.shape() == std::array<std::size_t, 4>{1, 1, 4, 4});
    REQUIRE(y(0, 0, 0, 0) == 11.0);
    REQUIRE(y(0, 0, 0, 1) == 21.5);
    REQUIRE(y(0, 0, 0, 2) == 31.0);
    REQUIRE(y(0, 0, 0, 3) == 41.0);
  }

  SECTION("2D")
  {
    xt::xtensor<float64, 4> x({3, 5, 5, 4});
    for (size_t iy = 0; iy < x.shape(1); ++iy) {
      for (size_t ix = 0; ix < x.shape(2); ++ix) {
        x(1, iy, ix, 0) = 10.0 + iy * 10.0 + ix;
        x(1, iy, ix, 1) = 20.0 + iy * 10.0 + ix;
        x(1, iy, ix, 2) = 30.0 + iy * 10.0 + ix;
        x(1, iy, ix, 3) = 40.0 + iy * 10.0 + ix;
      }
    }

    auto y = packer.colocate_current(x, MockData{1, 3, 1, 3, 1, 1});
    REQUIRE(y.shape() == std::array<std::size_t, 4>{1, 3, 3, 4});
    REQUIRE(y(0, 0, 0, 1) == 31.5);
    REQUIRE(y(0, 0, 0, 2) == 46.0);
    REQUIRE(y(0, 0, 0, 3) == 51.0);
  }

  SECTION("3D")
  {
    xt::xtensor<float64, 4> x({5, 5, 5, 4});
    for (size_t iz = 0; iz < x.shape(0); ++iz) {
      for (size_t iy = 0; iy < x.shape(1); ++iy) {
        for (size_t ix = 0; ix < x.shape(2); ++ix) {
          x(iz, iy, ix, 0) = 10.0 + iz * 100.0 + iy * 10.0 + ix;
          x(iz, iy, ix, 1) = 20.0 + iz * 100.0 + iy * 10.0 + ix;
          x(iz, iy, ix, 2) = 30.0 + iz * 100.0 + iy * 10.0 + ix;
          x(iz, iy, ix, 3) = 40.0 + iz * 100.0 + iy * 10.0 + ix;
        }
      }
    }

    auto y = packer.colocate_current(x, MockData{1, 3, 1, 3, 1, 3});
    REQUIRE(y.shape() == std::array<std::size_t, 4>{3, 3, 3, 4});
    REQUIRE(y(0, 0, 0, 1) == 131.5);
    REQUIRE(y(0, 0, 0, 2) == 146.0);
    REQUIRE(y(0, 0, 0, 3) == 201.0);
  }
}

TEST_CASE("PIC-NIX field layout metadata identifies staggered components")
{
  REQUIRE(picnix::insitu::raw_schema_version == 1);
  REQUIRE(picnix::insitu::uf_components[0].name == "Ex");
  REQUIRE(picnix::insitu::uf_components[0].association == "x-face");
  REQUIRE(picnix::insitu::uf_components[0].normalized_xyz == std::array<double, 3>{0.0, 0.5, 0.5});
  REQUIRE(picnix::insitu::uj_components[0].association == "cell");
  REQUIRE(picnix::insitu::um_components[13] == "zx");
}

TEST_CASE("XtensorPacker3D pack_moment decimates by averaging blocks")
{
  XtensorPacker3D         packer;
  xt::xtensor<float64, 4> x({4, 4, 4, 1});
  for (size_t iz = 0; iz < x.shape(0); ++iz) {
    for (size_t iy = 0; iy < x.shape(1); ++iy) {
      for (size_t ix = 0; ix < x.shape(2); ++ix) {
        x(iz, iy, ix, 0) = static_cast<float64>(iz * 100 + iy * 10 + ix);
      }
    }
  }

  MockData     data{0, 3, 0, 3, 0, 3};
  const int    decimate       = 2;
  const size_t expected_count = sizeof(float64) * 2 * 2 * 2 * 1;

  std::vector<uint8_t> buffer(expected_count, 0);
  REQUIRE(packer.pack_moment(x, data, decimate, buffer.data(), 0) == expected_count);

  xt::xtensor<float64, 4> expected_view({2, 2, 2, 1});
  for (int bz = 0; bz < 2; ++bz) {
    for (int by = 0; by < 2; ++by) {
      for (int bx = 0; bx < 2; ++bx) {
        float64 sum = 0.0;
        for (int iz = 0; iz < 2; ++iz) {
          for (int iy = 0; iy < 2; ++iy) {
            for (int ix = 0; ix < 2; ++ix) {
              sum += x(bz * 2 + iz, by * 2 + iy, bx * 2 + ix, 0);
            }
          }
        }
        expected_view(bz, by, bx, 0) = sum / 8.0;
      }
    }
  }

  auto  expected = gather_expected_4d(expected_view, 2, 2, 2, 1);
  auto* out      = reinterpret_cast<float64*>(buffer.data());
  for (size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(out[i] == Catch::Approx(expected[i]));
  }
}

TEST_CASE("XtensorPacker3D pack_particle respects index order")
{
  XtensorPacker3D packer;
  auto            particle = std::make_shared<XtensorParticle>();
  particle->allocate(3, true);
  particle->Np = 3;

  for (int ip = 0; ip < 3; ++ip) {
    for (int c = 0; c < Particle::Nc; ++c) {
      particle->xu(ip, c) = static_cast<float64>(ip * 10 + c);
    }
  }

  xt::xtensor<int32, 1> index          = {2, 0};
  const size_t          expected_count = 2 * Particle::get_particle_size();

  std::vector<uint8_t> buffer(expected_count, 0);
  REQUIRE(packer.pack_particle(particle, index, buffer.data(), 0) == expected_count);

  auto* out = reinterpret_cast<float64*>(buffer.data());
  for (int i = 0; i < 2; ++i) {
    int ip = index(i);
    for (int c = 0; c < Particle::Nc; ++c) {
      REQUIRE(out[i * Particle::Nc + c] == particle->xu(ip, c));
    }
  }
}

TEST_CASE("XtensorPacker3D pack_tracer packs negative IDs only")
{
  XtensorPacker3D packer;
  auto            particle = std::make_shared<XtensorParticle>();
  particle->allocate(3, true);
  particle->Np = 3;

  for (int ip = 0; ip < 3; ++ip) {
    for (int c = 0; c < Particle::Nc; ++c) {
      particle->xu(ip, c) = static_cast<float64>(ip * 100 + c);
    }
  }

  set_particle_id(*particle, 0, -1);
  set_particle_id(*particle, 1, 42);
  set_particle_id(*particle, 2, -7);

  const size_t         expected_count = 2 * Particle::get_particle_size();
  std::vector<uint8_t> buffer(expected_count, 0);
  REQUIRE(packer.pack_tracer(particle, buffer.data(), 0) == expected_count);

  auto* out = buffer.data();
  REQUIRE(std::memcmp(out, &particle->xu(0, 0), Particle::get_particle_size()) == 0);
  REQUIRE(std::memcmp(out + Particle::get_particle_size(), &particle->xu(2, 0),
                      Particle::get_particle_size()) == 0);
}
