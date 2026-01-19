// -*- C++ -*-

#include "pic_chunk.hpp"
#include "pic_engine.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{
json make_config(int order, float64 friedman)
{
  json config;
  config["option"]                  = json::object();
  config["option"]["vectorization"] = "scalar";
  config["option"]["order"]         = order;
  config["option"]["pusher"]        = "Boris";
  config["option"]["interpolation"] = "MC";
  config["option"]["seed_type"]     = "fixed";
  config["option"]["friedman"]      = friedman;
  config["option"]["cell_load"]     = 1.0;
  config["option"]["buffer_ratio"]  = 0.2;
  return config;
}

void setup_chunk(PicChunk& chunk, const nix::Dims3D& dims, int order, float64 friedman)
{
  int offset[3] = {0, 0, 0};
  int gdims[3]  = {dims[0], dims[1], dims[2]};

  chunk.set_global_context(offset, gdims);
  chunk.set_coordinate(1.0, 1.0, 1.0);
  auto config = make_config(order, friedman);
  chunk.setup(config);
  chunk.allocate();

  auto data = chunk.get_internal_data();
  data.cc   = 1.0;
}

template <typename T>
void fill_sequence(T& array, typename T::value_type base, typename T::value_type step)
{
  auto* ptr = array.data();
  for (size_t i = 0; i < array.size(); i++) {
    ptr[i] = base + step * static_cast<typename T::value_type>(i);
  }
}

template <typename T>
void require_allclose(const T& actual, const T& expected, float64 rtol, float64 atol)
{
  REQUIRE(actual.size() == expected.size());
  auto* a = actual.data();
  auto* b = expected.data();
  for (size_t i = 0; i < actual.size(); i++) {
    const float64 diff = std::abs(static_cast<float64>(a[i]) - static_cast<float64>(b[i]));
    const float64 tol  = atol + rtol * std::abs(static_cast<float64>(b[i]));
    INFO("index " << i);
    REQUIRE(diff <= tol);
  }
}

struct SmokeParticleDiagnostics {
  float64 sum_rho;
  float64 sum_jx;
  float64 sum_jy;
  float64 sum_jz;
  float64 sum_x;
  float64 sum_y;
  float64 sum_z;
  float64 sum_ux;
  float64 sum_uy;
  float64 sum_uz;
};

struct SmokeFieldDiagnostics {
  float64 e_energy;
  float64 b_energy;
};

SmokeFieldDiagnostics compute_smoke_field(const PicChunk::DataContainer& data)
{
  SmokeFieldDiagnostics diag = {};

  for (int iz = data.Lbz; iz <= data.Ubz; iz++) {
    for (int iy = data.Lby; iy <= data.Uby; iy++) {
      for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
        const float64 ex = data.uf(iz, iy, ix, 0);
        const float64 ey = data.uf(iz, iy, ix, 1);
        const float64 ez = data.uf(iz, iy, ix, 2);
        const float64 bx = data.uf(iz, iy, ix, 3);
        const float64 by = data.uf(iz, iy, ix, 4);
        const float64 bz = data.uf(iz, iy, ix, 5);
        diag.e_energy += 0.5 * (ex * ex + ey * ey + ez * ez);
        diag.b_energy += 0.5 * (bx * bx + by * by + bz * bz);
      }
    }
  }

  return diag;
}

SmokeParticleDiagnostics compute_smoke_particle(const PicChunk::DataContainer& data)
{
  SmokeParticleDiagnostics diag = {};

  for (int iz = data.Lbz; iz <= data.Ubz; iz++) {
    for (int iy = data.Lby; iy <= data.Uby; iy++) {
      for (int ix = data.Lbx; ix <= data.Ubx; ix++) {
        diag.sum_rho += data.uj(iz, iy, ix, 0);
        diag.sum_jx += data.uj(iz, iy, ix, 1);
        diag.sum_jy += data.uj(iz, iy, ix, 2);
        diag.sum_jz += data.uj(iz, iy, ix, 3);
      }
    }
  }

  for (const auto& species : data.up) {
    for (int ip = 0; ip < species->Np; ip++) {
      diag.sum_x += species->xu(ip, 0);
      diag.sum_y += species->xu(ip, 1);
      diag.sum_z += species->xu(ip, 2);
      diag.sum_ux += species->xu(ip, 3);
      diag.sum_uy += species->xu(ip, 4);
      diag.sum_uz += species->xu(ip, 5);
    }
  }

  return diag;
}

json smoke_field_to_json(const SmokeFieldDiagnostics& diag)
{
  return {
      {"e_energy", diag.e_energy},
      {"b_energy", diag.b_energy},
  };
}

json smoke_particle_to_json(const SmokeParticleDiagnostics& diag)
{
  return {
      {"sum_rho", diag.sum_rho}, {"sum_jx", diag.sum_jx}, {"sum_jy", diag.sum_jy},
      {"sum_jz", diag.sum_jz},   {"sum_x", diag.sum_x},   {"sum_y", diag.sum_y},
      {"sum_z", diag.sum_z},     {"sum_ux", diag.sum_ux}, {"sum_uy", diag.sum_uy},
      {"sum_uz", diag.sum_uz},
  };
}

SmokeFieldDiagnostics smoke_field_from_json(const json& j)
{
  SmokeFieldDiagnostics diag = {};
  diag.e_energy              = j.at("e_energy").get<float64>();
  diag.b_energy              = j.at("b_energy").get<float64>();
  return diag;
}

SmokeParticleDiagnostics smoke_particle_from_json(const json& j)
{
  SmokeParticleDiagnostics diag = {};
  diag.sum_rho                  = j.at("sum_rho").get<float64>();
  diag.sum_jx                   = j.at("sum_jx").get<float64>();
  diag.sum_jy                   = j.at("sum_jy").get<float64>();
  diag.sum_jz                   = j.at("sum_jz").get<float64>();
  diag.sum_x                    = j.at("sum_x").get<float64>();
  diag.sum_y                    = j.at("sum_y").get<float64>();
  diag.sum_z                    = j.at("sum_z").get<float64>();
  diag.sum_ux                   = j.at("sum_ux").get<float64>();
  diag.sum_uy                   = j.at("sum_uy").get<float64>();
  diag.sum_uz                   = j.at("sum_uz").get<float64>();
  return diag;
}

std::filesystem::path smoke_field_golden_path(const std::string& tag)
{
  return std::filesystem::path(__FILE__).parent_path() / "data" /
         ("pic_chunk_smoke_" + tag + "_field.msgpack");
}

std::filesystem::path smoke_particle_golden_path(const std::string& tag)
{
  return std::filesystem::path(__FILE__).parent_path() / "data" /
         ("pic_chunk_smoke_" + tag + "_particle.msgpack");
}

void write_smoke_field_golden(const std::filesystem::path& path, const SmokeFieldDiagnostics& diag)
{
  std::filesystem::create_directories(path.parent_path());
  json                 payload = smoke_field_to_json(diag);
  std::vector<uint8_t> msgpack = json::to_msgpack(payload);
  std::ofstream        ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(msgpack.data()),
            static_cast<std::streamsize>(msgpack.size()));
}

void write_smoke_particle_golden(const std::filesystem::path&    path,
                                 const SmokeParticleDiagnostics& diag)
{
  std::filesystem::create_directories(path.parent_path());
  json                 payload = smoke_particle_to_json(diag);
  std::vector<uint8_t> msgpack = json::to_msgpack(payload);
  std::ofstream        ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(msgpack.data()),
            static_cast<std::streamsize>(msgpack.size()));
}

SmokeFieldDiagnostics read_smoke_field_golden(const std::filesystem::path& path)
{
  std::ifstream        ifs(path, std::ios::binary);
  std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
  json                 payload = json::from_msgpack(buffer);
  return smoke_field_from_json(payload);
}

SmokeParticleDiagnostics read_smoke_particle_golden(const std::filesystem::path& path)
{
  std::ifstream        ifs(path, std::ios::binary);
  std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
  json                 payload = json::from_msgpack(buffer);
  return smoke_particle_from_json(payload);
}

struct SmokeCase {
  std::string                                         tag;
  nix::Dims3D                                         dims;
  nix::Bool3D                                         has_dim;
  std::array<float64, 6>                              field;
  std::vector<std::array<float64, nix::Particle::Nc>> particles;
  float64                                             delt;
};

void apply_uniform_field(PicChunk::DataContainer& data, const std::array<float64, 6>& field)
{
  auto shape = data.uf.shape();
  for (size_t iz = 0; iz < shape[0]; iz++) {
    for (size_t iy = 0; iy < shape[1]; iy++) {
      for (size_t ix = 0; ix < shape[2]; ix++) {
        data.uf(iz, iy, ix, 0) = field[0];
        data.uf(iz, iy, ix, 1) = field[1];
        data.uf(iz, iy, ix, 2) = field[2];
        data.uf(iz, iy, ix, 3) = field[3];
        data.uf(iz, iy, ix, 4) = field[4];
        data.uf(iz, iy, ix, 5) = field[5];
      }
    }
  }
}

void init_particles(PicChunk& chunk, PicChunk::DataContainer& data,
                    const std::vector<std::array<float64, nix::Particle::Nc>>& particles)
{
  data.up.resize(1);
  data.up[0]     = std::make_shared<ParticleType>(static_cast<int>(particles.size()), chunk);
  data.up[0]->q  = 1.0;
  data.up[0]->m  = 1.0;
  data.up[0]->Np = static_cast<int>(particles.size());

  for (int ip = 0; ip < data.up[0]->Np; ip++) {
    for (int c = 0; c < nix::Particle::Nc; c++) {
      data.up[0]->xu(ip, c) = particles[ip][c];
      data.up[0]->xv(ip, c) = particles[ip][c];
    }
  }
}

void run_smoke_step(PicChunk& chunk, float64 delt)
{
  chunk.push_velocity(delt);
  chunk.push_position(delt);
  chunk.deposit_current(delt);
  chunk.push_efd(delt);
  chunk.push_bfd(delt);
}

void compare_smoke(const std::string& tag, const SmokeFieldDiagnostics& field_diag,
                   const SmokeParticleDiagnostics& particle_diag)
{
  const auto field_golden_path    = smoke_field_golden_path(tag);
  const auto particle_golden_path = smoke_particle_golden_path(tag);
  const bool update_golden        = std::getenv("PICNIX_UPDATE_GOLDEN") != nullptr;
  if (update_golden) {
    write_smoke_field_golden(field_golden_path, field_diag);
    write_smoke_particle_golden(particle_golden_path, particle_diag);
    SUCCEED("Updated smoke golden data");
    return;
  }

  if (!std::filesystem::exists(field_golden_path)) {
    FAIL("Missing field smoke golden data; set PICNIX_UPDATE_GOLDEN=1 to generate it");
  }
  if (!std::filesystem::exists(particle_golden_path)) {
    FAIL("Missing particle smoke golden data; set PICNIX_UPDATE_GOLDEN=1 to generate it");
  }

  const SmokeFieldDiagnostics    expected_field = read_smoke_field_golden(field_golden_path);
  const SmokeParticleDiagnostics expected_particle =
      read_smoke_particle_golden(particle_golden_path);

  INFO("sum_rho=" << particle_diag.sum_rho);
  INFO("sum_jx=" << particle_diag.sum_jx);
  INFO("sum_jy=" << particle_diag.sum_jy);
  INFO("sum_jz=" << particle_diag.sum_jz);
  INFO("e_energy=" << field_diag.e_energy);
  INFO("b_energy=" << field_diag.b_energy);
  INFO("sum_x=" << particle_diag.sum_x);
  INFO("sum_y=" << particle_diag.sum_y);
  INFO("sum_z=" << particle_diag.sum_z);
  INFO("sum_ux=" << particle_diag.sum_ux);
  INFO("sum_uy=" << particle_diag.sum_uy);
  INFO("sum_uz=" << particle_diag.sum_uz);

  auto approx = [](float64 value) { return Catch::Approx(value).epsilon(1.0e-5).margin(1.0e-9); };

  REQUIRE(field_diag.e_energy == approx(expected_field.e_energy));
  REQUIRE(field_diag.b_energy == approx(expected_field.b_energy));
  REQUIRE(particle_diag.sum_rho == approx(expected_particle.sum_rho));
  REQUIRE(particle_diag.sum_jx == approx(expected_particle.sum_jx));
  REQUIRE(particle_diag.sum_jy == approx(expected_particle.sum_jy));
  REQUIRE(particle_diag.sum_jz == approx(expected_particle.sum_jz));
  REQUIRE(particle_diag.sum_x == approx(expected_particle.sum_x));
  REQUIRE(particle_diag.sum_y == approx(expected_particle.sum_y));
  REQUIRE(particle_diag.sum_z == approx(expected_particle.sum_z));
  REQUIRE(particle_diag.sum_ux == approx(expected_particle.sum_ux));
  REQUIRE(particle_diag.sum_uy == approx(expected_particle.sum_uy));
  REQUIRE(particle_diag.sum_uz == approx(expected_particle.sum_uz));
}

void run_smoke_case(const SmokeCase& smoke_case)
{
  PicChunk chunk(smoke_case.dims, smoke_case.has_dim, 0);
  setup_chunk(chunk, smoke_case.dims, 1, 0.0);

  auto data = chunk.get_internal_data();
  data.cc   = 1.0;

  data.uf.fill(0.0);
  data.uj.fill(0.0);
  data.ff.fill(0.0);

  apply_uniform_field(data, smoke_case.field);
  init_particles(chunk, data, smoke_case.particles);

  run_smoke_step(chunk, smoke_case.delt);

  auto field_diag    = compute_smoke_field(data);
  auto particle_diag = compute_smoke_particle(data);
  compare_smoke(smoke_case.tag, field_diag, particle_diag);
}
} // namespace

TEST_CASE("PicChunk pack/unpack round-trip")
{
  nix::Dims3D dims    = {1, 1, 8};
  nix::Bool3D has_dim = {false, false, true};

  PicChunk chunk(dims, has_dim, 0);
  setup_chunk(chunk, dims, 1, 0.0);

  auto data = chunk.get_internal_data();
  data.cc   = 2.0;

  fill_sequence(data.uf, static_cast<float64>(0.1), static_cast<float64>(0.01));
  fill_sequence(data.uj, static_cast<float64>(0.2), static_cast<float64>(0.02));
  fill_sequence(data.ff, static_cast<float64>(0.3), static_cast<float64>(0.04));

  data.up.resize(1);
  data.up[0]     = std::make_shared<ParticleType>(2, chunk);
  data.up[0]->q  = 1.5;
  data.up[0]->m  = 2.5;
  data.up[0]->Np = 2;

  fill_sequence(data.up[0]->xu, static_cast<float64>(0.1), static_cast<float64>(0.05));
  fill_sequence(data.up[0]->xv, static_cast<float64>(-0.2), static_cast<float64>(0.04));
  fill_sequence(data.up[0]->gindex, static_cast<int32>(0), static_cast<int32>(1));
  fill_sequence(data.up[0]->pindex, static_cast<int32>(0), static_cast<int32>(2));
  fill_sequence(data.up[0]->pcount, static_cast<int32>(0), static_cast<int32>(1));

  std::vector<char> buffer(4 * 1024 * 1024, 0);
  int               byte1 = chunk.pack(buffer.data(), 0);

  PicChunk unpack(dims, has_dim, 0);
  int      byte2 = unpack.unpack(buffer.data(), 0);

  REQUIRE(byte1 == byte2);
  REQUIRE(unpack.get_order() == chunk.get_order());

  auto udata = unpack.get_internal_data();
  REQUIRE(udata.Ns == data.Ns);
  REQUIRE(udata.cc == Catch::Approx(data.cc));

  require_allclose(udata.uf, data.uf, 1.0e-12, 1.0e-14);
  require_allclose(udata.uj, data.uj, 1.0e-12, 1.0e-14);
  require_allclose(udata.ff, data.ff, 1.0e-12, 1.0e-14);

  REQUIRE(udata.up.size() == data.up.size());
  REQUIRE(udata.up[0]->Np == data.up[0]->Np);
  REQUIRE(udata.up[0]->Np_total == data.up[0]->Np_total);
  REQUIRE(udata.up[0]->q == Catch::Approx(data.up[0]->q));
  REQUIRE(udata.up[0]->m == Catch::Approx(data.up[0]->m));

  require_allclose(udata.up[0]->xu, data.up[0]->xu, 1.0e-12, 1.0e-14);
  require_allclose(udata.up[0]->xv, data.up[0]->xv, 1.0e-12, 1.0e-14);
  require_allclose(udata.up[0]->gindex, data.up[0]->gindex, 0.0, 0.0);
  require_allclose(udata.up[0]->pindex, data.up[0]->pindex, 0.0, 0.0);
  require_allclose(udata.up[0]->pcount, data.up[0]->pcount, 0.0, 0.0);
}

TEST_CASE("PicChunk integration smoke 1D")
{
  nix::Dims3D dims    = {1, 1, 8};
  nix::Bool3D has_dim = {false, false, true};

  std::array<float64, 6>                              field = {0.06, 0.02, -0.01, 0.0, 0.04, 0.01};
  std::vector<std::array<float64, nix::Particle::Nc>> particles = {
      {1.3, 0.0, 0.0, 0.04, 0.02, 0.0, -1.0},
      {5.4, 0.0, 0.0, -0.03, 0.01, 0.0, -2.0},
  };

  run_smoke_case({"1d", dims, has_dim, field, particles, 0.1});
}

TEST_CASE("PicChunk integration smoke 2D")
{
  nix::Dims3D dims    = {1, 4, 4};
  nix::Bool3D has_dim = {false, true, true};

  std::array<float64, 6>                              field = {0.1, -0.05, 0.02, 0.0, 0.03, -0.04};
  std::vector<std::array<float64, nix::Particle::Nc>> particles = {
      {1.2, 1.6, 0.0, 0.05, -0.02, 0.0, -1.0},
      {2.7, 0.8, 0.0, -0.03, 0.06, 0.0, -2.0},
  };

  run_smoke_case({"2d", dims, has_dim, field, particles, 0.1});
}

TEST_CASE("PicChunk integration smoke 3D")
{
  nix::Dims3D dims    = {3, 3, 3};
  nix::Bool3D has_dim = {true, true, true};

  std::array<float64, 6> field = {0.02, -0.01, 0.03, 0.01, -0.02, 0.04};
  std::vector<std::array<float64, nix::Particle::Nc>> particles = {
      {0.8, 1.2, 1.5, 0.02, -0.01, 0.03, -1.0},
      {1.7, 0.4, 2.1, -0.03, 0.04, -0.02, -2.0},
  };

  run_smoke_case({"3d", dims, has_dim, field, particles, 0.1});
}
