// -*- C++ -*-

#include "pic_chunk.hpp"
#include "pic_engine.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

namespace
{
struct SmokeOptions {
  std::string vectorization;
  int         order;
  std::string pusher;
  std::string interpolation;
};

struct SmokeCase {
  std::string                                         tag;
  nix::Dims3D                                         dims;
  nix::Bool3D                                         has_dim;
  std::array<float64, 6>                              field;
  std::vector<std::array<float64, nix::Particle::Nc>> particles;
  float64                                             delt;
};

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

struct SmokeDiagnostics {
  SmokeFieldDiagnostics    field;
  SmokeParticleDiagnostics particle;
};

struct SmokeState {
  SmokeDiagnostics                  diag;
  std::vector<float64>              uf;
  std::vector<float64>              uj;
  std::vector<float64>              ff;
  std::vector<std::vector<float64>> xu;
  std::vector<std::vector<float64>> xv;
  std::vector<std::vector<int32>>   gindex;
  std::vector<std::vector<int32>>   pindex;
  std::vector<std::vector<int32>>   pcount;
};

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

template <typename T>
std::vector<typename T::value_type> copy_flat(const T& array)
{
  return {array.data(), array.data() + array.size()};
}

json make_config(const SmokeOptions& options, float64 friedman)
{
  json config;
  config["option"]                  = json::object();
  config["option"]["vectorization"] = options.vectorization;
  config["option"]["order"]         = options.order;
  config["option"]["pusher"]        = options.pusher;
  config["option"]["interpolation"] = options.interpolation;
  config["option"]["seed_type"]     = "fixed";
  config["option"]["friedman"]      = friedman;
  config["option"]["cell_load"]     = 1.0;
  config["option"]["buffer_ratio"]  = 0.2;
  return config;
}

void setup_chunk(PicChunk& chunk, const nix::Dims3D& dims, const SmokeOptions& options,
                 float64 friedman)
{
  int offset[3] = {0, 0, 0};
  int gdims[3]  = {dims[0], dims[1], dims[2]};

  chunk.set_global_context(offset, gdims);
  chunk.set_coordinate(1.0, 1.0, 1.0);
  auto config = make_config(options, friedman);
  chunk.setup(config);
  chunk.allocate();

  auto data = chunk.get_internal_data();
  data.cc   = 1.0;
}

std::string smoke_tag_for_options(const SmokeOptions& options)
{
  return options.vectorization + "_o" + std::to_string(options.order) + "_" + options.pusher + "_" +
         options.interpolation;
}

std::vector<SmokeOptions> make_smoke_options()
{
  std::vector<SmokeOptions> options;
  for (const auto& vectorization : {"scalar", "vector"}) {
    for (int order = 1; order <= 4; order++) {
      for (const auto& pusher : {"Boris", "Vay", "HigueraCary"}) {
        for (const auto& interpolation : {"MC", "WT"}) {
          options.push_back({vectorization, order, pusher, interpolation});
        }
      }
    }
  }
  return options;
}

float64 sinusoidal_mode1(float64 coord, float64 min, float64 length)
{
  if (length <= 0.0) {
    return 0.0;
  }
  const float64 two_pi = 2.0 * std::acos(-1.0);
  return std::sin(two_pi * (coord - min) / length);
}

void apply_sinusoidal_field(PicChunk::DataContainer& data, const nix::Bool3D& has_dim,
                            const std::array<float64, 6>& field)
{
  const float64 ex     = field[0];
  const float64 ey     = field[1];
  const float64 ez     = field[2];
  const float64 bx_amp = field[3];
  const float64 by_amp = field[4];
  const float64 bz_amp = field[5];

  const float64 xmin = data.xlim[0];
  const float64 ymin = data.ylim[0];
  const float64 zmin = data.zlim[0];
  const float64 lx   = has_dim[2] ? data.xlim[1] - data.xlim[0] : 0.0;
  const float64 ly   = has_dim[1] ? data.ylim[1] - data.ylim[0] : 0.0;
  const float64 lz   = has_dim[0] ? data.zlim[1] - data.zlim[0] : 0.0;

  auto shape = data.uf.shape();
  for (size_t iz = 0; iz < shape[0]; iz++) {
    for (size_t iy = 0; iy < shape[1]; iy++) {
      for (size_t ix = 0; ix < shape[2]; ix++) {
        const float64 x        = xmin + (static_cast<float64>(ix) - data.Lbx + 0.5) * data.delx;
        const float64 y        = ymin + (static_cast<float64>(iy) - data.Lby + 0.5) * data.dely;
        const float64 z        = zmin + (static_cast<float64>(iz) - data.Lbz + 0.5) * data.delz;
        const float64 bx       = has_dim[2] ? bx_amp * sinusoidal_mode1(x, xmin, lx) : 0.0;
        const float64 by       = has_dim[1] ? by_amp * sinusoidal_mode1(y, ymin, ly) : 0.0;
        const float64 bz       = has_dim[0] ? bz_amp * sinusoidal_mode1(z, zmin, lz) : 0.0;
        data.uf(iz, iy, ix, 0) = ex;
        data.uf(iz, iy, ix, 1) = ey;
        data.uf(iz, iy, ix, 2) = ez;
        data.uf(iz, iy, ix, 3) = bx;
        data.uf(iz, iy, ix, 4) = by;
        data.uf(iz, iy, ix, 5) = bz;
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
  return std::filesystem::path(__FILE__).parent_path() / "testdata" /
         ("pic_chunk_smoke_" + tag + "_field.msgpack");
}

std::filesystem::path smoke_particle_golden_path(const std::string& tag)
{
  return std::filesystem::path(__FILE__).parent_path() / "testdata" /
         ("pic_chunk_smoke_" + tag + "_particle.msgpack");
}

json read_smoke_payload(const std::filesystem::path& path)
{
  std::ifstream        ifs(path, std::ios::binary);
  std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
  return json::from_msgpack(buffer);
}

void write_smoke_payload(const std::filesystem::path& path, const json& payload)
{
  std::filesystem::create_directories(path.parent_path());
  std::vector<uint8_t> msgpack = json::to_msgpack(payload);
  std::ofstream        ofs(path, std::ios::binary);
  ofs.write(reinterpret_cast<const char*>(msgpack.data()),
            static_cast<std::streamsize>(msgpack.size()));
}

void run_smoke_step(PicChunk& chunk, float64 delt)
{
  chunk.push_velocity(delt);
  chunk.push_position(delt);
  chunk.deposit_current(delt);
  chunk.push_efd(delt);
  chunk.push_bfd(delt);
}

void compare_smoke(const std::string& case_tag, const std::string& option_tag,
                   const SmokeFieldDiagnostics&    field_diag,
                   const SmokeParticleDiagnostics& particle_diag)
{
  const auto field_golden_path    = smoke_field_golden_path(case_tag);
  const auto particle_golden_path = smoke_particle_golden_path(case_tag);
  const bool update_golden        = std::getenv("PICNIX_UPDATE_GOLDEN") != nullptr;
  if (update_golden) {
    json field_payload    = json::object();
    json particle_payload = json::object();
    if (std::filesystem::exists(field_golden_path)) {
      field_payload = read_smoke_payload(field_golden_path);
      if (!field_payload.is_object()) {
        field_payload = json::object();
      }
    }
    if (std::filesystem::exists(particle_golden_path)) {
      particle_payload = read_smoke_payload(particle_golden_path);
      if (!particle_payload.is_object()) {
        particle_payload = json::object();
      }
    }
    field_payload[option_tag]    = smoke_field_to_json(field_diag);
    particle_payload[option_tag] = smoke_particle_to_json(particle_diag);
    write_smoke_payload(field_golden_path, field_payload);
    write_smoke_payload(particle_golden_path, particle_payload);
    SUCCEED("Updated smoke golden data for " << option_tag);
    return;
  }

  if (!std::filesystem::exists(field_golden_path)) {
    FAIL("Missing field smoke golden data; set PICNIX_UPDATE_GOLDEN=1 to generate it");
  }
  if (!std::filesystem::exists(particle_golden_path)) {
    FAIL("Missing particle smoke golden data; set PICNIX_UPDATE_GOLDEN=1 to generate it");
  }

  const json field_payload    = read_smoke_payload(field_golden_path);
  const json particle_payload = read_smoke_payload(particle_golden_path);
  if (!field_payload.is_object() || !particle_payload.is_object()) {
    FAIL("Smoke golden payload format has changed; set PICNIX_UPDATE_GOLDEN=1 to regenerate it");
  }
  if (!field_payload.contains(option_tag) || !particle_payload.contains(option_tag)) {
    FAIL("Missing smoke golden data for options; set PICNIX_UPDATE_GOLDEN=1 to generate it");
  }

  const SmokeFieldDiagnostics expected_field = smoke_field_from_json(field_payload.at(option_tag));
  const SmokeParticleDiagnostics expected_particle =
      smoke_particle_from_json(particle_payload.at(option_tag));

  INFO("options=" << option_tag);
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

void compare_smoke_diagnostics(const SmokeDiagnostics& actual, const SmokeDiagnostics& expected)
{
  auto approx = [](float64 value) { return Catch::Approx(value).epsilon(1.0e-12).margin(1.0e-14); };

  REQUIRE(actual.field.e_energy == approx(expected.field.e_energy));
  REQUIRE(actual.field.b_energy == approx(expected.field.b_energy));
  REQUIRE(actual.particle.sum_rho == approx(expected.particle.sum_rho));
  REQUIRE(actual.particle.sum_jx == approx(expected.particle.sum_jx));
  REQUIRE(actual.particle.sum_jy == approx(expected.particle.sum_jy));
  REQUIRE(actual.particle.sum_jz == approx(expected.particle.sum_jz));
  REQUIRE(actual.particle.sum_x == approx(expected.particle.sum_x));
  REQUIRE(actual.particle.sum_y == approx(expected.particle.sum_y));
  REQUIRE(actual.particle.sum_z == approx(expected.particle.sum_z));
  REQUIRE(actual.particle.sum_ux == approx(expected.particle.sum_ux));
  REQUIRE(actual.particle.sum_uy == approx(expected.particle.sum_uy));
  REQUIRE(actual.particle.sum_uz == approx(expected.particle.sum_uz));
}

SmokeState capture_smoke_state(const PicChunk::DataContainer& data)
{
  SmokeState state    = {};
  state.diag.field    = compute_smoke_field(data);
  state.diag.particle = compute_smoke_particle(data);
  state.uf            = copy_flat(data.uf);
  state.uj            = copy_flat(data.uj);
  state.ff            = copy_flat(data.ff);
  state.xu.reserve(data.up.size());
  state.xv.reserve(data.up.size());
  state.gindex.reserve(data.up.size());
  state.pindex.reserve(data.up.size());
  state.pcount.reserve(data.up.size());
  for (const auto& species : data.up) {
    state.xu.push_back(copy_flat(species->xu));
    state.xv.push_back(copy_flat(species->xv));
    state.gindex.push_back(copy_flat(species->gindex));
    state.pindex.push_back(copy_flat(species->pindex));
    state.pcount.push_back(copy_flat(species->pcount));
  }
  return state;
}

void compare_smoke_state(const SmokeState& actual, const SmokeState& expected)
{
  compare_smoke_diagnostics(actual.diag, expected.diag);
  require_allclose(actual.uf, expected.uf, 1.0e-12, 1.0e-14);
  require_allclose(actual.uj, expected.uj, 1.0e-12, 1.0e-14);
  require_allclose(actual.ff, expected.ff, 1.0e-12, 1.0e-14);
  REQUIRE(actual.xu.size() == expected.xu.size());
  for (size_t i = 0; i < actual.xu.size(); i++) {
    require_allclose(actual.xu[i], expected.xu[i], 1.0e-12, 1.0e-14);
    require_allclose(actual.xv[i], expected.xv[i], 1.0e-12, 1.0e-14);
    require_allclose(actual.gindex[i], expected.gindex[i], 0.0, 0.0);
    require_allclose(actual.pindex[i], expected.pindex[i], 0.0, 0.0);
    require_allclose(actual.pcount[i], expected.pcount[i], 0.0, 0.0);
  }
}

SmokeState run_smoke_case(const SmokeCase& smoke_case, const SmokeOptions& options,
                          bool compare_golden)
{
  PicChunk chunk(smoke_case.dims, smoke_case.has_dim, 0);
  setup_chunk(chunk, smoke_case.dims, options, 0.0);

  auto data = chunk.get_internal_data();
  data.cc   = 1.0;

  data.uf.fill(0.0);
  data.uj.fill(0.0);
  data.ff.fill(0.0);

  apply_sinusoidal_field(data, smoke_case.has_dim, smoke_case.field);
  init_particles(chunk, data, smoke_case.particles);
  chunk.sort_particle(data.up);

  run_smoke_step(chunk, smoke_case.delt);

  SmokeState state = capture_smoke_state(data);
  if (compare_golden) {
    compare_smoke(smoke_case.tag, smoke_tag_for_options(options), state.diag.field,
                  state.diag.particle);
  }
  return state;
}

void run_smoke_case_full_sweep(const SmokeCase& smoke_case)
{
  const auto                        options = make_smoke_options();
  std::map<std::string, SmokeState> scalar_results;
  for (const auto& option : options) {
    const std::string tag = smoke_tag_for_options(option);
    INFO("options=" << smoke_tag_for_options(option));
    const bool compare_golden = option.vectorization == "scalar";
    SmokeState diag           = run_smoke_case(smoke_case, option, compare_golden);
    if (option.vectorization == "scalar") {
      const std::string key =
          std::to_string(option.order) + "_" + option.pusher + "_" + option.interpolation;
      scalar_results[key] = diag;
      continue;
    }
    if (option.vectorization == "vector") {
      const std::string key =
          std::to_string(option.order) + "_" + option.pusher + "_" + option.interpolation;
      auto it = scalar_results.find(key);
      if (it == scalar_results.end()) {
        FAIL("Missing scalar reference for vectorization option: " + tag);
      }
      compare_smoke_state(diag, it->second);
    }
  }
}
} // namespace

TEST_CASE("PicChunk pack/unpack round-trip")
{
  nix::Dims3D dims    = {1, 1, 8};
  nix::Bool3D has_dim = {false, false, true};

  const SmokeOptions options = {"scalar", 1, "Boris", "MC"};
  PicChunk           chunk(dims, has_dim, 0);
  setup_chunk(chunk, dims, options, 0.0);

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

TEST_CASE("Maxwell divergence excludes staggered physical-boundary stencils")
{
  nix::Dims3D dims    = {1, 8, 8};
  nix::Bool3D has_dim = {false, true, true};

  const SmokeOptions options = {"scalar", 2, "Boris", "MC"};
  PicChunk           chunk(dims, has_dim, 0);
  setup_chunk(chunk, dims, options, 0.0);

  auto data = chunk.get_internal_data();
  std::fill(data.uf.begin(), data.uf.end(), 0.0);
  std::fill(data.uj.begin(), data.uj.end(), 0.0);

  const int nb = data.boundary_margin;

  // These values contaminate only the old shared loop bounds.
  data.uf(data.Lbz, data.Lby, data.Lbx + nb - 1, 3) = 1.0;
  data.uf(data.Lbz, data.Lby, data.Ubx - nb + 1, 0) = 1.0;

  float64 efd        = 0.0;
  float64 bfd        = 0.0;
  int64   ecount     = 0;
  int64   bcount     = 0;
  int     xbuffer[2] = {nb, nb};
  int     ybuffer[2] = {0, 0};

  pic_engine::BaseMaxwell maxwell(data);
  maxwell.get_diverror_2d(data.uf, data.uj, efd, bfd, ecount, bcount, xbuffer, ybuffer);

  REQUIRE(efd == Catch::Approx(0.0));
  REQUIRE(bfd == Catch::Approx(0.0));
  REQUIRE(ecount == 24);
  REQUIRE(bcount == 24);
}

TEST_CASE("Maxwell divergence is undefined without valid stencils")
{
  nix::Dims3D dims    = {1, 1, 2};
  nix::Bool3D has_dim = {false, false, true};

  const SmokeOptions options = {"scalar", 2, "Boris", "MC"};
  PicChunk           chunk(dims, has_dim, 0);
  setup_chunk(chunk, dims, options, 0.0);

  auto data = chunk.get_internal_data();
  std::fill(data.uf.begin(), data.uf.end(), 0.0);
  std::fill(data.uj.begin(), data.uj.end(), 0.0);

  float64 efd        = 0.0;
  float64 bfd        = 0.0;
  int64   ecount     = 0;
  int64   bcount     = 0;
  int     xbuffer[2] = {data.boundary_margin, data.boundary_margin};

  pic_engine::BaseMaxwell maxwell(data);
  maxwell.get_diverror_1d(data.uf, data.uj, efd, bfd, ecount, bcount, xbuffer);

  REQUIRE(efd == Catch::Approx(0.0));
  REQUIRE(bfd == Catch::Approx(0.0));
  REQUIRE(ecount == 0);
  REQUIRE(bcount == 0);
}

TEST_CASE("Maxwell 3D divergence excludes staggered x-boundary stencils")
{
  nix::Dims3D dims    = {8, 8, 8};
  nix::Bool3D has_dim = {true, true, true};

  const SmokeOptions options = {"scalar", 2, "Boris", "MC"};
  PicChunk           chunk(dims, has_dim, 0);
  setup_chunk(chunk, dims, options, 0.0);

  auto data = chunk.get_internal_data();
  std::fill(data.uf.begin(), data.uf.end(), 0.0);
  std::fill(data.uj.begin(), data.uj.end(), 0.0);

  const int nb = data.boundary_margin;

  data.uf(data.Lbz, data.Lby, data.Lbx + nb - 1, 3) = 1.0;
  data.uf(data.Lbz, data.Lby, data.Ubx - nb + 1, 0) = 1.0;

  float64 efd        = 0.0;
  float64 bfd        = 0.0;
  int64   ecount     = 0;
  int64   bcount     = 0;
  int     xbuffer[2] = {nb, nb};
  int     ybuffer[2] = {0, 0};
  int     zbuffer[2] = {0, 0};

  pic_engine::BaseMaxwell maxwell(data);
  maxwell.get_diverror_3d(data.uf, data.uj, efd, bfd, ecount, bcount, xbuffer, ybuffer, zbuffer);

  REQUIRE(efd == Catch::Approx(0.0));
  REQUIRE(bfd == Catch::Approx(0.0));
  REQUIRE(ecount == 192);
  REQUIRE(bcount == 192);
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

  run_smoke_case_full_sweep({"1d", dims, has_dim, field, particles, 0.1});
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

  run_smoke_case_full_sweep({"2d", dims, has_dim, field, particles, 0.1});
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

  run_smoke_case_full_sweep({"3d", dims, has_dim, field, particles, 0.1});
}
