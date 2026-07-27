// -*- C++ -*-
#include "hybrid_application.hpp"
#include "hybrid_chunk.hpp"

#include "engine/field.hpp"
#include "engine/moment.hpp"
#include "nix/cfgparser.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr std::array<char, 8> shared_magic   = {'H', 'Y', 'B', 'R', 'I', 'D', 'R', '6'};
constexpr std::uint32_t       shared_version = 2;

template <typename T>
T read_value(std::ifstream& input, const char* description)
{
  T value = {};
  if (!input.read(reinterpret_cast<char*>(&value), sizeof(value))) {
    throw std::runtime_error(std::string("Cannot read shared Hybrid ") + description);
  }
  return value;
}

class SharedStateApplication final : public hybrid::HybridApplication
{
public:
  using HybridApplication::HybridApplication;

protected:
  void pcc2_stage_completed(hybrid::engine::Pcc2Stage stage) override
  {
    if (!hybrid::engine::pcc2_is_field_stage(stage)) {
      return;
    }
    for (const auto& chunk_ptr : chunkvec) {
      auto data = static_cast<hybrid::HybridChunk&>(*chunk_ptr).get_internal_data();
      for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
        for (int component = 0; component < hybrid::num_field_components; ++component) {
          const auto reference = data.work_field_cell(data.Lbz, data.Lby, ix, component);
          for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
            for (int iy = data.Lby; iy <= data.Uby; ++iy) {
              if (std::abs(data.work_field_cell(iz, iy, ix, component) - reference) > 1.0e-12) {
                throw std::runtime_error(
                    std::string("Shared Hybrid transverse symmetry fails after ") +
                    hybrid::engine::pcc2_stage_name(stage));
              }
            }
          }
        }
      }
    }
  }

  void setup_chunks() override
  {
    HybridApplication::setup_chunks();
    if (nprocess != 1 || chunkvec.size() != 1) {
      throw std::runtime_error("Shared Hybrid state requires one MPI rank and one chunk");
    }

    const auto    path = cfgparser->get_parameter().at("shared_state").get<std::string>();
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("Cannot open shared Hybrid state: " + path);
    }
    std::array<char, 8> magic = {};
    if (!input.read(magic.data(), magic.size()) || magic != shared_magic) {
      throw std::runtime_error("Invalid shared Hybrid state magic");
    }
    if (read_value<std::uint32_t>(input, "version") != shared_version) {
      throw std::runtime_error("Unsupported shared Hybrid state version");
    }
    const auto step        = read_value<std::uint32_t>(input, "step");
    const auto nz          = read_value<std::int32_t>(input, "Nz");
    const auto ny          = read_value<std::int32_t>(input, "Ny");
    const auto nx          = read_value<std::int32_t>(input, "Nx");
    const auto num_species = read_value<std::int32_t>(input, "species count");
    const auto time        = read_value<nix::float64>(input, "time");
    const auto time_step   = read_value<nix::float64>(input, "time step");

    auto&      chunk = static_cast<hybrid::HybridChunk&>(*chunkvec.front());
    auto       data  = chunk.get_internal_data();
    const auto dims  = chunk.get_dims();
    if (step != 0 || nz != dims[0] || ny != dims[1] || nx != dims[2] ||
        num_species != data.num_species || time != 0 || time_step != cfgparser->get_delt()) {
      throw std::runtime_error("Shared Hybrid state metadata does not match configuration");
    }

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          for (int component = 0; component < hybrid::num_field_components; ++component) {
            data.field_cell(iz, iy, ix, component) = read_value<nix::float64>(input, "cell field");
          }
        }
      }
    }
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          for (int component = 0; component < hybrid::num_fluid_components; ++component) {
            data.fluid(iz, iy, ix, component) = read_value<nix::float64>(input, "fluid");
          }
        }
      }
    }
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          for (int species = 0; species < data.num_species; ++species) {
            for (int component = 0; component < hybrid::num_moment_components; ++component) {
              data.moment_kinetic(iz, iy, ix, species, component) =
                  read_value<nix::float64>(input, "kinetic moment");
            }
          }
        }
      }
    }
    const auto fixture_moment = data.moment_kinetic;
    for (int species = 0; species < data.num_species; ++species) {
      const auto count = read_value<std::int64_t>(input, "particle count");
      if (count < 0 || count > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Invalid shared Hybrid particle count");
      }
      auto& particle = *data.particles[species];
      particle.Np    = 0;
      particle.resize(static_cast<int>(count));
      particle.Np = static_cast<int>(count);
      for (int ip = 0; ip < particle.Np; ++ip) {
        for (int component = 0; component < 6; ++component) {
          particle.xu(ip, component) = read_value<nix::float64>(input, "particle state");
        }
        const auto id = read_value<std::int64_t>(input, "particle ID");
        std::memcpy(&particle.xu(ip, 6), &id, sizeof(id));
      }
      particle.xv = particle.xu;
      particle.count(0, particle.Np - 1, true, data.order);
      particle.sort();
    }

    has_step1 = (read_value<std::uint32_t>(input, "step-1 marker")) != 0;

    expected_field.resize(dims[0] * dims[1] * dims[2] * hybrid::num_field_components);
    input.read(reinterpret_cast<char*>(expected_field.data()),
               static_cast<std::streamsize>(expected_field.size() * sizeof(nix::float64)));
    expected_fluid.resize(dims[0] * dims[1] * dims[2] * hybrid::num_fluid_components);
    input.read(reinterpret_cast<char*>(expected_fluid.data()),
               static_cast<std::streamsize>(expected_fluid.size() * sizeof(nix::float64)));
    expected_moment.resize(dims[0] * dims[1] * dims[2] * num_species *
                           hybrid::num_moment_components);
    input.read(reinterpret_cast<char*>(expected_moment.data()),
               static_cast<std::streamsize>(expected_moment.size() * sizeof(nix::float64)));

    for (int species = 0; species < num_species; ++species) {
      const auto                count = read_value<std::int64_t>(input, "step-1 particle count");
      std::vector<std::int64_t> ids(count);
      std::vector<std::array<nix::float64, 6>> states(count);
      for (std::int64_t ip = 0; ip < count; ++ip) {
        for (int component = 0; component < 6; ++component) {
          states[ip][component] = read_value<nix::float64>(input, "step-1 particle state");
        }
        ids[ip] = read_value<std::int64_t>(input, "step-1 particle ID");
      }
      expected_particle_ids.push_back(std::move(ids));
      expected_particle_states.push_back(std::move(states));
    }

    const auto ssor_offset_count = read_value<std::uint32_t>(input, "SSOR offset count");
    ssor_offset.resize(ssor_offset_count);
    input.read(reinterpret_cast<char*>(ssor_offset.data()),
               static_cast<std::streamsize>(ssor_offset.size() * sizeof(std::int32_t)));

    const auto ssor_iter_count = read_value<std::uint32_t>(input, "SSOR iter count");
    input.ignore(ssor_iter_count * sizeof(std::int32_t));

    const auto ssor_res_count = read_value<std::uint32_t>(input, "SSOR residual count");
    input.ignore(ssor_res_count * sizeof(nix::float64));

    if (input.peek() != std::ifstream::traits_type::eof()) {
      throw std::runtime_error("Shared Hybrid state has trailing data");
    }

    restore_accepted_halos();
    update_kinetic_moments();
    nix::float64 moment_difference = 0;
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          for (int species = 0; species < data.num_species; ++species) {
            for (int component = 0; component < hybrid::num_moment_components; ++component) {
              moment_difference = std::max(
                  moment_difference, std::abs(data.moment_kinetic(iz, iy, ix, species, component) -
                                              fixture_moment(iz, iy, ix, species, component)));
            }
          }
        }
      }
    }
    data.moment_kinetic = fixture_moment;
    if (moment_difference > 1.0e-12) {
      throw std::runtime_error("Shared Hybrid initial moment differs after recomputation: " +
                               std::to_string(moment_difference));
    }
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          for (int component = hybrid::field_component::electric_x;
               component <= hybrid::field_component::electric_z; ++component) {
            data.field_staggered(iz, iy, ix, component) = 0;
          }
          data.field_staggered(iz, iy, ix, hybrid::field_component::magnetic_x) =
              hybrid::engine::magnetic_cell_to_face(
                  hybrid::field_component::magnetic_x,
                  data.field_cell(iz, iy, ix, hybrid::field_component::magnetic_x),
                  data.field_cell(iz, iy, ix + 1, hybrid::field_component::magnetic_x));
          data.field_staggered(iz, iy, ix, hybrid::field_component::magnetic_y) =
              hybrid::engine::magnetic_cell_to_face(
                  hybrid::field_component::magnetic_y,
                  data.field_cell(iz, iy, ix, hybrid::field_component::magnetic_y),
                  data.field_cell(iz, iy + 1, ix, hybrid::field_component::magnetic_y));
          data.field_staggered(iz, iy, ix, hybrid::field_component::magnetic_z) =
              hybrid::engine::magnetic_cell_to_face(
                  hybrid::field_component::magnetic_z,
                  data.field_cell(iz, iy, ix, hybrid::field_component::magnetic_z),
                  data.field_cell(iz + 1, iy, ix, hybrid::field_component::magnetic_z));
        }
      }
    }
    restore_accepted_halos();
    hybrid::engine::derive_current(data);
    chunk.reset_load();
    curstep = static_cast<int>(step);
    curtime = time;
  }

  void validate_step1()
  {
    auto& chunk = static_cast<hybrid::HybridChunk&>(*chunkvec.front());
    auto  data  = chunk.get_internal_data();

    nix::float64 field_diff  = 0;
    nix::float64 fluid_diff  = 0;
    nix::float64 moment_diff = 0;
    size_t       idx         = 0;

    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          for (int c = 0; c < hybrid::num_field_components; ++c) {
            field_diff = std::max(field_diff,
                                  std::abs(data.field_cell(iz, iy, ix, c) - expected_field[idx++]));
          }
        }
      }
    }
    if (field_diff > 1e-13) {
      throw std::runtime_error("Shared Hybrid step-1 field differs: " + std::to_string(field_diff));
    }

    idx = 0;
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          for (int c = 0; c < hybrid::num_fluid_components; ++c) {
            fluid_diff =
                std::max(fluid_diff, std::abs(data.fluid(iz, iy, ix, c) - expected_fluid[idx++]));
          }
        }
      }
    }
    if (fluid_diff > 1e-13) {
      throw std::runtime_error("Shared Hybrid step-1 fluid differs: " + std::to_string(fluid_diff));
    }

    idx = 0;
    for (int iz = data.Lbz; iz <= data.Ubz; ++iz) {
      for (int iy = data.Lby; iy <= data.Uby; ++iy) {
        for (int ix = data.Lbx; ix <= data.Ubx; ++ix) {
          for (int species = 0; species < data.num_species; ++species) {
            for (int c = 0; c < hybrid::num_moment_components; ++c) {
              moment_diff =
                  std::max(moment_diff, std::abs(data.moment_kinetic(iz, iy, ix, species, c) -
                                                 expected_moment[idx++]));
            }
          }
        }
      }
    }
    if (moment_diff > 1e-13) {
      throw std::runtime_error("Shared Hybrid step-1 moment differs: " +
                               std::to_string(moment_diff));
    }

    for (int species = 0; species < data.num_species; ++species) {
      auto&       particle     = *data.particles[species];
      const auto& expected_ids = expected_particle_ids[species];
      const auto& expected_st  = expected_particle_states[species];
      if (particle.Np != static_cast<int>(expected_ids.size())) {
        throw std::runtime_error("Shared Hybrid step-1 particle count differs for species " +
                                 std::to_string(species));
      }
      std::vector<std::pair<std::int64_t, int>> sorted;
      for (int ip = 0; ip < particle.Np; ++ip) {
        std::int64_t id = 0;
        std::memcpy(&id, &particle.xu(ip, 6), sizeof(id));
        sorted.emplace_back(id, ip);
      }
      std::sort(sorted.begin(), sorted.end());
      nix::float64 particle_diff = 0;
      for (int ip = 0; ip < particle.Np; ++ip) {
        const auto actual_id = sorted[ip].first;
        if (actual_id != expected_ids[ip]) {
          throw std::runtime_error("Shared Hybrid step-1 particle ID differs for species " +
                                   std::to_string(species) + " particle " + std::to_string(ip));
        }
        for (int c = 0; c < 6; ++c) {
          particle_diff = std::max(
              particle_diff, std::abs(particle.xu(sorted[ip].second, c) - expected_st[ip][c]));
        }
      }
      if (particle_diff > 1e-13) {
        throw std::runtime_error("Shared Hybrid step-1 particle state differs for species " +
                                 std::to_string(species) + ": " + std::to_string(particle_diff));
      }
    }
    const std::array<int, 4> legacy_offsets = {0, 7, 13, 19};
    if (ssor_offset.size() != legacy_offsets.size() ||
        !std::equal(ssor_offset.begin(), ssor_offset.end(), legacy_offsets.begin())) {
      throw std::runtime_error("Shared Hybrid step-1 SSOR offsets differ");
    }
  }

  void finalize() override
  {
    if (has_step1) {
      validate_step1();
    }
    HybridApplication::finalize();
  }

private:
  bool                                                  has_step1 = false;
  std::vector<nix::float64>                             expected_field;
  std::vector<nix::float64>                             expected_fluid;
  std::vector<nix::float64>                             expected_moment;
  std::vector<std::vector<std::int64_t>>                expected_particle_ids;
  std::vector<std::vector<std::array<nix::float64, 6>>> expected_particle_states;
  std::vector<std::int32_t>                             ssor_offset;
};
} // namespace

int main(int argc, char** argv)
{
  auto                   interface = std::make_shared<hybrid::HybridApplicationInterface>();
  SharedStateApplication application(argc, argv, interface);
  return application.main();
}
