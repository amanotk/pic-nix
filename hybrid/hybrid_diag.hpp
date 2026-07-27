// -*- C++ -*-
#ifndef _HYBRID_DIAG_HPP_
#define _HYBRID_DIAG_HPP_

#include "hybrid_chunk.hpp"

#include "nix/nix.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hybrid::diag
{
struct SnapshotMetadata {
  int                       rank;
  int                       chunk_id;
  std::array<int, 3>        offset;
  std::array<int, 3>        local_dims;
  std::array<int, 3>        global_dims;
  int                       step;
  nix::float64              time;
  nix::float64              time_step;
  std::vector<nix::float64> particle_mass;
  std::vector<nix::float64> particle_charge;
};

inline bool is_little_endian()
{
  const std::uint16_t value = 1;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

template <typename T>
inline void write_little_endian(std::ostream& stream, const T& value)
{
  std::array<char, sizeof(T)> bytes = {};
  std::memcpy(bytes.data(), &value, sizeof(T));
  if (!is_little_endian()) {
    std::reverse(bytes.begin(), bytes.end());
  }
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

inline std::ofstream open_output(const std::filesystem::path& path)
{
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.exceptions(std::ios::badbit | std::ios::failbit);
  return stream;
}

inline void write_npy_header(std::ostream& stream, const std::vector<size_t>& shape,
                             const std::string& dtype)
{
  stream.write("\x93NUMPY\x01\x00", 8);
  std::ostringstream header_stream;
  header_stream << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      header_stream << ", ";
    }
    header_stream << shape[i];
  }
  if (shape.size() == 1) {
    header_stream << ",";
  }
  header_stream << "), }";
  std::string header = header_stream.str();
  while (header.size() % 64 != 63) {
    header.push_back(' ');
  }
  header.push_back('\n');
  if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
    throw std::runtime_error("Hybrid NPY header exceeds version 1 size limit");
  }
  const auto header_len = static_cast<std::uint16_t>(header.size());
  write_little_endian(stream, header_len);
  stream.write(header.data(), static_cast<std::streamsize>(header.size()));
}

inline void write_npy(const std::filesystem::path& path, const nix::Array4D<nix::float64>& array,
                      int iz_start, int iz_end, int iy_start, int iy_end, int ix_start, int ix_end)
{
  const size_t nz   = static_cast<size_t>(iz_end - iz_start);
  const size_t ny   = static_cast<size_t>(iy_end - iy_start);
  const size_t nx   = static_cast<size_t>(ix_end - ix_start);
  const size_t nc   = array.shape()[3];
  auto         file = open_output(path);
  write_npy_header(file, {nz, ny, nx, nc}, "<f8");
  for (int iz = iz_start; iz < iz_end; ++iz) {
    for (int iy = iy_start; iy < iy_end; ++iy) {
      for (int ix = ix_start; ix < ix_end; ++ix) {
        for (size_t component = 0; component < nc; ++component) {
          write_little_endian(file, array(iz, iy, ix, component));
        }
      }
    }
  }
}

inline void write_npy(const std::filesystem::path& path, const nix::Array5D<nix::float64>& array,
                      int iz_start, int iz_end, int iy_start, int iy_end, int ix_start, int ix_end)
{
  const size_t nz   = static_cast<size_t>(iz_end - iz_start);
  const size_t ny   = static_cast<size_t>(iy_end - iy_start);
  const size_t nx   = static_cast<size_t>(ix_end - ix_start);
  const size_t ns   = array.shape()[3];
  const size_t nc   = array.shape()[4];
  auto         file = open_output(path);
  write_npy_header(file, {nz, ny, nx, ns, nc}, "<f8");
  for (int iz = iz_start; iz < iz_end; ++iz) {
    for (int iy = iy_start; iy < iy_end; ++iy) {
      for (int ix = ix_start; ix < ix_end; ++ix) {
        for (size_t species = 0; species < ns; ++species) {
          for (size_t component = 0; component < nc; ++component) {
            write_little_endian(file, array(iz, iy, ix, species, component));
          }
        }
      }
    }
  }
}

inline void write_particles_npy(const std::filesystem::path&                 path,
                                const std::shared_ptr<nix::XtensorParticle>& particle)
{
  const int                 count = particle->Np;
  std::vector<std::int64_t> ids(static_cast<size_t>(count));
  std::vector<int>          order(static_cast<size_t>(count));
  std::iota(order.begin(), order.end(), 0);
  for (int ip = 0; ip < count; ++ip) {
    std::memcpy(&ids[static_cast<size_t>(ip)], &particle->xu(ip, 6), sizeof(std::int64_t));
  }
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return ids[static_cast<size_t>(lhs)] < ids[static_cast<size_t>(rhs)];
  });

  auto file = open_output(path);
  write_npy_header(file, {static_cast<size_t>(count), static_cast<size_t>(7)}, "<f8");
  for (int ip : order) {
    for (int component = 0; component < 7; ++component) {
      write_little_endian(file, particle->xu(ip, component));
    }
  }
}

inline void write_json_array(std::ostream& stream, const std::array<int, 3>& value)
{
  stream << "[" << value[0] << ", " << value[1] << ", " << value[2] << "]";
}

inline void write_json_array(std::ostream& stream, const std::vector<nix::float64>& value)
{
  stream << "[";
  for (size_t i = 0; i < value.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << value[i];
  }
  stream << "]";
}

inline void write_diagnostics(const HybridChunk::DataContainer& data,
                              const std::filesystem::path& root, const SnapshotMetadata& metadata)
{
  const auto step_dir  = root / "snapshots" / ("step_" + std::to_string(metadata.step));
  const auto chunk_dir = step_dir / ("chunk_" + std::to_string(metadata.chunk_id));
  std::filesystem::create_directories(chunk_dir);

  write_npy(chunk_dir / "field.npy", data.field_cell, data.Lbz, data.Ubz + 1, data.Lby,
            data.Uby + 1, data.Lbx, data.Ubx + 1);
  write_npy(chunk_dir / "fluid.npy", data.fluid, data.Lbz, data.Ubz + 1, data.Lby, data.Uby + 1,
            data.Lbx, data.Ubx + 1);
  write_npy(chunk_dir / "moment.npy", data.moment_kinetic, data.Lbz, data.Ubz + 1, data.Lby,
            data.Uby + 1, data.Lbx, data.Ubx + 1);
  for (size_t species = 0; species < data.particles.size(); ++species) {
    write_particles_npy(chunk_dir / ("particle_" + std::to_string(species) + ".npy"),
                        data.particles[species]);
  }

  auto meta = open_output(chunk_dir / "meta.json");
  meta << std::setprecision(std::numeric_limits<nix::float64>::max_digits10);
  meta << "{\n";
  meta << "  \"rank\": " << metadata.rank << ",\n";
  meta << "  \"chunk_id\": " << metadata.chunk_id << ",\n";
  meta << "  \"offset\": ";
  write_json_array(meta, metadata.offset);
  meta << ",\n  \"local_dims\": ";
  write_json_array(meta, metadata.local_dims);
  meta << ",\n  \"global_dims\": ";
  write_json_array(meta, metadata.global_dims);
  meta << ",\n  \"step\": " << metadata.step << ",\n";
  meta << "  \"time\": " << metadata.time << ",\n";
  meta << "  \"time_step\": " << metadata.time_step << ",\n";
  meta << "  \"num_species\": " << data.num_species << ",\n";
  meta << "  \"particle_mass\": ";
  write_json_array(meta, metadata.particle_mass);
  meta << ",\n  \"particle_charge\": ";
  write_json_array(meta, metadata.particle_charge);
  meta << "\n}\n";
}
} // namespace hybrid::diag

#endif
