// -*- C++ -*-
#ifndef _HYBRID_DIAG_HPP_
#define _HYBRID_DIAG_HPP_

#include "hybrid_chunk.hpp"

#include "nix/nix.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace hybrid
{
namespace diag
{
inline void write_npy_header(std::ostream& os, const std::vector<size_t>& shape,
                             const std::string& dtype)
{
  os.write("\x93NUMPY\x01\x00", 8);
  std::ostringstream header_stream;
  header_stream << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      header_stream << ", ";
    header_stream << shape[i];
  }
  if (shape.size() == 1)
    header_stream << ",";
  header_stream << "), }";
  std::string header = header_stream.str();
  while (header.size() % 64 != 63) {
    header.push_back(' ');
  }
  header.push_back('\n');
  uint16_t header_len = static_cast<uint16_t>(header.size());
  os.write(reinterpret_cast<const char*>(&header_len), 2);
  os.write(header.data(), header.size());
}

inline void write_npy(const std::string& path, const nix::Array4D<nix::float64>& array,
                      int iz_start, int iz_end, int iy_start, int iy_end, int ix_start, int ix_end)
{
  const size_t  nz = static_cast<size_t>(iz_end - iz_start);
  const size_t  ny = static_cast<size_t>(iy_end - iy_start);
  const size_t  nx = static_cast<size_t>(ix_end - ix_start);
  const size_t  nc = array.shape()[3];
  std::ofstream file(path, std::ios::binary);
  write_npy_header(file, {nz, ny, nx, nc}, "<f8");
  for (size_t iz = static_cast<size_t>(iz_start); iz < static_cast<size_t>(iz_end); ++iz) {
    for (size_t iy = static_cast<size_t>(iy_start); iy < static_cast<size_t>(iy_end); ++iy) {
      for (size_t ix = static_cast<size_t>(ix_start); ix < static_cast<size_t>(ix_end); ++ix) {
        for (size_t c = 0; c < nc; ++c) {
          nix::float64 value = array(iz, iy, ix, c);
          file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
      }
    }
  }
}

inline void write_npy(const std::string& path, const nix::Array5D<nix::float64>& array,
                      int iz_start, int iz_end, int iy_start, int iy_end, int ix_start, int ix_end)
{
  const size_t  nz = static_cast<size_t>(iz_end - iz_start);
  const size_t  ny = static_cast<size_t>(iy_end - iy_start);
  const size_t  nx = static_cast<size_t>(ix_end - ix_start);
  const size_t  ns = array.shape()[3];
  const size_t  nc = array.shape()[4];
  std::ofstream file(path, std::ios::binary);
  write_npy_header(file, {nz, ny, nx, ns, nc}, "<f8");
  for (size_t iz = static_cast<size_t>(iz_start); iz < static_cast<size_t>(iz_end); ++iz) {
    for (size_t iy = static_cast<size_t>(iy_start); iy < static_cast<size_t>(iy_end); ++iy) {
      for (size_t ix = static_cast<size_t>(ix_start); ix < static_cast<size_t>(ix_end); ++ix) {
        for (size_t s = 0; s < ns; ++s) {
          for (size_t c = 0; c < nc; ++c) {
            nix::float64 value = array(iz, iy, ix, s, c);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
          }
        }
      }
    }
  }
}
inline void write_particles_npy(const std::string&                           path,
                                const std::shared_ptr<nix::XtensorParticle>& particle)
{
  const int                 Np = particle->Np;
  std::vector<int64_t>      ids(static_cast<size_t>(Np));
  std::vector<int>          order(static_cast<size_t>(Np));
  std::vector<nix::float64> data(static_cast<size_t>(Np) * 7);

  for (int ip = 0; ip < Np; ++ip) {
    std::memcpy(&ids[static_cast<size_t>(ip)], &particle->xu(ip, 6), sizeof(int64_t));
    for (int c = 0; c < 7; ++c) {
      data[static_cast<size_t>(ip) * 7 + static_cast<size_t>(c)] = particle->xu(ip, c);
    }
  }

  for (int ip = 0; ip < Np; ++ip) {
    order[static_cast<size_t>(ip)] = ip;
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return ids[static_cast<size_t>(a)] < ids[static_cast<size_t>(b)];
  });

  std::ofstream file(path, std::ios::binary);
  write_npy_header(file, {static_cast<size_t>(Np), static_cast<size_t>(7)}, "<f8");
  for (int ip_idx = 0; ip_idx < Np; ++ip_idx) {
    const int sorted_ip = order[static_cast<size_t>(ip_idx)];
    for (int c = 0; c < 7; ++c) {
      nix::float64 value = data[static_cast<size_t>(sorted_ip) * 7 + static_cast<size_t>(c)];
      file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
}

inline void write_diagnostics(const HybridChunk::DataContainer& data, const std::string& dir,
                              int rank, int chunk_index)
{
  const std::string chunk_dir =
      dir + "/rank_" + std::to_string(rank) + "_chunk_" + std::to_string(chunk_index);
  std::string mkdir_cmd = "mkdir -p " + chunk_dir;
  std::system(mkdir_cmd.c_str());

  write_npy(chunk_dir + "/field.npy", data.field_cell, data.Lbz, data.Ubz + 1, data.Lby,
            data.Uby + 1, data.Lbx, data.Ubx + 1);
  write_npy(chunk_dir + "/fluid.npy", data.fluid, data.Lbz, data.Ubz + 1, data.Lby, data.Uby + 1,
            data.Lbx, data.Ubx + 1);
  write_npy(chunk_dir + "/moment.npy", data.moment_kinetic, data.Lbz, data.Ubz + 1, data.Lby,
            data.Uby + 1, data.Lbx, data.Ubx + 1);

  for (size_t species = 0; species < data.particles.size(); ++species) {
    write_particles_npy(chunk_dir + "/particle_" + std::to_string(species) + ".npy",
                        data.particles[species]);
  }

  {
    std::ofstream meta(chunk_dir + "/meta.json");
    meta << "{\n";
    meta << "  \"rank\": " << rank << ",\n";
    meta << "  \"chunk_index\": " << chunk_index << ",\n";
    meta << "  \"Lbx\": " << data.Lbx << ",\n";
    meta << "  \"Ubx\": " << data.Ubx << ",\n";
    meta << "  \"Lby\": " << data.Lby << ",\n";
    meta << "  \"Uby\": " << data.Uby << ",\n";
    meta << "  \"Lbz\": " << data.Lbz << ",\n";
    meta << "  \"Ubz\": " << data.Ubz << ",\n";
    meta << "  \"num_species\": " << data.num_species << "\n";
    meta << "}\n";
  }
}
} // namespace diag
} // namespace hybrid

#endif
