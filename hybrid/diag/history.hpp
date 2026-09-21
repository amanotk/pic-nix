// -*- C++ -*-
#ifndef _HYBRID_HISTORY_DIAG_HPP_
#define _HYBRID_HISTORY_DIAG_HPP_

#include "hybrid/diag/hybrid_diag.hpp"
#include "hybrid/hybrid_chunk.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace hybrid
{
class HistoryDiag : public HybridDiag
{
public:
  static constexpr const char* diag_name = "history";

  HistoryDiag(PtrInterface interface) : HybridDiag(diag_name, interface)
  {
  }

  void operator()(nix::json& config) override
  {
    auto data = interface->get_data();
    auto Ns   = interface->get_num_species();

    if (require_diagnostic(data.curstep, config) == false)
      return;

    std::vector<nix::float64> history(Ns + 1);

    std::fill(history.begin(), history.end(), 0.0);

    for (int i = 0; i < data.chunkvec.size(); i++) {
      auto  chunk     = static_cast<HybridChunk*>(data.chunkvec[i].get());
      auto  chunkdata = chunk->get_internal_data();
      auto& field     = chunkdata.field_cell;

      // magnetic energy: B^2/2
      for (int iz = chunkdata.Lbz; iz <= chunkdata.Ubz; ++iz) {
        for (int iy = chunkdata.Lby; iy <= chunkdata.Uby; ++iy) {
          for (int ix = chunkdata.Lbx; ix <= chunkdata.Ubx; ++ix) {
            nix::float64 bx = field(iz, iy, ix, field_component::magnetic_x);
            nix::float64 by = field(iz, iy, ix, field_component::magnetic_y);
            nix::float64 bz = field(iz, iy, ix, field_component::magnetic_z);
            history[0] += 0.5 * (bx * bx + by * by + bz * bz);
          }
        }
      }

      // per-species kinetic energy
      for (int is = 0; is < Ns && is < static_cast<int>(chunkdata.particles.size()); ++is) {
        auto& particle = chunkdata.particles[is];
        for (int ip = 0; ip < particle->Np; ++ip) {
          nix::float64 vx = particle->xu(ip, 3);
          nix::float64 vy = particle->xu(ip, 4);
          nix::float64 vz = particle->xu(ip, 5);
          history[is + 1] += 0.5 * particle->m * (vx * vx + vy * vy + vz * vz);
        }
      }
    }

    {
      void* sndptr = history.data();
      void* rcvptr = nullptr;

      if (data.thisrank == 0) {
        sndptr = MPI_IN_PLACE;
        rcvptr = history.data();
      }

      MPI_Reduce(sndptr, rcvptr, Ns + 1, MPI_FLOAT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (data.thisrank == 0) {
      std::string dirname  = format_dirname("");
      std::string filename = dirname + "history.txt";
      std::string msg      = "";

      if (is_initial_step(data.curstep, config) == true) {
        msg += fmt::format("# {:>8} {:>13}", "step", "time");
        msg += fmt::format(" {:>13}", "B^2/2");
        for (int is = 0; is < Ns; ++is) {
          msg += fmt::format(" {:>13}", fmt::format("kinetic_{:02d}", is));
        }
        msg += "\n";

        std::filesystem::remove(filename);
      }

      msg += fmt::format("  {:>8} {:13.6e}", nix::format_step(data.curstep), data.curtime);
      for (int is = 0; is < Ns + 1; ++is) {
        msg += fmt::format(" {:13.6e}", history[is]);
      }
      msg += "\n";

      std::cout << msg << std::flush;

      if (make_sure_directory_exists(filename) == true) {
        std::ofstream ofs(filename, nix::text_append);
        ofs << msg << std::flush;
        ofs.close();
      }
    }
  }
};
} // namespace hybrid

#endif
