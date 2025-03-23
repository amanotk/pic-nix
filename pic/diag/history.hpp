// -*- C++ -*-
#ifndef _HISTORY_DIAG_HPP_
#define _HISTORY_DIAG_HPP_

#include "base.hpp"

///
/// @brief Diagnostic for time history
///
template <typename App, typename Data>
class HistoryDiag : public BaseDiag<App, Data>
{
protected:
  using BaseDiag<App, Data>::info;

public:
  // constructor
  HistoryDiag(std::shared_ptr<DiagInfo> info) : BaseDiag<App, Data>("history", info)
  {
  }

  void operator()(json& config, App& app, Data& data) override
  {
    if (this->require_diagnostic(data.curstep, config) == false)
      return;

    const int            Ns = this->get_num_species();
    std::vector<float64> history(Ns + 4);

    // clear
    std::fill(history.begin(), history.end(), 0.0);

    // calculate moment if not cached
    app.calculate_moment();

    // calculate divergence error and energy
    for (int i = 0; i < data.chunkvec.size(); i++) {
      float64 div_e = 0;
      float64 div_b = 0;
      float64 ene_e = 0;
      float64 ene_b = 0;
      float64 ene_p[Ns];

      data.chunkvec[i]->get_diverror(div_e, div_b);
      data.chunkvec[i]->get_energy(ene_e, ene_b, ene_p);

      history[0] += div_e;
      history[1] += div_b;
      history[2] += ene_e;
      history[3] += ene_b;
      for (int is = 0; is < Ns; is++) {
        history[is + 4] += ene_p[is];
      }
    }

    {
      void* sndptr = history.data();
      void* rcvptr = nullptr;

      if (data.thisrank == 0) {
        sndptr = MPI_IN_PLACE;
        rcvptr = history.data();
      }

      MPI_Reduce(sndptr, rcvptr, Ns + 4, MPI_FLOAT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // output from root
    if (data.thisrank == 0) {
      std::string dirname  = this->format_dirname("");
      std::string filename = dirname + "history.txt";
      std::string msg      = "";

      // initial call
      if (this->is_initial_step(data.curstep, config) == true) {
        // header
        msg += tfm::format("# %8s %13s", "step", "time");
        msg += tfm::format(" %13s", "div(E)");
        msg += tfm::format(" %13s", "div(B)");
        msg += tfm::format(" %13s", "E^2/2");
        msg += tfm::format(" %13s", "B^2/2");
        for (int is = 0; is < Ns; is++) {
          std::string label = tfm::format("Particle #%02d", is);
          msg += tfm::format(" %13s", label);
        }
        msg += "\n";

        // clear file
        std::filesystem::remove(filename);
      }

      msg += tfm::format("  %8s %13.6e", nix::format_step(data.curstep), data.curtime);
      msg += tfm::format(" %13.6e", history[0]);
      msg += tfm::format(" %13.6e", history[1]);
      msg += tfm::format(" %13.6e", history[2]);
      msg += tfm::format(" %13.6e", history[3]);
      for (int is = 0; is < Ns; is++) {
        msg += tfm::format(" %13.6e", history[is + 4]);
      }
      msg += "\n";

      // output to steam
      std::cout << msg << std::flush;

      // append to file
      if (this->make_sure_directory_exists(filename) == true) {
        std::ofstream ofs(filename, nix::text_append);
        ofs << msg << std::flush;
        ofs.close();
      }
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
