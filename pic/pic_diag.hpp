// -*- C++ -*-
#ifndef _PIC_DIAG_HPP_
#define _PIC_DIAG_HPP_

#include "nix/nixio.hpp"
#include "pic.hpp"

using namespace nix::typedefs;
using namespace nixio;

class PicDiag : public nix::Diag<PicApplication>
{
public:
  using base_type = nix::Diag<PicApplication>;
  using base_type::base_type; // inherit constructors

  // type alias
  using app_type  = PicDiag::app_type;
  using data_type = PicDiag::app_type::data_type;

  // check if the diagnostic is required
  bool require_diagnostic(int curstep, json& config)
  {
    int interval = config.value("interval", 1);
    int begin    = config.value("begin", 0);
    int end      = config.value("end", std::numeric_limits<int>::max());

    return (curstep >= begin) && (curstep <= end) && ((curstep - begin) % interval == 0);
  }

  bool is_json_required()
  {
    bool is_json_required_mpiio = info->iomode == "mpiio" && info->world_rank == 0;
    bool is_json_required_posix = info->iomode == "posix" && info->intra_rank == 0;

    return is_json_required_mpiio || is_json_required_posix;
  }

  // check if the given step is the initial step
  bool is_initial_step(int curstep, json& config)
  {
    int begin = config.value("begin", 0);
    return curstep == begin;
  }

  std::string get_prefix(json& config, std::string prefix)
  {
    return config.value("prefix", prefix);
  }

  // calculate percentile assuming pre-sorted data
  template <typename T>
  float64 percentile(T& data, float64 p, bool is_sorted)
  {
    int     size  = data.size();
    int     index = p * (size - 1);
    float64 frac  = p * (size - 1) - index;

    if (is_sorted == false) {
      std::sort(data.begin(), data.end());
    }

    if (index >= 0 && index < size - 1) {
      // linear interpolation
      return data[index] * (1 - frac) + data[index + 1] * frac;
    } else {
      // error
      return -1;
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
