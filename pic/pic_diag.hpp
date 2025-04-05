// -*- C++ -*-
#ifndef _PIC_DIAG_HPP_
#define _PIC_DIAG_HPP_

#include "nix/nixio.hpp"
#include "pic.hpp"

using namespace nix::typedefs;
using namespace nixio;

class PicDiag : public nix::Diag
{
public:
  using base_type = nix::Diag;
  using base_type::base_type; // inherit constructors

  // type alias
  using info_type    = base_type::info_type;
  using app_type     = PicApplication;
  using data_type    = app_type::data_type;
  using PtrInterface = std::shared_ptr<PicApplicationInterface>;

protected:
  PtrInterface interface; ///< interface

public:
  // constructor
  PicDiag(std::string name, PtrInterface interface) : nix::Diag(name), interface(interface)
  {
    make_sure_directory_exists(format_dirname(""));
  }

  // check if the diagnostic is required
  virtual bool require_diagnostic(int curstep, json& config)
  {
    int interval = config.value("interval", 1);
    int begin    = config.value("begin", 0);
    int end      = config.value("end", std::numeric_limits<int>::max());

    return (curstep >= begin) && (curstep <= end) && ((curstep - begin) % interval == 0);
  }

  virtual bool is_json_required()
  {
    bool is_json_required_mpiio = info->iomode == "mpiio" && info->world_rank == 0;
    bool is_json_required_posix = info->iomode == "posix" && info->intra_rank == 0;

    return is_json_required_mpiio || is_json_required_posix;
  }

  // check if the given step is the initial step
  virtual bool is_initial_step(int curstep, json& config)
  {
    int begin = config.value("begin", 0);
    return curstep == begin;
  }

  virtual std::string get_prefix(json& config, std::string prefix)
  {
    return config.value("prefix", prefix);
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
