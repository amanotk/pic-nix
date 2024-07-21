// -*- C++ -*-
#ifndef _BASE_DIAG_HPP_
#define _BASE_DIAG_HPP_

#include "nix/buffer.hpp"
#include "nix/debug.hpp"
#include "nix/nix.hpp"
#include "nix/nixio.hpp"
#include "nix/random.hpp"
#include "nix/xtensor_packer3d.hpp"

using namespace nix::typedefs;
using nix::json;
using nix::Buffer;
using nix::XtensorPacker3D;

template <typename App, typename Data>
class BaseDiag
{
protected:
  std::string name;
  std::string basedir;

public:
  // constructor
  BaseDiag(std::string basedir, std::string name) : basedir(basedir), name(name)
  {
  }

  // destructor
  virtual ~BaseDiag() = default;

  // main operation
  virtual void operator()(json& config, App& app, Data& data) = 0;

  // check if the given key matches the name
  bool match(std::string key)
  {
    return key == name;
  }

  // check if the diagnostic is required
  bool require_diagnostic(int curstep, json& config)
  {
    int interval = config.value("interval", 1);
    int begin    = config.value("begin", 0);
    int end      = config.value("end", std::numeric_limits<int>::max());

    return (curstep >= begin) && (curstep <= end) && ((curstep - begin) % interval == 0);
  }

  // check if the given step is the initial step
  bool is_initial_step(int curstep, json& config)
  {
    int begin = config.value("begin", 0);
    return curstep == begin;
  }

  // check if the parent directory of the given path exists
  bool make_sure_directory_exists(std::string path)
  {
    namespace fs = std::filesystem;

    fs::path filepath(path);
    fs::path dirpath = filepath.parent_path();

    if (fs::exists(dirpath) == true) {
      return true;
    }

    if (fs::create_directory(dirpath) == true) {
      return true;
    }

    ERROR << tfm::format("Failed to create directory: %s", path);

    return false;
  }

  // return formatted filename without directory
  std::string format_filename(json& config, std::string ext, std::string prefix, int step = -1)
  {
    namespace fs = std::filesystem;

    prefix = config.value("prefix", prefix);

    if (step >= 0) {
      prefix = tfm::format("%s_%s", prefix, nix::format_step(step));
    }

    return prefix + ext;
  }

  // return formatted filename
  std::string format_filename(json& config, std::string ext, std::string basedir, std::string path,
                              std::string prefix, int step = -1)
  {
    namespace fs = std::filesystem;

    path   = config.value("path", path);
    prefix = config.value("prefix", prefix);

    if (step >= 0) {
      prefix = tfm::format("%s_%s", prefix, nix::format_step(step));
    }

    return (fs::path(basedir) / path / prefix).string() + ext;
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
