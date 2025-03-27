// -*- C++ -*-
#ifndef _ARGPARSER_HPP_
#define _ARGPARSER_HPP_

#include "cmdline.hpp"
#include "nix.hpp"

NIX_NAMESPACE_BEGIN

class ArgParser : public cmdline::parser
{
protected:
  virtual void setup()
  {
    const float64 etmax = 60 * 60;
    const float64 ptmax = std::numeric_limits<float64>::max();

    this->add<std::string>("config", 'c', "configuration file", "config.json");
    this->add<std::string>("load", 'l', "prefix of snapshot to load", false, "");
    this->add<std::string>("save", 's', "prefix of snapshot to save", false, "");
    this->add<float64>("tmax", 't', "maximum physical time", false, ptmax);
    this->add<float64>("emax", 'e', "maximum elapsed time [sec]", false, etmax);
    this->add<int>("verbose", 'v', "verbosity level", false, 0);
  }

public:
  ArgParser() : cmdline::parser()
  {
    setup();
  }

  std::string get_config() const
  {
    return this->get<std::string>("config");
  }

  std::string get_load() const
  {
    return this->get<std::string>("load");
  }

  std::string get_save() const
  {
    return this->get<std::string>("save");
  }

  float64 get_physical_time_max() const
  {
    return this->get<float64>("tmax");
  }

  float64 get_elapsed_time_max() const
  {
    return this->get<float64>("emax");
  }

  int get_verbosity() const
  {
    return this->get<int>("verbose");
  }

  static std::vector<const char*> convert_to_clargs(const std::vector<std::string>& args)
  {
    int argc = static_cast<int>(args.size());

    std::vector<const char*> cl_args(argc);
    for (int i = 0; i < argc; i++)
      cl_args[i] = args[i].c_str();

    return cl_args;
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
