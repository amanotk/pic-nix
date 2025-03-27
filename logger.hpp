// -*- C++ -*-
#ifndef _LOGGER_HPP_
#define _LOGGER_HPP_

#include "nix.hpp"
#include "tinyformat.hpp"
#include <nlohmann/json.hpp>

NIX_NAMESPACE_BEGIN

static constexpr int logger_flush_interval = 10;

static const json default_config = R"(
{
  "prefix": "log",
  "path": ".",
  "interval": 100,
  "flush" : 10.0
}
)"_json;

///
/// @brief Simple Logger class
///
class Logger
{
protected:
  int               thisrank;     ///< MPI rank
  float64           last_flushed; ///< last flushed time
  std::string       basedir;      ///< base directory
  json              config;       ///< configuration
  std::vector<json> content;      ///< log content

public:
  Logger(json& object, std::string basedir = "", int thisrank = 0, bool overwrite = true)
      : basedir(basedir), thisrank(thisrank)
  {
    initialize_content();

    // set configuration (use default if not specified)
    if (object.is_null() == true) {
      config = default_config;
    } else {
      for (auto& element : default_config.items()) {
        config[element.key()] = object.value(element.key(), element.value());
      }
    }

    if (overwrite == true) {
      std::filesystem::remove(get_filename());
    }

    last_flushed = wall_clock();
  }

  virtual int get_interval()
  {
    return config["interval"];
  }

  virtual std::string get_filename()
  {
    namespace fs = std::filesystem;

    std::string path   = config["path"];
    std::string prefix = config["prefix"];

    return (fs::path(basedir) / fs::path(path) / fs::path(prefix + ".msgpack")).string();
  }

  virtual bool is_flush_required()
  {
    return (wall_clock() - last_flushed > config["flush"]);
  }

  virtual void initialize_content()
  {
    content.resize(0);
    content.push_back({});
  }

  virtual void log(int curstep)
  {
    bool is_interval_step = curstep % get_interval() == 0;

    if (is_interval_step || is_flush_required()) {
      flush();
    }
  }

  virtual void append(int curstep, std::string name, json& obj)
  {
    json& last = content.back();

    // check the last element and append new element if needed
    if (last.is_null() == true) {
      last = {{"rank", thisrank}, {"step", curstep}};
    } else if (last.contains("step") == true && last.value("step", -1) != curstep) {
      content.push_back({{"rank", thisrank}, {"step", curstep}});
    }

    // data
    json& element = content.back();

    element[name] = {};
    for (auto it = obj.begin(); it != obj.end(); ++it) {
      element[name][it.key()] = it.value();
    }
  }

  virtual void flush()
  {
    if (thisrank == 0) {
      std::ofstream ofs(get_filename(), std::ios::binary | std::ios::app);

      for (auto it = content.begin(); it != content.end(); ++it) {
        std::vector<std::uint8_t> buffer = json::to_msgpack(*it);
        ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
      }
      ofs.close();
    }

    initialize_content();
    last_flushed = wall_clock();
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
