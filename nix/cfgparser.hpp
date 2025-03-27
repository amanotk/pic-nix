// -*- C++ -*-
#ifndef _CFGPARSER_HPP_
#define _CFGPARSER_HPP_

#include "debug.hpp"
#include "nix.hpp"

NIX_NAMESPACE_BEGIN

class CfgParser
{
protected:
  json root;

  json toml_to_json(const toml::value& toml_data)
  {
    json result;

    if (toml_data.is_table()) {
      for (const auto& [key, value] : toml_data.as_table()) {
        result[key] = toml_to_json(value);
      }
    } else if (toml_data.is_array()) {
      for (const auto& elem : toml_data.as_array()) {
        result.push_back(toml_to_json(elem));
      }
    } else if (toml_data.is_boolean()) {
      result = toml_data.as_boolean();
    } else if (toml_data.is_integer()) {
      result = toml_data.as_integer();
    } else if (toml_data.is_floating()) {
      result = toml_data.as_floating();
    } else if (toml_data.is_string()) {
      result = toml_data.as_string();
    }

    return result;
  }

public:
  json get_root()
  {
    return root;
  }

  json get_application()
  {
    return root["application"];
  }

  json get_parameter()
  {
    return root["parameter"];
  }

  json get_diagnostic()
  {
    return root["diagnostic"];
  }

  virtual int get_Nx()
  {
    return root["parameter"]["Nx"].get<int>();
  }

  virtual int get_Ny()
  {
    return root["parameter"]["Ny"].get<int>();
  }

  virtual int get_Nz()
  {
    return root["parameter"]["Nz"].get<int>();
  }

  virtual int get_Cx()
  {
    return root["parameter"]["Cx"].get<int>();
  }

  virtual int get_Cy()
  {
    return root["parameter"]["Cy"].get<int>();
  }

  virtual int get_Cz()
  {
    return root["parameter"]["Cz"].get<int>();
  }

  virtual float64 get_delt()
  {
    return root["parameter"]["delt"].get<float64>();
  }

  virtual float64 get_delx()
  {
    return root["parameter"]["delh"].get<float64>();
  }

  virtual float64 get_dely()
  {
    return root["parameter"]["delh"].get<float64>();
  }

  virtual float64 get_delz()
  {
    return root["parameter"]["delh"].get<float64>();
  }

  bool parse_file(std::string filename, bool exit_on_error = true)
  {
    namespace fs = std::filesystem;

    fs::path    path(filename);
    std::string ext = path.extension().string();

    if (ext == ".json") {
      std::ifstream ifs(filename.c_str());
      root = json::parse(ifs, nullptr, true, true);
    } else if (ext == ".toml") {
      root = toml_to_json(toml::parse(filename));
    } else {
      std::cerr << tfm::format("Unknown file extension `%s`\n", ext);
      exit(1);
    }

    bool status = validate(root);

    if (status == false && exit_on_error == true) {
      std::cerr << tfm::format("Failed to parse `%s`\n", filename);
      exit(1);
    }

    return status;
  }

  void overwrite(json& object)
  {
    assert(validate(object) == true);

    root = object;
  }

  virtual bool validate(json& object)
  {
    bool status = true;

    status = status & check_mandatory_sections(object);

    // make sure that the option section exists in the application section
    if (root["application"]["option"].is_null() == true) {
      root["application"]["option"] = {};
    }

    // check the parameter section
    if (object["parameter"].is_null() == false) {
      status = status & check_mandatory_parameters(object["parameter"]);
      status = status & check_dimensions(object["parameter"]);
    } else {
      status = status & false;
    }

    return status;
  }

  virtual bool check_mandatory_sections(json& object)
  {
    bool status = true;

    std::vector<std::string> mandatory_sections = {"application", "diagnostic", "parameter"};

    for (auto section : mandatory_sections) {
      if (object[section].is_null()) {
        std::cerr << tfm::format("Configuration misses `%s` section\n", section);
        status = false;
      }
    }

    return status;
  }

  virtual bool check_mandatory_parameters(json& parameter)
  {
    bool status = true;

    std::vector<std::string> mandatory_parameters = {"Nx", "Ny", "Nz",   "Cx",
                                                     "Cy", "Cz", "delt", "delh"};

    for (auto key : mandatory_parameters) {
      if (parameter[key].is_null()) {
        std::cerr << tfm::format("Configuration misses `%s` parameter\n", key);
        status = false;
      }
    }

    return status;
  }

  virtual bool check_dimensions(json& parameter)
  {
    int  nx     = parameter.value("Nx", 1);
    int  ny     = parameter.value("Ny", 1);
    int  nz     = parameter.value("Nz", 1);
    int  cx     = parameter.value("Cx", 1);
    int  cy     = parameter.value("Cy", 1);
    int  cz     = parameter.value("Cz", 1);
    bool status = (nz % cz == 0) && (ny % cy == 0) && (nx % cx == 0);

    if (status == false) {
      std::cerr << tfm::format("Number of grid must be divisible by number of chunk\n");
      std::cerr << tfm::format("Nx, Ny, Nz = [%4d, %4d, %4d]\n", nx, ny, nz);
      std::cerr << tfm::format("Cx, Cy, Cz = [%4d, %4d, %4d]\n", cx, cy, cz);
    }

    return status;
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
