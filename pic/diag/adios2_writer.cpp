// -*- C++ -*-

#include "adios2_writer.hpp"

#if PICNIX_ENABLE_ADIOS2

#include <adios2.h>

#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

namespace
{
std::string parameter_value(const nix::json& value)
{
  if (value.is_string()) {
    return value.get<std::string>();
  }
  if (value.is_boolean()) {
    return value.get<bool>() ? "true" : "false";
  }
  if (value.is_number()) {
    return value.dump();
  }

  throw std::invalid_argument("ADIOS2 parameters must be scalar values");
}
} // namespace

namespace nix
{
struct Adios2Writer::Impl {
  using Dims = Adios2Writer::Dims;

  std::shared_ptr<Diag::info_type> info;
  std::unique_ptr<adios2::ADIOS>   adios;
  std::unique_ptr<adios2::IO>      io;
  std::unique_ptr<adios2::Engine>  engine;
  std::filesystem::path            filename;

  std::optional<adios2::Variable<std::int64_t>>          step_variable;
  std::optional<adios2::Variable<double>>                time_variable;
  std::map<std::string, adios2::Variable<double>>        double_variables;
  std::map<std::string, adios2::Variable<std::int32_t>>  int32_variables;
  std::map<std::string, adios2::Variable<std::uint64_t>> uint64_variables;

  bool step_open = false;

  explicit Impl(std::shared_ptr<Diag::info_type> info) : info(std::move(info))
  {
  }
};

Adios2Writer::Adios2Writer(std::shared_ptr<Diag::info_type> info)
    : impl(std::make_unique<Impl>(std::move(info)))
{
}

Adios2Writer::~Adios2Writer()
{
  if (impl != nullptr && impl->engine != nullptr) {
    try {
      close();
    } catch (...) {
    }
  }
}

void Adios2Writer::initialize(const std::string& diagnostic, const std::string& prefix,
                              const json& config)
{
  if (impl->adios != nullptr) {
    throw std::logic_error("ADIOS2 writer initialized more than once");
  }

  const json application  = config.value("application", json::object());
  const json adios_config = application.value("adios2", json::object());
  if (adios_config.is_object() == false) {
    throw std::invalid_argument("application.adios2 must be a table");
  }

  const std::string engine_name = adios_config.value("engine", "BP5");
  const json        parameters  = adios_config.value("parameters", json::object());
  if (parameters.is_object() == false) {
    throw std::invalid_argument("application.adios2.parameters must be a table");
  }

  impl->adios = std::make_unique<adios2::ADIOS>(MPI_COMM_WORLD);
  impl->io    = std::make_unique<adios2::IO>(impl->adios->DeclareIO("PICNIX"));
  impl->io->SetEngine(engine_name);
  impl->filename = std::filesystem::path(impl->info->basedir) / "adios2" / (prefix + ".bp");

  adios2::Params adios_parameters{{"AsyncWrite", "false"}};
  for (auto it = parameters.begin(); it != parameters.end(); ++it) {
    adios_parameters[it.key()] = parameter_value(it.value());
  }
  impl->io->SetParameters(adios_parameters);

  impl->io->DefineAttribute<std::string>("picnix_schema", "diagnostic-bp-v1");
  impl->io->DefineAttribute<std::string>("diagnostic", diagnostic);
  impl->io->DefineAttribute<std::string>("prefix", prefix);

  impl->step_variable = impl->io->DefineVariable<std::int64_t>("step");
  impl->time_variable = impl->io->DefineVariable<double>("time");
}

void Adios2Writer::define_global_double(const std::string& name, const Dims& shape,
                                        const Dims& start, const Dims& count)
{
  if (impl->io == nullptr || impl->engine != nullptr) {
    throw std::logic_error("ADIOS2 variables must be defined before opening the engine");
  }
  if (impl->double_variables.count(name) != 0 || impl->int32_variables.count(name) != 0 ||
      impl->uint64_variables.count(name) != 0) {
    throw std::invalid_argument("duplicate ADIOS2 variable: " + name);
  }

  impl->double_variables.emplace(
      name, impl->io->DefineVariable<double>(name, shape, start, count, false));
}

void Adios2Writer::define_global_int32(const std::string& name, const Dims& shape,
                                       const Dims& start, const Dims& count)
{
  if (impl->io == nullptr || impl->engine != nullptr) {
    throw std::logic_error("ADIOS2 variables must be defined before opening the engine");
  }
  if (impl->double_variables.count(name) != 0 || impl->int32_variables.count(name) != 0 ||
      impl->uint64_variables.count(name) != 0) {
    throw std::invalid_argument("duplicate ADIOS2 variable: " + name);
  }

  impl->int32_variables.emplace(
      name, impl->io->DefineVariable<std::int32_t>(name, shape, start, count, false));
}

void Adios2Writer::define_joined_double(const std::string& name, std::size_t width,
                                        std::size_t count)
{
  if (impl->io == nullptr || impl->engine != nullptr) {
    throw std::logic_error("ADIOS2 variables must be defined before opening the engine");
  }
  if (impl->double_variables.count(name) != 0 || impl->int32_variables.count(name) != 0 ||
      impl->uint64_variables.count(name) != 0) {
    throw std::invalid_argument("duplicate ADIOS2 variable: " + name);
  }

  impl->double_variables.emplace(name,
                                 impl->io->DefineVariable<double>(name, {adios2::JoinedDim, width},
                                                                  {}, {count, width}, false));
}

void Adios2Writer::define_joined_uint64(const std::string& name, std::size_t count)
{
  if (impl->io == nullptr || impl->engine != nullptr) {
    throw std::logic_error("ADIOS2 variables must be defined before opening the engine");
  }
  if (impl->double_variables.count(name) != 0 || impl->int32_variables.count(name) != 0 ||
      impl->uint64_variables.count(name) != 0) {
    throw std::invalid_argument("duplicate ADIOS2 variable: " + name);
  }

  impl->uint64_variables.emplace(
      name, impl->io->DefineVariable<std::uint64_t>(name, {adios2::JoinedDim}, {}, {count}, false));
}

void Adios2Writer::open()
{
  if (impl->io == nullptr || impl->engine != nullptr) {
    throw std::logic_error("invalid ADIOS2 engine open state");
  }

  const std::filesystem::path directory = impl->filename.parent_path();
  if (impl->info->world_rank == 0) {
    std::filesystem::create_directories(directory);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  impl->engine = std::make_unique<adios2::Engine>(
      impl->io->Open(impl->filename.string(), adios2::Mode::Write));
}

void Adios2Writer::begin_step(std::int64_t step, double time)
{
  if (impl->engine == nullptr || impl->step_open) {
    throw std::logic_error("invalid ADIOS2 BeginStep state");
  }

  impl->engine->BeginStep();
  if (impl->info->world_rank == 0) {
    impl->engine->Put(*impl->step_variable, &step, adios2::Mode::Sync);
    impl->engine->Put(*impl->time_variable, &time, adios2::Mode::Sync);
  }
  impl->step_open = true;
}

void Adios2Writer::put_global_double(const std::string& name, const Dims& start, const Dims& count,
                                     const double* data, std::size_t size)
{
  auto it = impl->double_variables.find(name);
  if (it == impl->double_variables.end() || impl->step_open == false) {
    throw std::invalid_argument("unknown or inactive ADIOS2 variable: " + name);
  }

  it->second.SetSelection({start, count});
  static const double empty = 0.0;
  impl->engine->Put(it->second, size == 0 ? &empty : data, adios2::Mode::Sync);
}

void Adios2Writer::put_global_int32(const std::string& name, const Dims& start, const Dims& count,
                                    const std::int32_t* data, std::size_t size)
{
  auto it = impl->int32_variables.find(name);
  if (it == impl->int32_variables.end() || impl->step_open == false) {
    throw std::invalid_argument("unknown or inactive ADIOS2 variable: " + name);
  }

  it->second.SetSelection({start, count});
  static const std::int32_t empty = 0;
  impl->engine->Put(it->second, size == 0 ? &empty : data, adios2::Mode::Sync);
}

void Adios2Writer::put_joined_double(const std::string& name, std::size_t count, const double* data)
{
  auto it = impl->double_variables.find(name);
  if (it == impl->double_variables.end() || impl->step_open == false) {
    throw std::invalid_argument("unknown or inactive ADIOS2 variable: " + name);
  }

  const auto current_count = it->second.Count();
  if (current_count.size() != 2) {
    throw std::logic_error("invalid ADIOS2 joined variable dimensions: " + name);
  }
  it->second.SetSelection({{}, {count, current_count[1]}});
  static const double empty = 0.0;
  impl->engine->Put(it->second, count == 0 ? &empty : data, adios2::Mode::Sync);
}

void Adios2Writer::put_joined_uint64(const std::string& name, std::size_t count,
                                     const std::uint64_t* data)
{
  auto it = impl->uint64_variables.find(name);
  if (it == impl->uint64_variables.end() || impl->step_open == false) {
    throw std::invalid_argument("unknown or inactive ADIOS2 variable: " + name);
  }

  it->second.SetSelection({{}, {count}});
  static const std::uint64_t empty = 0;
  impl->engine->Put(it->second, count == 0 ? &empty : data, adios2::Mode::Sync);
}

void Adios2Writer::end_step()
{
  if (impl->engine == nullptr || impl->step_open == false) {
    throw std::logic_error("invalid ADIOS2 EndStep state");
  }
  impl->engine->EndStep();
  impl->step_open = false;
}

void Adios2Writer::close()
{
  if (impl->engine == nullptr) {
    return;
  }
  if (impl->step_open) {
    end_step();
  }
  impl->engine->Close();
  impl->engine.reset();
}
} // namespace nix

#endif
