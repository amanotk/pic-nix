// -*- C++ -*-

#include "adios.hpp"

#include <adios2.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <limits>
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

std::filesystem::path segment_path(const std::filesystem::path& base, int index)
{
  if (index == 0) {
    return base;
  }
  const std::string name =
      fmt::format("{}.part{:04d}{}", base.stem().string(), index, base.extension().string());
  return base.parent_path() / name;
}

std::filesystem::path temporary_path(const std::filesystem::path& path)
{
  return path.string() + ".tmp";
}

std::optional<int> segment_index(const std::filesystem::path& base,
                                 const std::filesystem::path& candidate)
{
  if (candidate.extension() != base.extension()) {
    return std::nullopt;
  }

  const std::string prefix = base.stem().string() + ".part";
  const std::string stem   = candidate.stem().string();
  if (stem.size() <= prefix.size() || stem.compare(0, prefix.size(), prefix) != 0) {
    return std::nullopt;
  }

  const std::string digits = stem.substr(prefix.size());
  if (std::all_of(digits.begin(), digits.end(),
                  [](unsigned char ch) { return std::isdigit(ch); }) == false) {
    return std::nullopt;
  }

  int index = 0;
  for (const char digit : digits) {
    const int value = digit - '0';
    if (index > (std::numeric_limits<int>::max() - value) / 10) {
      return std::nullopt;
    }
    index = index * 10 + value;
  }
  if (candidate.filename() != segment_path(base, index).filename()) {
    return std::nullopt;
  }
  return index;
}

std::optional<int> dataset_index(const std::filesystem::path& base,
                                 const std::filesystem::path& candidate)
{
  if (candidate == base) {
    return 0;
  }
  return segment_index(base, candidate);
}

std::optional<int> temporary_dataset_index(const std::filesystem::path& base,
                                           const std::filesystem::path& candidate)
{
  const std::string suffix = ".tmp";
  const std::string name   = candidate.string();
  if (name.size() <= suffix.size() ||
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
    return std::nullopt;
  }
  return dataset_index(base, name.substr(0, name.size() - suffix.size()));
}

int next_segment_index(const std::filesystem::path& base)
{
  std::vector<int> indices;
  if (std::filesystem::exists(base)) {
    indices.push_back(0);
  }
  if (std::filesystem::exists(base.parent_path())) {
    for (const auto& entry : std::filesystem::directory_iterator(base.parent_path())) {
      const auto index = segment_index(base, entry.path());
      if (index.has_value()) {
        indices.push_back(*index);
      }
    }
  }
  if (indices.empty()) {
    return 0;
  }

  std::sort(indices.begin(), indices.end());
  for (int expected = 0; expected < static_cast<int>(indices.size()); expected++) {
    if (indices[expected] != expected) {
      throw std::runtime_error("ADIOS2 diagnostic segments are not contiguous");
    }
  }
  if (indices.back() == std::numeric_limits<int>::max()) {
    throw std::overflow_error("ADIOS2 diagnostic segment index overflow");
  }
  return indices.back() + 1;
}

void remove_segments(const std::filesystem::path& base)
{
  std::filesystem::remove_all(base);
  if (std::filesystem::exists(base.parent_path()) == false) {
    return;
  }

  std::vector<std::filesystem::path> segments;
  for (const auto& entry : std::filesystem::directory_iterator(base.parent_path())) {
    if (dataset_index(base, entry.path()).has_value() ||
        temporary_dataset_index(base, entry.path()).has_value()) {
      segments.push_back(entry.path());
    }
  }
  for (const auto& segment : segments) {
    std::filesystem::remove_all(segment);
  }
}

void remove_temporary_segments(const std::filesystem::path& base)
{
  if (std::filesystem::exists(base.parent_path()) == false) {
    return;
  }

  std::vector<std::filesystem::path> segments;
  for (const auto& entry : std::filesystem::directory_iterator(base.parent_path())) {
    if (temporary_dataset_index(base, entry.path()).has_value()) {
      segments.push_back(entry.path());
    }
  }
  for (const auto& segment : segments) {
    std::filesystem::remove_all(segment);
  }
}

void broadcast_rank0_error(std::string& error)
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 0;
  if (rank == 0) {
    if (error.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      error = "rank 0 filesystem error message is too long";
    }
    size = static_cast<int>(error.size());
  }
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    error.resize(size);
  }
  if (size > 0) {
    MPI_Bcast(error.data(), size, MPI_CHAR, 0, MPI_COMM_WORLD);
    throw std::runtime_error(error);
  }
}
} // namespace

namespace nix
{
struct AdiosWriter::Impl {
  using Dims = AdiosWriter::Dims;

  std::shared_ptr<Diag::info_type> info;
  std::unique_ptr<adios2::ADIOS>   adios;
  std::unique_ptr<adios2::IO>      io;
  std::unique_ptr<adios2::Engine>  engine;
  std::filesystem::path            filename;
  std::filesystem::path            temporary_filename;

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

bool AdiosWriter::available()
{
  return true;
}

void AdiosWriter::prepare_fresh_run(const std::string& basedir, const std::string& prefix)
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string error;
  if (rank == 0) {
    try {
      const std::filesystem::path base =
          std::filesystem::path(basedir) / "adios" / (prefix + ".bp");
      std::filesystem::create_directories(base.parent_path());
      remove_segments(base);
    } catch (const std::exception& exception) {
      error = exception.what();
    }
  }
  broadcast_rank0_error(error);
}

AdiosWriter::AdiosWriter(std::shared_ptr<Diag::info_type> info)
    : impl(std::make_unique<Impl>(std::move(info)))
{
}

AdiosWriter::~AdiosWriter()
{
  if (impl != nullptr && impl->engine != nullptr) {
    try {
      close();
    } catch (...) {
    }
  }
}

void AdiosWriter::initialize(const std::string& diagnostic, const std::string& prefix,
                             const json& config)
{
  if (impl->adios != nullptr) {
    throw std::logic_error("ADIOS2 writer initialized more than once");
  }

  const json application  = config.value("application", json::object());
  const json adios_config = application.value("adios", json::object());
  if (adios_config.is_object() == false) {
    throw std::invalid_argument("application.adios must be a table");
  }

  if (adios_config.contains("engine")) {
    throw std::invalid_argument("ADIOS2 engine is fixed to BP5; remove application.adios.engine");
  }
  impl->adios = std::make_unique<adios2::ADIOS>(MPI_COMM_WORLD);
  impl->io    = std::make_unique<adios2::IO>(impl->adios->DeclareIO("PICNIX"));
  impl->io->SetEngine("BP5");
  impl->filename = std::filesystem::path(impl->info->basedir) / "adios" / (prefix + ".bp");

  adios2::Params adios_parameters{{"AsyncWrite", "false"}};
  for (auto it = adios_config.begin(); it != adios_config.end(); ++it) {
    adios_parameters[it.key()] = parameter_value(it.value());
  }
  impl->io->SetParameters(adios_parameters);

  impl->io->DefineAttribute<std::string>("picnix_schema", "diagnostic-bp-v1");
  impl->io->DefineAttribute<std::string>("diagnostic", diagnostic);
  impl->io->DefineAttribute<std::string>("prefix", prefix);

  impl->step_variable = impl->io->DefineVariable<std::int64_t>("step");
  impl->time_variable = impl->io->DefineVariable<double>("time");
}

void AdiosWriter::define_global_double(const std::string& name, const Dims& shape,
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

void AdiosWriter::define_global_int32(const std::string& name, const Dims& shape, const Dims& start,
                                      const Dims& count)
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

void AdiosWriter::define_joined_double(const std::string& name, std::size_t width,
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

void AdiosWriter::define_joined_uint64(const std::string& name, std::size_t count)
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

void AdiosWriter::open()
{
  if (impl->io == nullptr || impl->engine != nullptr) {
    throw std::logic_error("invalid ADIOS2 engine open state");
  }

  const std::filesystem::path directory = impl->filename.parent_path();
  int                         segment   = 0;
  std::string                 error;
  if (impl->info->world_rank == 0) {
    try {
      std::filesystem::create_directories(directory);
      if (impl->info->is_restart) {
        if (impl->info->restart_step < 0) {
          throw std::logic_error("ADIOS2 restart step was not initialized");
        }
        remove_temporary_segments(impl->filename);
        segment = next_segment_index(impl->filename);
      } else {
        remove_segments(impl->filename);
      }
    } catch (const std::exception& exception) {
      error = exception.what();
    }
  }
  broadcast_rank0_error(error);
  MPI_Bcast(&segment, 1, MPI_INT, 0, MPI_COMM_WORLD);

  impl->filename           = segment_path(impl->filename, segment);
  impl->temporary_filename = temporary_path(impl->filename);
  impl->io->DefineAttribute<std::int32_t>("segment_index", segment);
  impl->io->DefineAttribute<std::int64_t>("restart_step", impl->info->restart_step);

  impl->engine = std::make_unique<adios2::Engine>(
      impl->io->Open(impl->temporary_filename.string(), adios2::Mode::Write));
}

void AdiosWriter::begin_step(std::int64_t step, double time)
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

void AdiosWriter::put_global_double(const std::string& name, const Dims& start, const Dims& count,
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

void AdiosWriter::put_global_int32(const std::string& name, const Dims& start, const Dims& count,
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

void AdiosWriter::put_joined_double(const std::string& name, std::size_t count, const double* data)
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

void AdiosWriter::put_joined_uint64(const std::string& name, std::size_t count,
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

void AdiosWriter::end_step()
{
  if (impl->engine == nullptr || impl->step_open == false) {
    throw std::logic_error("invalid ADIOS2 EndStep state");
  }
  impl->engine->EndStep();
  impl->step_open = false;
}

void AdiosWriter::close()
{
  if (impl->engine == nullptr) {
    return;
  }

  std::string error;
  try {
    if (impl->step_open) {
      end_step();
    }
    impl->engine->Close();
  } catch (const std::exception& exception) {
    error = exception.what();
  }
  impl->engine.reset();

  int local_status  = error.empty() ? 1 : 0;
  int global_status = 0;
  MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (global_status == 0) {
    if (impl->info->world_rank == 0 && error.empty()) {
      error = "ADIOS2 diagnostic close failed on another rank";
    }
    broadcast_rank0_error(error);
  }

  if (impl->info->world_rank == 0) {
    try {
      std::filesystem::rename(impl->temporary_filename, impl->filename);
      nix::sync_directory(impl->filename.parent_path().string());
    } catch (const std::exception& exception) {
      error = exception.what();
    }
  }
  broadcast_rank0_error(error);
}
} // namespace nix
