// -*- C++ -*-
#ifndef _NIX_DIAG_ADIOS_HPP_
#define _NIX_DIAG_ADIOS_HPP_

#include "diag.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nix
{
/// Common lifecycle and typed variable support for persistent ADIOS diagnostic writers.
class AdiosWriter
{
public:
  using Dims = std::vector<std::size_t>;

  explicit AdiosWriter(std::shared_ptr<Diag::info_type> info);
  ~AdiosWriter();

  static bool available();
  static void prepare_fresh_run(const std::string& basedir, const std::string& prefix);

  void initialize(const std::string& diagnostic, const std::string& prefix, const json& config);

  void define_global_double(const std::string& name, const Dims& shape, const Dims& start,
                            const Dims& count);
  void define_global_int32(const std::string& name, const Dims& shape, const Dims& start,
                           const Dims& count);
  void define_joined_double(const std::string& name, std::size_t width, std::size_t count);
  void define_joined_uint64(const std::string& name, std::size_t count);

  void open();
  void begin_step(std::int64_t step, double time);
  void put_global_double(const std::string& name, const Dims& start, const Dims& count,
                         const double* data, std::size_t size);
  void put_global_int32(const std::string& name, const Dims& start, const Dims& count,
                        const std::int32_t* data, std::size_t size);
  void put_joined_double(const std::string& name, std::size_t count, const double* data);
  void put_joined_uint64(const std::string& name, std::size_t count, const std::uint64_t* data);
  void end_step();
  void rotate();
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;

  void open_segment(int segment, std::int64_t restart_step);
  void close_segment();
};
} // namespace nix

#endif
