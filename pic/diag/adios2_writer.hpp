// -*- C++ -*-
#ifndef _PIC_ADIOS2_WRITER_HPP_
#define _PIC_ADIOS2_WRITER_HPP_

#include "nix/diag.hpp"

#if PICNIX_ENABLE_ADIOS2

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nix
{
/// Common lifecycle and typed variable support for persistent ADIOS2 diagnostic writers.
class Adios2Writer
{
public:
  using Dims = std::vector<std::size_t>;

  explicit Adios2Writer(std::shared_ptr<Diag::info_type> info);
  ~Adios2Writer();

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
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace nix

#endif

#endif
