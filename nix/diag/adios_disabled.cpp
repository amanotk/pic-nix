// -*- C++ -*-

#include "adios.hpp"

#include <stdexcept>

namespace nix
{
struct AdiosWriter::Impl {
};

namespace
{
[[noreturn]] void unavailable()
{
  throw std::runtime_error("ADIOS support is unavailable; rebuild with PICNIX_ENABLE_ADIOS2=ON");
}
} // namespace

bool AdiosWriter::available()
{
  return false;
}

void AdiosWriter::prepare_fresh_run(const std::string&, const std::string&)
{
  unavailable();
}

AdiosWriter::AdiosWriter(std::shared_ptr<Diag::info_type>)
{
  unavailable();
}

AdiosWriter::~AdiosWriter() = default;

void AdiosWriter::initialize(const std::string&, const std::string&, const json&)
{
  unavailable();
}

void AdiosWriter::define_global_double(const std::string&, const Dims&, const Dims&, const Dims&)
{
  unavailable();
}

void AdiosWriter::define_global_int32(const std::string&, const Dims&, const Dims&, const Dims&)
{
  unavailable();
}

void AdiosWriter::define_joined_double(const std::string&, std::size_t, std::size_t)
{
  unavailable();
}

void AdiosWriter::define_joined_uint64(const std::string&, std::size_t)
{
  unavailable();
}

void AdiosWriter::open()
{
  unavailable();
}

void AdiosWriter::begin_step(std::int64_t, double)
{
  unavailable();
}

void AdiosWriter::put_global_double(const std::string&, const Dims&, const Dims&, const double*,
                                    std::size_t)
{
  unavailable();
}

void AdiosWriter::put_global_int32(const std::string&, const Dims&, const Dims&,
                                   const std::int32_t*, std::size_t)
{
  unavailable();
}

void AdiosWriter::put_joined_double(const std::string&, std::size_t, const double*)
{
  unavailable();
}

void AdiosWriter::put_joined_uint64(const std::string&, std::size_t, const std::uint64_t*)
{
  unavailable();
}

void AdiosWriter::end_step()
{
  unavailable();
}

void AdiosWriter::rotate()
{
  unavailable();
}

void AdiosWriter::close()
{
  unavailable();
}
} // namespace nix
