// -*- C++ -*-

#include "diag/chunk_writer.hpp"
#include "diag/io_handler.hpp"
#include "diag/load.hpp"
#include "diag/resource.hpp"
#include "nix.hpp"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <type_traits>
#include <utility>

using namespace nix;

namespace
{
class MockResourceDiagBase
{
public:
  struct Interface;
  using PtrInterface = std::shared_ptr<Interface>;
  using data_type    = int;
  using chunk_type   = Chunk;
  using info_type    = Diag::info_type;

protected:
  PtrInterface interface;

public:
  MockResourceDiagBase(std::string name, PtrInterface interface)
      : interface(interface), name(std::move(name))
  {
  }

  virtual ~MockResourceDiagBase() = default;

  virtual void operator()(json& config)
  {
  }

  virtual bool require_diagnostic(int curstep, json& config)
  {
    return Diag::require_diagnostic_impl(curstep, config);
  }

  virtual bool is_initial_step(int curstep, json& config)
  {
    return Diag::is_initial_step_impl(curstep, config);
  }

  bool make_sure_directory_exists(std::string path)
  {
    return true;
  }

  std::string format_dirname(std::string prefix)
  {
    return prefix;
  }

  std::string                              name;
  static inline std::shared_ptr<info_type> info;
};

struct MockChunkData {
  std::vector<float64> load;
};

class MockPacker
{
public:
  using chunk_data_type = MockChunkData;

  virtual ~MockPacker() = default;

  virtual size_t operator()(chunk_data_type data, uint8_t* buffer, int address)
  {
    return address + data.load.size() * sizeof(float64);
  }

  virtual size_t operator()(Chunk* chunk, uint8_t* buffer, int address)
  {
    return address;
  }
};

template <typename T, typename = void>
struct has_is_completed : std::false_type {
};

template <typename T>
struct has_is_completed<T, std::void_t<decltype(std::declval<T&>().is_completed())>>
    : std::true_type {
};

template <typename T, typename = void>
struct has_wait : std::false_type {
};

template <typename T>
struct has_wait<T, std::void_t<decltype(std::declval<T&>().wait(0))>> : std::true_type {
};

template <typename T, typename = void>
struct has_wait_all : std::false_type {
};

template <typename T>
struct has_wait_all<T, std::void_t<decltype(std::declval<T&>().wait_all())>> : std::true_type {
};

template <typename T, typename = void>
struct has_test_all : std::false_type {
};

template <typename T>
struct has_test_all<T, std::void_t<decltype(std::declval<T&>().test_all())>> : std::true_type {
};
} // namespace

TEST_CASE("diagnostic trigger helpers are available from nix::Diag")
{
  json every_step = {{"interval", 1}};
  REQUIRE(Diag::require_diagnostic_impl(0, every_step));
  REQUIRE(Diag::require_diagnostic_impl(1, every_step));

  json bounded = {{"begin", 2}, {"end", 6}, {"interval", 2}};
  REQUIRE_FALSE(Diag::require_diagnostic_impl(1, bounded));
  REQUIRE(Diag::require_diagnostic_impl(2, bounded));
  REQUIRE_FALSE(Diag::require_diagnostic_impl(3, bounded));
  REQUIRE(Diag::require_diagnostic_impl(4, bounded));
  REQUIRE_FALSE(Diag::require_diagnostic_impl(7, bounded));

  REQUIRE(Diag::is_initial_step_impl(2, bounded));
  REQUIRE_FALSE(Diag::is_initial_step_impl(4, bounded));

  REQUIRE(Diag::get_prefix_impl(bounded, "fallback") == "fallback");
  bounded["prefix"] = "custom";
  REQUIRE(Diag::get_prefix_impl(bounded, "fallback") == "custom");
}

TEST_CASE("moved diagnostics have module-facing template shapes")
{
  using ResourceDiagType    = ResourceDiag<MockResourceDiagBase>;
  using ChunkDiagWriterType = ChunkDiagWriter<MockResourceDiagBase, MockPacker>;
  using LoadDiagType        = LoadDiag<MockResourceDiagBase, MockPacker>;

  REQUIRE(std::is_base_of_v<MockResourceDiagBase, ResourceDiagType>);
  REQUIRE(std::is_base_of_v<MockResourceDiagBase, ChunkDiagWriterType>);
  REQUIRE(std::is_base_of_v<ChunkDiagWriterType, LoadDiagType>);
  REQUIRE(std::is_constructible_v<ResourceDiagType, MockResourceDiagBase::PtrInterface>);
  REQUIRE(std::is_constructible_v<LoadDiagType, MockResourceDiagBase::PtrInterface>);
}

TEST_CASE("diagnostic handlers expose concrete backends")
{
  using WriteSignature = size_t (DiagIoHandler::*)(Buffer&, size_t&);

  REQUIRE(std::is_abstract_v<DiagIoHandler>);
  REQUIRE(std::is_base_of_v<DiagIoHandler, MpiioDiagIoHandler>);
  REQUIRE(std::is_base_of_v<DiagIoHandler, PosixDiagIoHandler>);
  REQUIRE(std::is_same_v<decltype(&DiagIoHandler::write), WriteSignature>);
  REQUIRE_FALSE(has_is_completed<DiagIoHandler>::value);
  REQUIRE_FALSE(has_wait<DiagIoHandler>::value);
  REQUIRE_FALSE(has_wait_all<DiagIoHandler>::value);
  REQUIRE_FALSE(has_test_all<DiagIoHandler>::value);
}
