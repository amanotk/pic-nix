// -*- C++ -*-

#include "diag/adios.hpp"

#include <catch2/catch_test_macros.hpp>

#include <adios2.h>
#include <mpi.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
class TestDiag : public nix::Diag
{
public:
  TestDiag() : Diag("adios-test")
  {
  }

  std::shared_ptr<info_type> get_info()
  {
    return info;
  }
};

void write_segment(const std::filesystem::path& basedir, bool is_restart, std::int64_t step,
                   double value)
{
  nix::Diag::initialize(basedir.string(), "adios", "", is_restart);
  if (is_restart) {
    nix::Diag::set_restart_step(static_cast<int>(step));
  }
  {
    TestDiag         diag;
    nix::AdiosWriter writer(diag.get_info());
    const nix::json  config = {
        {"application", {{"adios", nix::json::object()}}},
    };

    writer.initialize("field", "segments", config);
    writer.define_global_double("field", {1}, {0}, {1});
    writer.open();
    writer.begin_step(step, 0.5 * step);
    writer.put_global_double("field", {0}, {1}, &value, 1);
    writer.end_step();
    writer.close();
  }
  nix::Diag::finalize();
}

template <typename Callable>
std::string thrown_message(Callable&& callable)
{
  try {
    callable();
  } catch (const std::exception& exception) {
    return exception.what();
  }
  return "";
}

std::pair<std::int64_t, double> read_segment(const std::filesystem::path& path)
{
  adios2::ADIOS adios(MPI_COMM_WORLD);
  auto          io     = adios.DeclareIO("segment-reader");
  auto          engine = io.Open(path.string(), adios2::Mode::ReadRandomAccess);
  auto          step   = io.InquireVariable<std::int64_t>("step");
  auto          field  = io.InquireVariable<double>("field");
  field.SetSelection({{0}, {1}});

  std::int64_t        actual_step = -1;
  std::vector<double> actual_value(1);
  engine.Get(step, actual_step, adios2::Mode::Sync);
  engine.Get(field, actual_value, adios2::Mode::Sync);
  engine.Close();
  return {actual_step, actual_value[0]};
}
} // namespace

TEST_CASE("ADIOS writer round trip")
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE(rank == 0);

  const auto basedir = std::filesystem::temp_directory_path() / "picnix-adios-writer-test";
  std::filesystem::remove_all(basedir);
  std::filesystem::create_directories(basedir);
  MPI_Barrier(MPI_COMM_WORLD);

  nix::Diag::initialize(basedir.string(), "adios", "");
  {
    TestDiag         diag;
    nix::AdiosWriter writer(diag.get_info());
    const nix::json  config = {
        {"application", {{"adios", nix::json::object()}}},
    };

    writer.initialize("field", "roundtrip", config);
    writer.define_global_double("field", {1, 2}, {0, 0}, {1, 2});
    writer.define_global_int32("rank", {1}, {0}, {1});
    writer.define_joined_double("particles", 2, 2);
    writer.define_joined_uint64("particles_id", 2);
    writer.open();

    for (std::int64_t step = 0; step < 2; step++) {
      writer.begin_step(step, 0.25 * step);
      const double        values[]    = {1.0 + 2.0 * step, 2.0 + 2.0 * step};
      const double        particles[] = {10.0 + step, 11.0 + step, 12.0 + step, 13.0 + step};
      const std::uint64_t ids[]       = {100 + static_cast<std::uint64_t>(step),
                                         200 + static_cast<std::uint64_t>(step)};
      const std::int32_t  rank_value  = 3;
      writer.put_global_double("field", {0, 0}, {1, 2}, values, 2);
      writer.put_global_int32("rank", {0}, {1}, &rank_value, 1);
      writer.put_joined_double("particles", 2, particles);
      writer.put_joined_uint64("particles_id", 2, ids);
      writer.end_step();
    }
    writer.close();

    adios2::ADIOS adios(MPI_COMM_WORLD);
    auto          io = adios.DeclareIO("reader");
    auto          engine =
        io.Open((basedir / "adios" / "roundtrip.bp").string(), adios2::Mode::ReadRandomAccess);
    auto field        = io.InquireVariable<double>("field");
    auto particles    = io.InquireVariable<double>("particles");
    auto particle_ids = io.InquireVariable<std::uint64_t>("particles_id");
    auto rank         = io.InquireVariable<std::int32_t>("rank");
    auto step         = io.InquireVariable<std::int64_t>("step");
    auto time         = io.InquireVariable<double>("time");

    REQUIRE(field);
    REQUIRE(particles);
    REQUIRE(particle_ids);
    REQUIRE(rank);
    REQUIRE(step);
    REQUIRE(time);

    for (std::int64_t expected_step = 0; expected_step < 2; expected_step++) {
      field.SetStepSelection({static_cast<std::size_t>(expected_step), 1});
      particles.SetStepSelection({static_cast<std::size_t>(expected_step), 1});
      particle_ids.SetStepSelection({static_cast<std::size_t>(expected_step), 1});
      rank.SetStepSelection({static_cast<std::size_t>(expected_step), 1});
      step.SetStepSelection({static_cast<std::size_t>(expected_step), 1});
      time.SetStepSelection({static_cast<std::size_t>(expected_step), 1});
      field.SetSelection({{0, 0}, {1, 2}});
      particles.SetSelection({{0, 0}, {2, 2}});
      particle_ids.SetSelection({{0}, {2}});
      rank.SetSelection({{0}, {1}});

      std::vector<double>        values(2);
      std::vector<double>        particle_values(4);
      std::vector<std::uint64_t> ids(2);
      std::vector<std::int32_t>  rank_values(1);
      std::int64_t               actual_step = -1;
      double                     actual_time = -1.0;
      engine.Get(field, values, adios2::Mode::Sync);
      engine.Get(particles, particle_values, adios2::Mode::Sync);
      engine.Get(particle_ids, ids, adios2::Mode::Sync);
      engine.Get(rank, rank_values, adios2::Mode::Sync);
      engine.Get(step, actual_step, adios2::Mode::Sync);
      engine.Get(time, actual_time, adios2::Mode::Sync);
      REQUIRE(actual_step == expected_step);
      REQUIRE(actual_time == 0.25 * expected_step);
      REQUIRE(values[0] == 1.0 + 2.0 * expected_step);
      REQUIRE(values[1] == 2.0 + 2.0 * expected_step);
      REQUIRE(particle_values[0] == 10.0 + expected_step);
      REQUIRE(particle_values[3] == 13.0 + expected_step);
      REQUIRE(ids[0] == 100 + static_cast<std::uint64_t>(expected_step));
      REQUIRE(ids[1] == 200 + static_cast<std::uint64_t>(expected_step));
      REQUIRE(rank_values[0] == 3);
    }

    engine.Close();
  }
  nix::Diag::finalize();
  std::filesystem::remove_all(basedir);
}

TEST_CASE("ADIOS writer completes asynchronous BP5 output on close")
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE(rank == 0);

  const auto basedir = std::filesystem::temp_directory_path() / "picnix-adios-async-test";
  std::filesystem::remove_all(basedir);
  std::filesystem::create_directories(basedir);
  MPI_Barrier(MPI_COMM_WORLD);

  nix::Diag::initialize(basedir.string(), "adios", "");
  {
    TestDiag         diag;
    nix::AdiosWriter writer(diag.get_info());
    const nix::json  config = {
        {"application", {{"adios", {{"AsyncWrite", true}}}}},
    };

    writer.initialize("field", "async", config);
    writer.define_global_double("field", {1}, {0}, {1});
    writer.open();
    writer.begin_step(7, 1.5);
    const double value = 42.0;
    writer.put_global_double("field", {0}, {1}, &value, 1);
    writer.end_step();
    writer.close();

    adios2::ADIOS adios(MPI_COMM_WORLD);
    auto          io = adios.DeclareIO("async-reader");
    auto          engine =
        io.Open((basedir / "adios" / "async.bp").string(), adios2::Mode::ReadRandomAccess);
    auto field = io.InquireVariable<double>("field");
    REQUIRE(field);

    std::vector<double> actual(1);
    field.SetStepSelection({0, 1});
    field.SetSelection({{0}, {1}});
    engine.Get(field, actual, adios2::Mode::Sync);
    REQUIRE(actual[0] == value);
    engine.Close();
  }
  nix::Diag::finalize();
  std::filesystem::remove_all(basedir);
}

TEST_CASE("ADIOS writer segments restart output and replaces fresh output")
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE(rank == 0);

  const auto basedir = std::filesystem::temp_directory_path() / "picnix-adios-segment-test";
  std::filesystem::remove_all(basedir);
  std::filesystem::create_directories(basedir);
  MPI_Barrier(MPI_COMM_WORLD);

  const auto base  = basedir / "adios" / "segments.bp";
  const auto part1 = basedir / "adios" / "segments.part0001.bp";
  const auto part2 = basedir / "adios" / "segments.part0002.bp";
  const auto temp2 = basedir / "adios" / "segments.part0002.bp.tmp";
  write_segment(basedir, false, 0, 10.0);
  write_segment(basedir, true, 1, 20.0);
  std::filesystem::create_directories(temp2);
  write_segment(basedir, true, 2, 30.0);

  REQUIRE(std::filesystem::exists(base));
  REQUIRE(std::filesystem::exists(part1));
  REQUIRE(std::filesystem::exists(part2));
  REQUIRE_FALSE(std::filesystem::exists(temp2));
  REQUIRE(read_segment(base) == std::pair<std::int64_t, double>{0, 10.0});
  REQUIRE(read_segment(part1) == std::pair<std::int64_t, double>{1, 20.0});
  REQUIRE(read_segment(part2) == std::pair<std::int64_t, double>{2, 30.0});

  write_segment(basedir, false, 3, 40.0);
  REQUIRE(std::filesystem::exists(base));
  REQUIRE_FALSE(std::filesystem::exists(part1));
  REQUIRE_FALSE(std::filesystem::exists(part2));
  REQUIRE(read_segment(base) == std::pair<std::int64_t, double>{3, 40.0});

  std::filesystem::remove_all(basedir);
}

TEST_CASE("ADIOS writer rejects invalid restart segment state")
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE(rank == 0);

  const auto basedir = std::filesystem::temp_directory_path() / "picnix-adios-segment-error-test";
  std::filesystem::remove_all(basedir);
  std::filesystem::create_directories(basedir);
  MPI_Barrier(MPI_COMM_WORLD);

  const nix::json config = {
      {"application", {{"adios", nix::json::object()}}},
  };
  const auto base = basedir / "adios" / "segments.bp";
  const auto temp = basedir / "adios" / "segments.bp.tmp";

  SECTION("engine is fixed to BP5")
  {
    nix::Diag::initialize(basedir.string(), "adios", "");
    {
      TestDiag         diag;
      nix::AdiosWriter writer(diag.get_info());
      const nix::json  config = {
          {"application", {{"adios", {{"engine", "SST"}}}}},
      };
      REQUIRE(thrown_message([&] { writer.initialize("field", "segments", config); }) ==
              "ADIOS2 engine is fixed to BP5; remove application.adios.engine");
    }
    nix::Diag::finalize();
  }

  SECTION("restart step must be initialized")
  {
    write_segment(basedir, false, 0, 10.0);

    nix::Diag::initialize(basedir.string(), "adios", "", true);
    {
      TestDiag         diag;
      nix::AdiosWriter writer(diag.get_info());
      writer.initialize("field", "segments", config);
      writer.define_global_double("field", {1}, {0}, {1});
      REQUIRE(thrown_message([&] { writer.open(); }) == "ADIOS2 restart step was not initialized");
    }
    nix::Diag::finalize();

    REQUIRE(std::filesystem::exists(base));
    REQUIRE_FALSE(std::filesystem::exists(temp));
  }

  SECTION("segments must be contiguous")
  {
    write_segment(basedir, false, 0, 10.0);
    std::filesystem::rename(base, basedir / "adios" / "segments.part0002.bp");

    nix::Diag::initialize(basedir.string(), "adios", "", true);
    nix::Diag::set_restart_step(1);
    {
      TestDiag         diag;
      nix::AdiosWriter writer(diag.get_info());
      writer.initialize("field", "segments", config);
      writer.define_global_double("field", {1}, {0}, {1});
      REQUIRE(thrown_message([&] { writer.open(); }) ==
              "ADIOS2 diagnostic segments are not contiguous");
    }
    nix::Diag::finalize();

    REQUIRE(std::filesystem::exists(basedir / "adios" / "segments.part0002.bp"));
    REQUIRE_FALSE(std::filesystem::exists(temp));
  }

  std::filesystem::remove_all(basedir);
}

TEST_CASE("ADIOS fresh run cleanup is eager")
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  REQUIRE(rank == 0);

  const auto basedir = std::filesystem::temp_directory_path() / "picnix-adios-eager-cleanup-test";
  std::filesystem::remove_all(basedir);
  std::filesystem::create_directories(basedir);
  MPI_Barrier(MPI_COMM_WORLD);

  const auto base  = basedir / "adios" / "segments.bp";
  const auto part1 = basedir / "adios" / "segments.part0001.bp";
  const auto temp2 = basedir / "adios" / "segments.part0002.bp.tmp";
  write_segment(basedir, false, 0, 10.0);
  write_segment(basedir, true, 1, 20.0);
  std::filesystem::create_directories(temp2);

  REQUIRE(std::filesystem::exists(base));
  REQUIRE(std::filesystem::exists(part1));
  REQUIRE(std::filesystem::exists(temp2));

  nix::AdiosWriter::prepare_fresh_run(basedir.string(), "segments");

  REQUIRE_FALSE(std::filesystem::exists(base));
  REQUIRE_FALSE(std::filesystem::exists(part1));
  REQUIRE_FALSE(std::filesystem::exists(temp2));
  std::filesystem::remove_all(basedir);
}
