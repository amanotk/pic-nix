// -*- C++ -*-

#include "diag/adios.hpp"

#include <catch2/catch_test_macros.hpp>

#include <adios2.h>
#include <mpi.h>

#include <cstdint>
#include <filesystem>
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
        {"application", {{"adios", {{"engine", "BP5"}}}}},
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
