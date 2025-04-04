// -*- C++ -*-

#include "application.hpp"
#include "chunk.hpp"
#include "chunkmap.hpp"
#include "diag.hpp"

#include "catch.hpp"

using namespace nix;

class TestApplication;

const std::string config_filename = "config.json";
const std::string config_content  = R"(
{
  "application": {
    "log": {
      "prefix": "log",
      "path": ".",
      "interval": 100
    },
    "rebalance": {
      "loglevel": 1,
      "interval": 100
    }
  },
  "diagnostic": [
    {
      "name": "foo",
      "prefix": "foo",
      "path": ".",
      "interval": 100
    },
    {
      "name": "bar",
      "prefix": "bar",
      "path": ".",
      "interval": 100
    }
  ],
  "parameter": {
    "Nx": 16,
    "Ny": 16,
    "Nz": 16,
    "Cx": 2,
    "Cy": 2,
    "Cz": 2,
    "delt": 1.0,
    "delh": 1.0
  }
}
)";

class MockChunk : public Chunk
{
public:
  using Chunk::Chunk;

  MockChunk(const int dims[3], const bool has_dim[3], int id) : Chunk(dims, has_dim, id)
  {
  }

  virtual int set_boundary_query(int mode, int sendrecv) override
  {
    return 0;
  }

  virtual void set_boundary_pack(int mode) override
  {
  }

  virtual void set_boundary_unpack(int mode) override
  {
  }

  virtual void set_boundary_begin(int mode) override
  {
  }

  virtual void set_boundary_end(int mode) override
  {
  }

  virtual void setup(json& config)
  {
  }

  virtual void set_global_context(const int* offset, const int* gdims)
  {
  }
};

class MockDiag : public Diag
{
};

class TestApplication : public Application<MockChunk, MockDiag>
{
public:
  TestApplication() : Application<MockChunk, MockDiag>()
  {
    is_mpi_init_already_called = true;

    std::ofstream ofs(config_filename);
    ofs << config_content;
  }

  ~TestApplication()
  {
    std::filesystem::remove(config_filename);
  }

  void test_main()
  {
    std::vector<std::string> args = {"./a.out", "-c", "config.json", "--emax", "1"};
    std::vector<const char*> argv = ArgParser::convert_to_clargs(args);

    cl_argc = argv.size();
    cl_argv = const_cast<char**>(&argv[0]);

    REQUIRE(main() == 0);

    std::filesystem::remove("profile.msgpack");
    std::filesystem::remove("log.msgpack");
  }
};

TEST_CASE_METHOD(TestApplication, "test_main")
{
  test_main();
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
