// -*- C++ -*-
#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

#include "nix.hpp"

#include "argparser.hpp"
#include "balancer.hpp"
#include "cfgparser.hpp"
#include "chunkmap.hpp"
#include "chunkvector.hpp"
#include "logger.hpp"
#include "statehandler.hpp"

NIX_NAMESPACE_BEGIN

// forward declaration
class Chunk;
class Diag;

///
/// @brief Base Application class
///
class Application
{
public:
  struct InternalData; // forward declaration
  using this_type       = Application;
  using chunk_type      = Chunk;
  using diag_type       = Diag;
  using data_type       = InternalData;
  using PtrArgParser    = std::unique_ptr<ArgParser>;
  using PtrCfgParser    = std::unique_ptr<CfgParser>;
  using PtrStateHandler = std::unique_ptr<StateHandler>;
  using PtrBalancer     = std::unique_ptr<Balancer>;
  using PtrLogger       = std::unique_ptr<Logger>;
  using PtrChunkMap     = std::unique_ptr<ChunkMap>;
  using PtrChunk        = std::unique_ptr<Chunk>;
  using PtrDiag         = std::unique_ptr<Diag>;
  using ChunkVec        = ChunkVector<PtrChunk>;
  using DiagVec         = std::vector<PtrDiag>;

  ///
  /// @brief internal data struct
  ///
  struct InternalData {
    int*         ndims;
    int*         cdims;
    int&         thisrank;
    int&         nprocess;
    int&         nthread;
    int&         curstep;
    float64&     curtime;
    PtrChunkMap& chunkmap;
    ChunkVec&    chunkvec;
  };

  ///
  /// @brief return internal data struct
  ///
  InternalData get_internal_data()
  {
    return {ndims, cdims, thisrank, nprocess, nthread, curstep, curtime, chunkmap, chunkvec};
  }

protected:
  PtrArgParser    argparser;    ///< argument parser
  PtrCfgParser    cfgparser;    ///< configuration parser
  PtrStateHandler statehandler; ///< state handler
  PtrBalancer     balancer;     ///< load balancer
  PtrLogger       logger;       ///< logger
  PtrChunkMap     chunkmap;     ///< chunkmap
  ChunkVec        chunkvec;     ///< local chunks
  DiagVec         diagvec;      ///< diagnostic objects

  int     thisrank; ///< my rank
  int     nprocess; ///< number of mpi processes
  int     nthread;  ///< number of threads
  int     cl_argc;  ///< command-line argc
  char**  cl_argv;  ///< command-line argv
  float64 wclock;   ///< wall clock time at initialization
  int     ndims[4]; ///< global grid dimensions
  int     cdims[4]; ///< chunk dimensions
  int     curstep;  ///< current iteration step
  float64 curtime;  ///< current time

  bool is_mpi_init_already_called; ///< flag for testing purpose

public:
  /// @brief default constructor
  Application() : Application(0, nullptr)
  {
  }

  ///
  /// @brief constructor
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  Application(int argc, char** argv) : is_mpi_init_already_called(false)
  {
    cl_argc = argc;
    cl_argv = argv;
  }

  ///
  /// @brief factory to create argument parser
  /// @return parser object
  ///
  virtual std::unique_ptr<ArgParser> create_argparser()
  {
    return std::make_unique<ArgParser>();
  }

  ///
  /// @brief factory to create config parser
  /// @return parser object
  ///
  virtual std::unique_ptr<CfgParser> create_cfgparser()
  {
    return std::make_unique<CfgParser>();
  }

  ///
  /// @brief factory to create state handler
  /// @return state handler object
  ///
  virtual std::unique_ptr<StateHandler> create_statehandler()
  {
    return std::make_unique<StateHandler>(get_basedir());
  }

  ///
  /// @brief factory to create balancer
  /// @return balancer object
  ///
  virtual std::unique_ptr<Balancer> create_balancer()
  {
    auto parameter = cfgparser->get_parameter();

    int Cx = parameter.value("Cx", 1);
    int Cy = parameter.value("Cy", 1);
    int Cz = parameter.value("Cz", 1);
    return std::make_unique<Balancer>(Cz * Cy * Cx);
  }

  ///
  /// @brief factory to create logger
  /// @return logger object
  ///
  virtual std::unique_ptr<Logger> create_logger()
  {
    auto config = cfgparser->get_application()["log"];

    return std::make_unique<Logger>(config, get_basedir(), thisrank, is_initial_run());
  }

  ///
  /// @brief factory to create chunkmap
  /// @return chunkmap object
  ///
  virtual std::unique_ptr<ChunkMap> create_chunkmap()
  {
    auto parameter = cfgparser->get_parameter();

    int Cx = parameter.value("Cx", 1);
    int Cy = parameter.value("Cy", 1);
    int Cz = parameter.value("Cz", 1);
    return std::make_unique<ChunkMap>(Cz, Cy, Cx);
  }

  ///
  /// @brief factory to create chunk object
  /// @param dims local number of grids in each direction
  /// @param id chunk ID
  /// @return chunk object
  ///
  virtual std::unique_ptr<Chunk> create_chunk(const int dims[], const bool has_dim[], int id)
  {
    return std::make_unique<Chunk>(dims, has_dim, id);
  }

  ///
  /// @brief convert internal data to json object
  /// @return json object
  ///
  virtual json to_json();

  ///
  /// @brief restore internal data from json object
  /// @param obj json object
  ///
  virtual bool from_json(json& obj);

  ///
  /// @brief main loop of simulation
  /// @return return code of application
  ///
  virtual int main();

protected:
  ///
  /// @brief initialize application
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  virtual void initialize(int argc, char** argv);

  ///
  /// @brief finalize application
  ///
  virtual void finalize();

  ///
  /// @brief initialize MPI
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  void initialize_mpi(int* argc, char*** argv);

  ///
  /// @brief finalize MPI
  ///
  void finalize_mpi();

  ///
  /// @brief assert
  ///
  void assert_mpi(bool condition, std::string msg);

  ///
  /// @brief initialize base directory
  ///
  void initialize_base_directory();

  ///
  /// @brief initialize debug printing
  ///
  void initialize_debugprinting();

  ///
  /// @brief initialize dimensions
  ///
  virtual void initialize_dimensions();

  ///
  /// @brief initialize domain
  ///
  virtual void initialize_domain();

  ///
  /// @brief initialize work load array
  ///
  virtual void initialize_workload();

  ///
  /// @brief initialize diagnostic
  ///
  virtual void initialize_diagnostic();

  ///
  /// @brief setup chunks with initial condition
  ///
  virtual void setup_chunks_init();

  ///
  /// @brief setup chunks
  ///
  virtual void setup_chunks();

  ///
  /// @brief check the validity of chunks
  /// @return true if the chunks are appropriate
  ///
  virtual bool validate_chunks();

  ///
  /// @brief performing load balancing
  /// @return return true if rebalancing is performed and false otherwise
  ///
  virtual bool rebalance();

  ///
  /// @brief perform various diagnostics output
  ///
  virtual void diagnostic();

  ///
  /// @brief advance physical quantities by one step
  ///
  virtual void push()
  {
    // override me
  }

  ///
  /// @brief save profile of run
  ///
  virtual void save_profile();

  ///
  /// @brief check if this is the initial run or not
  /// @return true if no snapshot is specified and false otherwise
  ///
  virtual bool is_initial_run();

  ///
  /// @brief check if further push is needed or not
  /// @return true if the maximum physical time is not yet reached and false otherwise
  ///
  virtual bool is_push_needed();

  ///
  /// @brief get basedir from configuration file
  /// @return return basedir
  ///
  virtual std::string get_basedir();

  ///
  /// @brief get I/O mode from configuration file
  /// @return return I/O mode
  ///
  virtual std::string get_iomode();

  ///
  /// @brief get available elapsed time
  /// @return available elapsed time in second
  ///
  virtual float64 get_available_etime();

  ///
  /// @brief take log
  ///
  virtual void take_log();

  ///
  /// @brief increment step and physical time
  ///
  virtual void increment_time();
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
