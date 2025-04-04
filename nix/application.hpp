// -*- C++ -*-
#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

#include "argparser.hpp"
#include "balancer.hpp"
#include "buffer.hpp"
#include "cfgparser.hpp"
#include "chunkmap.hpp"
#include "chunkvector.hpp"
#include "logger.hpp"
#include "mpistream.hpp"
#include "nix.hpp"
#include "statehandler.hpp"
#include "tinyformat.hpp"
#include <nlohmann/json.hpp>

NIX_NAMESPACE_BEGIN

///
/// @brief Base Application class
/// @tparam Chunk Chunk type
/// @tparam ChunkMap ChunkMap type
///
template <typename Chunk, typename Diag>
class Application
{
public:
  struct InternalData; // forward declaration
  using this_type       = Application<Chunk, Diag>;
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
    std::string basedir = get_basedir();

    return std::make_unique<StateHandler>(basedir);
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
    auto        config  = cfgparser->get_application()["log"];
    std::string basedir = get_basedir();

    return std::make_unique<Logger>(config, basedir, thisrank, is_initial_run());
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
  }

  ///
  /// @brief save profile of run
  ///
  virtual void save_profile()
  {
    if (is_initial_run() == true) {
      statehandler->save_application(*this, get_internal_data(), "profile");
    }
  }

  ///
  /// @brief check if this is the initial run or not
  /// @return true if no snapshot is specified and false otherwise
  ///
  virtual bool is_initial_run()
  {
    return argparser->get_load() == "";
  }

  ///
  /// @brief check if further push is needed or not
  /// @return true if the maximum physical time is not yet reached and false otherwise
  ///
  virtual bool is_push_needed()
  {
    if (curtime < argparser->get_physical_time_max() + cfgparser->get_delt()) {
      return true;
    }
    return false;
  }

  ///
  /// @brief get basedir from configuration file
  /// @return return basedir
  ///
  virtual std::string get_basedir()
  {
    return cfgparser->get_application().value("basedir", "");
  }

  ///
  /// @brief get I/O mode from configuration file
  /// @return return I/O mode
  ///
  virtual std::string get_iomode()
  {
    return cfgparser->get_application().value("iomode", "mpiio");
  }

  ///
  /// @brief get available elapsed time
  /// @return available elapsed time in second
  ///
  virtual float64 get_available_etime()
  {
    float64 etime;

    if (thisrank == 0) {
      etime = wall_clock() - wclock;
    }
    MPI_Bcast(&etime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return argparser->get_elapsed_time_max() - etime;
  }

  ///
  /// @brief take log
  ///
  virtual void take_log()
  {
    // timestamp
    json log = {{"unixtime", nix::wall_clock()}};
    logger->append(curstep, "timestamp", log);

    logger->log(curstep);
  }

  ///
  /// @brief increment step and physical time
  ///
  virtual void increment_time()
  {
    curtime += cfgparser->get_delt();
    curstep++;
  }
};

//
// implementation follows
//

#define DEFINE_MEMBER(type, name)                                                                  \
  template <typename Chunk, typename Diag>                                                         \
  type Application<Chunk, Diag>::name

DEFINE_MEMBER(json, to_json)()
{
  json state = {{"timestamp", nix::wall_clock()},
                {"wclock", wclock},
                {"ndims", ndims},
                {"cdims", cdims},
                {"curstep", curstep},
                {"curtime", curtime},
                {"thisrank", thisrank},
                {"nprocess", nprocess},
                {"nthread", nthread},
                {"configuration", cfgparser->get_root()},
                {"chunkmap", chunkmap->to_json()}};

  return state;
}

DEFINE_MEMBER(bool, from_json)(json& state)
{
  json current_state = to_json();

  // check consistency
  bool consistency = true;

  consistency &= current_state["ndims"] == state["ndims"];
  consistency &= current_state["cdims"] == state["cdims"];
  consistency &= current_state["nprocess"] == state["nprocess"];
  consistency &= current_state["configuration"]["parameter"] == state["configuration"]["parameter"];

  if (consistency == false) {
    ERROR << tfm::format("Trying to load inconsistent state");
  } else {
    curstep = state["curstep"].get<int>();
    curtime = state["curtime"].get<float64>();
    chunkmap->from_json(state["chunkmap"]);
  }

  return consistency;
}

DEFINE_MEMBER(int, main)()
{
  //
  // initialize the application
  //
  initialize(cl_argc, cl_argv);
  DEBUG1 << tfm::format("initialize");

  //
  // set initial condition
  //
  setup_chunks();
  DEBUG1 << tfm::format("setup_chunks");

  //
  // save profile
  //
  save_profile();

  //
  // main loop
  //
  while (is_push_needed()) {
    //
    // output diagnostics
    //
    diagnostic();
    DEBUG1 << tfm::format("step[%s] diagnostic", format_step(curstep));

    //
    // advance physical quantities by one step
    //
    push();
    DEBUG1 << tfm::format("step[%s] push", format_step(curstep));

    //
    // perform rebalance
    //
    rebalance();
    DEBUG1 << tfm::format("step[%s] rebalance", format_step(curstep));

    //
    // take log
    //
    take_log();
    DEBUG1 << tfm::format("step[%s] logging", format_step(curstep));

    //
    // increment step and time
    //
    increment_time();

    //
    // exit if elapsed time exceeds the limit
    //
    if (get_available_etime() < 0) {
      DEBUG1 << tfm::format("step[%s] run out of time", format_step(curstep));
      break;
    }
  }

  //
  // finalize the application
  //
  DEBUG1 << tfm::format("finalize");
  finalize();

  return 0;
}

DEFINE_MEMBER(void, initialize)(int argc, char** argv)
{
  curstep = 0;
  curtime = 0.0;

  // parse command line arguments
  argparser = create_argparser();
  argparser->parse_check(argc, argv);

  // parse configuration file
  cfgparser = create_cfgparser();
  cfgparser->parse_file(argparser->get_config());

  initialize_mpi(&argc, &argv);

  // object initialization
  statehandler = create_statehandler();
  balancer     = create_balancer();
  logger       = create_logger();
  chunkmap     = create_chunkmap();

  // misc
  initialize_debugprinting();
  initialize_dimensions();
  initialize_domain();
  initialize_diagnostic();
}

DEFINE_MEMBER(void, finalize)()
{
  logger->flush();

  // save snapshot
  if (argparser->get_save() != "") {
    statehandler->save(*this, get_internal_data(), argparser->get_save());
  }

  finalize_mpi();
}

DEFINE_MEMBER(void, initialize_base_directory)()
{
  if (thisrank == 0 && is_initial_run() == true) {
    namespace fs = std::filesystem;

    std::string basedir = get_basedir();
    if (basedir != "" && fs::exists(basedir) == false) {
      fs::create_directory(basedir);
      nix::sync_directory(basedir);
    }
  }
  // synchronize
  MPI_Barrier(MPI_COMM_WORLD);
}

DEFINE_MEMBER(void, initialize_mpi)(int* argc, char*** argv)
{
  nthread = nix::get_max_threads();

  // initialize MPI with thread support
  {
    int thread_required = NIX_MPI_THREAD_LEVEL;
    int thread_provided = -1;

    if (is_mpi_init_already_called == false) {
      MPI_Init_thread(argc, argv, thread_required, &thread_provided);
      is_mpi_init_already_called = true;
    } else {
      // MPI_Init should be already called when doing unit test
      MPI_Init_thread(nullptr, nullptr, thread_required, &thread_provided);
    }

    if (thread_provided < thread_required) {
      ERROR << tfm::format("Your MPI does not support required thread level!");
      MPI_Finalize();
      exit(-1);
    }
  }

  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
  MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);

  wclock = wall_clock();
  MPI_Bcast(&wclock, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // base directory and stdout/stderr redirection
  {
    namespace fs = std::filesystem;

    json        config            = cfgparser->get_application();
    std::string path              = "";
    int         max_files_per_dir = 1000;

    initialize_base_directory();

    if (config.contains("mpistream") == false) {
      // redirect to /dev/null except for rank 0 by default
      MpiStream::initialize(path, max_files_per_dir);
    } else if (config["mpistream"].is_object() == true) {
      // redirect with user setting
      config            = config["mpistream"];
      path              = fs::path(get_basedir()) / config.value("path", path);
      max_files_per_dir = config.value("max_files_per_dir", max_files_per_dir);
      MpiStream::initialize(path, max_files_per_dir);
    } else if (config["mpistream"] == false) {
      // no redirection
    } else {
      ERROR << tfm::format("Ignore invalid configuration for mpistream\n");
    }
  }
}

DEFINE_MEMBER(void, finalize_mpi)()
{
  // these must be called before MPI_Finalize
  MpiStream::finalize();
  Diag::finalize();

  MPI_Finalize();
}

DEFINE_MEMBER(void, assert_mpi)(bool condition, std::string msg)
{
  if (condition == false) {
    MpiStream::finalize();
    ERROR << msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}

DEFINE_MEMBER(void, initialize_debugprinting)()
{
  DebugPrinter::init();
  DebugPrinter::set_level(argparser->get_verbosity());
}

DEFINE_MEMBER(void, initialize_dimensions)()
{
  ndims[0] = cfgparser->get_Nz();
  ndims[1] = cfgparser->get_Ny();
  ndims[2] = cfgparser->get_Nx();
  ndims[3] = ndims[0] * ndims[1] * ndims[2];

  cdims[0] = cfgparser->get_Cz();
  cdims[1] = cfgparser->get_Cy();
  cdims[2] = cfgparser->get_Cx();
  cdims[3] = cdims[0] * cdims[1] * cdims[2];
}

DEFINE_MEMBER(void, initialize_domain)()
{
  // not necessary by default
}

DEFINE_MEMBER(void, initialize_workload)()
{
  balancer->fill_load(1.0);
}

DEFINE_MEMBER(void, initialize_diagnostic)()
{
  Diag::initialize(get_basedir(), get_iomode());
}

DEFINE_MEMBER(void, setup_chunks_init)()
{
  // error check
  {
    const int numchunk_global = cdims[3];

    if (numchunk_global < nprocess) {
      ERROR << tfm::format("Number of processes should not exceed number of chunks");
      ERROR << tfm::format("* number of processes = %8d", nprocess);
      ERROR << tfm::format("* number of chunks    = %8d", numchunk_global);
      finalize();
      exit(-1);
    }
  }

  // initial assignment
  initialize_workload();
  auto boundary = balancer->assign_initial(nprocess);
  chunkmap->set_rank_boundary(boundary);

  // create local chunks
  int  numchunk   = boundary[thisrank + 1] - boundary[thisrank];
  bool has_dim[3] = {
      (ndims[0] == 1 && cdims[0] == 1) ? false : true,
      (ndims[1] == 1 && cdims[1] == 1) ? false : true,
      (ndims[2] == 1 && cdims[2] == 1) ? false : true,
  };
  int dims[3]{
      ndims[0] / cdims[0],
      ndims[1] / cdims[1],
      ndims[2] / cdims[2],
  };

  // check dimension
  {
    bool is_1d = has_dim[0] == false && has_dim[1] == false && has_dim[2] == true;
    bool is_2d = has_dim[0] == false && has_dim[1] == true && has_dim[2] == true;
    bool is_3d = has_dim[0] == true && has_dim[1] == true && has_dim[2] == true;

    if (is_1d == false && is_2d == false && is_3d == false) {
      ERROR << tfm::format("Invalid dimension");
      ERROR << tfm::format("* has_dim = %d %d %d", has_dim[0], has_dim[1], has_dim[2]);
      finalize();
      exit(-1);
    }
  }

  chunkvec.resize(numchunk);

  for (int i = 0, id = boundary[thisrank]; id < boundary[thisrank + 1]; i++, id++) {
    chunkvec[i] = create_chunk(dims, has_dim, id);
  }
  chunkvec.set_neighbors(chunkmap);

  // set auxiliary information for chunk
  for (int i = 0; i < chunkvec.size(); i++) {
    int ix, iy, iz;
    int offset[3];

    chunkmap->get_coordinate(chunkvec[i]->get_id(), iz, iy, ix);
    offset[0] = iz * ndims[0] / cdims[0];
    offset[1] = iy * ndims[1] / cdims[1];
    offset[2] = ix * ndims[2] / cdims[2];
    chunkvec[i]->set_global_context(offset, ndims);
  }

  // setup initial condition
  for (int i = 0; i < chunkvec.size(); i++) {
    auto config      = cfgparser->get_parameter();
    config["option"] = cfgparser->get_application()["option"];
    chunkvec[i]->setup(config);
  }
}

DEFINE_MEMBER(void, setup_chunks)()
{
  if (argparser->get_load() != "") {
    statehandler->load(*this, get_internal_data(), argparser->get_load());
  } else {
    setup_chunks_init();
  }

  assert_mpi(validate_chunks() == true, "invalid chunks after setup_chunks");
}

DEFINE_MEMBER(bool, validate_chunks)()
{
  bool status = chunkvec.validate(chunkmap);

  MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

  return status;
}

DEFINE_MEMBER(bool, rebalance)()
{
  const int nchunk_global = cdims[3];

  bool status   = false;
  json log      = {};
  json config   = cfgparser->get_application()["rebalance"];
  int  interval = 100;
  int  loglevel = 0;

  DEBUG2 << "rebalance() start";
  float64 wclock1 = nix::wall_clock();

  if (config.is_null() == false) {
    interval = config.value("interval", interval);
    loglevel = config.value("loglevel", loglevel);
  }

  if (curstep > 0 && curstep % interval == 0) {
    // update global load of chunks
    balancer->update_global_load(get_internal_data());

    // find new assignment
    auto boundary = chunkmap->get_rank_boundary();
    boundary      = balancer->assign(boundary);

    // sned/recv chunks
    balancer->sendrecv_chunk(*this, get_internal_data(), boundary);
    chunkmap->set_rank_boundary(boundary);
    chunkvec.set_neighbors(chunkmap);

    assert_mpi(validate_chunks() == true, "invalid chunks after rebalance");

    status = true;
  }

  if (loglevel >= 1 && curstep % interval == 0) {
    // log assignment result
    log["boundary"] = chunkmap->get_rank_boundary();
  }

  DEBUG2 << "rebalance() end";
  float64 wclock2 = nix::wall_clock();

  log["elapsed"] = wclock2 - wclock1;
  log["status"]  = status;
  logger->append(curstep, "rebalance", log);

  return status;
}

DEFINE_MEMBER(void, diagnostic)()
{
  DEBUG2 << "diagnostic() start";
  float64 wclock1 = nix::wall_clock();

  json config = cfgparser->get_diagnostic();

  for (json::iterator it = config.begin(); it != config.end(); ++it) {
    auto element = *it;

    if (element.contains("name") == false)
      return;

    for (auto& diag : diagvec) {
      if (diag->match(element["name"])) {
        (*diag)(element);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  DEBUG2 << "diagnostic() end";
  float64 wclock2 = nix::wall_clock();

  json log = {{"elapsed", wclock2 - wclock1}};
  logger->append(curstep, "diagnostic", log);
}

#undef DEFINE_MEMBER

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
