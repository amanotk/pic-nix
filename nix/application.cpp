// -*- C++ -*-

#include "application.hpp"

#include "chunk.hpp"
#include "diag.hpp"
#include "mpistream.hpp"

NIX_NAMESPACE_BEGIN

json Application::to_json()
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

bool Application::from_json(json& state)
{
  json current_state = to_json();

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

int Application::main()
{

  initialize(cl_argc, cl_argv);
  DEBUG1 << tfm::format("initialize");

  setup_chunks();
  DEBUG1 << tfm::format("setup_chunks");

  save_profile();

  while (is_push_needed()) {

    diagnostic();
    DEBUG1 << tfm::format("step[%s] diagnostic", format_step(curstep));

    push();
    DEBUG1 << tfm::format("step[%s] push", format_step(curstep));

    rebalance();
    DEBUG1 << tfm::format("step[%s] rebalance", format_step(curstep));

    take_log();
    DEBUG1 << tfm::format("step[%s] logging", format_step(curstep));

    increment_time();

    if (get_available_etime() < 0) {
      DEBUG1 << tfm::format("step[%s] run out of time", format_step(curstep));
      break;
    }
  }

  DEBUG1 << tfm::format("finalize");
  finalize();

  return 0;
}

void Application::initialize(int argc, char** argv)
{
  curstep = 0;
  curtime = 0.0;

  argparser = create_argparser();
  argparser->parse_check(argc, argv);

  cfgparser = create_cfgparser();
  cfgparser->parse_file(argparser->get_config());

  initialize_mpi(&argc, &argv);

  statehandler = create_statehandler();
  balancer     = create_balancer();
  logger       = create_logger();
  chunkmap     = create_chunkmap();

  initialize_debugprinting();
  initialize_dimensions();
  initialize_domain();
  initialize_diagnostic();
}

void Application::finalize()
{
  logger->flush();

  if (argparser->get_save() != "") {
    statehandler->save(*this, get_internal_data(), argparser->get_save());
  }

  finalize_mpi();
}

void Application::initialize_mpi(int* argc, char*** argv)
{
  nthread = nix::get_max_threads();

  {
    int thread_required = NIX_MPI_THREAD_LEVEL;
    int thread_provided = -1;

    if (is_mpi_init_already_called == false) {
      MPI_Init_thread(argc, argv, thread_required, &thread_provided);
      is_mpi_init_already_called = true;
    } else {

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

  {
    namespace fs = std::filesystem;

    json        config            = cfgparser->get_application();
    std::string path              = "";
    int         max_files_per_dir = 1000;

    initialize_base_directory();

    if (config.contains("mpistream") == false) {

      MpiStream::initialize(path, max_files_per_dir);
    } else if (config["mpistream"].is_object() == true) {

      config            = config["mpistream"];
      path              = fs::path(get_basedir()) / config.value("path", path);
      max_files_per_dir = config.value("max_files_per_dir", max_files_per_dir);
      MpiStream::initialize(path, max_files_per_dir);
    } else if (config["mpistream"] == false) {

    } else {
      ERROR << tfm::format("Ignore invalid configuration for mpistream\n");
    }
  }
}

void Application::finalize_mpi()
{

  MpiStream::finalize();
  Diag::finalize();

  MPI_Finalize();
}

void Application::assert_mpi(bool condition, std::string msg)
{
  if (condition == false) {
    MpiStream::finalize();
    ERROR << msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}

void Application::initialize_base_directory()
{
  if (thisrank == 0 && is_initial_run() == true) {
    namespace fs = std::filesystem;

    std::string basedir = get_basedir();
    if (basedir != "" && fs::exists(basedir) == false) {
      fs::create_directory(basedir);
      nix::sync_directory(basedir);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

void Application::initialize_debugprinting()
{
  DebugPrinter::init();
  DebugPrinter::set_level(argparser->get_verbosity());
}

void Application::initialize_dimensions()
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

void Application::initialize_domain()
{
}

void Application::initialize_workload()
{
  balancer->fill_load(1.0);
}

void Application::initialize_diagnostic()
{
  Diag::initialize(get_basedir(), get_iomode());
}

void Application::setup_chunks_init()
{
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

  initialize_workload();
  auto boundary = balancer->assign_initial(nprocess);
  chunkmap->set_rank_boundary(boundary);

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

  for (int i = 0; i < chunkvec.size(); i++) {
    int ix, iy, iz;
    int offset[3];

    chunkmap->get_coordinate(chunkvec[i]->get_id(), iz, iy, ix);
    offset[0] = iz * ndims[0] / cdims[0];
    offset[1] = iy * ndims[1] / cdims[1];
    offset[2] = ix * ndims[2] / cdims[2];
    chunkvec[i]->set_global_context(offset, ndims);
  }

  for (int i = 0; i < chunkvec.size(); i++) {
    auto config      = cfgparser->get_parameter();
    config["option"] = cfgparser->get_application()["option"];
    chunkvec[i]->setup(config);
  }
}

void Application::setup_chunks()
{
  if (argparser->get_load() != "") {
    statehandler->load(*this, get_internal_data(), argparser->get_load());
  } else {
    setup_chunks_init();
  }

  assert_mpi(validate_chunks() == true, "invalid chunks after setup_chunks");
}

bool Application::validate_chunks()
{
  bool status = chunkvec.validate(chunkmap);

  MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

  return status;
}

bool Application::rebalance()
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

    balancer->update_global_load(get_internal_data());

    auto boundary = chunkmap->get_rank_boundary();
    boundary      = balancer->assign(boundary);

    balancer->sendrecv_chunk(*this, get_internal_data(), boundary);
    chunkmap->set_rank_boundary(boundary);
    chunkvec.set_neighbors(chunkmap);

    assert_mpi(validate_chunks() == true, "invalid chunks after rebalance");

    status = true;
  }

  if (loglevel >= 1 && curstep % interval == 0) {

    log["boundary"] = chunkmap->get_rank_boundary();
  }

  DEBUG2 << "rebalance() end";
  float64 wclock2 = nix::wall_clock();

  log["elapsed"] = wclock2 - wclock1;
  log["status"]  = status;
  logger->append(curstep, "rebalance", log);

  return status;
}

void Application::diagnostic()
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

void Application::save_profile()
{
  if (is_initial_run() == true) {
    statehandler->save_application(*this, get_internal_data(), "profile");
  }
}

bool Application::is_initial_run()
{
  return argparser->get_load() == "";
}

bool Application::is_push_needed()
{
  if (curtime < argparser->get_physical_time_max() + cfgparser->get_delt()) {
    return true;
  }
  return false;
}

std::string Application::get_basedir()
{
  return cfgparser->get_application().value("basedir", "");
}

std::string Application::get_iomode()
{
  return cfgparser->get_application().value("iomode", "mpiio");
}

float64 Application::get_available_etime()
{
  float64 etime;

  if (thisrank == 0) {
    etime = wall_clock() - wclock;
  }
  MPI_Bcast(&etime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return argparser->get_elapsed_time_max() - etime;
}

void Application::take_log()
{
  // timestamp
  json log = {{"unixtime", nix::wall_clock()}};
  logger->append(curstep, "timestamp", log);

  logger->log(curstep);
}

void Application::increment_time()
{
  curtime += cfgparser->get_delt();
  curstep++;
}

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
