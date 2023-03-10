// -*- C++ -*-
#include "expic3d.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Order>                                                                             \
  type ExPIC3D<Order>::name

DEFINE_MEMBER(, ExPIC3D)
(int argc, char** argv) : BaseApp(argc, argv), Ns(1), momstep(-1)
{
}

DEFINE_MEMBER(void, parse_cfg)()
{
  // read configuration file
  {
    std::ifstream f(cfg_file.c_str());
    cfg_json = json::parse(f, nullptr, true, true);
  }

  // application
  {
    json application = cfg_json["application"];
  }

  // parameters
  {
    json parameter = cfg_json["parameter"];

    // number of grids and chunks
    int nx = parameter["Nx"].get<int>();
    int ny = parameter["Ny"].get<int>();
    int nz = parameter["Nz"].get<int>();
    int cx = parameter["Cx"].get<int>();
    int cy = parameter["Cy"].get<int>();
    int cz = parameter["Cz"].get<int>();

    // other parameters
    float64 delh = parameter["delh"].get<float64>();

    Ns   = parameter["Ns"].get<int>();
    cc   = parameter["cc"].get<float64>();
    delt = parameter["delt"].get<float64>();
    delx = delh;
    dely = delh;
    delz = delh;

    // check dimensions
    if (!(nz % cz == 0 && ny % cy == 0 && nx % cx == 0)) {
      ERROR << tfm::format("Number of grid must be divisible by number of chunk");
      ERROR << tfm::format("Nx, Ny, Nz = [%4d, %4d, %4d]", nx, ny, nz);
      ERROR << tfm::format("Cx, Cy, Cz = [%4d, %4d, %4d]", cx, cy, cz);
      this->finalize(-1);
      exit(-1);
    }

    // global number of grid
    ndims[0] = nz;
    ndims[1] = ny;
    ndims[2] = nx;
    ndims[3] = ndims[0] * ndims[1] * ndims[2];

    // global number of chunk
    cdims[0] = cz;
    cdims[1] = cy;
    cdims[2] = cx;
    cdims[3] = cdims[0] * cdims[1] * cdims[2];

    // global domain size
    xlim[0] = 0;
    xlim[1] = delx * ndims[2];
    xlim[2] = xlim[1] - xlim[0];
    ylim[0] = 0;
    ylim[1] = dely * ndims[1];
    ylim[2] = ylim[1] - ylim[0];
    zlim[0] = 0;
    zlim[1] = delz * ndims[0];
    zlim[2] = zlim[1] - zlim[0];
  }

  // check diagnostic
  if (cfg_json["diagnostic"].is_array() == false) {
    ERROR << tfm::format("Invalid diagnostic");
  }
}

DEFINE_MEMBER(void, diagnostic_field)(std::ostream& out, json& obj)
{
  const int nz = ndims[0] / cdims[0];
  const int ny = ndims[1] / cdims[1];
  const int nx = ndims[2] / cdims[2];
  const int ns = Ns;
  const int nc = cdims[3];

  // get parameters from json
  std::string prefix = obj.value("prefix", "field");
  std::string path   = obj.value("path", ".") + "/";

  // filename
  std::string fn_prefix = tfm::format("%s_%s", prefix, nix::format_step(curstep));
  std::string fn_json   = fn_prefix + ".json";
  std::string fn_data   = fn_prefix + ".data";

  MPI_File fh;
  size_t   disp;
  json     dataset;

  jsonio::open_file((path + fn_data).c_str(), &fh, &disp, "w");

  //
  // coordinate
  //
  {
    const char name[]  = "xc";
    const char desc[]  = "x coordinate";
    const int  ndim    = 2;
    const int  dims[2] = {nc, nx};
    const int  size    = nc * nx * sizeof(float64);

    jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
    this->write_chunk_all(fh, disp, Chunk::DiagnosticX);
  }

  {
    const char name[]  = "yc";
    const char desc[]  = "y coordinate";
    const int  ndim    = 2;
    const int  dims[2] = {nc, ny};
    const int  size    = nc * ny * sizeof(float64);

    jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
    this->write_chunk_all(fh, disp, Chunk::DiagnosticY);
  }

  {
    const char name[]  = "zc";
    const char desc[]  = "z coordinate";
    const int  ndim    = 2;
    const int  dims[2] = {nc, nz};
    const int  size    = nc * nz * sizeof(float64);

    jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
    this->write_chunk_all(fh, disp, Chunk::DiagnosticZ);
  }

  //
  // electromagnetic field
  //
  {
    const char name[]  = "uf";
    const char desc[]  = "electromagnetic field";
    const int  ndim    = 5;
    const int  dims[5] = {nc, nz, ny, nx, 6};
    const int  size    = nc * nz * ny * nx * 6 * sizeof(float64);

    jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
    this->write_chunk_all(fh, disp, Chunk::DiagnosticEmf);
  }

  //
  // current
  //
  {
    const char name[]  = "uj";
    const char desc[]  = "current";
    const int  ndim    = 5;
    const int  dims[5] = {nc, nz, ny, nx, 4};
    const int  size    = nc * nz * ny * nx * 4 * sizeof(float64);

    jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
    this->write_chunk_all(fh, disp, Chunk::DiagnosticCur);
  }

  //
  // moment
  //
  {
    const char name[]  = "um";
    const char desc[]  = "moment";
    const int  ndim    = 6;
    const int  dims[6] = {nc, nz, ny, nx, ns, 11};
    const int  size    = nc * nz * ny * nx * ns * 11 * sizeof(float64);

    // calculate moment if not cached
    calculate_moment();

    // write
    jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
    this->write_chunk_all(fh, disp, Chunk::DiagnosticMom);
  }

  jsonio::close_file(&fh);

  //
  // output json file
  //
  {
    json root;
    json cmap;

    // convert chunkmap into json
    chunkmap->save_json(cmap);

    // meta data
    root["meta"] = {{"endian", nix::get_endian_flag()},
                    {"rawfile", fn_data},
                    {"order", 1},
                    {"time", curtime},
                    {"step", curstep}};
    // chunkmap
    root["chunkmap"] = cmap;
    // dataset
    root["dataset"] = dataset;

    if (thisrank == 0) {
      std::ofstream ofs(path + fn_json);
      ofs << std::setw(2) << root;
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

DEFINE_MEMBER(void, diagnostic_particle)(std::ostream& out, json& obj)
{
  const int nz = ndims[0] / cdims[0];
  const int ny = ndims[1] / cdims[1];
  const int nx = ndims[2] / cdims[2];
  const int ns = Ns;
  const int nc = cdims[3];

  // get parameters from json
  std::string prefix = obj.value("prefix", "particle");
  std::string path   = obj.value("path", ".") + "/";

  // filename
  std::string fn_prefix = tfm::format("%s_%s", prefix, nix::format_step(curstep));
  std::string fn_json   = fn_prefix + ".json";
  std::string fn_data   = fn_prefix + ".data";

  MPI_File fh;
  size_t   disp;
  json     dataset;

  jsonio::open_file((path + fn_data).c_str(), &fh, &disp, "w");

  //
  // for each particle
  //
  for (int is = 0; is < Ns; is++) {
    // write particles
    size_t disp0 = disp;
    int    mode  = Chunk::DiagnosticParticle + is;
    this->write_chunk_all(fh, disp, mode);

    // meta data
    {
      std::string name = tfm::format("up%02d", is);
      std::string desc = tfm::format("particle species %02d", is);

      const int Np      = (disp - disp0) / (Particle::Nc * sizeof(float64));
      const int ndim    = 2;
      const int dims[2] = {Np, Particle::Nc};
      const int size    = Np * Particle::Nc * sizeof(float64);

      jsonio::put_metadata(dataset, name.c_str(), "f8", desc.c_str(), disp0, size, ndim, dims);
    }
  }

  jsonio::close_file(&fh);

  //
  // output json file
  //
  {
    json root;

    // meta data
    root["meta"] = {{"endian", nix::get_endian_flag()},
                    {"rawfile", fn_data},
                    {"order", 1},
                    {"time", curtime},
                    {"step", curstep}};
    // dataset
    root["dataset"] = dataset;

    if (thisrank == 0) {
      std::ofstream ofs(path + fn_json);
      ofs << std::setw(2) << root;
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

DEFINE_MEMBER(void, calculate_moment)()
{
  if (curstep == momstep)
    return;

  std::set<int> queue;

  for (int i = 0; i < numchunk; i++) {
    chunkvec[i]->deposit_moment();
    chunkvec[i]->set_boundary_begin(Chunk::BoundaryMom);
    queue.insert(i);
  }

  this->wait_bc_exchange(queue, Chunk::BoundaryMom);

  // cache
  momstep = curstep;
}

DEFINE_MEMBER(void, diagnostic_history)(std::ostream& out, json& obj)
{
  std::vector<float64> history(Ns + 4);

  // clear
  std::fill(history.begin(), history.end(), 0.0);

  // calculate moment if not cached
  calculate_moment();

  // calculate divergence error and energy
  for (int i = 0; i < numchunk; i++) {
    float64 div_e = 0;
    float64 div_b = 0;
    float64 ene_e = 0;
    float64 ene_b = 0;
    float64 ene_p[Ns];

    chunkvec[i]->get_diverror(div_e, div_b);
    chunkvec[i]->get_energy(ene_e, ene_b, ene_p);

    history[0] += div_e;
    history[1] += div_b;
    history[2] += ene_e;
    history[3] += ene_b;
    for (int is = 0; is < Ns; is++) {
      history[is + 4] += ene_p[is];
    }
  }

  {
    void* sndptr = history.data();
    void* rcvptr = nullptr;

    if (thisrank == 0) {
      sndptr = MPI_IN_PLACE;
      rcvptr = history.data();
    }

    MPI_Reduce(sndptr, rcvptr, Ns + 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  // output from root
  if (thisrank == 0) {
    // get parameters from json
    std::string prefix = obj.value("prefix", "history");
    std::string path   = obj.value("path", ".") + "/";
    std::string msg    = "";

    // initial call
    if (curstep == 0) {
      // header
      msg += tfm::format("# %8s %15s", "step", "time");
      msg += tfm::format(" %15s", "div(E)");
      msg += tfm::format(" %15s", "div(B)");
      msg += tfm::format(" %15s", "E^2/2");
      msg += tfm::format(" %15s", "B^2/2");
      for (int is = 0; is < Ns; is++) {
        msg += tfm::format("    Particle #%02d", is);
      }
      msg += "\n";

      // clear file
      std::ofstream ofs(path + prefix + ".txt", nix::text_write);
      ofs.close();
    }

    msg += tfm::format("  %8s %15.6e", nix::format_step(curstep), curtime);
    msg += tfm::format(" %15.6e", history[0]);
    msg += tfm::format(" %15.6e", history[1]);
    msg += tfm::format(" %15.6e", history[2]);
    msg += tfm::format(" %15.6e", history[3]);
    for (int is = 0; is < Ns; is++) {
      msg += tfm::format(" %15.6e", history[is + 4]);
    }
    msg += "\n";

    // output to steam
    out << msg << std::flush;

    // append to file
    {
      std::ofstream ofs(path + prefix + ".txt", nix::text_append);
      ofs << msg << std::flush;
      ofs.close();
    }
  }
}

DEFINE_MEMBER(void, initialize)(int argc, char** argv)
{
  // parse command line arguments
  this->parse_cmd(argc, argv);

  // parse configuration file
  this->parse_cfg();

  // some initial setup
  curstep = 0;
  curtime = 0.0;
  this->initialize_mpi(&argc, &argv);
  this->initialize_debugprinting(this->debug);
  this->initialize_chunkmap();

  // load balancer
  balancer = this->create_balancer();

  // buffer allocation
  int bufsize = 1024 * 16;
  sendbuf.resize(bufsize);
  recvbuf.resize(bufsize);

  // set auxiliary information for chunk
  for (int i = 0; i < numchunk; i++) {
    int ix, iy, iz;
    int offset[3];

    chunkmap->get_coordinate(chunkvec[i]->get_id(), iz, iy, ix);
    offset[0] = iz * ndims[0] / cdims[0];
    offset[1] = iy * ndims[1] / cdims[1];
    offset[2] = ix * ndims[2] / cdims[2];
    chunkvec[i]->set_global_context(offset, ndims);
  }

  // initialize communicators
  for (int mode = 0; mode < Chunk::NumBoundaryMode; mode++) {
    for (int iz = 0; iz < 3; iz++) {
      for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
          MPI_Comm_dup(MPI_COMM_WORLD, &mpicommvec(mode, iz, iy, ix));
        }
      }
    }
  }
}

DEFINE_MEMBER(void, set_chunk_communicator)()
{
  for (int i = 0; i < numchunk; i++) {
    for (int mode = 0; mode < Chunk::NumBoundaryMode; mode++) {
      for (int iz = 0; iz < 3; iz++) {
        for (int iy = 0; iy < 3; iy++) {
          for (int ix = 0; ix < 3; ix++) {
            chunkvec[i]->set_mpi_communicator(mode, iz, iy, ix, mpicommvec(mode, iz, iy, ix));
          }
        }
      }
    }
  }
}

DEFINE_MEMBER(void, setup)()
{
  // set MPI communicator for each mode
  set_chunk_communicator();

  // setup for each chunk with boundary condition
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      // setup for physical conditions
      chunkvec[i]->setup(cfg_json["parameter"]);

      // begin boundary exchange for field
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryEmf);
    }
  }
}
DEFINE_MEMBER(bool, rebuild_chunkmap)()
{
  if (BaseApp::rebuild_chunkmap()) {
    // reset MPI communicator for each mode
    set_chunk_communicator();
    return true;
  }

  return false;
}

DEFINE_MEMBER(void, push)()
{
  float64 wclock1, wclock2;

  wclock1 = nix::wall_clock();

#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      // reset load
      chunkvec[i]->reset_load();

      // push B for a half step
      chunkvec[i]->push_bfd(0.5 * delt);

      // push particle
      chunkvec[i]->push_velocity(delt);
      chunkvec[i]->push_position(delt);

      // calculate current
      chunkvec[i]->deposit_current(delt);

      // begin boundary exchange for current
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryCur);

      // begin boundary exchange for particle
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryParticle);

      // push B for a half step
      chunkvec[i]->push_bfd(0.5 * delt);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryCur);

      // push E
      chunkvec[i]->push_efd(delt);

      // begin boundary exchange for field
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryParticle);
    }

#pragma omp for schedule(dynamic)
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_boundary_end(Chunk::BoundaryEmf);
    }
  }

  wclock2 = nix::wall_clock();

  // log
  {
    json push     = {{"start", wclock1}, {"end", wclock2}, {"elapsed", wclock2 - wclock1}};
    json log_step = {{"push", push}};
    this->append_log_step(log_step);
  }
}

DEFINE_MEMBER(void, diagnostic)(std::ostream& out)
{
  json diagnostic = cfg_json["diagnostic"];

  // iterate over diagnostics
  for (json::iterator it = diagnostic.begin(); it != diagnostic.end(); ++it) {
    json        obj = *it;
    std::string name;
    int         interval;

    try {
      name     = obj["name"].get<std::string>();
      interval = obj["interval"].get<int>();
    } catch (json::exception& e) {
      std::string msg = tfm::format("Error in diagnostic (ignored)\n%s\n", e.what());
      ERROR << tfm::format(msg.c_str());
      continue;
    }

    if (curstep % interval != 0)
      continue;

    //
    // call specific diagnostic routines
    //
    if (name == "history") {
      diagnostic_history(out, obj);
    }

    if (name == "field") {
      diagnostic_field(out, obj);
    }

    if (name == "particle") {
      diagnostic_particle(out, obj);
    }
  }
}

template class ExPIC3D<1>;
// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
