// -*- C++ -*-
#include "expic3d.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Order>                                                                             \
  type ExPIC3D<Order>::name

DEFINE_MEMBER(, ExPIC3D)(int argc, char **argv) : BaseApp(argc, argv), Ns(1)
{
}

DEFINE_MEMBER(void, parse_cfg)()
{
  // read configuration file
  {
    std::ifstream f(cfg_file.c_str());
    cfg_json = json::parse(f, nullptr, true, true);
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
    Ns   = parameter["Ns"].get<int>();
    cc   = parameter["cc"].get<float64>();
    delt = parameter["delt"].get<float64>();
    delh = parameter["delh"].get<float64>();

    // check dimensions
    if (!(nz % cz == 0 && ny % cy == 0 && nx % cx == 0)) {
      ERRORPRINT("Number of grid must be divisible by number of chunk\n"
                 "Nx, Ny, Nz = [%4d, %4d, %4d]\n"
                 "Cx, Cy, Cz = [%4d, %4d, %4d]\n",
                 nx, ny, nz, cx, cy, cz);
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
    xlim[1] = delh * ndims[2];
    xlim[2] = xlim[1] - xlim[0];
    ylim[0] = 0;
    ylim[1] = delh * ndims[1];
    ylim[2] = ylim[1] - ylim[0];
    zlim[0] = 0;
    zlim[1] = delh * ndims[0];
    zlim[2] = zlim[1] - zlim[0];
  }

  // check diagnostic
  if (cfg_json["diagnostic"].is_array() == false) {
    ERRORPRINT("Invalid diagnostic\n");
  }
}

DEFINE_MEMBER(void, diagnostic_load)(std::ostream &out, json &obj)
{
  // get parameters from json
  std::string prefix = obj.value("prefix", "load");
  std::string path   = obj.value("path", ".") + "/";

  // filename
  std::string fn_json = prefix + ".json";
  std::string fn_data = prefix + ".data";

  MPI_File fh;
  size_t   disp;

  //
  // initial setup
  //
  if (curstep == 0) {
    // data file
    jsonio::open_file((path + fn_data).c_str(), &fh, &disp, "w");
    jsonio::close_file(&fh);

    // json file
    if (thisrank == 0) {
      const char name[]  = "load";
      const char desc[]  = "chunk work load";
      const int  ndim    = 3;
      const int  dims[3] = {0, cdims[3], Chunk::NumLoadMode};
      const int  size    = 0;

      json root;
      json dataset;

      // meta data
      root["meta"] = {{"endian", common::get_endian_flag()},
                      {"rawfile", fn_data},
                      {"order", 1},
                      {"header", {"field push", "current deposit", "particle push"}}};

      // dataset
      jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
      root["dataset"] = dataset;

      std::ofstream ofs(path + fn_json);
      ofs << std::setw(2) << root;
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  //
  // output data file
  //
  {
    jsonio::open_file((path + fn_data).c_str(), &fh, &disp, "a");

    this->write_chunk_all(fh, disp, Chunk::DiagnosticLoad);

    jsonio::close_file(&fh);
  }

  //
  // output json file
  //
  {
    json root;

    // read json file
    {
      std::ifstream ifs(path + fn_json);
      ifs >> root;
    }

    // modify shape and size
    std::vector<int> shape = root["dataset"]["load"]["shape"];
    shape[0]++;

    root["dataset"]["load"]["shape"] = shape;
    root["dataset"]["load"]["size"]  = shape[0] * shape[1] * shape[2] * sizeof(float64);

    // write
    if (thisrank == 0) {
      std::ofstream ofs(path + fn_json);
      ofs << std::setw(2) << root;
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // reset
  for (int i = 0; i < numchunk; i++) {
    chunkvec[i]->reset_load();
  }
}

DEFINE_MEMBER(void, diagnostic_field)(std::ostream &out, json &obj)
{
  const int nz = ndims[0] / cdims[0];
  const int ny = ndims[1] / cdims[1];
  const int nx = ndims[2] / cdims[2];
  const int ns = Ns;
  const int nc = cdims[3];

  // console message
  if (thisrank == 0) {
    tfm::format(out, "step = %8d, time = %15.6e\n", curstep, curtime);
  }

  // get parameters from json
  std::string prefix = obj.value("prefix", "field");
  std::string path   = obj.value("path", ".") + "/";

  // filename
  std::string fn_prefix = tfm::format("%s_%06d", prefix, curstep);
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
    const int  dims[6] = {nc, nz, ny, nx, ns, 10};
    const int  size    = nc * nz * ny * nx * ns * 10 * sizeof(float64);

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
    root["meta"] = {{"endian", common::get_endian_flag()},
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

DEFINE_MEMBER(void, diagnostic_particle)(std::ostream &out, json &obj)
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
  std::string fn_prefix = tfm::format("%s_%06d", prefix, curstep);
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
    root["meta"] = {{"endian", common::get_endian_flag()},
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

DEFINE_MEMBER(void, initialize)(int argc, char **argv)
{
  // parse command line arguments
  this->parse_cmd(argc, argv);

  // parse configuration file
  this->parse_cfg();

  // some initial setup
  curstep     = 0;
  curtime     = 0.0;
  periodic[0] = 1;
  periodic[1] = 1;
  periodic[2] = 1;
  this->initialize_mpi_default(&argc, &argv);
  this->initialize_chunkmap();
  balancer.reset(new Balancer());

  // buffer allocation
  bufsize = 1024 * 16;
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

  // communicators
  mpicommvec.resize(Chunk::NumBoundaryMode);
  for (int mode = 0; mode < Chunk::NumBoundaryMode; mode++) {
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    mpicommvec[mode] = comm;
  }
}

DEFINE_MEMBER(void, setup)()
{
  // set MPI communicator for each mode
  for (int mode = 0; mode < Chunk::NumBoundaryMode; mode++) {
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_mpi_communicator(mode, mpicommvec[mode]);
    }
  }

  // setup for each chunk with boundary condition
  {
    std::set<int> bc_queue;

    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->setup(cfg_json["parameter"]);
      chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
      bc_queue.insert(i);
    }

    this->wait_bc_exchange(bc_queue, Chunk::BoundaryEmf);
  }
}

DEFINE_MEMBER(void, rebuild_chunkmap)()
{
  BaseApp::rebuild_chunkmap();

  // set MPI communicator for each mode
  for (int mode = 0; mode < Chunk::NumBoundaryMode; mode++) {
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_mpi_communicator(mode, mpicommvec[mode]);
    }
  }
}

DEFINE_MEMBER(void, push)()
{
  std::set<int> bc_queue_uf;
  std::set<int> bc_queue_uj;
  std::set<int> bc_queue_up;

  for (int i = 0; i < numchunk; i++) {
    // push B for a half step
    chunkvec[i]->push_bfd(0.5 * delt);

    // push particle
    chunkvec[i]->push_velocity(delt);
    chunkvec[i]->push_position(delt);

    // calculate current
    chunkvec[i]->deposit_current(delt);

    // begin boundary exchange for current and particles
    chunkvec[i]->set_boundary_begin(Chunk::BoundaryCur);
    bc_queue_uj.insert(i);
    chunkvec[i]->set_boundary_begin(Chunk::BoundaryParticle);
    bc_queue_up.insert(i);

    // push B for a half step
    chunkvec[i]->push_bfd(0.5 * delt);
  }

  // wait for current boundary exchange
  this->wait_bc_exchange(bc_queue_uj, Chunk::BoundaryCur);

  for (int i = 0; i < numchunk; i++) {
    // push E
    chunkvec[i]->push_efd(delt);

    // begin boundary exchange for field
    chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
    bc_queue_uf.insert(i);
  }

  // wait for particle and field boundary exchange
  this->wait_bc_exchange(bc_queue_up, Chunk::BoundaryParticle);
  this->wait_bc_exchange(bc_queue_uf, Chunk::BoundaryEmf);

  curtime += delt;
  curstep++;
}

DEFINE_MEMBER(void, diagnostic)(std::ostream &out)
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
    } catch (json::exception &e) {
      std::string msg = tfm::format("Error in diagnostic (ignored)\n%s\n", e.what());
      ERRORPRINT(msg.c_str());
      continue;
    }

    if (curstep % interval != 0)
      continue;

    //
    // call specific diagnostic routines
    //
    if (name == "load") {
      diagnostic_load(out, obj);
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
