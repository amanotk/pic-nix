// -*- C++ -*-
#include "expic3d.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Nb>                                                                                \
  type ExPIC3D<Nb>::name

DEFINE_MEMBER(void, parse_cfg)()
{
  // read configuration file
  {
    std::ifstream f(cfg_file.c_str());
    cfg_json = json::parse(f, nullptr, true, true);
  }

  // parameters
  {
    parameters = cfg_json["parameter"];

    int nx = parameters["Nx"].get<int>();
    int ny = parameters["Ny"].get<int>();
    int nz = parameters["Nz"].get<int>();
    int cx = parameters["Cx"].get<int>();
    int cy = parameters["Cy"].get<int>();
    int cz = parameters["Cz"].get<int>();

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

    // other parameters
    tmax = parameters["tmax"].get<float64>();
    delt = parameters["delt"].get<float64>();
    delh = parameters["delt"].get<float64>();
    cc   = parameters["cc"].get<float64>();
  }

  // diagnostic
  {
    json diagnostic = cfg_json["diagnostic"];

    datadir           = diagnostic.get<std::string>();
    prefix_field      = diagnostic.get<std::string>();
    interval_field    = diagnostic.get<int>();
    prefix_particle   = diagnostic.get<std::string>();
    interval_particle = diagnostic.get<int>();
  }
}

DEFINE_MEMBER(void, write_allchunk)
(MPI_File &fh, json &dataset, size_t &disp, const char *name, const char *desc,
 const int size, const int ndim, const int *dims, const int mode)
{
  MPI_Request req[numchunk];

  // json metadata
  jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);

  // buffer size (assuming constant)
  int bufsize = chunkvec[0]->pack_diagnostic(mode, nullptr);

  for (int i = 0; i < numchunk; i++) {
    // pack
    assert(bufsize == chunkvec[i]->pack_diagnostic(mode, sendbuf.get()));

    // write
    size_t chunkdisp = disp + bufsize * chunkvec[i]->get_id();
    jsonio::write_contiguous_at(&fh, &disp, sendbuf.get(), bufsize, 1, &req[i]);
  }

  // wait
  MPI_Waitall(numchunk, req, MPI_STATUS_IGNORE);

  // update pointer
  disp += size;
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

  // set MPI communicator for each mode
  for (int mode = 0; mode < Chunk::NumBoundaryMode; mode++) {
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_mpi_communicator(mode, comm);
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
    chunkvec[i]->psh_mfd(0.5 * delt);

    // push particle
    chunkvec[i]->push_velocity(delt);
    chunkvec[i]->push_position(delt);

    // calculate current
    chunkvec[i]->deposit_current(delt);

    // begin boundary exchange for current and particles
    chunkvec[i]->set_boundary_begin(Chunk::BoundaryCur);
    chunkvec[i]->set_boundary_begin(Chunk::BoundaryParticle);
    bc_queue_uj.insert(i);
    bc_queue_up.insert(i);

    // push B for a half step
    chunkvec[i]->psh_mfd(0.5 * delt);
  }

  // wait for current boundary exchange
  wait_bc_exchange(bc_queue_uj, Chunk::BoundaryCur);

  for (int i = 0; i < numchunk; i++) {
    // push E
    chunkvec[i]->psh_efd(delt);

    // begin boundary exchange for field
    chunkvec[i]->set_boundary_begin(Chunk::BoundaryEmf);
    bc_queue_uf.insert(i);
  }

  // wait for particle and field boundary exchange
  wait_bc_exchange(bc_queue_up, Chunk::BoundaryParticle);
  wait_bc_exchange(bc_queue_uf, Chunk::BoundaryEmf);

  curtime += delt;
  curstep++;
}

DEFINE_MEMBER(void, diagnostic)(std::ostream &out)
{
  if (curstep % interval_field != 0) {
    diagnostic_field();
  }
}

DEFINE_MEMBER(void, diagnostic_field)()
{
  const int nz = ndims[0] / cdims[0];
  const int ny = ndims[1] / cdims[1];
  const int nx = ndims[2] / cdims[2];
  const int ns = Ns;
  const int nc = cdims[3];

  // filename
  std::string fn_prefix = tfm::format("%s/%s_%05d", datadir, prefix_field, curstep);
  std::string fn_json   = fn_prefix + ".json";
  std::string fn_data   = fn_prefix + ".data";

  json json_root;
  json json_chunkmap;
  json json_dataset;

  MPI_File fh;
  size_t   disp;

  // open file
  jsonio::open_file(fn_data.c_str(), &fh, &disp, "w");

  // save chunkmap
  chunkmap->save(json_chunkmap, &fh, &disp);

  //
  // electromagnetic field
  //
  {
    const char name[] = "uf";
    const char desc[]  = "electromagnetic field";
    const int  ndim    = 5;
    const int  dims[5] = {nc, nz, ny, nx, 6};
    const int  size    = nc * nz * ny * nx * 6 * sizeof(float64);

    write_allchunk(fh, json_dataset, disp, name, desc, size, ndim, ndims, Chunk::PackEmf);
  }

  //
  // current
  //
  {
    const char name[] = "uj";
    const char desc[]  = "current";
    const int  ndim    = 5;
    const int  dims[5] = {nc, nz, ny, nx, 4};
    const int  size    = nc * nz * ny * nx * 4 * sizeof(float64);

    write_allchunk(fh, json_dataset, disp, name, desc, size, ndim, ndims, Chunk::PackCur);
  }

  //
  // moment
  //
  {
    const char name[] = "um";
    const char desc[]  = "moment";
    const int  ndim    = 6;
    const int  dims[5] = {nc, ns, nz, ny, nx, 10};
    const int  size    = nc * ns * nz * ny * nx * 10 * sizeof(float64);

    write_allchunk(fh, json_dataset, disp, name, desc, size, ndim, ndims, Chunk::PackMom);
  }

  jsonio::close_file(&fh);

  //
  // output json file
  //

  // meta data
  json_root["meta"] = {{"endian", common::get_endian_flag()},
                       {"rawfile", fn_data},
                       {"order", 1},
                       {"time", curtime},
                       {"step", curstep}};
  // chunkmap
  json_root["chunkmap"] = json_chunkmap;
  // dataset
  json_root["dataset"] = json_dataset;

  if (thisrank == 0) {
    std::ofstream ofs(fn_json);
    ofs << std::setw(2) << json_root;
    ofs.close();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

DEFINE_MEMBER(void, wait_bc_exchange)(std::set<int> &queue, const int mode)
{
  int recvmode = Chunk::RecvMode | mode;

  while (queue.empty() == false) {
    // find chunk for unpacking
    auto iter = std::find_if(queue.begin(), queue.end(),
                             [&](int i) { return chunkvec[i]->set_boundary_query(recvmode); });

    // not found
    if (iter == queue.end()) {
      continue;
    }

    // unpack
    chunkvec[*iter]->set_boundary_end(mode);
    queue.erase(*iter);
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
