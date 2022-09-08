// -*- C++ -*-
#include "expic3d.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Nb>                                                                                \
  type ExPIC3D<Nb>::name

DEFINE_MEMBER(void, initialize)(int argc, char **argv)
{
  // default initialize()
  BaseApp::initialize(argc, argv);

  // additional parameters
  interval = cfg_json["interval"].get();
  prefix   = cfg_json["prefix"].get();
  cc       = cfg_json["cc"].get();
  //interval = cfg_json["interval"].template get<int>();
  //prefix   = cfg_json["prefix"].template get<std::string>();
  //cc       = cfg_json["cc"].template get<float64>();

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
  for(int mode=0; mode < Chunk::NumBoundaryMode; mode++) {
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    for(int i=0; i < numchunk; i++) {
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
  if (curstep % interval != 0) {
    return;
  }

  // filename
  std::string filename = prefix + tfm::format("%05d", curstep);
  std::string fn_json  = filename + ".json";
  std::string fn_data  = filename + ".data";

  json     json_root;
  json     json_chunkmap;
  json     json_dataset;

  MPI_File fh;
  size_t   disp;
  int      bufsize;
  int      ndim    = 5;
  int      dims[5] = {cdims[3], ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2], 6};
  int      size    = dims[0] * dims[1] * dims[2] * dims[3] * dims[4] * sizeof(float64);

  // open file
  jsonio::open_file(fn_data.c_str(), &fh, &disp, "w");

  // save chunkmap
  chunkmap->save(json_chunkmap, &fh, &disp);

  // json metadata
  jsonio::put_metadata(json_dataset, "uf", "f8", "", disp, size, ndim, dims);

  // assume buffer size for each chunk is equal
  bufsize = chunkvec[0]->pack(Chunk::PackEmfQuery, nullptr);
  for (int i = 0; i < numchunk; i++) {
    assert(bufsize == chunkvec[i]->pack(Chunk::PackEmfQuery, nullptr));
  }
  sendbuf.resize(bufsize);
  disp += bufsize * chunkvec[0]->get_id();

  // write data for each chunk
  for (int i = 0; i < numchunk; i++) {
    MPI_Request req;

    chunkvec[i]->pack(Chunk::PackEmf, sendbuf.get());

    jsonio::write_contiguous_at(&fh, &disp, sendbuf.get(), bufsize, 1, &req);
    disp += bufsize;

    MPI_Wait(&req, MPI_STATUS_IGNORE);
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
