// -*- C++ -*-
#include "chunk3d.hpp"

NIX_NAMESPACE_BEGIN

Chunk3D::Chunk3D(const int dims[3], const bool has_dim[3], int id)
    : myid(id), delx(1.0), dely(1.0), delz(1.0), option({})
{
  for (int i = 0; i < 3; i++) {
    this->dims[i]    = dims[i];
    this->has_dim[i] = has_dim[i];
  }

  load.resize(1);
  reset_load();
}

int Chunk3D::pack(void* buffer, int address)
{
  address += memcpy_count(buffer, &myid, sizeof(int), address, 0);
  address += memcpy_count(buffer, nbid, nbsize * sizeof(int), address, 0);
  address += memcpy_count(buffer, nbrank, nbsize * sizeof(int), address, 0);
  address += memcpy_count(buffer, &boundary_margin, sizeof(int), address, 0);
  address += memcpy_count(buffer, gdims, 3 * sizeof(int), address, 0);
  address += memcpy_count(buffer, offset, 3 * sizeof(int), address, 0);
  address += memcpy_count(buffer, &delx, sizeof(float64), address, 0);
  address += memcpy_count(buffer, &dely, sizeof(float64), address, 0);
  address += memcpy_count(buffer, &delz, sizeof(float64), address, 0);
  address += memcpy_count(buffer, xlim, 3 * sizeof(float64), address, 0);
  address += memcpy_count(buffer, ylim, 3 * sizeof(float64), address, 0);
  address += memcpy_count(buffer, zlim, 3 * sizeof(float64), address, 0);
  address += memcpy_count(buffer, gxlim, 3 * sizeof(float64), address, 0);
  address += memcpy_count(buffer, gylim, 3 * sizeof(float64), address, 0);
  address += memcpy_count(buffer, gzlim, 3 * sizeof(float64), address, 0);

  // option
  {
    std::vector<uint8_t> msgpack = json::to_msgpack(option);
    int                  size    = msgpack.size();
    address += memcpy_count(buffer, &size, sizeof(int), address, 0);
    address += memcpy_count(buffer, msgpack.data(), size, address, 0);
  }

  // load
  {
    int size = load.size();
    address += memcpy_count(buffer, &size, sizeof(int), address, 0);
    address += memcpy_count(buffer, load.data(), sizeof(float64) * size, address, 0);
  }

  // MPI buffer
  {
    int nmode = mpibufvec.size();
    address += memcpy_count(buffer, &nmode, sizeof(int), address, 0);

    for (int mode = 0; mode < nmode; mode++) {
      address = mpibufvec[mode]->pack(buffer, address);
    }
  }

  return address;
}

int Chunk3D::unpack(void* buffer, int address)
{
  address += memcpy_count(&myid, buffer, sizeof(int), 0, address);
  address += memcpy_count(nbid, buffer, nbsize * sizeof(int), 0, address);
  address += memcpy_count(nbrank, buffer, nbsize * sizeof(int), 0, address);
  address += memcpy_count(&boundary_margin, buffer, sizeof(int), 0, address);
  address += memcpy_count(gdims, buffer, 3 * sizeof(int), 0, address);
  address += memcpy_count(offset, buffer, 3 * sizeof(int), 0, address);
  address += memcpy_count(&delx, buffer, sizeof(float64), 0, address);
  address += memcpy_count(&dely, buffer, sizeof(float64), 0, address);
  address += memcpy_count(&delz, buffer, sizeof(float64), 0, address);
  address += memcpy_count(xlim, buffer, 3 * sizeof(float64), 0, address);
  address += memcpy_count(ylim, buffer, 3 * sizeof(float64), 0, address);
  address += memcpy_count(zlim, buffer, 3 * sizeof(float64), 0, address);
  address += memcpy_count(gxlim, buffer, 3 * sizeof(float64), 0, address);
  address += memcpy_count(gylim, buffer, 3 * sizeof(float64), 0, address);
  address += memcpy_count(gzlim, buffer, 3 * sizeof(float64), 0, address);

  // option
  {
    int size = 0;
    address += memcpy_count(&size, buffer, sizeof(int), 0, address);

    std::vector<uint8_t> msgpack(size);
    address += memcpy_count(msgpack.data(), buffer, size, 0, address);

    option = json::from_msgpack(msgpack);
  }

  // load
  {
    int size = 0;
    address += memcpy_count(&size, buffer, sizeof(int), 0, address);
    load.resize(size);
    address += memcpy_count(load.data(), buffer, sizeof(float64) * size, 0, address);
  }

  // MPI buffer
  {
    int nmode = 0;
    address += memcpy_count(&nmode, buffer, sizeof(int), 0, address);
    mpibufvec.resize(nmode);

    for (int mode = 0; mode < nmode; mode++) {
      mpibufvec[mode] = std::make_shared<MpiBuffer>();
      address         = mpibufvec[mode]->unpack(buffer, address);
    }
  }

  set_index_bounds();

  return address;
}

void Chunk3D::set_index_bounds()
{

  indexlb[0] = 0;
  indexub[0] = 2;
  dirlb[0]   = -1;
  dirub[0]   = +1;
  indexlb[1] = 0;
  indexub[1] = 2;
  dirlb[1]   = -1;
  dirub[1]   = +1;
  indexlb[2] = 0;
  indexub[2] = 2;
  dirlb[2]   = -1;
  dirub[2]   = +1;

  Lbz = boundary_margin;
  Ubz = boundary_margin + this->dims[0] - 1;
  Lby = boundary_margin;
  Uby = boundary_margin + this->dims[1] - 1;
  Lbx = boundary_margin;
  Ubx = boundary_margin + this->dims[2] - 1;

  if (has_zdim() == false) {

    Lbz        = boundary_margin;
    Ubz        = boundary_margin;
    indexlb[0] = 1;
    indexub[0] = 1;
    dirlb[0]   = 0;
    dirub[0]   = 0;
  }

  if (has_ydim() == false) {

    Lby        = boundary_margin;
    Uby        = boundary_margin;
    indexlb[1] = 1;
    indexub[1] = 1;
    dirlb[1]   = 0;
    dirub[1]   = 0;
  }

  if (has_xdim() == false) {

    Lbx        = boundary_margin;
    Ubx        = boundary_margin;
    indexlb[2] = 1;
    indexub[2] = 1;
    dirlb[2]   = 0;
    dirub[2]   = 0;
  }

  sendlb[0][0] = Lbz;
  sendlb[0][1] = Lbz;
  sendlb[0][2] = Ubz - boundary_margin + 1;
  sendub[0][0] = Lbz + boundary_margin - 1;
  sendub[0][1] = Ubz;
  sendub[0][2] = Ubz;
  sendlb[1][0] = Lby;
  sendlb[1][1] = Lby;
  sendlb[1][2] = Uby - boundary_margin + 1;
  sendub[1][0] = Lby + boundary_margin - 1;
  sendub[1][1] = Uby;
  sendub[1][2] = Uby;
  sendlb[2][0] = Lbx;
  sendlb[2][1] = Lbx;
  sendlb[2][2] = Ubx - boundary_margin + 1;
  sendub[2][0] = Lbx + boundary_margin - 1;
  sendub[2][1] = Ubx;
  sendub[2][2] = Ubx;

  recvlb[0][0] = Lbz - boundary_margin;
  recvlb[0][1] = Lbz;
  recvlb[0][2] = Ubz + 1;
  recvub[0][0] = Lbz - 1;
  recvub[0][1] = Ubz;
  recvub[0][2] = Ubz + boundary_margin;
  recvlb[1][0] = Lby - boundary_margin;
  recvlb[1][1] = Lby;
  recvlb[1][2] = Uby + 1;
  recvub[1][0] = Lby - 1;
  recvub[1][1] = Uby;
  recvub[1][2] = Uby + boundary_margin;
  recvlb[2][0] = Lbx - boundary_margin;
  recvlb[2][1] = Lbx;
  recvlb[2][2] = Ubx + 1;
  recvub[2][0] = Lbx - 1;
  recvub[2][1] = Ubx;
  recvub[2][2] = Ubx + boundary_margin;
}

void Chunk3D::set_coordinate(float64 dz, float64 dy, float64 dx)
{

  delz = dz;
  dely = dy;
  delx = dx;

  zlim[0]  = offset[0] * delz;
  zlim[1]  = offset[0] * delz + dims[0] * delz;
  zlim[2]  = zlim[1] - zlim[0];
  gzlim[0] = 0.0;
  gzlim[1] = gdims[0] * delz;
  gzlim[2] = gzlim[1] - gzlim[0];

  ylim[0]  = offset[1] * dely;
  ylim[1]  = offset[1] * dely + dims[1] * dely;
  ylim[2]  = ylim[1] - ylim[0];
  gylim[0] = 0.0;
  gylim[1] = gdims[1] * dely;
  gylim[2] = gylim[1] - gylim[0];

  xlim[0]  = offset[2] * delx;
  xlim[1]  = offset[2] * delx + dims[2] * delx;
  xlim[2]  = xlim[1] - xlim[0];
  gxlim[0] = 0.0;
  gxlim[1] = gdims[2] * delx;
  gxlim[2] = gxlim[1] - gxlim[0];
}

void Chunk3D::set_global_context(const int* offset, const int* gdims)
{
  this->gdims[0]  = gdims[0];
  this->gdims[1]  = gdims[1];
  this->gdims[2]  = gdims[2];
  this->offset[0] = offset[0];
  this->offset[1] = offset[1];
  this->offset[2] = offset[2];
}

void Chunk3D::set_mpi_communicator(int mode, int iz, int iy, int ix, MPI_Comm& comm)
{
  if (mode >= 0 && mode < mpibufvec.size()) {
    mpibufvec[mode]->comm(iz, iy, ix) = comm;
  } else {
    ERROR << tfm::format("invalid index %d for mpibufvec", mode);
  }
}
void Chunk3D::set_mpi_buffer(MpiBufferPtr mpibuf, int mode, int headbyte, int elembyte)
{
  int size = 0;

  for (int iz = 0; iz <= 2; iz++) {
    for (int iy = 0; iy <= 2; iy++) {
      for (int ix = 0; ix <= 2; ix++) {
        if (iz == 1 && iy == 1 && ix == 1) {
          mpibuf->bufsize(iz, iy, ix) = 0;
          mpibuf->bufaddr(iz, iy, ix) = size;
        } else {
          int nz = recvub[0][iz] - recvlb[0][iz] + 1;
          int ny = recvub[1][iy] - recvlb[1][iy] + 1;
          int nx = recvub[2][ix] - recvlb[2][ix] + 1;

          mpibuf->bufsize(iz, iy, ix) = headbyte + elembyte * nz * ny * nx;
          mpibuf->bufaddr(iz, iy, ix) = size;
          size += mpibuf->bufsize(iz, iy, ix);
        }
      }
    }
  }

  if (mode == +1 || mode == 0) {
    mpibuf->sendbuf.resize(size);
  }
  if (mode == -1 || mode == 0) {
    mpibuf->recvbuf.resize(size);
  }
}

int Chunk3D::set_boundary_query(int mode, int sendrecv)
{
  int flag = 0;

  MpiBufferPtr mpibuf = mpibufvec[mode];

  OMP_MAYBE_CRITICAL
  if (sendrecv == 0) {

    MPI_Testall(27, mpibuf->sendreq.data(), &flag, MPI_STATUSES_IGNORE);
    MPI_Testall(27, mpibuf->recvreq.data(), &flag, MPI_STATUSES_IGNORE);
  } else if (sendrecv == +1) {

    MPI_Testall(27, mpibuf->sendreq.data(), &flag, MPI_STATUSES_IGNORE);
  } else if (sendrecv == -1) {

    MPI_Testall(27, mpibuf->recvreq.data(), &flag, MPI_STATUSES_IGNORE);
  }

  return flag;
}

int Chunk3D::probe_bc_exchange(MpiBufferPtr mpibuf)
{

  if (mpibuf->recvwait == true)
    return 1;

  bool is_everyone_ready = true;

  mpibuf->bufsize.fill(0);
  mpibuf->bufaddr.fill(0);

  OMP_MAYBE_CRITICAL
  for (int dirz = dirlb[0], iz = indexlb[0]; dirz <= dirub[0]; dirz++, iz++) {
    for (int diry = dirlb[1], iy = indexlb[1]; diry <= dirub[1]; diry++, iy++) {
      for (int dirx = dirlb[2], ix = indexlb[2]; dirx <= dirub[2]; dirx++, ix++) {

        if (iz == 1 && iy == 1 && ix == 1)
          continue;

        MPI_Status status;
        int        is_ready = 0;

        auto& recvcomm = mpibuf->comm(1 - dirz, 1 - diry, 1 - dirx);
        auto& recvtype = mpibuf->recvtype(iz, iy, ix);
        int   nbrank   = get_nb_rank(dirz, diry, dirx);
        int   recvtag  = get_rcvtag(dirz, diry, dirx);

        MPI_Iprobe(nbrank, recvtag, recvcomm, &is_ready, &status);

        if (is_ready) {
          int count    = 0;
          int typebyte = 0;
          MPI_Get_count(&status, recvtype, &count);
          MPI_Type_size(recvtype, &typebyte);
          mpibuf->bufsize(iz, iy, ix) = count * typebyte;
        } else {

          is_everyone_ready = false;
        }
      }
    }
  }

  if (is_everyone_ready == false)
    return 0;

  {
    int bufsize = 0;

    for (int iz = indexlb[0]; iz <= indexub[0]; iz++) {
      for (int iy = indexlb[1]; iy <= indexub[1]; iy++) {
        for (int ix = indexlb[2]; ix <= indexub[2]; ix++) {
          mpibuf->recvreq(iz, iy, ix) = MPI_REQUEST_NULL;
          mpibuf->bufaddr(iz, iy, ix) = bufsize;
          bufsize += mpibuf->bufsize(iz, iy, ix);
        }
      }
    }
    mpibuf->recvbuf.resize(bufsize);

    OMP_MAYBE_CRITICAL
    for (int dirz = dirlb[0], iz = indexlb[0]; dirz <= dirub[0]; dirz++, iz++) {
      for (int diry = dirlb[1], iy = indexlb[1]; diry <= dirub[1]; diry++, iy++) {
        for (int dirx = dirlb[2], ix = indexlb[2]; dirx <= dirub[2]; dirx++, ix++) {

          if (iz == 1 && iy == 1 && ix == 1)
            continue;

          auto& recvcomm = mpibuf->comm(1 - dirz, 1 - diry, 1 - dirx);
          auto& recvtype = mpibuf->recvtype(iz, iy, ix);
          auto& recvreq  = mpibuf->recvreq(iz, iy, ix);
          int   nbrank   = get_nb_rank(dirz, diry, dirx);
          int   recvtag  = get_rcvtag(dirz, diry, dirx);
          void* recvptr  = mpibuf->get_recv_buffer(iz, iy, ix);
          int   recvcnt  = mpibuf->bufsize(iz, iy, ix);

          MPI_Irecv(recvptr, recvcnt, recvtype, nbrank, recvtag, recvcomm, &recvreq);
        }
      }
    }

    mpibuf->recvwait = true;
  }

  return 1;
}

///
/// Implementation of MpiBuffer
///

int64_t Chunk3D::MpiBuffer::get_size_byte() const
{
  int64_t size = 0;
  size += sizeof(sendwait);
  size += sizeof(recvwait);
  size += sendbuf.size;
  size += recvbuf.size;
  size += bufsize.size() * sizeof(int);
  size += bufaddr.size() * sizeof(int);
  size += comm.size() * sizeof(MPI_Comm);
  size += sendreq.size() * sizeof(MPI_Request);
  size += recvreq.size() * sizeof(MPI_Request);
  size += sendtype.size() * sizeof(MPI_Datatype);
  size += recvtype.size() * sizeof(MPI_Datatype);
  return size;
}

int Chunk3D::MpiBuffer::pack(void* buffer, int address)
{
  int ssize = sendbuf.size;
  int rsize = recvbuf.size;
  int asize = bufsize.size() * sizeof(int);

  address += memcpy_count(buffer, &sendwait, sizeof(bool), address, 0);
  address += memcpy_count(buffer, &recvwait, sizeof(bool), address, 0);
  address += memcpy_count(buffer, &ssize, sizeof(int), address, 0);
  address += memcpy_count(buffer, &rsize, sizeof(int), address, 0);
  address += memcpy_count(buffer, bufsize.data(), asize, address, 0);
  address += memcpy_count(buffer, bufaddr.data(), asize, address, 0);

  return address;
}

int Chunk3D::MpiBuffer::unpack(void* buffer, int address)
{
  int ssize = 0;
  int rsize = 0;
  int asize = bufsize.size() * sizeof(int);

  address += memcpy_count(&sendwait, buffer, sizeof(bool), 0, address);
  address += memcpy_count(&recvwait, buffer, sizeof(bool), 0, address);
  address += memcpy_count(&ssize, buffer, sizeof(int), 0, address);
  address += memcpy_count(&rsize, buffer, sizeof(int), 0, address);
  address += memcpy_count(bufsize.data(), buffer, asize, 0, address);
  address += memcpy_count(bufaddr.data(), buffer, asize, 0, address);

  // memory allocation
  sendbuf.resize(ssize);
  recvbuf.resize(rsize);

  return address;
}

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
