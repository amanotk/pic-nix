// -*- C++ -*-
#ifndef _CHUNK3D_HPP_
#define _CHUNK3D_HPP_

#include "buffer.hpp"
#include "debug.hpp"
#include "nix.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Base class for 3D Chunk
///
class Chunk3D
{
public:
  static constexpr int nbsize = 27;

  struct MpiBuffer; ///< forward declaration
  using IntArray     = xt::xtensor_fixed<int, xt::xshape<3, 3, 3>>;
  using Comm         = xt::xtensor_fixed<MPI_Comm, xt::xshape<3, 3, 3>>;
  using Request      = xt::xtensor_fixed<MPI_Request, xt::xshape<3, 3, 3>>;
  using Datatype     = xt::xtensor_fixed<MPI_Datatype, xt::xshape<3, 3, 3>>;
  using MpiBufferPtr = std::shared_ptr<MpiBuffer>;
  using MpiBufferVec = std::vector<MpiBufferPtr>;

  ///
  /// @brief MPI buffer
  ///
  struct MpiBuffer {
    bool     sendwait;
    bool     recvwait;
    Buffer   sendbuf;
    Buffer   recvbuf;
    IntArray bufsize;
    IntArray bufaddr;
    Comm     comm;
    Request  sendreq;
    Request  recvreq;
    Datatype sendtype;
    Datatype recvtype;

    /// constructor
    MpiBuffer() : sendwait(false), recvwait(false)
    {
    }

    /// @brief return send buffer for given direction
    void* get_send_buffer(int iz, int iy, int ix)
    {
      return sendbuf.get(bufaddr(iz, iy, ix));
    }

    /// @brief return recv buffer for given direction
    void* get_recv_buffer(int iz, int iy, int ix)
    {
      return recvbuf.get(bufaddr(iz, iy, ix));
    }

    /// @brief get size of buffer in bytes
    int64_t get_size_byte() const;

    /// @brief pack the content into given `buffer`
    int pack(void* buffer, int address);

    /// @brief unpack the content from given `buffer`
    int unpack(void* buffer, int address);
  };

protected:
  int  myid;            ///< chunk ID
  int  nbid[nbsize];    ///< neighboring chunk ID
  int  nbrank[nbsize];  ///< neighboring chunk MPI rank
  int  boundary_margin; ///< boundary margin
  int  dims[3];         ///< number of grids
  bool has_dim[3];      ///< flag to indicate if the dimension is ignorable
  int  gdims[3];        ///< global number of grids
  int  offset[3];       ///< global index offset
  int  Lbx;             ///< lower bound in x
  int  Ubx;             ///< upper bound in x
  int  Lby;             ///< lower bound in y
  int  Uby;             ///< upper bound in y
  int  Lbz;             ///< lower bound in z
  int  Ubz;             ///< upper bound in z
  int  indexlb[3];      ///< index lower bound for MPI data exchange
  int  indexub[3];      ///< index upper bound for MPI data exchange
  int  dirlb[3];        ///< direction lower bound for MPI data exchange
  int  dirub[3];        ///< direction upper bound for MPI data exchange
  int  sendlb[3][3];    ///< lower bound for send
  int  sendub[3][3];    ///< upper bound for send
  int  recvlb[3][3];    ///< lower bound for recv
  int  recvub[3][3];    ///< upper bound for recv

  float64              delx;      ///< grid size in x
  float64              dely;      ///< grid size in y
  float64              delz;      ///< grid size in z
  float64              xlim[3];   ///< physical domain in x
  float64              ylim[3];   ///< physical domain in y
  float64              zlim[3];   ///< physical domain in z
  float64              gxlim[3];  ///< global physical domain in x
  float64              gylim[3];  ///< global physical domain in y
  float64              gzlim[3];  ///< global physical domain in z
  json                 option;    ///< internal option
  std::vector<float64> load;      ///< load array of chunk
  MpiBufferVec         mpibufvec; ///< MPI buffer vector

public:
  ///
  /// @brief constructor
  /// @param dims number of grids for each direction
  /// @param id Chunk ID
  ///
  Chunk3D(const int dims[3], const bool has_dim[3], int id = 0);

  /// @brief setup initial condition (pure virtual)
  virtual void setup(json& config) = 0;

  /// @brief return (approximate) size of Chunk in byte
  virtual int64_t get_size_byte() const
  {
    return 0; // override me
  }

  /// @brief pack the content of Chunk into given `buffer`
  virtual int pack(void* buffer, int address);

  /// @brief unpack the content of Chunk from given `buffer`
  virtual int unpack(void* buffer, int address);

  /// @brief set index bounds for MPI data exchange
  virtual void set_index_bounds();

  /// @brief set coordinate of Chunk (using gdims and offset)
  virtual void set_coordinate(float64 dz, float64 dy, float64 dx);

  /// @brief set the global context of Chunk
  virtual void set_global_context(const int offset[3], const int gdims[3]);

  /// @brief set MPI communicator to MpiBuffer of given `mode`
  virtual void set_mpi_communicator(int mode, int iz, int iy, int ix, MPI_Comm& comm);

  /// @brief reset load
  virtual void reset_load()
  {
    load.assign(load.size(), 0.0);
  }

  /// @brief get load array
  virtual std::vector<float64> get_load()
  {
    return load;
  }

  /// @brief get total load
  virtual float64 get_total_load()
  {
    return std::accumulate(load.begin(), load.end(), 0.0);
  }

  /// @brief  set chunk ID
  void set_id(int id)
  {
    myid = id;
  }

  /// @brief  get chunk ID
  int get_id() const
  {
    return myid;
  }

  /// @brief set neighbor chunk ID
  void set_nb_id(int dirz, int diry, int dirx, int id)
  {
    nbid[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)] = id;
  }

  /// @brief get neighbor chunk ID
  int get_nb_id(int dirz, int diry, int dirx) const
  {
    return nbid[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)];
  }

  /// @brief set neighbor MPI rank
  void set_nb_rank(int dirz, int diry, int dirx, int rank)
  {
    nbrank[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)] = rank;
  }

  /// @brief get neighbor MPI rank
  int get_nb_rank(int dirz, int diry, int dirx) const
  {
    return nbrank[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)];
  }

  /// @brief get send tag for MPI
  int get_sndtag(int dirz, int diry, int dirx) const
  {
    int dir = 9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1);
    // return dummy tag for invalid neighbor
    return nbid[dir] < 0 ? myid : nbid[dir] % MAX_CHUNK_PER_RANK;
  }

  /// @brief get recv tag for MPI
  int get_rcvtag(int dirz, int diry, int dirx) const
  {
    return myid % MAX_CHUNK_PER_RANK;
  }

  /// @brief setup MPI Buffer
  void set_mpi_buffer(MpiBufferPtr mpibuf, int mode, int headbyte, int elembyte);

  /// @brief return MpiBuffer of given mode of boundary exchange
  MpiBufferPtr get_mpi_buffer(int mode)
  {
    return mpibufvec[mode];
  }

  /// @brief get buffer ratio (relative to the required size)
  float64 get_buffer_ratio()
  {
    return option.value("buffer_ratio", 0.2);
  }

  /// @brief set boundary margin
  void set_boundary_margin(int margin)
  {
    boundary_margin = margin;
    set_index_bounds();
  }

  /// @brief get boundary margin
  int get_boundary_margin() const
  {
    return boundary_margin;
  }

  /// @brief return if x dimension is ignorable or not
  bool has_xdim() const
  {
    return has_dim[2];
  }

  /// @brief return if y dimension is ignorable or not
  bool has_ydim() const
  {
    return has_dim[1];
  }

  /// @brief return if z dimension is ignorable or not
  bool has_zdim() const
  {
    return has_dim[0];
  }

  /// @brief return grid size in x direction
  float64 get_delx() const
  {
    return delx;
  }

  /// @brief return grid size in y direction
  float64 get_dely() const
  {
    return dely;
  }

  /// @brief return grid size in z direction
  float64 get_delz() const
  {
    return delz;
  }

  /// @brief return index bounds in x direction
  auto get_xbound() const
  {
    return std::pair(Lbx, Ubx);
  }

  /// @brief return index bounds in y direction
  auto get_ybound() const
  {
    return std::pair(Lby, Uby);
  }

  /// @brief return index bounds in z direction
  auto get_zbound() const
  {
    return std::pair(Lbz, Ubz);
  }

  /// @brief return physical domain range in x direction
  auto get_xrange() const
  {
    float64 xmin = -std::numeric_limits<float64>::max();
    float64 xmax = +std::numeric_limits<float64>::max();

    if (has_xdim()) {
      xmin = xlim[0];
      xmax = xlim[1];
    }

    return std::pair(xmin, xmax);
  }

  /// @brief return physical domain range in y direction
  auto get_yrange() const
  {
    float64 ymin = -std::numeric_limits<float64>::max();
    float64 ymax = +std::numeric_limits<float64>::max();

    if (has_ydim()) {
      ymin = ylim[0];
      ymax = ylim[1];
    }

    return std::pair(ymin, ymax);
  }

  /// @brief return physical domain range in z direction
  auto get_zrange() const
  {
    float64 zmin = -std::numeric_limits<float64>::max();
    float64 zmax = +std::numeric_limits<float64>::max();

    if (has_zdim()) {
      zmin = zlim[0];
      zmax = zlim[1];
    }

    return std::pair(zmin, zmax);
  }

  /// @brief return global physical domain range in x direction
  auto get_xrange_global() const
  {
    return std::pair(gxlim[0], gxlim[1]);
  }

  /// @brief return global physical domain range in y direction
  auto get_yrange_global() const
  {
    return std::pair(gylim[0], gylim[1]);
  }

  /// @brief return global physical domain range in z direction
  auto get_zrange_global() const
  {
    return std::pair(gzlim[0], gzlim[1]);
  }

  /// @brief probe incoming messages and call recv if ready
  virtual bool set_boundary_probe(int mode, bool wait)
  {
    return false;
  }

  /// @brief pack for boundary exchange
  virtual void set_boundary_pack(int mode)
  {
  }

  /// @brief unpack for boundary exchange
  virtual void set_boundary_unpack(int mode)
  {
  }

  /// @brief begin boundary exchange
  virtual void set_boundary_begin(int mode)
  {
  }

  /// @brief end boundary exchange
  virtual void set_boundary_end(int mode)
  {
  }

  /// @brief query status of boundary exchange
  virtual int set_boundary_query(int mode = 0, int sendrecv = 0);

  /// @brief probe incoming messages
  virtual int probe_bc_exchange(MpiBufferPtr mpibuf);

  ///
  /// @brief pack for boundary exchange
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void pack_bc_exchange(MpiBufferPtr mpibuf, Halo& halo)
  {
    // pre-process
    halo.pre_pack(mpibuf, indexlb, indexub);

    for (int dirz = dirlb[0], iz = indexlb[0]; dirz <= dirub[0]; dirz++, iz++) {
      for (int diry = dirlb[1], iy = indexlb[1]; diry <= dirub[1]; diry++, iy++) {
        for (int dirx = dirlb[2], ix = indexlb[2]; dirx <= dirub[2]; dirx++, ix++) {
          // clang-format off
        int send_bound[3][2] = {
          sendlb[0][iz], sendub[0][iz],
          sendlb[1][iy], sendub[1][iy],
          sendlb[2][ix], sendub[2][ix]
        };
        int recv_bound[3][2] = {
          recvlb[0][iz], recvub[0][iz],
          recvlb[1][iy], recvub[1][iy],
          recvlb[2][ix], recvub[2][ix]
        };
          // clang-format on

          // pack
          bool status = halo.pack(mpibuf, iz, iy, ix, send_bound, recv_bound);

          mpibuf->sendreq(iz, iy, ix) = MPI_REQUEST_NULL;
          mpibuf->recvreq(iz, iy, ix) = MPI_REQUEST_NULL;
        }
      }
    }

    // post-process
    halo.post_pack(mpibuf, indexlb, indexub);
  }

  ///
  /// @brief unpack for boundary exchange
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void unpack_bc_exchange(MpiBufferPtr mpibuf, Halo& halo)
  {
    // pre-process
    halo.pre_unpack(mpibuf, indexlb, indexub);

    //
    // unpack recv buffer
    //
    for (int dirz = dirlb[0], iz = indexlb[0]; dirz <= dirub[0]; dirz++, iz++) {
      for (int diry = dirlb[1], iy = indexlb[1]; diry <= dirub[1]; diry++, iy++) {
        for (int dirx = dirlb[2], ix = indexlb[2]; dirx <= dirub[2]; dirx++, ix++) {
          // clang-format off
        int send_bound[3][2] = {
          sendlb[0][iz], sendub[0][iz],
          sendlb[1][iy], sendub[1][iy],
          sendlb[2][ix], sendub[2][ix]
        };
        int recv_bound[3][2] = {
          recvlb[0][iz], recvub[0][iz],
          recvlb[1][iy], recvub[1][iy],
          recvlb[2][ix], recvub[2][ix]
        };
          // clang-format on

          // unpack
          bool status = halo.unpack(mpibuf, iz, iy, ix, send_bound, recv_bound);
        }
      }
    }

    // post-proces
    halo.post_unpack(mpibuf, indexlb, indexub);
  }

  ///
  /// @brief pack and start boundary exchange
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void begin_bc_exchange(MpiBufferPtr mpibuf, Halo& halo)
  {
    static constexpr bool is_send_required = true;
    static constexpr bool is_recv_required = Halo::is_buffer_fixed == true;

    mpibuf->sendwait = false;
    mpibuf->recvwait = false;

    for (int iz = 0; iz <= 2; iz++) {
      for (int iy = 0; iy <= 2; iy++) {
        for (int ix = 0; ix <= 2; ix++) {
          mpibuf->sendreq(iz, iy, ix) = MPI_REQUEST_NULL;
          mpibuf->recvreq(iz, iy, ix) = MPI_REQUEST_NULL;
        }
      }
    }

    OMP_MAYBE_CRITICAL
    for (int dirz = dirlb[0], iz = indexlb[0]; dirz <= dirub[0]; dirz++, iz++) {
      for (int diry = dirlb[1], iy = indexlb[1]; diry <= dirub[1]; diry++, iy++) {
        for (int dirx = dirlb[2], ix = indexlb[2]; dirx <= dirub[2]; dirx++, ix++) {
          if (iz == 1 && iy == 1 && ix == 1)
            continue;

          int nbrank = get_nb_rank(dirz, diry, dirx);
          // send
          if constexpr (is_send_required == true) {
            int   sendtag  = get_sndtag(dirz, diry, dirx);
            auto& sendcomm = mpibuf->comm(1 + dirz, 1 + diry, 1 + dirx);
            auto& sendtype = mpibuf->sendtype(iz, iy, ix);
            auto& sendreq  = mpibuf->sendreq(iz, iy, ix);
            void* sendptr  = mpibuf->get_send_buffer(iz, iy, ix);
            int   sendcnt  = mpibuf->bufsize(iz, iy, ix);

            MPI_Isend(sendptr, sendcnt, sendtype, nbrank, sendtag, sendcomm, &sendreq);
          }

          // recv
          if constexpr (is_recv_required == true) {
            int   recvtag  = get_rcvtag(dirz, diry, dirx);
            auto& recvcomm = mpibuf->comm(1 - dirz, 1 - diry, 1 - dirx);
            auto& recvtype = mpibuf->recvtype(iz, iy, ix);
            auto& recvreq  = mpibuf->recvreq(iz, iy, ix);
            void* recvptr  = mpibuf->get_recv_buffer(iz, iy, ix);
            int   recvcnt  = mpibuf->bufsize(iz, iy, ix);

            MPI_Irecv(recvptr, recvcnt, recvtype, nbrank, recvtag, recvcomm, &recvreq);
          }
        }
      }
    }

    if constexpr (is_send_required == true) {
      mpibuf->sendwait = true;
    }

    if constexpr (is_recv_required == true) {
      mpibuf->recvwait = true;
    }
  }

  ///
  /// @brief wait boundary exchange and unpack
  /// @tparam Halo boundary halo class
  /// @param mpibuf MPI buffer
  /// @param halo boundary halo object
  ///
  template <typename Halo>
  void end_bc_exchange(MpiBufferPtr mpibuf, Halo& halo)
  {
    // wait for MPI send/recv calls to complete
    OMP_MAYBE_CRITICAL
    {
      if (mpibuf->sendwait == true) {
        MPI_Waitall(27, mpibuf->sendreq.data(), MPI_STATUSES_IGNORE);
        mpibuf->sendwait = false;
      }

      if (mpibuf->recvwait == true) {
        MPI_Waitall(27, mpibuf->recvreq.data(), MPI_STATUSES_IGNORE);
        mpibuf->recvwait = false;
      }
    }
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
