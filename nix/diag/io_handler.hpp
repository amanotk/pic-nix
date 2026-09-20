// -*- C++ -*-
#ifndef _IO_HANDLER_HPP_
#define _IO_HANDLER_HPP_

#include "buffer.hpp"
#include "diag.hpp"
#include "nixio.hpp"

NIX_NAMESPACE_BEGIN

class DiagIoHandler
{
protected:
  using info_type = Diag::info_type;

  std::shared_ptr<info_type> info;

public:
  DiagIoHandler(std::shared_ptr<info_type> info) : info(info)
  {
  }

  virtual ~DiagIoHandler()
  {
  }

  virtual void open_file(std::string filename, size_t* disp, const char* mode) = 0;

  virtual void close_file() = 0;

  virtual std::vector<int> get_chunk_id_range(int id_min, int id_max) = 0;

  /// Write the complete buffer before returning.
  virtual size_t write(Buffer& buffer, size_t& disp) = 0;
};

class MpiioDiagIoHandler : public DiagIoHandler
{
protected:
  MPI_File filehandle;
  bool     is_opened;

public:
  MpiioDiagIoHandler(std::shared_ptr<info_type> info) : DiagIoHandler(info), is_opened(false)
  {
  }

  virtual void open_file(std::string filename, size_t* disp, const char* mode) override
  {
    if (is_opened == false) {
      nixio::open_file(filename.c_str(), &filehandle, disp, mode);
      is_opened = true;
    }
  }

  virtual void close_file() override
  {
    if (is_opened == true) {
      nixio::close_file(&filehandle);
      is_opened = false;
    }
  }

  virtual std::vector<int> get_chunk_id_range(int id_min, int id_max) override
  {
    int global_id_min = std::numeric_limits<int>::max();
    int global_id_max = std::numeric_limits<int>::min();

    MPI_Reduce(&id_min, &global_id_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&id_max, &global_id_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    return std::vector<int>({global_id_min, global_id_max});
  }

  virtual size_t write(Buffer& buffer, size_t& disp) override
  {
    MPI_Request request = MPI_REQUEST_NULL;
    auto        count =
        nixio::write_contiguous(&filehandle, &disp, buffer.get(), buffer.size, 1, 1, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    return count;
  }
};

class PosixDiagIoHandler : public DiagIoHandler
{
protected:
  std::ofstream file;

public:
  PosixDiagIoHandler(std::shared_ptr<info_type> info) : DiagIoHandler(info)
  {
  }

  virtual void open_file(std::string filename, size_t* disp, const char* mode) override
  {
    if (file.is_open() == false) {
      std::string             mode_str = mode;
      std::ios_base::openmode openmode;
      if (mode_str == "w") {
        openmode = std::ios::out | std::ios::binary;
      } else if (mode_str == "a") {
        openmode = std::ios::app | std::ios::binary;
      } else if (mode_str == "r") {
        openmode = std::ios::in | std::ios::binary;
      }

      file.open(filename, openmode);
      file.seekp(*disp);
    }
  }

  virtual void close_file() override
  {
    if (file.is_open() == true) {
      file.flush();
      file.close();
    }
  }

  virtual std::vector<int> get_chunk_id_range(int id_min, int id_max) override
  {
    int node_id_min = std::numeric_limits<int>::max();
    int node_id_max = std::numeric_limits<int>::min();

    MPI_Reduce(&id_min, &node_id_min, 1, MPI_INT, MPI_MIN, 0, info->intra_comm);
    MPI_Reduce(&id_max, &node_id_max, 1, MPI_INT, MPI_MAX, 0, info->intra_comm);

    return std::vector<int>({node_id_min, node_id_max});
  }

  virtual size_t write(Buffer& buffer, size_t& disp) override
  {
    Buffer           totbuf;
    int              totcnt  = 0;
    int              sendcnt = static_cast<int>(buffer.size);
    std::vector<int> recvcnt(info->intra_size + 1, 0);
    std::vector<int> recvpos(info->intra_size + 1, 0);

    MPI_Gather(&sendcnt, 1, MPI_INT, recvcnt.data(), 1, MPI_INT, 0, info->intra_comm);

    std::partial_sum(recvcnt.begin(), recvcnt.end() - 1, recvpos.begin() + 1);

    totcnt = recvpos[info->intra_size];
    totbuf.resize(totcnt);
    MPI_Gatherv(buffer.get(), sendcnt, MPI_BYTE, totbuf.get(), recvcnt.data(), recvpos.data(),
                MPI_BYTE, 0, info->intra_comm);

    if (info->intra_rank == 0) {
      file.seekp(disp);
      file.write(reinterpret_cast<char*>(totbuf.get()), totcnt);
    }

    disp += totcnt;

    return totcnt;
  }
};

NIX_NAMESPACE_END

#endif
