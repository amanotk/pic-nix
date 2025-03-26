// -*- C++ -*-
#ifndef _DIAG_HANDLER_HPP_
#define _DIAG_HANDLER_HPP_

#include "pic.hpp"
#include "pic_diag.hpp"

class PicDiagHandler
{
protected:
  std::shared_ptr<nix::DiagInfo> info;

public:
  PicDiagHandler(std::shared_ptr<nix::DiagInfo> info) : info(info)
  {
  }

  virtual ~PicDiagHandler()
  {
  }

  virtual void open_file(std::string filename, size_t* disp, const char* mode) = 0;

  virtual void close_file() = 0;

  virtual bool is_completed() = 0;

  virtual void wait(int index) = 0;

  virtual void wait_all() = 0;

  virtual bool test_all() = 0;

  virtual std::vector<int> get_chunk_id_range(int id_min, int id_max) = 0;

  virtual size_t queue(int index, nix::Buffer& buffer, size_t& disp) = 0;
};

class MpiioHandler : public PicDiagHandler
{
protected:
  MPI_File                 filehandle;
  bool                     is_opened;
  std::vector<MPI_Request> request;

public:
  MpiioHandler(std::shared_ptr<nix::DiagInfo> info) : PicDiagHandler(info), is_opened(false)
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
    assert(is_completed() == true);

    if (is_opened == true) {
      nixio::close_file(&filehandle);
      is_opened = false;
    }
  }

  virtual bool is_completed() override
  {
    bool status = std::all_of(request.begin(), request.end(),
                              [](auto& req) { return req == MPI_REQUEST_NULL; });
    return status;
  }

  virtual void wait(int index) override
  {
    MPI_Wait(&request[index], MPI_STATUS_IGNORE);
  }

  virtual void wait_all() override
  {
    MPI_Waitall(request.size(), request.data(), MPI_STATUSES_IGNORE);
    std::fill(request.begin(), request.end(), MPI_REQUEST_NULL);
    close_file();
  }

  virtual bool test_all() override
  {
    int flag = 0;
    MPI_Testall(request.size(), request.data(), &flag, MPI_STATUSES_IGNORE);

    return flag;
  }

  virtual std::vector<int> get_chunk_id_range(int id_min, int id_max) override
  {
    int global_id_min = std::numeric_limits<int>::max();
    int global_id_max = std::numeric_limits<int>::min();

    MPI_Reduce(&id_min, &global_id_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&id_max, &global_id_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    return std::vector<int>({global_id_min, global_id_max});
  }

  virtual size_t queue(int index, nix::Buffer& buffer, size_t& disp) override
  {
    if (request.size() <= index) {
      request.resize(index + 1, MPI_REQUEST_NULL);
    }
    auto count = nixio::write_contiguous(&filehandle, &disp, buffer.get(), buffer.size, 1, 1,
                                         &request[index]);

    return count;
  }
};

class PosixHandler : public PicDiagHandler
{
protected:
  std::ofstream file;

public:
  PosixHandler(std::shared_ptr<nix::DiagInfo> info) : PicDiagHandler(info)
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
    assert(is_completed() == true);

    if (file.is_open() == true) {
      file.flush();
      file.close();
    }
  }

  virtual bool is_completed() override
  {
    return true;
  }

  virtual void wait(int index) override
  {
  }

  virtual void wait_all() override
  {
  }

  virtual bool test_all() override
  {
    return true;
  }

  virtual std::vector<int> get_chunk_id_range(int id_min, int id_max) override
  {
    int node_id_min = std::numeric_limits<int>::max();
    int node_id_max = std::numeric_limits<int>::min();

    MPI_Reduce(&id_min, &node_id_min, 1, MPI_INT, MPI_MIN, 0, info->intra_comm);
    MPI_Reduce(&id_max, &node_id_max, 1, MPI_INT, MPI_MAX, 0, info->intra_comm);

    return std::vector<int>({node_id_min, node_id_max});
  }

  virtual size_t queue(int index, nix::Buffer& buffer, size_t& disp) override
  {
    nix::Buffer      totbuf;
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

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
