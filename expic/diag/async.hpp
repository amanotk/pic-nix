// -*- C++ -*-
#ifndef _ASYNC_DIAG_HPP_
#define _ASYNC_DIAG_HPP_

#include "base.hpp"

class AsyncHandler
{
protected:
  std::shared_ptr<DiagInfo> info;

public:
  AsyncHandler(std::shared_ptr<DiagInfo> info) : info(info)
  {
  }

  virtual ~AsyncHandler()
  {
  }

  virtual void set_queue_size(int size) = 0;

  virtual void open_file(std::string filename, size_t* disp, const char* mode) = 0;

  virtual void close_file() = 0;

  virtual bool is_completed() = 0;

  virtual void wait(int index) = 0;

  virtual void wait_all() = 0;

  virtual bool test_all() = 0;

  virtual std::vector<int> get_chunk_id_range(int id_min, int id_max) = 0;

  virtual size_t write(int index, uint8_t* buf, size_t bufsize, size_t& disp) = 0;
};

class MpiioHandler : public AsyncHandler
{
protected:
  MPI_File                 filehandle;
  bool                     is_opened;
  std::vector<MPI_Request> request;

public:
  MpiioHandler(std::shared_ptr<DiagInfo> info) : AsyncHandler(info), is_opened(false)
  {
  }

  virtual void set_queue_size(int size) override
  {
    if (size == request.size())
      return;

    // resize
    request.resize(size);
    std::fill(request.begin(), request.end(), MPI_REQUEST_NULL);
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

  virtual size_t write(int index, uint8_t* buf, size_t bufsize, size_t& disp) override
  {
    auto size = nixio::write_contiguous(&filehandle, &disp, buf, bufsize, 1, 1, &request[index]);
    wait(index);

    return size;
  }
};

class PosixHandler : public AsyncHandler
{
protected:
  std::ofstream file;

public:
  PosixHandler(std::shared_ptr<DiagInfo> info) : AsyncHandler(info)
  {
  }

  virtual void set_queue_size(int size) override
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

  virtual size_t write(int index, uint8_t* buf, size_t bufsize, size_t& disp) override
  {
    Buffer           buffer;
    int              totsize = 0;
    int              sendcnt = static_cast<int>(bufsize);
    std::vector<int> recvcnt(info->intra_size + 1, 0);
    std::vector<int> recvpos(info->intra_size + 1, 0);

    MPI_Gather(&sendcnt, 1, MPI_INT, recvcnt.data(), 1, MPI_INT, 0, info->intra_comm);

    std::partial_sum(recvcnt.begin(), recvcnt.end() - 1, recvpos.begin() + 1);

    totsize = recvpos[info->intra_size];
    buffer.resize(totsize);
    MPI_Gatherv(buf, sendcnt, MPI_BYTE, buffer.get(), recvcnt.data(), recvpos.data(), MPI_BYTE, 0,
                info->intra_comm);

    if (info->intra_rank == 0) {
      file.seekp(disp);
      file.write(reinterpret_cast<char*>(buffer.get()), totsize);
    }

    disp += totsize;

    return totsize;
  }
};

template <typename App, typename Data>
class AsyncDiag : public BaseDiag<App, Data>
{
protected:
  std::unique_ptr<AsyncHandler> handler;
  std::vector<Buffer>           buffer;

  // check if the diagnostic is required
  bool require_diagnostic(int curstep, json& config)
  {
    bool status    = BaseDiag<App, Data>::require_diagnostic(curstep, config);
    bool completed = handler->is_completed();

    if (status == true) {
      // make sure all the requests are completed
      handler->wait_all();
    }

    if (status == false && completed == false) {
      // check if all the request are completed
      if (handler->test_all()) {
        handler->wait_all();
      }
    }

    return status;
  }

public:
  // constructor
  AsyncDiag(std::string name, std::shared_ptr<DiagInfo> info, int size = 0)
      : BaseDiag<App, Data>(name, info)
  {
    // create handler
    if (info->iomode == "mpiio") {
      handler = std::make_unique<MpiioHandler>(info);
    } else if (info->iomode == "posix") {
      handler = std::make_unique<PosixHandler>(info);
    }

    set_queue_size(size);
  }

  void set_queue_size(int size)
  {
    if (size != buffer.size()) {
      buffer.resize(size);
    }

    handler->set_queue_size(size);
  }

  // open file
  void open_file(std::string filename, size_t* disp, const char* mode)
  {
    handler->open_file(filename, disp, mode);
  }

  // close file
  void close_file()
  {
    handler->close_file();
  }

  // check if all the requests are completed
  bool is_completed()
  {
    return handler->is_completed();
  }

  // wait for the completion of the job
  void wait(int index)
  {
    handler->wait(index);
  }

  // wait for the completion of all the jobs and close the file
  void wait_all()
  {
    handler->wait_all();
  }

  bool test_all()
  {
    return handler->test_all();
  }

  std::vector<int> get_chunk_id_range(Data& data)
  {
    int id_min = std::numeric_limits<int>::max();
    int id_max = std::numeric_limits<int>::min();

    for (int i = 0; i < data.chunkvec.size(); i++) {
      id_min = std::min(id_min, data.chunkvec[i]->get_id());
      id_max = std::max(id_max, data.chunkvec[i]->get_id());
    }

    assert(id_max - id_min + 1 == data.chunkvec.size());

    return handler->get_chunk_id_range(id_min, id_max);
  }

  // launch asynchronous write
  template <typename DataPacker>
  size_t launch(int index, DataPacker packer, Data& data, size_t& disp)
  {
    size_t bufsize = 0;

    // calculate buffer size
    for (int i = 0; i < data.chunkvec.size(); i++) {
      bufsize += data.chunkvec[i]->pack_diagnostic(packer, nullptr, 0);
    }

    // pack data
    buffer[index].resize(bufsize);
    uint8_t* bufptr = buffer[index].get();

    for (int i = 0, address = 0; i < data.chunkvec.size(); i++) {
      address = data.chunkvec[i]->pack_diagnostic(packer, bufptr, address);
    }

    // write to the disk
    return handler->write(index, bufptr, bufsize, disp);
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
