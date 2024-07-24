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
public:
  PosixHandler(std::shared_ptr<DiagInfo> info) : AsyncHandler(info)
  {
  }

  virtual void set_queue_size(int size) override
  {
  }

  virtual void open_file(std::string filename, size_t* disp, const char* mode) override
  {
  }

  virtual void close_file() override
  {
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

  virtual size_t write(int index, uint8_t* buf, size_t bufsize, size_t& disp) override
  {
    return 0;
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
      handler->test_all();
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
