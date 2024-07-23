// -*- C++ -*-
#ifndef _ASYNC_DIAG_HPP_
#define _ASYNC_DIAG_HPP_

#include "base.hpp"

template <typename App, typename Data>
class AsyncDiag : public BaseDiag<App, Data>
{
protected:
  MPI_File                 filehandle;
  bool                     is_opened;
  std::vector<Buffer>      buffer;
  std::vector<MPI_Request> request;

  // check if the diagnostic is required
  bool require_diagnostic(int curstep, json& config)
  {
    bool status    = BaseDiag<App, Data>::require_diagnostic(curstep, config);
    bool completed = is_completed();

    if (status == true) {
      // make sure all the requests are completed
      wait_all();
    }

    if (status == false && completed == false) {
      // check if all the request are completed
      int flag = 0;
      MPI_Testall(request.size(), request.data(), &flag, MPI_STATUSES_IGNORE);
    }

    return status;
  }

public:
  // constructor
  AsyncDiag(std::string name, std::shared_ptr<DiagInfo> info, int size = 0)
      : BaseDiag<App, Data>(name, info), is_opened(false)
  {
    set_queue_size(size);
  }

  void set_queue_size(int size)
  {
    if (size == buffer.size() && size == request.size())
      return;

    // resize
    buffer.resize(size);
    request.resize(size);
    std::fill(request.begin(), request.end(), MPI_REQUEST_NULL);
  }

  // open file
  void open_file(std::string filename, size_t* disp, const char* mode)
  {
    if (is_opened == false) {
      nixio::open_file(filename.c_str(), &filehandle, disp, mode);
      is_opened = true;
    }
  }

  // close file
  void close_file()
  {
    assert(is_completed() == true);

    if (is_opened == true) {
      nixio::close_file(&filehandle);
      is_opened = false;
    }
  }

  // check if all the requests are completed
  bool is_completed()
  {
    bool status = std::all_of(request.begin(), request.end(),
                              [](auto& req) { return req == MPI_REQUEST_NULL; });
    return status;
  }

  // wait for the completion of the job
  void wait(int index)
  {
    MPI_Wait(&request[index], MPI_STATUS_IGNORE);
  }

  // wait for the completion of all the jobs and close the file
  void wait_all()
  {
    MPI_Waitall(request.size(), request.data(), MPI_STATUSES_IGNORE);
    std::fill(request.begin(), request.end(), MPI_REQUEST_NULL);
    close_file();
  }

  // launch asynchronous write
  template <typename DataPacker>
  void launch(int index, DataPacker packer, Data& data, size_t& disp)
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
    nixio::write_contiguous(&filehandle, &disp, bufptr, bufsize, 1, 1, &request[index]);
    wait(index);
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
