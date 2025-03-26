// -*- C++ -*-
#ifndef _PARALLEL_DIAG_HPP_
#define _PARALLEL_DIAG_HPP_

#include "pic.hpp"
#include "pic_diag.hpp"
#include "pic_packer.hpp"

#include "diag_handler.hpp"

class ParallelDiag : public PicDiag
{
protected:
  std::unique_ptr<PicDiagHandler> handler;
  std::vector<nix::Buffer>        buffer;

  // check if the diagnostic is required
  virtual bool require_diagnostic(int curstep, json& config) override
  {
    bool status    = PicDiag::require_diagnostic(curstep, config);
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
  ParallelDiag(std::string name, app_type& application, std::shared_ptr<info_type> info,
               int size = 0)
      : PicDiag(name, application, info)
  {
    // create handler
    if (info->iomode == "mpiio") {
      handler = std::make_unique<MpiioHandler>(info);
    } else if (info->iomode == "posix") {
      handler = std::make_unique<PosixHandler>(info);
    }
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
    buffer.erase(buffer.begin() + index);
  }

  // wait for the completion of all the jobs and close the file
  void wait_all()
  {
    handler->wait_all();
    buffer.clear();
  }

  bool test_all()
  {
    return handler->test_all();
  }

  std::vector<int> get_chunk_id_range(data_type& data)
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

  // queue write request
  size_t queue(PicPacker& packer, data_type& data, size_t& disp)
  {
    size_t bufsize = 0;

    // calculate buffer size
    for (int i = 0; i < data.chunkvec.size(); i++) {
      auto chunk_data = data.chunkvec[i]->get_internal_data();
      bufsize += packer(chunk_data, nullptr, 0);
    }

    // pack data
    buffer.emplace_back(bufsize);
    int  index  = buffer.size() - 1;
    auto bufptr = buffer[index].get();

    for (int i = 0, address = 0; i < data.chunkvec.size(); i++) {
      auto chunk_data = data.chunkvec[i]->get_internal_data();
      address         = packer(chunk_data, bufptr, address);
    }

    // write to the disk
    auto count = handler->queue(index, buffer[index], disp);

    // TODO: implement asynchronous write
    wait(index);

    return count;
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
