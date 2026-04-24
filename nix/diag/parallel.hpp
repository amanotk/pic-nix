// -*- C++ -*-
#ifndef _DIAG_PARALLEL_HPP_
#define _DIAG_PARALLEL_HPP_

#include "buffer.hpp"
#include "chunk.hpp"
#include "diag.hpp"
#include "diag/handler.hpp"

NIX_NAMESPACE_BEGIN

/// Parallel diagnostic base for module-specific diagnostics.
///
/// Module packer contract:
/// Packer must support the chunk-based call used by write_packed_chunks():
///
///   size_t operator()(typename BaseDiag::chunk_type* chunk, uint8_t* buffer, int address);
///
/// Modules may use adapters to forward to internal data packers when needed.
///
template <typename BaseDiag, typename Packer>
class ParallelDiag : public BaseDiag
{
protected:
  std::unique_ptr<DiagHandler> handler;
  std::vector<Buffer>          buffer;

  using data_type  = typename BaseDiag::data_type;
  using chunk_type = typename BaseDiag::chunk_type;
  using info_type  = typename BaseDiag::info_type;
  using info_ptr   = std::shared_ptr<info_type>;

  // check if the diagnostic is required
  virtual bool require_diagnostic(int curstep, json& config) override
  {
    bool status    = BaseDiag::require_diagnostic(curstep, config);
    bool completed = handler->is_completed();

    if (status == true) {
      handler->wait_all();
    }

    if (status == false && completed == false) {
      if (handler->test_all()) {
        handler->wait_all();
      }
    }

    return status;
  }

public:
  // constructor
  ParallelDiag(std::string name, typename BaseDiag::PtrInterface interface)
      : BaseDiag(name, interface)
  {
    if (BaseDiag::info->iomode == "mpiio") {
      handler = std::make_unique<MpiioDiagHandler>(BaseDiag::info);
    } else if (BaseDiag::info->iomode == "posix") {
      handler = std::make_unique<PosixDiagHandler>(BaseDiag::info);
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

  // write packed chunks to disk
  size_t write_packed_chunks(Packer& packer, data_type& data, size_t& disp)
  {
    size_t bufsize = 0;

    // calculate packed buffer size
    for (int i = 0; i < data.chunkvec.size(); i++) {
      auto chunk = static_cast<chunk_type*>(data.chunkvec[i].get());
      bufsize += packer(chunk, nullptr, 0);
    }

    // pack chunks into buffer
    buffer.emplace_back(bufsize);
    int  index  = buffer.size() - 1;
    auto bufptr = buffer[index].get();

    for (int i = 0, address = 0; i < data.chunkvec.size(); i++) {
      auto chunk = static_cast<chunk_type*>(data.chunkvec[i].get());
      address    = packer(chunk, bufptr, address);
    }

    // write packed buffer to disk
    auto count = handler->write(index, buffer[index], disp);

    // synchronous write: wait for completion immediately
    wait(index);

    return count;
  }
};

NIX_NAMESPACE_END

#endif
