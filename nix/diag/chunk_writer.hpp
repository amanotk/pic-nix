// -*- C++ -*-
#ifndef _CHUNK_WRITER_HPP_
#define _CHUNK_WRITER_HPP_

#include "buffer.hpp"
#include "chunk.hpp"
#include "diag.hpp"
#include "diag/io_handler.hpp"

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
class ChunkDiagWriter : public BaseDiag
{
protected:
  std::unique_ptr<DiagIoHandler> handler;

  using data_type  = typename BaseDiag::data_type;
  using chunk_type = typename BaseDiag::chunk_type;
  using info_type  = typename BaseDiag::info_type;
  using info_ptr   = std::shared_ptr<info_type>;

public:
  // constructor
  ChunkDiagWriter(std::string name, typename BaseDiag::PtrInterface interface)
      : BaseDiag(name, interface)
  {
    if (BaseDiag::info->iomode == "mpiio") {
      handler = std::make_unique<MpiioDiagIoHandler>(BaseDiag::info);
    } else if (BaseDiag::info->iomode == "posix") {
      handler = std::make_unique<PosixDiagIoHandler>(BaseDiag::info);
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
    Buffer packed_buffer(bufsize);
    auto   bufptr = packed_buffer.get();

    for (int i = 0, address = 0; i < data.chunkvec.size(); i++) {
      auto chunk = static_cast<chunk_type*>(data.chunkvec[i].get());
      address    = packer(chunk, bufptr, address);
    }

    // write packed buffer to disk
    return handler->write(packed_buffer, disp);
  }
};

NIX_NAMESPACE_END

#endif
