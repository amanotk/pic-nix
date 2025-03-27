// -*- C++ -*-
#ifndef _BUFFER_HPP_
#define _BUFFER_HPP_

#include "nix.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Buffer for MPI
///
struct Buffer {
  using Pointer = std::unique_ptr<uint8_t[]>;

  int     size; ///< size of buffer in byte
  Pointer data; ///< data pointer

  ///
  /// @brief Constructor
  /// @param s size of buffer
  ///
  Buffer(int s = 0) : size(s)
  {
    data = std::make_unique<uint8_t[]>(size);
  }

  ///
  /// @brief get raw pointer
  /// @param pos position in byte from the beginning of pointer
  /// @return return pointer
  ///
  uint8_t* get(int pos = 0)
  {
    return data.get() + pos;
  }

  ///
  /// @brief resize the buffer
  /// @param s new size for resize
  ///
  void resize(int s)
  {
    const int copysize = std::min(size, s);

    // allocate new memory and copy contents
    Pointer p = std::make_unique<uint8_t[]>(s);
    std::memcpy(p.get(), data.get(), copysize);

    // move
    data.reset(nullptr);
    data = std::move(p);
    size = s;
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
