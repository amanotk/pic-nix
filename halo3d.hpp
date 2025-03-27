// -*- C++ -*-
#ifndef _HALO3D_HPP_
#define _HALO3D_HPP_

#include "nix.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Halo3D
///
template <typename Data, typename Chunk, bool FixedBufferFlag>
class Halo3D
{
public:
  static constexpr bool is_buffer_fixed = FixedBufferFlag;

  Data*  data;
  Chunk* chunk;

  ///
  /// @brief constructor
  ///
  Halo3D(Data& data, Chunk& chunk)
  {
    this->data  = &data;
    this->chunk = &chunk;
  }

  ///
  /// @brief pre-processing for packing
  ///
  template <typename BufferPtr>
  void pre_pack(BufferPtr& mpibuf, int indexlb[3], int indexub[3])
  {
  }

  ///
  /// @brief post-processing for packing
  ///
  template <typename BufferPtr>
  void post_pack(BufferPtr& mpibuf, int indexlb[3], int indexub[3])
  {
  }

  ///
  /// @brief pre-processing for unpacking
  ///
  template <typename BufferPtr>
  void pre_unpack(BufferPtr& mpibuf, int indexlb[3], int indexub[3])
  {
  }

  ///
  /// @brief post-processing for unpacking
  ///
  template <typename BufferPtr>
  void post_unpack(BufferPtr& mpibuf, int indexlb[3], int indexub[3])
  {
  }

  ///
  /// @brief perform packing; return false if send/recv is not required
  /// @param mpibuf pointer to MpiBuffer
  /// @param iz direction in z (either 0, 1, 2)
  /// @param iy direction in y (either 0, 1, 2)
  /// @param ix direction in x (either 0, 1, 2)
  /// @param send_bound send lower- and upper- bounds
  /// @param recv_bound recv lower- and upper- bounds
  ///
  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2]);

  ///
  /// @brief perform unpacking; return false if send/recv is not required
  /// @param mpibuf pointer to MpiBuffer
  /// @param iz direction in z (either 0, 1, 2)
  /// @param iy direction in y (either 0, 1, 2)
  /// @param ix direction in x (either 0, 1, 2)
  /// @param send_bound send lower- and upper- bounds
  /// @param recv_bound recv lower- and upper- bounds
  ///
  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2],
              int recv_bound[3][2]);
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
