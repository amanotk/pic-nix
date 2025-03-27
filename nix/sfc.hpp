// -*- C++ -*-
#ifndef _SFC_HPP_
#define _SFC_HPP_

#include "tinyformat.hpp"
#include "xtensorall.hpp"

///
/// Space-Filling Curve (SFC) module
///
/// Following implementation of SFC is according to the python code for Generalized Hilbert Curve
/// or pseudo Hilbert curve of arbitrary sizes available at:
///   https://github.com/jakubcerveny/gilbert
/// However, if the size is odd, it requires a non-local (i.e., diagonal) path. The following code
/// thus accepts only even numbers as the size in each direction, which guaranteees the locality of
/// resulting SFC.
///

namespace sfc
{
using array1d = xt::xtensor<int, 1>;
using array2d = xt::xtensor<int, 2>;
using array3d = xt::xtensor<int, 3>;

///
/// @brief construct 1D SFC map
/// @param Nx number of cells in x
/// @param index indices or IDs of cells (1D array)
/// @param coord x coordiante of cells (1D array)
///
void get_map1d(size_t Nx, array1d& index, array2d& coord);

///
/// @brief construct 2D SFC map
/// @param Ny number of cells in y
/// @param Nx number of cells in x
/// @param index indices or IDs of cells (1D array)
/// @param coord x and y coordinates of cells (2D array)
///
void get_map2d(size_t Ny, size_t Nx, array2d& index, array2d& coord);

///
/// @brief construct 3D SFC map
/// @param Nz number of cells in z
/// @param Ny number of cells in y
/// @param Nx number of cells in x
/// @param index indices or IDs of cells (1D array)
/// @param coord x, y, and z coordinates of cells (2D array)
///
void get_map3d(size_t Nz, size_t Ny, size_t Nx, array3d& index, array2d& coord);

///
/// @brief check index array
/// @tparam T typename of arrays
/// @param index index of cells
/// @return true if it is valid and false otherwise
///
template <typename T>
bool check_index(T& index);

///
/// @brief check locality of 2D map
/// @param coord coordinate array to be checked
/// @param distmax2 maximum allowable distance square between neighboring cells
/// @return true if it is local and false otherwise
///
bool check_locality2d(array2d& coord, const int distmax2 = 1);

///
/// @brief check locality of 3D map
/// @param coord coordinate array to be checked
/// @param distmax2 maximum allowable distance square between neighboring cells
/// @return true if it is local and false otherwise
///
bool check_locality3d(array2d& coord, const int distmax2 = 1);

} // namespace sfc

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
