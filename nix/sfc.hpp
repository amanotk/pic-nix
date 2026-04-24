// -*- C++ -*-
#ifndef _SFC_HPP_
#define _SFC_HPP_

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

#include <cstddef>
#include <vector>

namespace sfc
{
///
/// @brief construct 2D SFC map
/// @param Ny number of cells in y
/// @param Nx number of cells in x
/// @param ncol number of columns in coord (2 for standalone, 3 when called from 3D)
/// @param index indices or IDs of cells (flat Ny*Nx, row-major)
/// @param coord coordinates of cells (flat (Ny*Nx)*ncol, row-major)
///
void get_map2d(size_t Ny, size_t Nx, int ncol, std::vector<int>& index, std::vector<int>& coord);

///
/// @brief construct 3D SFC map
/// @param Nz number of cells in z
/// @param Ny number of cells in y
/// @param Nx number of cells in x
/// @param index indices or IDs of cells (flat Nz*Ny*Nx, row-major z-y-x)
/// @param coord coordinates of cells (flat (Nz*Ny*Nx)*3, row-major)
///
void get_map3d(size_t Nz, size_t Ny, size_t Nx, std::vector<int>& index, std::vector<int>& coord);

///
/// @brief check that index is a valid permutation of [0..N-1]
/// @param index flat index array
/// @return true if valid
///
bool check_index(const std::vector<int>& index);

///
/// @brief check locality of 2D map
/// @param coord coordinate array (flat, ncol=2, row-major)
/// @param N number of cells
/// @param distmax2 maximum allowable distance square between neighboring cells
/// @return true if local
///
bool check_locality2d(const std::vector<int>& coord, size_t N, int distmax2 = 1);

///
/// @brief check locality of 3D map
/// @param coord coordinate array (flat, ncol=3, row-major)
/// @param N number of cells
/// @param distmax2 maximum allowable distance square between neighboring cells
/// @return true if local
///
bool check_locality3d(const std::vector<int>& coord, size_t N, int distmax2 = 1);

} // namespace sfc

#endif
