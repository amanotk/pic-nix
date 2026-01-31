#ifndef _PETSC_MATRIX_HELPERS_HPP_
#define _PETSC_MATRIX_HELPERS_HPP_

#include "nix.hpp"

#include <petscdmda.h>
#include <petscmat.h>

namespace elliptic
{

using nix::typedefs::float64;

void build_poisson_matrix_1d(Mat matrix, DM dm, float64 diag, float64 ofdx);
void build_poisson_matrix_2d(Mat matrix, DM dm, float64 diag, float64 ofdx, float64 ofdy);
void build_poisson_matrix_3d(Mat matrix, DM dm, float64 diag, float64 ofdx, float64 ofdy,
                             float64 ofdz);

} // namespace elliptic

#endif
