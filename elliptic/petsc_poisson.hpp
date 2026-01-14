#ifndef _PETSC_POISSON_HPP_
#define _PETSC_POISSON_HPP_

#include "elliptic.hpp"
#include "petsc_interface.hpp"

namespace elliptic
{
using namespace nix::typedefs;
using nix::Dims3D;

class PetscPoisson3D : public PetscInterface
{
public:
  PetscPoisson3D(Dims3D dims) : PetscInterface(dims)
  {
    set_matrix(1.0, 1.0, 1.0);
  }

protected:
  virtual void set_matrix(float64 hx, float64 hy, float64 hz) override
  {
    const float64 hx2_inv = 1.0 / (hx * hx);
    const float64 hy2_inv = 1.0 / (hy * hy);
    const float64 hz2_inv = 1.0 / (hz * hz);
    const float64 diag    = 2.0 * hx2_inv + 2.0 * hy2_inv + 2.0 * hz2_inv;
    const float64 ofdx    = -1.0 * hx2_inv;
    const float64 ofdy    = -1.0 * hy2_inv;
    const float64 ofdz    = -1.0 * hz2_inv;

    DMDALocalInfo info;
    DMDAGetLocalInfo(dm_obj, &info);

    // create matrix
    if (matrix != nullptr) {
      MatDestroy(&matrix);
    }
    DMCreateMatrix(dm_obj, &matrix);

    for (int iz = info.zs; iz < info.zs + info.zm; ++iz) {
      for (int iy = info.ys; iy < info.ys + info.ym; ++iy) {
        for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
          MatStencil row;
          MatStencil col[7];
          float64    vals[7];
          int        ncols = 0;

          row.i = ix;
          row.j = iy;
          row.k = iz;
          row.c = 0;

          col[ncols].i = ix;
          col[ncols].j = iy;
          col[ncols].k = iz;
          col[ncols].c = 0;
          vals[ncols]  = diag;
          ncols++;

          col[ncols].i = ix - 1;
          col[ncols].j = iy;
          col[ncols].k = iz;
          col[ncols].c = 0;
          vals[ncols]  = ofdx;
          ncols++;

          col[ncols].i = ix + 1;
          col[ncols].j = iy;
          col[ncols].k = iz;
          col[ncols].c = 0;
          vals[ncols]  = ofdx;
          ncols++;

          col[ncols].i = ix;
          col[ncols].j = iy - 1;
          col[ncols].k = iz;
          col[ncols].c = 0;
          vals[ncols]  = ofdy;
          ncols++;

          col[ncols].i = ix;
          col[ncols].j = iy + 1;
          col[ncols].k = iz;
          col[ncols].c = 0;
          vals[ncols]  = ofdy;
          ncols++;

          col[ncols].i = ix;
          col[ncols].j = iy;
          col[ncols].k = iz - 1;
          col[ncols].c = 0;
          vals[ncols]  = ofdz;
          ncols++;

          col[ncols].i = ix;
          col[ncols].j = iy;
          col[ncols].k = iz + 1;
          col[ncols].c = 0;
          vals[ncols]  = ofdz;
          ncols++;

          MatSetValuesStencil(matrix, 1, &row, ncols, col, vals, INSERT_VALUES);
        }
      }
    }

    // assemble matrix
    MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);

    // set null space
    {
      MatNullSpace ns;
      MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &ns);
      MatSetNullSpace(matrix, ns);
      MatNullSpaceDestroy(&ns);
    }
  }
};

} // namespace elliptic

#endif //_PETSC_POISSON_HPP_