#include "petsc_matrix_helpers.hpp"

#include <petscdmda.h>

namespace elliptic
{

void build_poisson_matrix_1d(Mat matrix, DM dm, float64 diag, float64 ofdx)
{
  DMDALocalInfo info;
  DMDAGetLocalInfo(dm, &info);

  MatStencil row;
  MatStencil col[3];
  float64    vals[3];

  for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
    int ncols = 0;

    row.i = ix;
    row.j = 0;
    row.k = 0;
    row.c = 0;

    col[ncols]    = row;
    vals[ncols++] = diag;

    col[ncols].i  = ix - 1;
    col[ncols].j  = 0;
    col[ncols].k  = 0;
    col[ncols].c  = 0;
    vals[ncols++] = ofdx;

    col[ncols].i  = ix + 1;
    col[ncols].j  = 0;
    col[ncols].k  = 0;
    col[ncols].c  = 0;
    vals[ncols++] = ofdx;

    MatSetValuesStencil(matrix, 1, &row, ncols, col, vals, ADD_VALUES);
  }

  MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);
}

void build_poisson_matrix_2d(Mat matrix, DM dm, float64 diag, float64 ofdx, float64 ofdy)
{
  DMDALocalInfo info;
  DMDAGetLocalInfo(dm, &info);

  for (int iy = info.ys; iy < info.ys + info.ym; ++iy) {
    for (int ix = info.xs; ix < info.xs + info.xm; ++ix) {
      MatStencil row;
      MatStencil col[5];
      float64    vals[5];
      int        ncols = 0;

      row.i = ix;
      row.j = iy;
      row.k = 0;
      row.c = 0;

      col[ncols]    = row;
      vals[ncols++] = diag;

      col[ncols].i  = ix - 1;
      col[ncols].j  = iy;
      col[ncols].k  = 0;
      col[ncols].c  = 0;
      vals[ncols++] = ofdx;

      col[ncols].i  = ix + 1;
      col[ncols].j  = iy;
      col[ncols].k  = 0;
      col[ncols].c  = 0;
      vals[ncols++] = ofdx;

      col[ncols].i  = ix;
      col[ncols].j  = iy - 1;
      col[ncols].k  = 0;
      col[ncols].c  = 0;
      vals[ncols++] = ofdy;

      col[ncols].i  = ix;
      col[ncols].j  = iy + 1;
      col[ncols].k  = 0;
      col[ncols].c  = 0;
      vals[ncols++] = ofdy;

      MatSetValuesStencil(matrix, 1, &row, ncols, col, vals, ADD_VALUES);
    }
  }

  MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);
}

void build_poisson_matrix_3d(Mat matrix, DM dm, float64 diag, float64 ofdx, float64 ofdy,
                             float64 ofdz)
{
  DMDALocalInfo info;
  DMDAGetLocalInfo(dm, &info);

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

        MatSetValuesStencil(matrix, 1, &row, ncols, col, vals, ADD_VALUES);
      }
    }
  }

  MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY);
}

} // namespace elliptic
