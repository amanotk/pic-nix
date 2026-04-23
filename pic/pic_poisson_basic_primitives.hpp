// -*- C++ -*-
#pragma once

#include "pic.hpp"

namespace pic_poisson_basic_primitives
{

struct OperatorChunkData {
  const float64* in   = nullptr;
  float64*       out  = nullptr;
  int            nz   = 0;
  int            ny   = 0;
  int            nx   = 0;
  int            Lbx  = 0;
  int            Ubx  = 0;
  int            Lby  = 0;
  int            Uby  = 0;
  int            Lbz  = 0;
  int            Ubz  = 0;
  float64        delx = 1.0;
  float64        dely = 1.0;
  float64        delz = 1.0;
};

struct PreconditionerChunkData {
  const float64* r     = nullptr;
  float64*       z     = nullptr;
  int            nz    = 0;
  int            ny    = 0;
  int            nx    = 0;
  int            Lbx   = 0;
  int            Ubx   = 0;
  int            Lby   = 0;
  int            Uby   = 0;
  int            Lbz   = 0;
  int            Ubz   = 0;
  float64        delx  = 1.0;
  float64        dely  = 1.0;
  float64        delz  = 1.0;
  float64        omega = 1.0;
};

inline int flat_index(const OperatorChunkData& cw, int iz, int iy, int ix)
{
  return iz * (cw.ny * cw.nx) + iy * cw.nx + ix;
}

inline int flat_index(const PreconditionerChunkData& cw, int iz, int iy, int ix)
{
  return iz * (cw.ny * cw.nx) + iy * cw.nx + ix;
}

inline void apply_operator_1d_primitive(const OperatorChunkData& cw)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 ofdx = -dx2;

  for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
    for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
      for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
        const int     idx    = flat_index(cw, iz, iy, ix);
        const int     idx_xm = flat_index(cw, iz, iy, ix - 1);
        const int     idx_xp = flat_index(cw, iz, iy, ix + 1);
        const float64 diag   = 2.0 * dx2;
        const float64 sum    = ofdx * (cw.in[idx_xm] + cw.in[idx_xp]);
        cw.out[idx]          = diag * cw.in[idx] + sum;
      }
    }
  }
}

inline void apply_operator_2d_primitive(const OperatorChunkData& cw)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 dy2  = 1.0 / (cw.dely * cw.dely);
  const float64 ofdx = -dx2;
  const float64 ofdy = -dy2;

  for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
    for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
      for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
        const int     idx    = flat_index(cw, iz, iy, ix);
        const int     idx_xm = flat_index(cw, iz, iy, ix - 1);
        const int     idx_xp = flat_index(cw, iz, iy, ix + 1);
        const int     idx_ym = flat_index(cw, iz, iy - 1, ix);
        const int     idx_yp = flat_index(cw, iz, iy + 1, ix);
        const float64 diag   = 2.0 * dx2 + 2.0 * dy2;
        const float64 sum =
            ofdx * (cw.in[idx_xm] + cw.in[idx_xp]) + ofdy * (cw.in[idx_ym] + cw.in[idx_yp]);
        cw.out[idx] = diag * cw.in[idx] + sum;
      }
    }
  }
}

inline void apply_operator_3d_primitive(const OperatorChunkData& cw)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 dy2  = 1.0 / (cw.dely * cw.dely);
  const float64 dz2  = 1.0 / (cw.delz * cw.delz);
  const float64 ofdx = -dx2;
  const float64 ofdy = -dy2;
  const float64 ofdz = -dz2;

  for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
    for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
      for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
        const int     idx    = flat_index(cw, iz, iy, ix);
        const int     idx_xm = flat_index(cw, iz, iy, ix - 1);
        const int     idx_xp = flat_index(cw, iz, iy, ix + 1);
        const int     idx_ym = flat_index(cw, iz, iy - 1, ix);
        const int     idx_yp = flat_index(cw, iz, iy + 1, ix);
        const int     idx_zm = flat_index(cw, iz - 1, iy, ix);
        const int     idx_zp = flat_index(cw, iz + 1, iy, ix);
        const float64 diag   = 2.0 * dx2 + 2.0 * dy2 + 2.0 * dz2;
        const float64 sum    = ofdx * (cw.in[idx_xm] + cw.in[idx_xp]) +
                            ofdy * (cw.in[idx_ym] + cw.in[idx_yp]) +
                            ofdz * (cw.in[idx_zm] + cw.in[idx_zp]);
        cw.out[idx] = diag * cw.in[idx] + sum;
      }
    }
  }
}

inline void preconditioner_forward_1d_primitive(const PreconditionerChunkData& cw)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 ofdx = -dx2;

  for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
    for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
      for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
        const int     idx    = flat_index(cw, iz, iy, ix);
        const int     idx_xm = flat_index(cw, iz, iy, ix - 1);
        const int     idx_xp = flat_index(cw, iz, iy, ix + 1);
        const float64 diag   = 2.0 * dx2;
        float64       sum    = cw.r[idx];
        sum -= ofdx * (cw.z[idx_xm] + cw.z[idx_xp]);
        const float64 z_new = (diag > 0.0) ? sum / diag : cw.r[idx];
        cw.z[idx]           = (1.0 - cw.omega) * cw.z[idx] + cw.omega * z_new;
      }
    }
  }
}

inline void preconditioner_forward_2d_primitive(const PreconditionerChunkData& cw)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 dy2  = 1.0 / (cw.dely * cw.dely);
  const float64 ofdx = -dx2;
  const float64 ofdy = -dy2;

  for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
    for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
      for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
        const int     idx    = flat_index(cw, iz, iy, ix);
        const int     idx_xm = flat_index(cw, iz, iy, ix - 1);
        const int     idx_xp = flat_index(cw, iz, iy, ix + 1);
        const int     idx_ym = flat_index(cw, iz, iy - 1, ix);
        const int     idx_yp = flat_index(cw, iz, iy + 1, ix);
        const float64 diag   = 2.0 * dx2 + 2.0 * dy2;
        float64       sum    = cw.r[idx];
        sum -= ofdx * (cw.z[idx_xm] + cw.z[idx_xp]);
        sum -= ofdy * (cw.z[idx_ym] + cw.z[idx_yp]);
        const float64 z_new = (diag > 0.0) ? sum / diag : cw.r[idx];
        cw.z[idx]           = (1.0 - cw.omega) * cw.z[idx] + cw.omega * z_new;
      }
    }
  }
}

inline void preconditioner_forward_3d_primitive(const PreconditionerChunkData& cw)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 dy2  = 1.0 / (cw.dely * cw.dely);
  const float64 dz2  = 1.0 / (cw.delz * cw.delz);
  const float64 ofdx = -dx2;
  const float64 ofdy = -dy2;
  const float64 ofdz = -dz2;

  for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
    for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
      for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
        const int     idx    = flat_index(cw, iz, iy, ix);
        const int     idx_xm = flat_index(cw, iz, iy, ix - 1);
        const int     idx_xp = flat_index(cw, iz, iy, ix + 1);
        const int     idx_ym = flat_index(cw, iz, iy - 1, ix);
        const int     idx_yp = flat_index(cw, iz, iy + 1, ix);
        const int     idx_zm = flat_index(cw, iz - 1, iy, ix);
        const int     idx_zp = flat_index(cw, iz + 1, iy, ix);
        const float64 diag   = 2.0 * dx2 + 2.0 * dy2 + 2.0 * dz2;
        float64       sum    = cw.r[idx];
        sum -= ofdx * (cw.z[idx_xm] + cw.z[idx_xp]);
        sum -= ofdy * (cw.z[idx_ym] + cw.z[idx_yp]);
        sum -= ofdz * (cw.z[idx_zm] + cw.z[idx_zp]);
        const float64 z_new = (diag > 0.0) ? sum / diag : cw.r[idx];
        cw.z[idx]           = (1.0 - cw.omega) * cw.z[idx] + cw.omega * z_new;
      }
    }
  }
}

inline void preconditioner_backward_1d_primitive(const PreconditionerChunkData& cw)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 ofdx = -dx2;

  for (int iz = cw.Ubz; iz >= cw.Lbz; --iz) {
    for (int iy = cw.Uby; iy >= cw.Lby; --iy) {
      for (int ix = cw.Ubx; ix >= cw.Lbx; --ix) {
        const int     idx    = flat_index(cw, iz, iy, ix);
        const int     idx_xm = flat_index(cw, iz, iy, ix - 1);
        const int     idx_xp = flat_index(cw, iz, iy, ix + 1);
        const float64 diag   = 2.0 * dx2;
        float64       sum    = cw.r[idx];
        sum -= ofdx * (cw.z[idx_xm] + cw.z[idx_xp]);
        const float64 z_new = (diag > 0.0) ? sum / diag : cw.r[idx];
        cw.z[idx]           = (1.0 - cw.omega) * cw.z[idx] + cw.omega * z_new;
      }
    }
  }
}

inline void preconditioner_backward_2d_primitive(const PreconditionerChunkData& cw)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 dy2  = 1.0 / (cw.dely * cw.dely);
  const float64 ofdx = -dx2;
  const float64 ofdy = -dy2;

  for (int iz = cw.Ubz; iz >= cw.Lbz; --iz) {
    for (int iy = cw.Uby; iy >= cw.Lby; --iy) {
      for (int ix = cw.Ubx; ix >= cw.Lbx; --ix) {
        const int     idx    = flat_index(cw, iz, iy, ix);
        const int     idx_xm = flat_index(cw, iz, iy, ix - 1);
        const int     idx_xp = flat_index(cw, iz, iy, ix + 1);
        const int     idx_ym = flat_index(cw, iz, iy - 1, ix);
        const int     idx_yp = flat_index(cw, iz, iy + 1, ix);
        const float64 diag   = 2.0 * dx2 + 2.0 * dy2;
        float64       sum    = cw.r[idx];
        sum -= ofdx * (cw.z[idx_xm] + cw.z[idx_xp]);
        sum -= ofdy * (cw.z[idx_ym] + cw.z[idx_yp]);
        const float64 z_new = (diag > 0.0) ? sum / diag : cw.r[idx];
        cw.z[idx]           = (1.0 - cw.omega) * cw.z[idx] + cw.omega * z_new;
      }
    }
  }
}

inline void preconditioner_backward_3d_primitive(const PreconditionerChunkData& cw)
{
  const float64 dx2  = 1.0 / (cw.delx * cw.delx);
  const float64 dy2  = 1.0 / (cw.dely * cw.dely);
  const float64 dz2  = 1.0 / (cw.delz * cw.delz);
  const float64 ofdx = -dx2;
  const float64 ofdy = -dy2;
  const float64 ofdz = -dz2;

  for (int iz = cw.Ubz; iz >= cw.Lbz; --iz) {
    for (int iy = cw.Uby; iy >= cw.Lby; --iy) {
      for (int ix = cw.Ubx; ix >= cw.Lbx; --ix) {
        const int     idx    = flat_index(cw, iz, iy, ix);
        const int     idx_xm = flat_index(cw, iz, iy, ix - 1);
        const int     idx_xp = flat_index(cw, iz, iy, ix + 1);
        const int     idx_ym = flat_index(cw, iz, iy - 1, ix);
        const int     idx_yp = flat_index(cw, iz, iy + 1, ix);
        const int     idx_zm = flat_index(cw, iz - 1, iy, ix);
        const int     idx_zp = flat_index(cw, iz + 1, iy, ix);
        const float64 diag   = 2.0 * dx2 + 2.0 * dy2 + 2.0 * dz2;
        float64       sum    = cw.r[idx];
        sum -= ofdx * (cw.z[idx_xm] + cw.z[idx_xp]);
        sum -= ofdy * (cw.z[idx_ym] + cw.z[idx_yp]);
        sum -= ofdz * (cw.z[idx_zm] + cw.z[idx_zp]);
        const float64 z_new = (diag > 0.0) ? sum / diag : cw.r[idx];
        cw.z[idx]           = (1.0 - cw.omega) * cw.z[idx] + cw.omega * z_new;
      }
    }
  }
}

} // namespace pic_poisson_basic_primitives
