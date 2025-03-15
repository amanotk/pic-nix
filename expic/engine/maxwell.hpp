// -*- C++ -*-
#ifndef _ENGINE_MAXWELL_HPP_
#define _ENGINE_MAXWELL_HPP_

#include "nix/nix.hpp"

namespace engine
{

class Maxwell
{
public:
  int     boundary_margin;
  int     lbx;
  int     lby;
  int     lbz;
  int     ubx;
  int     uby;
  int     ubz;
  float64 cc;
  float64 dt;
  float64 dx;
  float64 dy;
  float64 dz;
  float64 theta;

  template <typename T_data>
  Maxwell(const T_data& data)
  {
    boundary_margin = data.boundary_margin;
    lbx             = data.Lbx;
    lby             = data.Lby;
    lbz             = data.Lbz;
    ubx             = data.Ubx;
    uby             = data.Uby;
    ubz             = data.Ubz;
    cc              = data.cc;
    dx              = data.delx;
    dy              = data.dely;
    dz              = data.delz;
    theta           = data.option.value("friedman", 0.0);
  }

  template <typename T_field, typename T_current, typename T_friedman>
  void push_efd_1d(T_field& uf, T_current& uj, T_friedman& ff, float64 delt)
  {
    const int     Nb   = boundary_margin;
    const float64 cflx = cc * delt / dx;
    const float64 cfly = cc * delt / dy;
    const float64 cflz = cc * delt / dz;

    // update for Friedman filter first (boundary condition has already been set)
    {
      int iz = lbz;
      {
        int iy = lby;
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          // Ex
          ff(iz, iy, ix, 2, 0) = ff(iz, iy, ix, 1, 0) + theta * ff(iz, iy, ix, 2, 0);
          ff(iz, iy, ix, 1, 0) = uf(iz, iy, ix, 0);
          // Ey
          ff(iz, iy, ix, 2, 1) = ff(iz, iy, ix, 1, 1) + theta * ff(iz, iy, ix, 2, 1);
          ff(iz, iy, ix, 1, 1) = uf(iz, iy, ix, 1);
          // Ez
          ff(iz, iy, ix, 2, 2) = ff(iz, iy, ix, 1, 2) + theta * ff(iz, iy, ix, 2, 2);
          ff(iz, iy, ix, 1, 2) = uf(iz, iy, ix, 2);
        }
      }
    }

    // Ex
    {
      int iz = lbz;
      {
        int iy = lby;
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 0) += -delt * uj(iz, iy, ix, 1);
        }
      }
    }

    // Ey
    {
      int iz = lbz;
      {
        int iy = lby;
        for (int ix = lbx - Nb; ix <= ubx + Nb - 1; ix++) {
          uf(iz, iy, ix, 1) +=
              (-cflx) * (uf(iz, iy, ix + 1, 5) - uf(iz, iy, ix, 5)) - delt * uj(iz, iy, ix, 2);
        }
      }
    }

    // Ez
    {
      int iz = lbz;
      {
        int iy = lby;
        for (int ix = lbx - Nb; ix <= ubx + Nb - 1; ix++) {
          uf(iz, iy, ix, 2) +=
              (+cflx) * (uf(iz, iy, ix + 1, 4) - uf(iz, iy, ix, 4)) + -delt * uj(iz, iy, ix, 3);
        }
      }
    }
  }

  template <typename T_field, typename T_current, typename T_friedman>
  void push_bfd_1d(T_field& uf, T_current& uj, T_friedman& ff, float64 delt)
  {
    const int     Nb   = boundary_margin;
    const float64 A    = 1 + 0.5 * theta;
    const float64 B    = -theta * (1 - 0.5 * theta);
    const float64 C    = 0.5 * theta * (1 - theta) * (1 - theta);
    const float64 cflx = cc * delt / dx;
    const float64 cfly = cc * delt / dy;
    const float64 cflz = cc * delt / dz;

    // update for Friedman filter first (boundary condition has already been set)
    {
      int iz = lbz;
      {
        int iy = lby;
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          ff(iz, iy, ix, 0, 0) =
              A * uf(iz, iy, ix, 0) + B * ff(iz, iy, ix, 1, 0) + C * ff(iz, iy, ix, 2, 0);
          ff(iz, iy, ix, 0, 1) =
              A * uf(iz, iy, ix, 1) + B * ff(iz, iy, ix, 1, 1) + C * ff(iz, iy, ix, 2, 1);
          ff(iz, iy, ix, 0, 2) =
              A * uf(iz, iy, ix, 2) + B * ff(iz, iy, ix, 1, 2) + C * ff(iz, iy, ix, 2, 2);
        }
      }
    }

    // Bx
    {
      // do nothing
    }

    // By
    {
      int iz = lbz;
      {
        int iy = lby;
        for (int ix = lbx - Nb + 1; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 4) += (+cflx) * (ff(iz, iy, ix, 0, 2) - ff(iz, iy, ix - 1, 0, 2));
        }
      }
    }

    // Bz
    {
      int iz = lbz;
      {
        int iy = lby;
        for (int ix = lbx - Nb + 1; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 5) += (-cflx) * (ff(iz, iy, ix, 0, 1) - ff(iz, iy, ix - 1, 0, 1));
        }
      }
    }
  }

  template <typename T_field, typename T_current, typename T_friedman>
  void push_efd_2d(T_field& uf, T_current& uj, T_friedman& ff, float64 delt)
  {
    const int     Nb   = boundary_margin;
    const float64 cflx = cc * delt / dx;
    const float64 cfly = cc * delt / dy;
    const float64 cflz = cc * delt / dz;

    // update for Friedman filter first (boundary condition has already been set)
    {
      int iz = lbz;
      for (int iy = lby - Nb; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          // Ex
          ff(iz, iy, ix, 2, 0) = ff(iz, iy, ix, 1, 0) + theta * ff(iz, iy, ix, 2, 0);
          ff(iz, iy, ix, 1, 0) = uf(iz, iy, ix, 0);
          // Ey
          ff(iz, iy, ix, 2, 1) = ff(iz, iy, ix, 1, 1) + theta * ff(iz, iy, ix, 2, 1);
          ff(iz, iy, ix, 1, 1) = uf(iz, iy, ix, 1);
          // Ez
          ff(iz, iy, ix, 2, 2) = ff(iz, iy, ix, 1, 2) + theta * ff(iz, iy, ix, 2, 2);
          ff(iz, iy, ix, 1, 2) = uf(iz, iy, ix, 2);
        }
      }
    }

    // Ex
    {
      int iz = lbz;
      for (int iy = lby - Nb; iy <= uby + Nb - 1; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 0) +=
              (+cfly) * (uf(iz, iy + 1, ix, 5) - uf(iz, iy, ix, 5)) - delt * uj(iz, iy, ix, 1);
        }
      }
    }

    // Ey
    {
      int iz = lbz;
      for (int iy = lby - Nb; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb - 1; ix++) {
          uf(iz, iy, ix, 1) +=
              (-cflx) * (uf(iz, iy, ix + 1, 5) - uf(iz, iy, ix, 5)) - delt * uj(iz, iy, ix, 2);
        }
      }
    }

    // Ez
    {
      int iz = lbz;
      for (int iy = lby - Nb; iy <= uby + Nb - 1; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb - 1; ix++) {
          uf(iz, iy, ix, 2) += (+cflx) * (uf(iz, iy, ix + 1, 4) - uf(iz, iy, ix, 4)) +
                               (-cfly) * (uf(iz, iy + 1, ix, 3) - uf(iz, iy, ix, 3)) -
                               delt * uj(iz, iy, ix, 3);
        }
      }
    }
  }

  template <typename T_field, typename T_current, typename T_friedman>
  void push_bfd_2d(T_field& uf, T_current& uj, T_friedman& ff, float64 delt)
  {
    const int     Nb   = boundary_margin;
    const float64 A    = 1 + 0.5 * theta;
    const float64 B    = -theta * (1 - 0.5 * theta);
    const float64 C    = 0.5 * theta * (1 - theta) * (1 - theta);
    const float64 cflx = cc * delt / dx;
    const float64 cfly = cc * delt / dy;
    const float64 cflz = cc * delt / dz;

    // update for Friedman filter first (boundary condition has already been set)
    {
      int iz = lbz;
      for (int iy = lby - Nb; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          ff(iz, iy, ix, 0, 0) =
              A * uf(iz, iy, ix, 0) + B * ff(iz, iy, ix, 1, 0) + C * ff(iz, iy, ix, 2, 0);
          ff(iz, iy, ix, 0, 1) =
              A * uf(iz, iy, ix, 1) + B * ff(iz, iy, ix, 1, 1) + C * ff(iz, iy, ix, 2, 1);
          ff(iz, iy, ix, 0, 2) =
              A * uf(iz, iy, ix, 2) + B * ff(iz, iy, ix, 1, 2) + C * ff(iz, iy, ix, 2, 2);
        }
      }
    }

    // Bx
    {
      int iz = lbz;
      for (int iy = lby - Nb + 1; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 3) += (-cfly) * (ff(iz, iy, ix, 0, 2) - ff(iz, iy - 1, ix, 0, 2));
        }
      }
    }

    // By
    {
      int iz = lbz;
      for (int iy = lby - Nb; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb + 1; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 4) += (+cflx) * (ff(iz, iy, ix, 0, 2) - ff(iz, iy, ix - 1, 0, 2));
        }
      }
    }

    // Bz
    {
      int iz = lbz;
      for (int iy = lby - Nb + 1; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb + 1; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 5) += (-cflx) * (ff(iz, iy, ix, 0, 1) - ff(iz, iy, ix - 1, 0, 1)) +
                               (+cfly) * (ff(iz, iy, ix, 0, 0) - ff(iz, iy - 1, ix, 0, 0));
        }
      }
    }
  }

  template <typename T_field, typename T_current, typename T_friedman>
  void push_efd_3d(T_field& uf, T_current& uj, T_friedman& ff, float64 delt)
  {
    const int     Nb   = boundary_margin;
    const float64 cflx = cc * delt / dx;
    const float64 cfly = cc * delt / dy;
    const float64 cflz = cc * delt / dz;

    // update for Friedman filter first (boundary condition has already been set)
    for (int iz = lbz - Nb; iz <= ubz + Nb; iz++) {
      for (int iy = lby - Nb; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          // Ex
          ff(iz, iy, ix, 2, 0) = ff(iz, iy, ix, 1, 0) + theta * ff(iz, iy, ix, 2, 0);
          ff(iz, iy, ix, 1, 0) = uf(iz, iy, ix, 0);
          // Ey
          ff(iz, iy, ix, 2, 1) = ff(iz, iy, ix, 1, 1) + theta * ff(iz, iy, ix, 2, 1);
          ff(iz, iy, ix, 1, 1) = uf(iz, iy, ix, 1);
          // Ez
          ff(iz, iy, ix, 2, 2) = ff(iz, iy, ix, 1, 2) + theta * ff(iz, iy, ix, 2, 2);
          ff(iz, iy, ix, 1, 2) = uf(iz, iy, ix, 2);
        }
      }
    }

    // Ex
    for (int iz = lbz - Nb; iz <= ubz + Nb - 1; iz++) {
      for (int iy = lby - Nb; iy <= uby + Nb - 1; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 0) += (+cfly) * (uf(iz, iy + 1, ix, 5) - uf(iz, iy, ix, 5)) +
                               (-cflz) * (uf(iz + 1, iy, ix, 4) - uf(iz, iy, ix, 4)) -
                               delt * uj(iz, iy, ix, 1);
        }
      }
    }

    // Ey
    for (int iz = lbz - Nb; iz <= ubz + Nb - 1; iz++) {
      for (int iy = lby - Nb; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb - 1; ix++) {
          uf(iz, iy, ix, 1) += (+cflz) * (uf(iz + 1, iy, ix, 3) - uf(iz, iy, ix, 3)) +
                               (-cflx) * (uf(iz, iy, ix + 1, 5) - uf(iz, iy, ix, 5)) -
                               delt * uj(iz, iy, ix, 2);
        }
      }
    }

    // Ez
    for (int iz = lbz - Nb; iz <= ubz + Nb; iz++) {
      for (int iy = lby - Nb; iy <= uby + Nb - 1; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb - 1; ix++) {
          uf(iz, iy, ix, 2) += (+cflx) * (uf(iz, iy, ix + 1, 4) - uf(iz, iy, ix, 4)) +
                               (-cfly) * (uf(iz, iy + 1, ix, 3) - uf(iz, iy, ix, 3)) -
                               delt * uj(iz, iy, ix, 3);
        }
      }
    }
  }

  template <typename T_field, typename T_current, typename T_friedman>
  void push_bfd_3d(T_field& uf, T_current& uj, T_friedman& ff, float64 delt)
  {
    const int     Nb   = boundary_margin;
    const float64 A    = 1 + 0.5 * theta;
    const float64 B    = -theta * (1 - 0.5 * theta);
    const float64 C    = 0.5 * theta * (1 - theta) * (1 - theta);
    const float64 cflx = cc * delt / dx;
    const float64 cfly = cc * delt / dy;
    const float64 cflz = cc * delt / dz;

    // update for Friedman filter first (boundary condition has already been set)
    for (int iz = lbz - Nb; iz <= ubz + Nb; iz++) {
      for (int iy = lby - Nb; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          ff(iz, iy, ix, 0, 0) =
              A * uf(iz, iy, ix, 0) + B * ff(iz, iy, ix, 1, 0) + C * ff(iz, iy, ix, 2, 0);
          ff(iz, iy, ix, 0, 1) =
              A * uf(iz, iy, ix, 1) + B * ff(iz, iy, ix, 1, 1) + C * ff(iz, iy, ix, 2, 1);
          ff(iz, iy, ix, 0, 2) =
              A * uf(iz, iy, ix, 2) + B * ff(iz, iy, ix, 1, 2) + C * ff(iz, iy, ix, 2, 2);
        }
      }
    }

    // Bx
    for (int iz = lbz - Nb + 1; iz <= ubz + Nb; iz++) {
      for (int iy = lby - Nb + 1; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 3) += (-cfly) * (ff(iz, iy, ix, 0, 2) - ff(iz, iy - 1, ix, 0, 2)) +
                               (+cflz) * (ff(iz, iy, ix, 0, 1) - ff(iz - 1, iy, ix, 0, 1));
        }
      }
    }

    // By
    for (int iz = lbz - Nb + 1; iz <= ubz + Nb; iz++) {
      for (int iy = lby - Nb; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb + 1; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 4) += (-cflz) * (ff(iz, iy, ix, 0, 0) - ff(iz - 1, iy, ix, 0, 0)) +
                               (+cflx) * (ff(iz, iy, ix, 0, 2) - ff(iz, iy, ix - 1, 0, 2));
        }
      }
    }

    // Bz
    for (int iz = lbz - Nb; iz <= ubz + Nb; iz++) {
      for (int iy = lby - Nb + 1; iy <= uby + Nb; iy++) {
        for (int ix = lbx - Nb + 1; ix <= ubx + Nb; ix++) {
          uf(iz, iy, ix, 5) += (-cflx) * (ff(iz, iy, ix, 0, 1) - ff(iz, iy, ix - 1, 0, 1)) +
                               (+cfly) * (ff(iz, iy, ix, 0, 0) - ff(iz, iy - 1, ix, 0, 0));
        }
      }
    }
  }
};

} // namespace engine

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif