// -*- C++ -*-

//
// Implementation for Vectorized Version with xsimd
//
// This file is to be included from exchunk3d.cpp
//

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Order>                                                                             \
  type ExChunk3D<Order>::name##_impl_xsimd

DEFINE_MEMBER(void, push_position)(const float64 delt)
{
  using namespace exchunk3d_impl;
  const simd_i64 index = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  Position<Order, simd_f64> LoopBodyV(cc);
  Position<Order, float64>  LoopBodyS(cc);

  for (int is = 0; is < Ns; is++) {
    int np_simd = (up[is]->Np / simd_f64::size) * simd_f64::size;

    //
    // vectorized loop
    //
    for (int ip = 0; ip < np_simd; ip += simd_f64::size) {
      // local SIMD register
      simd_f64 xu[ParticleType::Nc];
      simd_f64 xv[ParticleType::Nc];

      // load particles to SIMD register
      for (int i = 0; i < ParticleType::Nc; i++) {
        xu[i] = simd_f64::gather(&up[is]->xu(ip, i), index);
        xv[i] = simd_f64::gather(&up[is]->xv(ip, i), index);
      }

      LoopBodyV(xv, xu, delt);

      // store particles to memory
      xu[0].scatter(&up[is]->xu(ip, 0), index);
      xu[1].scatter(&up[is]->xu(ip, 1), index);
      xu[2].scatter(&up[is]->xu(ip, 2), index);
      for (int i = 0; i < ParticleType::Nc; i++) {
        xv[i].scatter(&up[is]->xv(ip, i), index);
      }
    }

    //
    // scalar loop for reminder
    //
    for (int ip = np_simd; ip < up[is]->Np; ip++) {
      float64* xu = &up[is]->xu(ip, 0);
      float64* xv = &up[is]->xv(ip, 0);

      LoopBodyS(xv, xu, delt);
    }

    // boundary condition before counting
    this->set_boundary_particle(up[is], 0, up[is]->Np - 1, is);
    // count
    this->count_particle(up[is], 0, up[is]->Np - 1, true);
  }
}

DEFINE_MEMBER(template <int Interpolation> void, push_velocity)(const float64 delt)
{
  using namespace exchunk3d_impl;
  constexpr int  is_odd   = Order % 2 == 0 ? 0 : 1;
  const int      stride_x = 1;
  const int      stride_y = stride_x * (Ubx - Lbx + 2);
  const int      stride_z = stride_y * (Uby - Lby + 2);
  const int      lbx      = Lbx;
  const int      lby      = Lby;
  const int      lbz      = Lbz;
  const int      ubx      = Ubx + is_odd;
  const int      uby      = Uby + is_odd;
  const int      ubz      = Ubz + is_odd;
  const simd_i64 index    = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  Velocity<Order, simd_f64, Interpolation> LoopBodyV(delt, delx, dely, delz, xlim, ylim, zlim, Lbx,
                                                     Lby, Lbz, cc);
  Velocity<Order, float64, Interpolation>  LoopBodyS(delt, delx, dely, delz, xlim, ylim, zlim, Lbx,
                                                     Lby, Lbz, cc);

  for (int iz = lbz, jz = 0; iz <= ubz; iz++, jz++) {
    for (int iy = lby, jy = 0; iy <= uby; iy++, jy++) {
      for (int ix = lbx, jx = 0; ix <= ubx; ix++, jx++) {
        int ii = jz * stride_z + jy * stride_y + jx * stride_x; // 1D grid index

        // process particles in the cell
        for (int is = 0; is < Ns; is++) {
          int     ip_zero = up[is]->pindex(ii);
          int     np_cell = up[is]->pindex(ii + 1) - ip_zero;
          int     np_simd = (np_cell / simd_f64::size) * simd_f64::size;
          float64 qmdt    = 0.5 * up[is]->q / up[is]->m * delt;

          //
          // vectorized loop
          //
          for (int ip = ip_zero; ip < ip_zero + np_simd; ip += simd_f64::size) {
            // local SIMD register
            simd_f64 xu[6];

            // load particles to SIMD register
            xu[0] = simd_f64::gather(&up[is]->xu(ip, 0), index);
            xu[1] = simd_f64::gather(&up[is]->xu(ip, 1), index);
            xu[2] = simd_f64::gather(&up[is]->xu(ip, 2), index);
            xu[3] = simd_f64::gather(&up[is]->xu(ip, 3), index);
            xu[4] = simd_f64::gather(&up[is]->xu(ip, 4), index);
            xu[5] = simd_f64::gather(&up[is]->xu(ip, 5), index);

            LoopBodyV.sorted(uf, iz, iy, ix, xu, simd_f64(qmdt));

            // store particles to memory
            xu[3].scatter(&up[is]->xu(ip, 3), index);
            xu[4].scatter(&up[is]->xu(ip, 4), index);
            xu[5].scatter(&up[is]->xu(ip, 5), index);
          }

          //
          // scalar loop for reminder
          //
          for (int ip = ip_zero + np_simd; ip < ip_zero + np_cell; ip++) {
            LoopBodyS.sorted(uf, iz, iy, ix, &up[is]->xu(ip, 0), qmdt);
          }
        }
      }
    }
  }
}

DEFINE_MEMBER(template <int Interpolation> void, push_velocity_unsorted)(const float64 delt)
{
  using namespace exchunk3d_impl;
  const simd_i64 index = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  Velocity<Order, simd_f64, Interpolation> LoopBodyV(delt, delx, dely, delz, xlim, ylim, zlim, Lbx,
                                                     Lby, Lbz, cc);
  Velocity<Order, float64, Interpolation>  LoopBodyS(delt, delx, dely, delz, xlim, ylim, zlim, Lbx,
                                                     Lby, Lbz, cc);

  for (int is = 0; is < Ns; is++) {
    int     np_simd = (up[is]->Np / simd_f64::size) * simd_f64::size;
    float64 qmdt    = 0.5 * up[is]->q / up[is]->m * delt;

    //
    // vectorized loop
    //
    for (int ip = 0; ip < np_simd; ip += simd_f64::size) {
      // local SIMD register
      simd_f64 xu[6];

      // load particles to SIMD register
      xu[0] = simd_f64::gather(&up[is]->xu(ip, 0), index);
      xu[1] = simd_f64::gather(&up[is]->xu(ip, 1), index);
      xu[2] = simd_f64::gather(&up[is]->xu(ip, 2), index);
      xu[3] = simd_f64::gather(&up[is]->xu(ip, 3), index);
      xu[4] = simd_f64::gather(&up[is]->xu(ip, 4), index);
      xu[5] = simd_f64::gather(&up[is]->xu(ip, 5), index);

      LoopBodyV.unsorted(uf, xu, simd_f64(qmdt));

      // store particles to memory
      xu[3].scatter(&up[is]->xu(ip, 3), index);
      xu[4].scatter(&up[is]->xu(ip, 4), index);
      xu[5].scatter(&up[is]->xu(ip, 5), index);
    }

    //
    // scalar loop for reminder
    //
    for (int ip = np_simd; ip < up[is]->Np; ip++) {
      LoopBodyS.unsorted(uf, &up[is]->xu(ip, 0), qmdt);
    }
  }
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  using namespace exchunk3d_impl;
  constexpr int  size     = Order + 3;
  constexpr int  is_odd   = Order % 2 == 0 ? 0 : 1;
  const int      stride_x = 1;
  const int      stride_y = stride_x * (Ubx - Lbx + 2);
  const int      stride_z = stride_y * (Uby - Lby + 2);
  const int      lbx      = Lbx;
  const int      lby      = Lby;
  const int      lbz      = Lbz;
  const int      ubx      = Ubx + is_odd;
  const int      uby      = Uby + is_odd;
  const int      ubz      = Ubz + is_odd;
  const simd_i64 index    = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  Current<Order, simd_f64> LoopBodyV(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);
  Current<Order, float64>  LoopBodyS(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);

  // clear charge/current density
  uj.fill(0);

  for (int iz = lbz, jz = 0; iz <= ubz; iz++, jz++) {
    for (int iy = lby, jy = 0; iy <= uby; iy++, jy++) {
      for (int ix = lbx, jx = 0; ix <= ubx; ix++, jx++) {
        int ii = jz * stride_z + jy * stride_y + jx * stride_x; // 1D grid index

        // local current
        float64  cur[size][size][size][4]      = {0}; // scalar
        simd_f64 cur_simd[size][size][size][4] = {0}; // SIMD register

        // process particles in the cell
        for (int is = 0; is < Ns; is++) {
          int ip_zero = up[is]->pindex(ii);
          int np_cell = up[is]->pindex(ii + 1) - ip_zero;
          int np_simd = (np_cell / simd_f64::size) * simd_f64::size;

          //
          // vectorized loop
          //
          for (int ip = ip_zero; ip < ip_zero + np_simd; ip += simd_f64::size) {
            // local SIMD register
            simd_f64 xv[3];
            simd_f64 xu[3];
            simd_f64 qs = up[is]->q;

            // load particles to SIMD register
            xv[0] = simd_f64::gather(&up[is]->xv(ip, 0), index);
            xv[1] = simd_f64::gather(&up[is]->xv(ip, 1), index);
            xv[2] = simd_f64::gather(&up[is]->xv(ip, 2), index);
            xu[0] = simd_f64::gather(&up[is]->xu(ip, 0), index);
            xu[1] = simd_f64::gather(&up[is]->xu(ip, 1), index);
            xu[2] = simd_f64::gather(&up[is]->xu(ip, 2), index);

            LoopBodyV.calc_local_current(xv, xu, qs, cur_simd);
          }

          //
          // scalar loop for reminder
          //
          for (int ip = ip_zero + np_simd; ip < ip_zero + np_cell; ip++) {
            float64* xv = &up[is]->xv(ip, 0);
            float64* xu = &up[is]->xu(ip, 0);
            float64  qs = up[is]->q;

            LoopBodyS.calc_local_current(xv, xu, qs, cur);
          }
        }

        // deposit to global array
        LoopBodyS.deposit_global_current(uj, iz, iy, ix, cur);
        LoopBodyV.deposit_global_current(uj, iz, iy, ix, cur_simd);
      }
    }
  }
}

DEFINE_MEMBER(void, deposit_current_unsorted)(const float64 delt)
{
  using namespace exchunk3d_impl;
  const simd_i64 index = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  Current<Order, simd_f64> LoopBodyV(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);
  Current<Order, float64>  LoopBodyS(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);

  // clear charge/current density
  uj.fill(0);

  for (int is = 0; is < Ns; is++) {
    int np_simd = (up[is]->Np / simd_f64::size) * simd_f64::size;

    //
    // vectorized loop
    //
    for (int ip = 0; ip < np_simd; ip += simd_f64::size) {
      // local SIMD register
      simd_f64 xv[3];
      simd_f64 xu[3];

      // load particles to SIMD register
      xv[0] = simd_f64::gather(&up[is]->xv(ip, 0), index);
      xv[1] = simd_f64::gather(&up[is]->xv(ip, 1), index);
      xv[2] = simd_f64::gather(&up[is]->xv(ip, 2), index);
      xu[0] = simd_f64::gather(&up[is]->xu(ip, 0), index);
      xu[1] = simd_f64::gather(&up[is]->xu(ip, 1), index);
      xu[2] = simd_f64::gather(&up[is]->xu(ip, 2), index);

      LoopBodyV.unsorted(uj, xv, xu, up[is]->q);
    }

    //
    // scalar loop for reminder
    //
    for (int ip = np_simd; ip < up[is]->Np; ip++) {
      float64* xv = &up[is]->xv(ip, 0);
      float64* xu = &up[is]->xu(ip, 0);

      LoopBodyS.unsorted(uj, xv, xu, up[is]->q);
    }
  }
}

DEFINE_MEMBER(void, deposit_moment)()
{
  using namespace exchunk3d_impl;
  constexpr int  size     = Order + 1;
  constexpr int  is_odd   = Order % 2 == 0 ? 0 : 1;
  const int      stride_x = 1;
  const int      stride_y = stride_x * (Ubx - Lbx + 2);
  const int      stride_z = stride_y * (Uby - Lby + 2);
  const int      lbx      = Lbx;
  const int      lby      = Lby;
  const int      lbz      = Lbz;
  const int      ubx      = Ubx + is_odd;
  const int      uby      = Uby + is_odd;
  const int      ubz      = Ubz + is_odd;
  const simd_i64 index    = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  Moment<Order, simd_f64> LoopBodyV(delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);
  Moment<Order, float64>  LoopBodyS(delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);

  // clear moment
  um.fill(0);

  for (int iz = lbz, jz = 0; iz <= ubz; iz++, jz++) {
    for (int iy = lby, jy = 0; iy <= uby; iy++, jy++) {
      for (int ix = lbx, jx = 0; ix <= ubx; ix++, jx++) {
        int ii = jz * stride_z + jy * stride_y + jx * stride_x; // 1D grid index

        // process particles in the cell
        for (int is = 0; is < Ns; is++) {
          // local moment
          float64  mom[size][size][size][14]      = {0}; // scalar
          simd_f64 mom_simd[size][size][size][14] = {0}; // SIMD register

          int ip_zero = up[is]->pindex(ii);
          int np_cell = up[is]->pindex(ii + 1) - ip_zero;
          int np_simd = (np_cell / simd_f64::size) * simd_f64::size;

          //
          // vectorized loop
          //
          for (int ip = ip_zero; ip < ip_zero + np_simd; ip += simd_f64::size) {
            // local SIMD register
            simd_f64 xu[6];
            simd_f64 ms = up[is]->m;

            // load particles to SIMD register
            xu[0] = simd_f64::gather(&up[is]->xu(ip, 0), index);
            xu[1] = simd_f64::gather(&up[is]->xu(ip, 1), index);
            xu[2] = simd_f64::gather(&up[is]->xu(ip, 2), index);
            xu[3] = simd_f64::gather(&up[is]->xu(ip, 3), index);
            xu[4] = simd_f64::gather(&up[is]->xu(ip, 4), index);
            xu[5] = simd_f64::gather(&up[is]->xu(ip, 5), index);

            LoopBodyV.calc_local_moment(xu, ms, mom_simd);
          }

          //
          // scalar loop for reminder
          //
          for (int ip = ip_zero + np_simd; ip < ip_zero + np_cell; ip++) {
            float64* xu = &up[is]->xu(ip, 0);
            float64  ms = up[is]->m;

            LoopBodyS.calc_local_moment(xu, ms, mom);
          }

          // deposit to global array
          LoopBodyS.deposit_global_moment(um, iz, iy, ix, is, mom);
          LoopBodyV.deposit_global_moment(um, iz, iy, ix, is, mom_simd);
        }
      }
    }
  }
}

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
