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
  using simd::simd_f64;
  using simd::simd_i64;
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

      LoopBodyV(xu, xv, delt);

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

      LoopBodyS(xu, xv, delt);
    }

    // count
    this->count_particle(up[is], 0, up[is]->Np - 1, true);
  }
}

DEFINE_MEMBER(void, push_velocity)(const float64 delt)
{
  using namespace exchunk3d_impl;
  using simd::simd_f64;
  using simd::simd_i64;
  constexpr int  size     = Order + 2;
  constexpr int  is_odd   = Order % 2 == 0 ? 0 : 1;
  const int      stride_x = 1;
  const int      stride_y = stride_x * (dims[2] + 1);
  const int      stride_z = stride_y * (dims[1] + 1);
  const int      lbx      = Lbx;
  const int      lby      = Lby;
  const int      lbz      = Lbz;
  const int      ubx      = Ubx + is_odd;
  const int      uby      = Uby + is_odd;
  const int      ubz      = Ubz + is_odd;
  const simd_i64 index    = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  Velocity<Order, simd_f64> LoopBodyV(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);
  Velocity<Order, float64>  LoopBodyS(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);

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

            LoopBodyV.sorted_mc(uf, iz, iy, ix, xu, simd_f64(qmdt));

            // store particles to memory
            xu[3].scatter(&up[is]->xu(ip, 3), index);
            xu[4].scatter(&up[is]->xu(ip, 4), index);
            xu[5].scatter(&up[is]->xu(ip, 5), index);
          }

          //
          // scalar loop for reminder
          //
          for (int ip = ip_zero + np_simd; ip < ip_zero + np_cell; ip++) {
            LoopBodyS.sorted_mc(uf, iz, iy, ix, &up[is]->xu(ip, 0), qmdt);
          }
        }
      }
    }
  }
}

DEFINE_MEMBER(void, push_velocity_unsorted)(const float64 delt)
{
  using namespace exchunk3d_impl;
  using simd::simd_f64;
  using simd::simd_i64;
  constexpr int  size   = Order + 2;
  constexpr int  is_odd = Order % 2 == 0 ? 0 : 1;
  const simd_i64 index  = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  Velocity<Order, simd_f64> LoopBodyV(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);
  Velocity<Order, float64>  LoopBodyS(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);

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

      LoopBodyV.unsorted_mc(uf, xu, simd_f64(qmdt));

      // store particles to memory
      xu[3].scatter(&up[is]->xu(ip, 3), index);
      xu[4].scatter(&up[is]->xu(ip, 4), index);
      xu[5].scatter(&up[is]->xu(ip, 5), index);
    }

    //
    // scalar loop for reminder
    //
    for (int ip = np_simd; ip < up[is]->Np; ip++) {
      LoopBodyS.unsorted_mc(uf, &up[is]->xu(ip, 0), qmdt);
    }
  }
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  using namespace exchunk3d_impl;
  using simd::simd_f64;
  using simd::simd_i64;
  constexpr int  size     = Order + 3;
  constexpr int  is_odd   = Order % 2 == 0 ? 0 : 1;
  const int      stride_x = 1;
  const int      stride_y = stride_x * (dims[2] + 1);
  const int      stride_z = stride_y * (dims[1] + 1);
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

            // load particles to SIMD register
            xv[0] = simd_f64::gather(&up[is]->xv(ip, 0), index);
            xv[1] = simd_f64::gather(&up[is]->xv(ip, 1), index);
            xv[2] = simd_f64::gather(&up[is]->xv(ip, 2), index);
            xu[0] = simd_f64::gather(&up[is]->xu(ip, 0), index);
            xu[1] = simd_f64::gather(&up[is]->xu(ip, 1), index);
            xu[2] = simd_f64::gather(&up[is]->xu(ip, 2), index);

            LoopBodyV.sorted(uj, iz, iy, ix, xv, xu, up[is]->q);
          }

          //
          // scalar loop for reminder
          //
          for (int ip = ip_zero + np_simd; ip < ip_zero + np_cell; ip++) {
            float64* xv = &up[is]->xv(ip, 0);
            float64* xu = &up[is]->xu(ip, 0);

            LoopBodyS.sorted(uj, iz, iy, ix, xv, xu, up[is]->q);
          }
        }
      }
    }
  }
}

DEFINE_MEMBER(void, deposit_current_unsorted)(const float64 delt)
{
  using namespace exchunk3d_impl;
  using simd::simd_f64;
  using simd::simd_i64;
  constexpr int  size   = Order + 3;
  constexpr int  is_odd = Order % 2 == 0 ? 0 : 1;
  const simd_i64 index  = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

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
  constexpr int size   = Order + 1;
  constexpr int is_odd = Order % 2 == 0 ? 0 : 1;

  const float64 rc    = 1 / cc;
  const float64 rdx   = 1 / delx;
  const float64 rdy   = 1 / dely;
  const float64 rdz   = 1 / delz;
  const float64 dx2   = 0.5 * delx;
  const float64 dy2   = 0.5 * dely;
  const float64 dz2   = 0.5 * delz;
  const float64 xmin  = xlim[0] + dx2 * is_odd;
  const float64 ymin  = ylim[0] + dy2 * is_odd;
  const float64 zmin  = zlim[0] + dz2 * is_odd;
  const float64 xgrid = xlim[0] + dx2;
  const float64 ygrid = ylim[0] + dy2;
  const float64 zgrid = zlim[0] + dz2;

  // clear moment
  um.fill(0);

  for (int is = 0; is < Ns; is++) {
    auto    ps = up[is];
    float64 ms = ps->m;

    // loop over particle
    auto& xu = ps->xu;
    for (int ip = 0; ip < ps->Np; ip++) {
      float64 wx[size]                  = {0};
      float64 wy[size]                  = {0};
      float64 wz[size]                  = {0};
      float64 mom[size][size][size][11] = {0};

      // grid indices
      int ix0 = digitize(xu(ip, 0), xmin, rdx);
      int iy0 = digitize(xu(ip, 1), ymin, rdy);
      int iz0 = digitize(xu(ip, 2), zmin, rdz);

      // weights
      shape<Order>(xu(ip, 0), xgrid + ix0 * delx, rdx, wx);
      shape<Order>(xu(ip, 1), ygrid + iy0 * dely, rdy, wy);
      shape<Order>(xu(ip, 2), zgrid + iz0 * delz, rdz, wz);

      // deposit to local array (this step is not necessary for scalar version)
      for (int jz = 0; jz < size; jz++) {
        for (int jy = 0; jy < size; jy++) {
          for (int jx = 0; jx < size; jx++) {
            float64 ww = ms * wz[jz] * wy[jy] * wx[jx];
            float64 gm = lorentz_factor(xu(ip, 3), xu(ip, 4), xu(ip, 5), rc);

            //
            //  0: mass density
            //  1: tx-component T^{0,1} (x-component of momentum density)
            //  2: ty-component T^{0,2} (y-component of momentum density)
            //  3: tz-component T^{0,3} (z-component of momentum density)
            //  4: tt-component T^{0,0} (energy density / c)
            //  5: xx-component T^{1,1}
            //  6: yy-component T^{2,2}
            //  7: zz-component T^{3,3}
            //  8: xy-component T^{1,2}
            //  9: xz-component T^{1,3}
            // 10: yz-component T^{2,3}
            //
            mom[jz][jy][jx][0]  = ww;
            mom[jz][jy][jx][1]  = ww * xu(ip, 3);
            mom[jz][jy][jx][2]  = ww * xu(ip, 4);
            mom[jz][jy][jx][3]  = ww * xu(ip, 5);
            mom[jz][jy][jx][4]  = ww * gm * cc;
            mom[jz][jy][jx][5]  = ww * xu(ip, 3) * xu(ip, 3) / gm;
            mom[jz][jy][jx][6]  = ww * xu(ip, 4) * xu(ip, 4) / gm;
            mom[jz][jy][jx][7]  = ww * xu(ip, 5) * xu(ip, 5) / gm;
            mom[jz][jy][jx][8]  = ww * xu(ip, 3) * xu(ip, 4) / gm;
            mom[jz][jy][jx][9]  = ww * xu(ip, 3) * xu(ip, 5) / gm;
            mom[jz][jy][jx][10] = ww * xu(ip, 4) * xu(ip, 5) / gm;
          }
        }
      }

      // deposit to global array
      ix0 += Lbx - (Order / 2);
      iy0 += Lby - (Order / 2);
      iz0 += Lbz - (Order / 2);
      for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
        for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
          for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
            for (int k = 0; k < 11; k++) {
              um(iz, iy, ix, is, k) += mom[jz][jy][jx][k];
            }
          }
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
