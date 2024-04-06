// -*- C++ -*-

//
// Implementation for Scalar Version
//
// This file is to be included from exchunk3d.cpp
//

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Order>                                                                             \
  type ExChunk3D<Order>::name##_impl_scalar

DEFINE_MEMBER(void, push_position)(const float64 delt)
{
  using namespace exchunk3d_impl;

  Position<Order, float64> LoopBody(cc);

  for (int is = 0; is < Ns; is++) {
    for (int ip = 0; ip < up[is]->Np; ip++) {
      float64* xu = &up[is]->xu(ip, 0);
      float64* xv = &up[is]->xv(ip, 0);

      LoopBody(xv, xu, delt);
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

  Velocity<Order, float64, Interpolation> LoopBody(delt, delx, dely, delz, xlim, ylim, zlim, Lbx,
                                                   Lby, Lbz, cc);

  for (int is = 0; is < Ns; is++) {
    float64 qmdt = 0.5 * up[is]->q / up[is]->m * delt;

    for (int ip = 0; ip < up[is]->Np; ip++) {
      LoopBody.unsorted(uf, &up[is]->xu(ip, 0), qmdt);
    }
  }
}

DEFINE_MEMBER(void, deposit_current)(const float64 delt)
{
  using namespace exchunk3d_impl;

  Current<Order, float64> LoopBody(delt, delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);

  // clear charge/current density
  uj.fill(0);

  for (int is = 0; is < Ns; is++) {
    for (int ip = 0; ip < up[is]->Np; ip++) {
      float64* xv = &up[is]->xv(ip, 0);
      float64* xu = &up[is]->xu(ip, 0);

      LoopBody.unsorted(uj, xv, xu, up[is]->q);
    }
  }
}

DEFINE_MEMBER(void, deposit_moment)()
{
  using namespace exchunk3d_impl;

  Moment<Order, float64> LoopBody(delx, dely, delz, xlim, ylim, zlim, Lbx, Lby, Lbz, cc);

  // clear moment
  um.fill(0);

  for (int is = 0; is < Ns; is++) {
    for (int ip = 0; ip < up[is]->Np; ip++) {
      float64* xu = &up[is]->xu(ip, 0);

      LoopBody.unsorted(um, is, xu, up[is]->m);
    }
  }
}

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
