// -*- C++ -*-

#include "diagnoser.hpp"
#include "expic3d.hpp"
#include "nix/random.hpp"

constexpr int order = PICNIX_SHAPE_ORDER;

class MainApplication;
using MainDiagnoser = Diagnoser;

class MainChunk : public ExChunk3D<order>
{
public:
  using ExChunk3D<order>::ExChunk3D; // inherit constructors

  float64 get_wall_xmin() const
  {
    return gxlim[0] + Nb * delx;
  }

  float64 get_wall_xmax() const
  {
    return gxlim[1] - Nb * delx;
  }

  float64 initial_flow_profile(const float64 x, const float64 L)
  {
    float64 xmin = get_wall_xmin();
    return -(x - xmin < L ? sin(0.5 * nix::math::pi * (x - xmin) / L) : 1.0);
  }

  virtual void setup(json& config) override
  {
    ExChunk3D<order>::setup(config);

    cc = 1.0;
    Ns = 2;

    int     nppc  = config["nppc"].get<int>();
    float64 delt  = config["delt"].get<float64>();
    float64 delh  = config["delh"].get<float64>();
    float64 wp    = config["wp"].get<float64>();
    float64 mime  = config["mime"].get<float64>();
    float64 mach  = config["mach"].get<float64>();
    float64 theta = config["theta"].get<float64>();
    float64 phi   = config["phi"].get<float64>();
    float64 sigma = config["sigma"].get<float64>();
    float64 betae = config["betae"].get<float64>();
    float64 betai = config["betai"].get<float64>();
    float64 taper = config["taper"].get<float64>();
    float64 me    = 1.0 / (sigma * nppc);
    float64 qe    = -wp * sqrt(sigma) * me;
    float64 mi    = me * mime;
    float64 qi    = -qe;
    float64 b0    = cc * sqrt(sigma) / std::abs(qe / me);
    float64 vae   = cc * sqrt(sigma);
    float64 vai   = cc * sqrt(sigma / mime);
    float64 vte   = vae * sqrt(0.5 * betae);
    float64 vti   = vai * sqrt(0.5 * betai);
    float64 vsh   = vai * mach;
    float64 gam   = sqrt(1 + vsh * vsh / (cc * cc));
    float64 Bx0   = b0 * cos(theta / 180 * nix::math::pi);
    float64 By0   = b0 * sin(theta / 180 * nix::math::pi) * cos(phi / 180 * nix::math::pi);
    float64 Bz0   = b0 * sin(theta / 180 * nix::math::pi) * sin(phi / 180 * nix::math::pi);
    float64 Ex0   = 0;
    float64 Ey0   = +(-vsh) * Bz0 / (gam * cc);
    float64 Ez0   = -(-vsh) * By0 / (gam * cc);

    // set grid size and coordinate
    set_coordinate(delh, delh, delh);

    //
    // initialize field
    //
    {
      float64 x0 = this->get_xmin();

      // memory allocation
      allocate();

      for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
        for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            float64 xi = (ix + 0.5) * delx + x0;
            float64 Ux = vsh * initial_flow_profile(xi, taper);
            float64 gm = sqrt(1 + Ux * Ux / (cc * cc));
            float64 bx = Bx0;
            float64 by = By0;
            float64 bz = Bz0;
            float64 ex = Ex0;
            float64 ey = +Ux * bz / (gm * cc);
            float64 ez = -Ux * by / (gm * cc);

            uf(iz, iy, ix, 0) = ex;
            uf(iz, iy, ix, 1) = ey;
            uf(iz, iy, ix, 2) = ez;
            uf(iz, iy, ix, 3) = bx;
            uf(iz, iy, ix, 4) = by;
            uf(iz, iy, ix, 5) = bz;
          }
        }
      }

      // allocate MPI buffer for field
      this->set_mpi_buffer(mpibufvec[BoundaryEmf], 0, 0, sizeof(float64) * 6);
      this->set_mpi_buffer(mpibufvec[BoundaryCur], 0, 0, sizeof(float64) * 4);
      this->set_mpi_buffer(mpibufvec[BoundaryMom], 0, 0, sizeof(float64) * Ns * 11);

      // setup for Friedman filter
      this->setup_friedman_filter();
    }

    //
    // initialize particles
    //
    {
      int                 random_seed = option["random_seed"].get<int>();
      std::mt19937_64     mtp(random_seed);
      std::mt19937_64     mtv(random_seed);
      nix::rand_uniform   uniform(0.0, 1.0);
      nix::MaxwellJuttner mj_ele(vte * vte, 0.0);
      nix::MaxwellJuttner mj_ion(vti * vti, 0.0);

      {
        int   nz = dims[0] + 2 * Nb;
        int   ny = dims[1] + 2 * Nb;
        int   nx = dims[2] + 2 * Nb;
        int   mp = nppc * dims[0] * dims[1] * dims[2];
        int64 id = static_cast<int64>(mp) * static_cast<int64>(this->myid);

        up.resize(Ns);

        // electron
        up[0]     = std::make_shared<ParticleType>(2 * mp, nz * ny * nx);
        up[0]->m  = me;
        up[0]->q  = qe;
        up[0]->Np = mp;

        // ion
        up[1]     = std::make_shared<ParticleType>(2 * mp, nz * ny * nx);
        up[1]->m  = mi;
        up[1]->q  = qi;
        up[1]->Np = mp;

        for (int ip = 0; ip < mp; ip++) {
          float64 x = uniform(mtp) * xlim[2] + xlim[0];
          float64 y = uniform(mtp) * ylim[2] + ylim[0];
          float64 z = uniform(mtp) * zlim[2] + zlim[0];
          float64 U = vsh * initial_flow_profile(x, taper);

          // electrons
          {
            auto [ux, uy, uz] = mj_ele(mtv);
            ux                = mj_ele.lorentz_boost(mtv, U, ux, uy, uz);

            up[0]->xu(ip, 0) = x;
            up[0]->xu(ip, 1) = y;
            up[0]->xu(ip, 2) = z;
            up[0]->xu(ip, 3) = ux;
            up[0]->xu(ip, 4) = uy;
            up[0]->xu(ip, 5) = uz;
          }

          // ions
          {
            auto [ux, uy, uz] = mj_ion(mtv);
            ux                = mj_ion.lorentz_boost(mtv, U, ux, uy, uz);

            up[1]->xu(ip, 0) = x;
            up[1]->xu(ip, 1) = y;
            up[1]->xu(ip, 2) = z;
            up[1]->xu(ip, 3) = ux;
            up[1]->xu(ip, 4) = uy;
            up[1]->xu(ip, 5) = uz;
          }
        }

        // initial sort
        this->sort_particle(up);

        // allocate MPI buffer for particle
        setup_particle_mpi_buffer(option["mpi_buffer_fraction"].get<float64>());
      }
    }

    // store option for injection
    option["injection"] = {{"step", 0},  {"nppc", nppc}, {"delt", delt}, {"vsh", vsh},
                           {"vte", vte}, {"vti", vti},   {"Ex0", Ex0},   {"Ey0", Ey0},
                           {"Ez0", Ez0}, {"Bx0", Bx0},   {"By0", By0},   {"Bz0", Bz0}};
  }

  void get_diverror(float64& efd, float64& bfd) override
  {
    const float64 rdx = 1 / delx;
    const float64 rdy = 1 / dely;
    const float64 rdz = 1 / delz;

    int lbz = Lbz;
    int ubz = Ubz;
    int lby = Lby;
    int uby = Uby;
    int lbx = Lbx;
    int ubx = Ubx;

    efd = 0;
    bfd = 0;

    if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
      lbx += Nb;
    }

    if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
      ubx -= Nb;
    }

    for (int iz = lbz; iz <= ubz; iz++) {
      for (int iy = lby; iy <= uby; iy++) {
        for (int ix = lbx; ix <= ubx; ix++) {
          // div(E) - rho
          efd += (uf(iz, iy, ix + 1, 0) - uf(iz, iy, ix, 0)) * rdx +
                 (uf(iz, iy + 1, ix, 1) - uf(iz, iy, ix, 1)) * rdy +
                 (uf(iz + 1, iy, ix, 2) - uf(iz, iy, ix, 2)) * rdz - uj(iz, iy, ix, 0);
          // div(B)
          bfd += (uf(iz, iy, ix, 3) - uf(iz, iy, ix - 1, 3)) * rdx +
                 (uf(iz, iy, ix, 4) - uf(iz, iy - 1, ix, 4)) * rdy +
                 (uf(iz, iy, ix, 5) - uf(iz - 1, iy, ix, 5)) * rdz;
        }
      }
    }
  }

  void set_boundary_emf()
  {
    const float64 delxy = delx / dely;
    const float64 delxz = delx / delz;

    //
    // lower boundary in x
    //
    if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
      // transverse E: Ey, Ez
      for (int ix = 0; ix < 2 * Nb - 1; ix++) {
        int ix1 = Lbx - ix + Nb - 1;
        int ix2 = Lbx + ix + Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
            uf(iz, iy, ix1, 1) = -uf(iz, iy, ix2, 1);
            uf(iz, iy, ix1, 2) = -uf(iz, iy, ix2, 2);
          }
        }
      }
      // normal E: Ex
      for (int ix = 0; ix < 2 * Nb; ix++) {
        int ix1 = Lbx - ix + Nb - 1;
        int ix2 = ix1 + 1;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb - 1; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb - 1; iy++) {
            uf(iz, iy, ix1, 0) = -delx * uj(iz, iy, ix1, 0) + uf(iz, iy, ix2, 0) +
                                 delxy * (uf(iz, iy + 1, ix1, 1) - uf(iz, iy, ix1, 1)) +
                                 delxz * (uf(iz + 1, iy, ix1, 2) - uf(iz, iy, ix1, 2));
          }
        }
      }
      // transverse B: By, Bz
      for (int ix = 0; ix < 2 * Nb; ix++) {
        int ix1 = Lbx - ix + Nb - 1;
        int ix2 = Lbx + ix + Nb + 1;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
            uf(iz, iy, ix1, 4) = uf(iz, iy, ix2, 4);
            uf(iz, iy, ix1, 5) = uf(iz, iy, ix2, 5);
          }
        }
      }
      // normal B: Bx
      for (int ix = 0; ix < 2 * Nb; ix++) {
        int ix1 = Lbx - ix + Nb - 1;
        int ix2 = ix1 + 1;
        for (int iz = Lbz - Nb + 1; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb + 1; iy <= Uby + Nb; iy++) {
            uf(iz, iy, ix1, 3) = uf(iz, iy, ix2, 3) +
                                 delxy * (uf(iz, iy, ix2, 4) - uf(iz, iy - 1, ix2, 4)) +
                                 delxz * (uf(iz, iy, ix2, 5) - uf(iz - 1, iy, ix2, 5));
          }
        }
      }
    }

    //
    // upper boundary in x
    //
    if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
      const float64 Ex0 = option["injection"]["Ex0"].get<float64>();
      const float64 Ey0 = option["injection"]["Ey0"].get<float64>();
      const float64 Ez0 = option["injection"]["Ez0"].get<float64>();
      const float64 Bx0 = option["injection"]["Bx0"].get<float64>();
      const float64 By0 = option["injection"]["By0"].get<float64>();
      const float64 Bz0 = option["injection"]["Bz0"].get<float64>();

      // transverse E: Ey, Ez
      for (int ix = 0; ix < 2 * Nb; ix++) {
        int ix1 = Ubx + ix - Nb + 1;
        int ix2 = Ubx - ix - Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
            uf(iz, iy, ix1, 1) = 1 * Ey0 + 0 * uf(iz, iy, ix2, 1);
            uf(iz, iy, ix1, 2) = 1 * Ez0 + 0 * uf(iz, iy, ix2, 2);
          }
        }
      }
      // normal E: Ex
      for (int ix = 0; ix < 2 * Nb - 1; ix++) {
        int ix1 = Ubx + ix - Nb + 2;
        int ix2 = ix1 - 1;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb - 1; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb - 1; iy++) {
            uf(iz, iy, ix1, 0) = +delx * uj(iz, iy, ix2, 0) + uf(iz, iy, ix2, 0) -
                                 delxy * (uf(iz, iy + 1, ix2, 1) - uf(iz, iy, ix2, 1)) -
                                 delxz * (uf(iz + 1, iy, ix2, 2) - uf(iz, iy, ix2, 2));
          }
        }
      }
      // transverse B: By, Bz
      for (int ix = 0; ix < 2 * Nb - 1; ix++) {
        int ix1 = Ubx + ix - Nb + 2;
        int ix2 = Ubx - ix - Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
            uf(iz, iy, ix1, 4) = 1 * By0 + 0 * uf(iz, iy, ix2, 4);
            uf(iz, iy, ix1, 5) = 1 * Bz0 + 0 * uf(iz, iy, ix2, 5);
          }
        }
      }
      // normal B: Bx
      for (int ix = 0; ix < 2 * Nb; ix++) {
        int ix1 = Ubx + ix - Nb + 1;
        int ix2 = ix1 - 1;
        for (int iz = Lbz - Nb + 1; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb + 1; iy <= Uby + Nb; iy++) {
            uf(iz, iy, ix1, 3) = uf(iz, iy, ix2, 3) -
                                 delxy * (uf(iz, iy, ix1, 4) - uf(iz, iy - 1, ix1, 4)) -
                                 delxz * (uf(iz, iy, ix1, 5) - uf(iz - 1, iy, ix1, 5));
          }
        }
      }
    }
  }

  void set_boundary_cur()
  {
    //
    // lower boundary in x
    //
    if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
      for (int ix = 0; ix < Nb; ix++) {
        int ix1 = Lbx - ix + Nb - 1;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
            uj(iz, iy, ix1, 0) = 0;
            uj(iz, iy, ix1, 1) = 0;
            uj(iz, iy, ix1, 2) = 0;
            uj(iz, iy, ix1, 3) = 0;
          }
        }
      }
    }

    //
    // upper boundary in x
    //
    if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
      for (int ix = 0; ix < Nb; ix++) {
        int ix1 = Ubx + ix - Nb + 1;
        int ix2 = ix1 + 1;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
            uj(iz, iy, ix1, 0) = 0;
            uj(iz, iy, ix2, 1) = 0;
            uj(iz, iy, ix1, 2) = 0;
            uj(iz, iy, ix1, 3) = 0;
          }
        }
      }
    }
  }

  void set_boundary_mom()
  {
    //
    // lower boundary in x
    //
    if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
      for (int ix = 0; ix < Nb; ix++) {
        int ix1 = Lbx - ix + Nb - 1;
        int ix2 = Lbx + Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
            for (int ik = 0; ik < 11; ik++) {
              um(iz, iy, ix1, ik) = um(iz, iy, ix2, ik);
            }
          }
        }
      }
    }

    //
    // upper boundary in x
    //
    if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
      for (int ix = 0; ix < Nb; ix++) {
        int ix1 = Ubx + ix - Nb + 1;
        int ix2 = Ubx - Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
            for (int ik = 0; ik < 11; ik++) {
              um(iz, iy, ix1, ik) = um(iz, iy, ix2, ik);
            }
          }
        }
      }
    }
  }

  void set_boundary_field(int mode) override
  {
    switch (mode) {
    case BoundaryEmf: {
      set_boundary_emf();
      break;
    }
    case BoundaryCur: {
      set_boundary_cur();
      break;
    }
    case BoundaryMom: {
      set_boundary_mom();
      break;
    }
    default:
      // ignore
      break;
    }
  }

  void set_boundary_particle(ParticlePtr particle, int Lbp, int Ubp) override
  {
    // NOTE: trick to take care of round-off error
    float64 ylength = gylim[2] - std::numeric_limits<float64>::epsilon();
    float64 zlength = gzlim[2] - std::numeric_limits<float64>::epsilon();

    float64 xmin = gxlim[0];
    float64 xmax = gxlim[1];

    // apply boundary condition
    auto& xu = particle->xu;
    for (int ip = Lbp; ip <= Ubp; ip++) {
      xu(ip, 1) += (xu(ip, 1) < gylim[0]) * ylength - (xu(ip, 1) >= gylim[1]) * ylength;
      xu(ip, 2) += (xu(ip, 2) < gzlim[0]) * zlength - (xu(ip, 2) >= gzlim[1]) * zlength;

      //
      // lower boundary in x
      //
      if (xu(ip, 0) < xmin) {
        xu(ip, 0) = -xu(ip, 0) + 2 * xmin;
        xu(ip, 3) = -xu(ip, 3);
      }

      //
      // upper boundary in x
      //
      if (xu(ip, 0) > xmax) {
        // do nothing remove from the system
      }
    }
  }

  void inject_particle(ParticleVec& particle) override
  {
    //
    // upper boundary in x
    //
    if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
      float64 rc   = 1.0 / cc;
      int     step = option["injection"]["step"].get<int>();
      int     nppc = option["injection"]["nppc"].get<int>();
      float64 delt = option["injection"]["delt"].get<float64>();
      float64 vsh  = option["injection"]["vsh"].get<float64>();
      float64 vte  = option["injection"]["vte"].get<float64>();
      float64 vti  = option["injection"]["vti"].get<float64>();
      int     mp   = nppc * dims[0] * dims[1];
      int     npe  = particle[0]->Np;
      int     npi  = particle[1]->Np;

      int                 random_seed = option["random_seed"].get<int>() + step;
      std::mt19937_64     mtp(random_seed);
      std::mt19937_64     mtv(random_seed);
      nix::rand_uniform   uniform(0.0, 1.0);
      nix::MaxwellJuttner mj_ele(vte * vte, -vsh);
      nix::MaxwellJuttner mj_ion(vti * vti, -vsh);

      for (int ip = 0; ip < mp; ip++) {
        float64 x = uniform(mtp) * delx + xlim[1];
        float64 y = uniform(mtp) * ylim[2] + ylim[0];
        float64 z = uniform(mtp) * zlim[2] + zlim[0];

        // electrons
        {
          auto [ux, uy, uz] = mj_ele(mtv);
          auto dx           = ux * delt / lorentz_factor(ux, uy, uz, rc);

          if (x + dx < xlim[1]) { // inside the domain
            up[0]->xu(npe, 0) = x + dx;
            up[0]->xu(npe, 1) = y;
            up[0]->xu(npe, 2) = z;
            up[0]->xu(npe, 3) = ux;
            up[0]->xu(npe, 4) = uy;
            up[0]->xu(npe, 5) = uz;
            npe++;
          }
        }

        // ions
        {
          auto [ux, uy, uz] = mj_ion(mtv);
          auto dx           = ux * delt / lorentz_factor(ux, uy, uz, rc);

          if (x + dx < xlim[1]) { // inside the domain
            up[1]->xu(npi, 0) = x + dx;
            up[1]->xu(npi, 1) = y;
            up[1]->xu(npi, 2) = z;
            up[1]->xu(npi, 3) = ux;
            up[1]->xu(npi, 4) = uy;
            up[1]->xu(npi, 5) = uz;
            npi++;
          }
        }
      }

      // count injected particles
      this->count_particle(up[0], up[0]->Np, npe - 1, false);
      this->count_particle(up[1], up[1]->Np, npi - 1, false);
      up[0]->Np = npe;
      up[1]->Np = npi;
    }

    option["injection"]["step"] = option["injection"]["step"].get<int>() + 1;
  }
};

class MainApplication : public ExPIC3D<MainChunk, MainDiagnoser>
{
public:
  using ExPIC3D<MainChunk, MainDiagnoser>::ExPIC3D; // inherit constructors

  PtrChunkMap create_chunkmap() override
  {
    auto ptr = ExPIC3D<MainChunk, MainDiagnoser>::create_chunkmap();
    ptr->set_periodicity(1, 1, 0); // set non-periodic in x
    return ptr;
  }

  std::unique_ptr<MainChunk> create_chunk(const int dims[], int id) override
  {
    return std::make_unique<MainChunk>(dims, id);
  }
};

//
// main
//
int main(int argc, char** argv)
{
  MainApplication app(argc, argv);
  return app.main();
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
