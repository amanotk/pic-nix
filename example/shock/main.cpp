// -*- C++ -*-

#include "expic3d.hpp"
#include "nix/random.hpp"

constexpr int order = PICNIX_SHAPE_ORDER;

class MainApplication;

class MainChunk : public ExChunk3D<order>
{
private:
  using MJ = nix::MaxwellJuttner;

  std::mt19937_64   mt;
  nix::rand_uniform uniform;

public:
  using ExChunk3D<order>::ExChunk3D; // inherit constructors

  void reset_random_number()
  {
    option["random_seed"] = option["random_seed"].get<int>() + 1;
    mt.seed(option["random_seed"].get<int>());
    uniform.reset();
  }

  template <typename Velocity>
  auto generate_injection_particle(Velocity& velocity, float64 delt)
  {
    const float64 rc = 1.0 / cc;

    float64 xx = 0;
    float64 dx = 0;
    float64 ux = 0;
    float64 uy = 0;
    float64 uz = 0;

    std::tie(ux, uy, uz) = velocity(mt);
    xx                   = xlim[1] + cc * delt * uniform(mt);
    dx                   = ux * delt / lorentz_factor(ux, uy, uz, rc);

    while (xx + dx >= xlim[1]) {
      std::tie(ux, uy, uz) = velocity(mt);
      xx                   = xlim[1] + cc * delt * uniform(mt);
      dx                   = ux * delt / lorentz_factor(ux, uy, uz, rc);
    }

    return std::make_tuple(xx + dx, ux, uy, uz);
  }

  float64 initial_flow_profile(const float64 x, const float64 L)
  {
    float64 xmin = gxlim[0] + Nb * delx;
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
    float64 u0    = config["u0"].get<float64>();
    float64 theta = config["theta"].get<float64>();
    float64 phi   = config["phi"].get<float64>();
    float64 sigma = config["sigma"].get<float64>();
    float64 betae = config["betae"].get<float64>();
    float64 betai = config["betai"].get<float64>();
    float64 taper = config["taper"].get<float64>();
    float64 gamma = sqrt(1.0 + (u0 * u0) / (cc * cc));
    float64 me    = 1.0 / nppc;
    float64 qe    = -wp / nppc * sqrt(gamma);
    float64 mi    = me * mime;
    float64 qi    = -qe;
    float64 b0    = cc * sqrt(sigma) / std::abs(qe / me);
    float64 vae   = cc * sqrt(sigma);
    float64 vai   = cc * sqrt(sigma / mime);
    float64 vte   = vae * sqrt(0.5 * betae);
    float64 vti   = vai * sqrt(0.5 * betai);
    float64 Bx0   = b0 * cos(theta / 180 * nix::math::pi);
    float64 By0   = b0 * sin(theta / 180 * nix::math::pi) * cos(phi / 180 * nix::math::pi);
    float64 Bz0   = b0 * sin(theta / 180 * nix::math::pi) * sin(phi / 180 * nix::math::pi);
    float64 Ex0   = 0;
    float64 Ey0   = +(-u0) * Bz0 / (gamma * cc);
    float64 Ez0   = -(-u0) * By0 / (gamma * cc);

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
            float64 Ux = u0 * initial_flow_profile(xi, taper);
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
      float64 target = 1 + this->get_buffer_ratio();

      // setup random number generator
      mt.seed(option["random_seed"].get<int>());

      // Maxwell-Juttner distribution
      std::vector<MJ> mj = {MJ(vte * vte, 0.0), MJ(vti * vti, 0.0)};

      {
        int   nz = dims[0] + 2 * Nb;
        int   ny = dims[1] + 2 * Nb;
        int   nx = dims[2] + 2 * Nb;
        int   mp = nppc * dims[0] * dims[1] * dims[2];
        int64 id = static_cast<int64>(mp) * static_cast<int64>(this->myid);

        up.resize(Ns);

        // electron
        up[0]     = std::make_shared<ParticleType>(mp * target, nz * ny * nx);
        up[0]->m  = me;
        up[0]->q  = qe;
        up[0]->Np = mp;

        // ion
        up[1]     = std::make_shared<ParticleType>(mp * target, nz * ny * nx);
        up[1]->m  = mi;
        up[1]->q  = qi;
        up[1]->Np = mp;

        int ip = 0;
        for (int iz = Lbz; iz <= Ubz; iz++) {
          for (int iy = Lby; iy <= Uby; iy++) {
            for (int ix = Lbx; ix <= Ubx; ix++) {
              for (int jp = 0; jp < nppc; ip++, jp++) {
                float64 x = xlim[0] + (ix - Lbx + uniform(mt)) * delx;
                float64 y = ylim[0] + (iy - Lby + uniform(mt)) * dely;
                float64 z = zlim[0] + (iz - Lbz + uniform(mt)) * delz;
                float64 U = u0 * initial_flow_profile(x, taper);

                for (int is = 0; is < Ns; is++) {
                  mj[is].set_drift(U);
                  auto [ux, uy, uz] = mj[is](mt);

                  up[is]->xu(ip, 0) = x;
                  up[is]->xu(ip, 1) = y;
                  up[is]->xu(ip, 2) = z;
                  up[is]->xu(ip, 3) = ux;
                  up[is]->xu(ip, 4) = uy;
                  up[is]->xu(ip, 5) = uz;
                }
              }
            }
          }
        }

        // initial sort
        this->sort_particle(up);
      }
    }

    // store option for boundary
    option["boundary"] = {{"nppc", nppc},
                          {"delt", delt},
                          {"u0", u0},
                          {"vte", vte},
                          {"vti", vti},
                          {"Ex0", Ex0},
                          {"Ey0", Ey0},
                          {"Ez0", Ez0},
                          {"Bx0", Bx0},
                          {"By0", By0},
                          {"Bz0", Bz0},
                          {"influx", std::array<int, 2>{0, 0}},
                          {"efflux", std::array<int, 2>{0, 0}}};
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
      for (int ix = 0; ix < 2 * Nb; ix++) {
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
      const float64 Ex0 = option["boundary"]["Ex0"].get<float64>();
      const float64 Ey0 = option["boundary"]["Ey0"].get<float64>();
      const float64 Ez0 = option["boundary"]["Ez0"].get<float64>();
      const float64 Bx0 = option["boundary"]["Bx0"].get<float64>();
      const float64 By0 = option["boundary"]["By0"].get<float64>();
      const float64 Bz0 = option["boundary"]["Bz0"].get<float64>();

      // transverse E: Ey, Ez
      for (int ix = 0; ix < 2 * Nb; ix++) {
        int ix1 = Ubx + ix - Nb + 1;
        int ix2 = Ubx - ix - Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
            uf(iz, iy, ix1, 1) = Ey0;
            uf(iz, iy, ix1, 2) = Ez0;
          }
        }
      }
      // normal E: Ex
      for (int ix = 0; ix < 2 * Nb - 1; ix++) {
        int ix1 = Ubx + ix - Nb + 2;
        int ix2 = ix1 - 1;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb - 1; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb - 1; iy++) {
            uf(iz, iy, ix1, 0) = Ex0;
          }
        }
      }
      // transverse B: By, Bz
      for (int ix = 0; ix < 2 * Nb - 1; ix++) {
        int ix1 = Ubx + ix - Nb + 2;
        int ix2 = Ubx - ix - Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
            uf(iz, iy, ix1, 4) = By0;
            uf(iz, iy, ix1, 5) = Bz0;
          }
        }
      }
      // normal B: Bx
      for (int ix = 0; ix < 2 * Nb; ix++) {
        int ix1 = Ubx + ix - Nb + 1;
        int ix2 = ix1 - 1;
        for (int iz = Lbz - Nb + 1; iz <= Ubz + Nb; iz++) {
          for (int iy = Lby - Nb + 1; iy <= Uby + Nb; iy++) {
            uf(iz, iy, ix1, 3) = Bx0;
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
    case BoundaryMom: {
      set_boundary_mom();
      break;
    }
    default:
      // ignore
      break;
    }
  }

  void set_boundary_particle(ParticlePtr particle, int Lbp, int Ubp, int species) override
  {
    //
    // lower boundary in x
    //
    if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
      auto& xu = particle->xu;
      for (int ip = Lbp; ip <= Ubp; ip++) {
        if (xu(ip, 0) < gxlim[0]) {
          xu(ip, 0) = -xu(ip, 0) + 2 * gxlim[0];
          xu(ip, 3) = -xu(ip, 3);
          xu(ip, 4) = -xu(ip, 4);
          xu(ip, 5) = -xu(ip, 5);
        }
      }
    }

    //
    // upper boundary in x
    //
    if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
      float64 delt = option["boundary"]["delt"].get<float64>();
      float64 u0   = option["boundary"]["u0"].get<float64>();
      float64 vte  = option["boundary"]["vte"].get<float64>();
      float64 vti  = option["boundary"]["vti"].get<float64>();

      // reset random number generator
      reset_random_number();

      int             efflux = 0;
      std::vector<MJ> mj     = {MJ(vte * vte, -u0), MJ(vti * vti, -u0)};

      // apply boundary condition
      auto& xu = particle->xu;
      for (int ip = Lbp; ip <= Ubp; ip++) {
        if (xu(ip, 0) >= gxlim[1]) {
          auto [x, ux, uy, uz] = generate_injection_particle(mj[species], delt);

          xu(ip, 0) = x;
          xu(ip, 3) = ux;
          xu(ip, 4) = uy;
          xu(ip, 5) = uz;

          efflux += 1;
        }
      }

      option["boundary"]["efflux"][species] = efflux;
    }
  }

  void inject_particle(ParticleVec& particle) override
  {
    //
    // upper boundary in x
    //
    if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
      const float64 target = 1 + this->get_buffer_ratio();

      float64 rc   = 1.0 / cc;
      int     nppc = option["boundary"]["nppc"].get<int>();
      float64 delt = option["boundary"]["delt"].get<float64>();
      float64 u0   = option["boundary"]["u0"].get<float64>();
      float64 vte  = option["boundary"]["vte"].get<float64>();
      float64 vti  = option["boundary"]["vti"].get<float64>();
      float64 flux = nppc * std::abs(u0) / sqrt(1 + u0 * u0 * rc * rc) * delt / delx;

      // number of injection particles
      size_t              nz           = dims[0] + 2 * Nb;
      size_t              ny           = dims[1] + 2 * Nb;
      int                 np_inj_total = 0;
      xt::xtensor<int, 2> np_inj({nz, ny});

      // reset random number generator
      reset_random_number();

      nix::rand_poisson poisson(flux);
      std::vector<MJ>   mj = {MJ(vte * vte, -u0), MJ(vti * vti, -u0)};

      // calculate number of injection particles and reallocate buffer if necessary
      np_inj.fill(0);

      for (int iz = Lbz; iz <= Ubz; iz++) {
        for (int iy = Lby; iy <= Uby; iy++) {
          np_inj(iz, iy) = poisson(mt);
          np_inj_total += np_inj(iz, iy);
        }
      }

      for (int is = 0; is < Ns; is++) {
        int np_next = particle[is]->Np + np_inj_total;
        if (np_next > particle[is]->Np_total) {
          particle[is]->resize(target * np_next);
        }
      }

      // inject particles
      for (int iz = Lbz; iz <= Ubz; iz++) {
        for (int iy = Lby; iy <= Uby; iy++) {
          for (int is = 0; is < Ns; is++) {
            int np_prev = particle[is]->Np;
            int np_next = particle[is]->Np + np_inj(iz, iy);
            for (int ip = np_prev; ip < np_next; ip++) {
              auto [x, ux, uy, uz] = generate_injection_particle(mj[is], delt);
              float64 y            = ylim[0] + (iy - Lby + uniform(mt)) * dely;
              float64 z            = zlim[0] + (iz - Lbz + uniform(mt)) * delz;

              particle[is]->xu(ip, 0) = x;
              particle[is]->xu(ip, 1) = y;
              particle[is]->xu(ip, 2) = z;
              particle[is]->xu(ip, 3) = ux;
              particle[is]->xu(ip, 4) = uy;
              particle[is]->xu(ip, 5) = uz;
            }

            this->count_particle(particle[is], np_prev, np_next - 1, false);
            particle[is]->Np = np_next;
          }
        }
      }

      for (int is = 0; is < Ns; is++) {
        option["boundary"]["influx"][is] = np_inj_total;
      }
    }
  }
};

class MainApplication : public ExPIC3D<MainChunk>
{
public:
  using ExPIC3D<MainChunk>::ExPIC3D; // inherit constructors

  PtrChunkMap create_chunkmap() override
  {
    auto ptr = ExPIC3D<MainChunk>::create_chunkmap();
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
