// -*- C++ -*-

#include "diagnoser.hpp"
#include "expic3d.hpp"

#if defined(SHAPE_ORDER) && 1 <= SHAPE_ORDER && SHAPE_ORDER <= 2
constexpr int order = SHAPE_ORDER;
#else
#error "Condition 1 <= SHAPE_ORDER <= 2 must be satisfied"
#endif

class MainApplication;
using MainDiagnoser = Diagnoser;

class MainChunk : public ExChunk3D<order>
{
public:
  using ExChunk3D<order>::ExChunk3D; // inherit constructors

  virtual void setup(json& config) override
  {
    field_load = config.value("field_load", 1.0);

    cc = 1.0;
    Ns = 2;

    float64 delt = config["delt"].get<float64>();
    float64 delh = config["delh"].get<float64>();

    float64 b0   = 1.0;
    float64 lcs  = config["lcs"].get<float64>();
    int     ncs  = config["ncs"].get<float64>();
    int     nbg  = config["nbg"].get<float64>();
    float64 mime = config["mime"].get<float64>();
    float64 tite = config["tite"].get<float64>();
    float64 bz   = config["bz"].get<float64>();
    float64 db   = config["db"].get<float64>();
    float64 qe   = -1.0 / ncs;
    float64 qi   = +1.0 / ncs;
    float64 me   = std::abs(qe);
    float64 mi   = me * mime;
    float64 vdi  = -cc * b0 / (qi * ncs * lcs) / (1 + tite) * tite;
    float64 vde  = +cc * b0 / (qi * ncs * lcs) / (1 + tite) * 1.0;
    float64 vti  = sqrt(0.5 * b0 * b0 / (ncs * mi) / (1 + tite) * tite);
    float64 vte  = sqrt(0.5 * b0 * b0 / (ncs * me) / (1 + tite) * 1.0);

    // set grid size and coordinate
    set_coordinate(delh, delh, delh);

    // current sheet location
    float64 ycs = 0.5 * (gylim[0] + gylim[1]);

    //
    // initialize field
    //
    {

      // memory allocation
      allocate();

      for (int iz = Lbz; iz <= Ubz; iz++) {
        for (int iy = Lby; iy <= Uby; iy++) {
          for (int ix = Lbx; ix <= Ubx; ix++) {
            float64 y         = ylim[0] + (iy - Lby + 0.5) * dely;
            float64 f         = tanh((y - ycs) / lcs);
            uf(iz, iy, ix, 0) = 0;
            uf(iz, iy, ix, 1) = 0;
            uf(iz, iy, ix, 2) = 0;
            uf(iz, iy, ix, 3) = b0 * f;
            uf(iz, iy, ix, 4) = 0;
            uf(iz, iy, ix, 5) = 0;
          }
        }
      }

      // allocate MPI buffer for field
      this->set_mpi_buffer(mpibufvec[BoundaryEmf], 0, 0, sizeof(float64) * 6);
      this->set_mpi_buffer(mpibufvec[BoundaryCur], 0, 0, sizeof(float64) * 4);
      this->set_mpi_buffer(mpibufvec[BoundaryMom], 0, 0, sizeof(float64) * Ns * 11);
    }

    //
    // initialize particles
    //
    {
      // random number generators
      int                                     random_seed = 0;
      std::mt19937                            mtp(0);
      std::mt19937                            mtv(0);
      std::uniform_real_distribution<float64> uniform(0.0, 1.0);
      std::normal_distribution<float64>       normal(0.0, 1.0);

      // random seed
      {
        std::string seed_type = config.value("seed_type", "random"); // random by default

        if (seed_type == "random") {
          random_seed = std::random_device()();
        } else if (seed_type == "chunkid") {
          random_seed = this->myid; // chunk ID
        } else {
          ERROR << tfm::format("Ignoring invalid seed_type: %s", seed_type);
        }

        mtp.seed(random_seed);
        mtv.seed(random_seed);
      }

      {
        int     numcell = dims[0] * dims[1] * dims[2];
        int     nz      = dims[0] + 2 * Nb;
        int     ny      = dims[1] + 2 * Nb;
        int     nx      = dims[2] + 2 * Nb;
        float64 ymin    = (ylim[0] - ycs) / lcs;
        float64 ymax    = (ylim[1] - ycs) / lcs;
        float64 rbg     = numcell * nbg;
        float64 rcs     = numcell * ncs * (tanh(ymax) - tanh(ymin)) / (ymax - ymin);
        int     mp      = rbg + rcs;

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
          float64 z = uniform(mtp) * zlim[2] + zlim[0];

          // electron
          up[0]->xu(ip, 0) = x;
          up[0]->xu(ip, 2) = z;
          up[0]->xu(ip, 3) = normal(mtv) * vte;
          up[0]->xu(ip, 4) = normal(mtv) * vte;

          // ion
          up[1]->xu(ip, 0) = x;
          up[1]->xu(ip, 2) = z;
          up[1]->xu(ip, 3) = normal(mtv) * vti;
          up[1]->xu(ip, 4) = normal(mtv) * vti;

          if (uniform(mtp) < rcs / (rcs + rbg)) {
            //
            // current sheet population
            //
            float64 r = uniform(mtp);
            float64 y = ycs + lcs * atanh(tanh(ymin) + r * (tanh(ymax) - tanh(ymin)));

            // electron
            up[0]->xu(ip, 1) = y;
            up[0]->xu(ip, 5) = normal(mtv) * vte + vde;

            // ion
            up[1]->xu(ip, 1) = y;
            up[1]->xu(ip, 5) = normal(mtv) * vti + vdi;
          } else {
            //
            // background population
            //
            float64 y = uniform(mtp) * ylim[2] + ylim[0];

            // electron
            up[0]->xu(ip, 1) = y;
            up[0]->xu(ip, 5) = normal(mtv) * vte;

            // ion
            up[1]->xu(ip, 1) = y;
            up[1]->xu(ip, 5) = normal(mtv) * vti;
          }
        }
      }

      // initial sort
      this->sort_particle(up);

      // use default MPI buffer allocator for particle
      float64 fraction = config.value("mpi_buffer_fraction", cc * delt / delh);
      setup_particle_mpi_buffer(fraction);
    }
  }

  void set_boundary_emf()
  {
    const float64 delyx = dely / delx;
    const float64 delyz = dely / delz;

    return;
    //
    // lower boundary in y
    //
    if (get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
      // transverse E: Ex, Ez
      for (int iy = 0; iy < Nb; iy++) {
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uf(iz, Lby - 1 - iy, ix, 0) = -uf(iz, Lby + iy, ix, 0);
            uf(iz, Lby - 1 - iy, ix, 2) = -uf(iz, Lby + iy, ix, 2);
          }
        }
      }
      // normal E: Ey
      for (int iy = Lby - 1; iy >= Lby - Nb; iy--) {
        for (int iz = Lbz - 1; iz <= Ubz + 1; iz++) {
          for (int ix = Lbx - 1; ix <= Ubx + 1; ix++) {
            uf(iz, iy, ix, 1) = uf(iz, iy + 1, ix, 1) +
                                delyx * (uf(iz, iy, ix + 1, 0) - uf(iz, iy, ix, 0)) +
                                delyz * (uf(iz + 1, iy, ix, 2) - uf(iz, iy, ix, 2));
          }
        }
      }
      // transverse B: Bx, Bz
      for (int iy = 0; iy < Nb; iy++) {
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uf(iz, Lby - 1 - iy, ix, 3) = +uf(iz, Lby + 1 + iy, ix, 3);
            uf(iz, Lby - 1 - iy, ix, 5) = +uf(iz, Lby + 1 + iy, ix, 5);
          }
        }
      }
      // normal B: By
      for (int iy = Lby; iy >= Lby - Nb + 1; iy--) {
        for (int iz = Lbz - 1; iz <= Ubz + 1; iz++) {
          for (int ix = Lbx - 1; ix <= Ubx + 1; ix++) {
            uf(iz, iy - 1, ix, 4) = +uf(iz, iy, ix, 4) +
                                    delyx * (uf(iz, iy, ix, 3) - uf(iz, iy, ix - 1, 3)) +
                                    delyz * (uf(iz, iy, ix, 5) - uf(iz - 1, iy, ix, 5));
          }
        }
      }
    }

    //
    // upper boundary in y
    //
    if (get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
      // transverse E: Ex, Ez
      for (int iy = 0; iy < Nb; iy++) {
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uf(iz, Uby + 1 + iy, ix, 0) = -uf(iz, Uby - iy, ix, 0);
            uf(iz, Uby + 1 + iy, ix, 2) = -uf(iz, Uby - iy, ix, 2);
          }
        }
      }
      // normal E: Ey
      for (int iy = Uby; iy <= Uby + Nb - 1; iy++) {
        for (int iz = Lbz - 1; iz <= Ubz + 1; iz++) {
          for (int ix = Lbx - 1; ix <= Ubx + 1; ix++) {
            uf(iz, iy + 1, ix, 1) = uf(iz, iy, ix, 1) -
                                    delyx * (uf(iz, iy, ix + 1, 0) - uf(iz, iy, ix, 0)) -
                                    delyz * (uf(iz + 1, iy, ix, 2) - uf(iz, iy, ix, 2));
          }
        }
      }
      // transverse B: Bx, Bz
      for (int iy = 1; iy < Nb; iy++) {
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uf(iz, Uby + 1 + iy, ix, 3) = +uf(iz, Uby + 1 - iy, ix, 3);
            uf(iz, Uby + 1 + iy, ix, 5) = +uf(iz, Uby + 1 - iy, ix, 5);
          }
        }
      }
      // normal B: By
      for (int iy = Uby + 1; iy <= Uby + Nb; iy++) {
        for (int iz = Lbz - 1; iz <= Ubz + 1; iz++) {
          for (int ix = Lbx - 1; ix <= Ubx + 1; ix++) {
            uf(iz, iy, ix, 4) = +uf(iz, iy - 1, ix, 4) -
                                delyx * (uf(iz, iy, ix, 3) - uf(iz, iy, ix - 1, 3)) -
                                delyz * (uf(iz, iy, ix, 5) - uf(iz - 1, iy, ix, 5));
          }
        }
      }
    }
  }

  void set_boundary_cur()
  {
    return;
    //
    // lower boundary in y
    //
    if (get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
      // add boundary contribution
      for (int iy = 0; iy < Nb; iy++) {
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uj(iz, Lby + iy, ix, 0) += uj(iz, Lby - 1 - iy, ix, 0);
            uj(iz, Lby + iy, ix, 1) -= uj(iz, Lby - 1 - iy, ix, 1);
            uj(iz, Lby + iy, ix, 2) -= uj(iz, Lby - 1 - iy, ix, 2);
            uj(iz, Lby + iy, ix, 3) -= uj(iz, Lby - 1 - iy, ix, 3);
          }
        }
      }
      // set boundary condition
      for (int iy = 0; iy < Nb; iy++) {
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uj(iz, Lby - 1 - iy, ix, 0) = 0;
            uj(iz, Lby - 1 - iy, ix, 1) = 0;
            uj(iz, Lby - 1 - iy, ix, 2) = 0;
            uj(iz, Lby - 1 - iy, ix, 3) = 0;
          }
        }
      }
    }

    //
    // upper boundary in y
    //
    if (get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
      // add boundary contribution
      for (int iy = 0; iy < Nb; iy++) {
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uj(iz, Uby - iy, ix, 0) += uj(iz, Uby + 1 + iy, ix, 0);
            uj(iz, Uby - iy, ix, 1) -= uj(iz, Uby + 1 + iy, ix, 1);
            uj(iz, Uby - iy, ix, 2) -= uj(iz, Uby + 1 + iy, ix, 2);
            uj(iz, Uby - iy, ix, 3) -= uj(iz, Uby + 1 + iy, ix, 3);
          }
        }
      }
      // set boundary condition
      for (int iy = 0; iy < Nb; iy++) {
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uj(iz, Uby + 1 + iy, ix, 0) = 0;
            uj(iz, Uby + 1 + iy, ix, 1) = 0;
            uj(iz, Uby + 1 + iy, ix, 2) = 0;
            uj(iz, Uby + 1 + iy, ix, 3) = 0;
          }
        }
      }
    }
  }

  void set_boundary_mom()
  {
    return;
    //
    // lower boundary in y
    //
    if (get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
      // add boundary contribution
      for (int iy = 0; iy < Nb; iy++) {
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            for (int ik = 0; ik < 11; ik++) {
              um(iz, Lby + iy, ix, ik) += um(iz, Lby - 1 - iy, ix, ik);
            }
          }
        }
      }
    }

    //
    // upper boundary in y
    //
    if (get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
      // add boundary contribution
      for (int iy = 0; iy < Nb; iy++) {
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            for (int ik = 0; ik < 11; ik++) {
              um(iz, Uby - iy, ix, ik) += um(iz, Uby + 1 + iy, ix, ik);
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
    float64 xlength = gxlim[2] - std::numeric_limits<float64>::epsilon();
    float64 ylength = gylim[2] - std::numeric_limits<float64>::epsilon();
    float64 zlength = gzlim[2] - std::numeric_limits<float64>::epsilon();

    // apply boundary condition
    auto& xu = particle->xu;
    for (int ip = Lbp; ip <= Ubp; ip++) {
      xu(ip, 0) += (xu(ip, 0) < gxlim[0]) * xlength - (xu(ip, 0) >= gxlim[1]) * xlength;
      xu(ip, 2) += (xu(ip, 2) < gzlim[0]) * zlength - (xu(ip, 2) >= gzlim[1]) * zlength;

      //
      // lower boundary in y
      //
      if (xu(ip, 1) < gylim[0]) {
        xu(ip, 1) = 2 * gylim[0] - xu(ip, 1);
        xu(ip, 3) = -xu(ip, 3);
        xu(ip, 4) = -xu(ip, 4);
        xu(ip, 5) = -xu(ip, 5);
      }

      //
      // upper boundary in y
      //
      if (xu(ip, 1) > gylim[1]) {
        xu(ip, 1) = 2 * gylim[1] - xu(ip, 1);
        xu(ip, 3) = -xu(ip, 3);
        xu(ip, 4) = -xu(ip, 4);
        xu(ip, 5) = -xu(ip, 5);
      }
    }
  }

  void inject_particle(ParticleVec& particle) override
  {
    // no injection is necessary
  }
};

class MainApplication : public ExPIC3D<MainChunk, MainDiagnoser>
{
public:
  using ExPIC3D<MainChunk, MainDiagnoser>::ExPIC3D; // inherit constructors

  PtrChunkMap create_chunkmap() override
  {
    auto ptr = ExPIC3D<MainChunk, MainDiagnoser>::create_chunkmap();
    ptr->set_periodicity(1, 0, 1); // set non-periodic in y
    return ptr;
  }

  std::unique_ptr<MainChunk> create_chunk(const int dims[], int id) override
  {
    return std::make_unique<MainChunk>(dims, id);
  }

  void initialize_workload() override
  {
    const int numchunk_global = cdims[3];

    int dims[3]       = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};
    int numcell_chunk = dims[0] * dims[1] * dims[2];

    json    parameter  = cfgparser->get_parameter();
    int     nbg        = parameter["nbg"].get<float64>();
    int     ncs        = parameter["ncs"].get<float64>();
    float64 lcs        = parameter["lcs"].get<float64>();
    float64 field_load = parameter["field_load"].get<float64>();
    float64 ycs        = 0.5 * (ylim[0] + ylim[1]);
    float64 ylen       = dely * dims[1];

    for (int i = 0; i < numchunk_global; i++) {
      int cx, cy, cz;
      chunkmap->get_coordinate(i, cz, cy, cx);

      float64 ymax      = (ylim[0] - ycs + ylen * (cy + 1)) / lcs;
      float64 ymin      = (ylim[0] - ycs + ylen * cy) / lcs;
      float64 rbg       = numcell_chunk * nbg;
      float64 rcs       = numcell_chunk * ncs * (tanh(ymax) - tanh(ymin)) / (ymax - ymin);
      balancer->load(i) = rbg + rcs + field_load;
    }
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
