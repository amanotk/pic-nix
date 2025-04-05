// -*- C++ -*-

#include "nix/random.hpp"
#include "pic_application.hpp"
#include "pic_chunk.hpp"
#include "pic_diag.hpp"

class MainChunk : public PicChunk
{
public:
  using PicChunk::PicChunk; // inherit constructors

  virtual void setup(json& config) override
  {
    PicChunk::setup(config);

    cc = 1.0;
    Ns = 2;

    float64 delt  = config["delt"].get<float64>();
    float64 delh  = config["delh"].get<float64>();
    float64 lcs   = config["lcs"].get<float64>();
    int     ncs   = config["ncs"].get<float64>();
    int     nbg   = config["nbg"].get<float64>();
    float64 sigma = config["sigma"].get<float64>();
    float64 mime  = config["mime"].get<float64>();
    float64 tite  = config["tite"].get<float64>();
    float64 bz    = config["bz"].get<float64>();
    float64 db    = config["db"].get<float64>();
    float64 b0    = sqrt(sigma);
    float64 qe    = -1.0 / ncs;
    float64 qi    = +1.0 / ncs;
    float64 me    = std::abs(qe);
    float64 mi    = me * mime;
    float64 vdi   = -cc * b0 / (qi * ncs * lcs) / (1 + tite) * tite;
    float64 vde   = +cc * b0 / (qi * ncs * lcs) / (1 + tite) * 1.0;
    float64 vti   = sqrt(0.5 * b0 * b0 / (ncs * mi) / (1 + tite) * tite);
    float64 vte   = sqrt(0.5 * b0 * b0 / (ncs * me) / (1 + tite) * 1.0);

    // set grid size and coordinate
    set_coordinate(delh, delh, delh);

    // center of current sheet
    float64 xcs = 0.5 * (gxlim[0] + gxlim[1]);
    float64 ycs = 0.5 * (gylim[0] + gylim[1]);

    //
    // initialize field
    //
    {
      const int Nb = boundary_margin;

      // memory allocation
      allocate();

      for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
        for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            float64 xi    = xlim[0] + (ix - Lbx + 0.5) * delx - xcs;
            float64 yi    = ylim[0] + (iy - Lby + 0.5) * dely - ycs;
            float64 xm    = xi - delx;
            float64 ym    = yi - dely;
            float64 bx    = tanh(yi / lcs);
            float64 az_00 = 2 * db * lcs * exp(-(xi * xi + yi * yi) / (4 * lcs * lcs));
            float64 az_01 = 2 * db * lcs * exp(-(xi * xi + ym * ym) / (4 * lcs * lcs));
            float64 az_10 = 2 * db * lcs * exp(-(xm * xm + yi * yi) / (4 * lcs * lcs));

            uf(iz, iy, ix, 0) = 0;
            uf(iz, iy, ix, 1) = 0;
            uf(iz, iy, ix, 2) = 0;
            uf(iz, iy, ix, 3) = b0 * (+(az_00 - az_01) / dely + bx);
            uf(iz, iy, ix, 4) = b0 * (-(az_00 - az_10) / delx);
            uf(iz, iy, ix, 5) = b0 * bz;
          }
        }
      }

      // allocate MPI buffer for field
      this->set_mpi_buffer(mpibufvec[BoundaryEmf], 0, 0, sizeof(float64) * 6);
      this->set_mpi_buffer(mpibufvec[BoundaryCur], 0, 0, sizeof(float64) * 4);
      this->set_mpi_buffer(mpibufvec[BoundaryMom], 0, 0, sizeof(float64) * Ns * 14);

      // setup for Friedman filter
      this->init_friedman();
    }

    //
    // initialize particles
    //
    {
      float64           target      = 1 + this->get_buffer_ratio();
      int               random_seed = option["random_seed"].get<int>();
      std::mt19937_64   mtp(random_seed);
      std::mt19937_64   mtv(random_seed);
      nix::rand_uniform uniform(0.0, 1.0);
      nix::rand_normal  normal(0.0, 1.0);

      {
        int     numcell = dims[0] * dims[1] * dims[2];
        float64 ymin    = (ylim[0] - ycs) / lcs;
        float64 ymax    = (ylim[1] - ycs) / lcs;
        float64 rbg     = numcell * nbg;
        float64 rcs     = numcell * ncs * (tanh(ymax) - tanh(ymin)) / (ymax - ymin);
        int     mp      = rbg + rcs;

        up.resize(Ns);

        // electron
        up[0]     = std::make_shared<ParticleType>(mp * target, *this);
        up[0]->m  = me;
        up[0]->q  = qe;
        up[0]->Np = mp;

        // ion
        up[1]     = std::make_shared<ParticleType>(mp * target, *this);
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
    }
  }

  void set_boundary_emf()
  {
    const float64 delyx = dely / delx * has_dim[2];
    const float64 delyz = dely / delz * has_dim[0];
    const int     Nb    = boundary_margin;

    //
    // lower boundary in y
    //
    if (get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
      // transverse E: Ex, Ez
      for (int iy = 0; iy < 2 * Nb; iy++) {
        int iy1 = Lby - iy + Nb - 1;
        int iy2 = Lby + iy + Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uf(iz, iy1, ix, 0) = -uf(iz, iy2, ix, 0);
            uf(iz, iy1, ix, 2) = -uf(iz, iy2, ix, 2);
          }
        }
      }
      // normal E: Ey
      for (int iy = 0; iy < 2 * Nb; iy++) {
        int iy1 = Lby - iy + Nb - 1;
        int iy2 = iy1 + 1;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb - 1; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb - 1; ix++) {
            uf(iz, iy1, ix, 1) = -dely * uj(iz, iy1, ix, 0) + uf(iz, iy2, ix, 1) +
                                 delyx * (uf(iz, iy1, ix + 1, 0) - uf(iz, iy1, ix, 0)) +
                                 delyz * (uf(iz + 1, iy1, ix, 2) - uf(iz, iy1, ix, 2));
          }
        }
      }
      // transverse B: Bx, Bz
      for (int iy = 0; iy < 2 * Nb; iy++) {
        int iy1 = Lby - iy + Nb - 1;
        int iy2 = Lby + iy + Nb + 1;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uf(iz, iy1, ix, 3) = uf(iz, iy2, ix, 3);
            uf(iz, iy1, ix, 5) = uf(iz, iy2, ix, 5);
          }
        }
      }
      // normal B: By
      for (int iy = 0; iy < 2 * Nb; iy++) {
        int iy1 = Lby - iy + Nb - 1;
        int iy2 = iy1 + 1;
        for (int iz = Lbz - Nb + 1; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb + 1; ix <= Ubx + Nb; ix++) {
            uf(iz, iy1, ix, 4) = uf(iz, iy2, ix, 4) +
                                 delyx * (uf(iz, iy2, ix, 3) - uf(iz, iy2, ix - 1, 3)) +
                                 delyz * (uf(iz, iy2, ix, 5) - uf(iz - 1, iy2, ix, 5));
          }
        }
      }
    }

    //
    // upper boundary in y
    //
    if (get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
      // transverse E: Ex, Ez
      for (int iy = 0; iy < 2 * Nb; iy++) {
        int iy1 = Uby + iy - Nb + 1;
        int iy2 = Uby - iy - Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uf(iz, iy1, ix, 0) = -uf(iz, iy2, ix, 0);
            uf(iz, iy1, ix, 2) = -uf(iz, iy2, ix, 2);
          }
        }
      }
      // normal E: Ey
      for (int iy = 0; iy < 2 * Nb - 1; iy++) {
        int iy1 = Uby + iy - Nb + 2;
        int iy2 = iy1 - 1;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb - 1; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb - 1; ix++) {
            uf(iz, iy1, ix, 1) = +dely * uj(iz, iy2, ix, 0) + uf(iz, iy2, ix, 1) -
                                 delyx * (uf(iz, iy2, ix + 1, 0) - uf(iz, iy2, ix, 0)) -
                                 delyz * (uf(iz + 1, iy2, ix, 2) - uf(iz, iy2, ix, 2));
            ;
          }
        }
      }
      // transverse B: Bx, Bz
      for (int iy = 0; iy < 2 * Nb - 1; iy++) {
        int iy1 = Uby + iy - Nb + 2;
        int iy2 = Uby - iy - Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            uf(iz, iy1, ix, 3) = uf(iz, iy2, ix, 3);
            uf(iz, iy1, ix, 5) = uf(iz, iy2, ix, 5);
          }
        }
      }
      // normal B: By
      for (int iy = 0; iy < 2 * Nb; iy++) {
        int iy1 = Uby + iy - Nb + 1;
        int iy2 = iy1 - 1;
        for (int iz = Lbz - Nb + 1; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb + 1; ix <= Ubx + Nb; ix++) {
            uf(iz, iy1, ix, 4) = uf(iz, iy2, ix, 4) -
                                 delyx * (uf(iz, iy1, ix, 3) - uf(iz, iy1, ix - 1, 3)) -
                                 delyz * (uf(iz, iy1, ix, 5) - uf(iz - 1, iy1, ix, 5));
          }
        }
      }
    }
  }

  void set_boundary_mom()
  {
    const int Nb = boundary_margin;

    //
    // lower boundary in y
    //
    if (get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
      for (int iy = 0; iy < Nb; iy++) {
        int iy1 = Lby - iy + Nb - 1;
        int iy2 = Lby + Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            for (int ik = 0; ik < 14; ik++) {
              um(iz, iy1, ix, ik) = um(iz, iy2, ix, ik);
            }
          }
        }
      }
    }

    //
    // upper boundary in y
    //
    if (get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
      for (int iy = 0; iy < Nb; iy++) {
        int iy1 = Uby + iy - Nb + 1;
        int iy2 = Uby - Nb;
        for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
          for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
            for (int ik = 0; ik < 14; ik++) {
              um(iz, iy1, ix, ik) = um(iz, iy2, ix, ik);
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

  void set_boundary_particle(ParticleVec& particle) override
  {
    //
    // lower boundary in x
    //
    if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
      for (int is = 0; is < Ns; is++) {
        auto& xu = particle[is]->xu;
        for (int ip = 0; ip < particle[is]->Np; ip++) {
          if (xu(ip, 1) < gylim[0]) {
            xu(ip, 1) = -xu(ip, 1) + 2 * gylim[0];
            xu(ip, 4) = -xu(ip, 4);
          }
        }
      }
    }

    //
    // upper boundary in x
    //
    if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
      for (int is = 0; is < Ns; is++) {
        auto& xu = particle[is]->xu;
        for (int ip = 0; ip < particle[is]->Np; ip++) {
          if (xu(ip, 1) >= gylim[1]) {
            xu(ip, 1) = -xu(ip, 1) + 2 * gylim[1];
            xu(ip, 4) = -xu(ip, 4);
          }
        }
      }
    }
  }
};

class MainInterface : public PicApplicationInterface
{
public:
  virtual PtrChunk create_chunk(const int dims[], const bool has_dim[], int id) override
  {
    return std::make_unique<MainChunk>(dims, has_dim, id);
  }
};

class MainApplication : public PicApplication
{
public:
  using PicApplication::PicApplication; // inherit constructors

  PtrChunkMap create_chunkmap() override
  {
    auto ptr = PicApplication::create_chunkmap();
    ptr->set_periodicity(1, 0, 1); // set non-periodic in y
    return ptr;
  }

  void initialize_workload() override
  {
    const int numchunk_global = cdims[3];

    int dims[3]       = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};
    int numcell_chunk = dims[0] * dims[1] * dims[2];

    json    parameter = cfgparser->get_parameter();
    json    option    = cfgparser->get_application()["option"];
    int     nbg       = parameter["nbg"].get<float64>();
    int     ncs       = parameter["ncs"].get<float64>();
    float64 lcs       = parameter["lcs"].get<float64>();
    float64 cell_load = option.value("cell_load", 1.0);
    float64 dely      = cfgparser->get_dely();
    float64 ymin      = 0.0;
    float64 ymax      = dely * ndims[1];
    float64 ycs       = 0.5 * (ymin + ymax);
    float64 ylen      = dely * dims[1];

    for (int i = 0; i < numchunk_global; i++) {
      auto [cz, cy, cx] = chunkmap->get_coordinate(i);

      float64 ylmax = (ymin - ycs + ylen * (cy + 1)) / lcs;
      float64 ylmin = (ymin - ycs + ylen * cy) / lcs;
      float64 rbg   = numcell_chunk * nbg;
      float64 rcs   = numcell_chunk * ncs * (tanh(ylmax) - tanh(ylmin)) / (ylmax - ylmin);

      balancer->load(i) = rbg + rcs + cell_load;
    }
  }
};

//
// main
//
int main(int argc, char** argv)
{
  MainApplication app(argc, argv, std::make_shared<MainInterface>());
  return app.main();
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
