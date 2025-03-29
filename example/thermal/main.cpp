// -*- C++ -*-

#include "nix/random.hpp"
#include "pic_application.hpp"
#include "pic_chunk.hpp"
#include "pic_diag.hpp"

class MainChunk;
class MainApplication;

class MainChunk : public PicChunk
{
public:
  using PicChunk::PicChunk; // inherit constructors

  virtual void setup(json& config) override
  {
    PicChunk::setup(config);

    float64 delt;
    float64 delh;

    Ns   = config["Ns"].get<int>();
    cc   = config["cc"].get<float64>();
    delt = config["delt"].get<float64>();
    delh = config["delh"].get<float64>();

    // set grid size and coordinate
    set_coordinate(delh, delh, delh);

    //
    // initialize field
    //
    {
      float64 Ex = config["Ex"].get<float64>();
      float64 Ey = config["Ey"].get<float64>();
      float64 Ez = config["Ez"].get<float64>();
      float64 Bx = config["Bx"].get<float64>();
      float64 By = config["By"].get<float64>();
      float64 Bz = config["Bz"].get<float64>();

      // memory allocation
      allocate();

      for (int iz = Lbz; iz <= Ubz; iz++) {
        for (int iy = Lby; iy <= Uby; iy++) {
          for (int ix = Lbx; ix <= Ubx; ix++) {
            uf(iz, iy, ix, 0) = Ex;
            uf(iz, iy, ix, 1) = Ey;
            uf(iz, iy, ix, 2) = Ez;
            uf(iz, iy, ix, 3) = Bx;
            uf(iz, iy, ix, 4) = By;
            uf(iz, iy, ix, 5) = Bz;
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

      json particle = config["particle"];

      up.resize(Ns);
      for (int is = 0; is < Ns; is++) {
        int     np = particle[is]["np"].get<int>();
        int     mp = np * dims[0] * dims[1] * dims[2];
        int64   id = mp;
        float64 ro = particle[is]["ro"].get<float64>();
        float64 qm = particle[is]["qm"].get<float64>();
        float64 vt = particle[is]["vt"].get<float64>();

        id *= this->myid;

        up[is]     = std::make_shared<ParticleType>(mp * target, *this);
        up[is]->m  = ro / np;
        up[is]->q  = qm * up[is]->m;
        up[is]->Np = mp;

        mtp.seed(random_seed); // for charge neutrality
        for (int ip = 0; ip < up[is]->Np; ip++) {
          float64* ptcl = &up[is]->xu(ip, 0);
          int64*   id64 = reinterpret_cast<int64*>(ptcl);

          ptcl[0] = uniform(mtp) * xlim[2] + xlim[0];
          ptcl[1] = uniform(mtp) * ylim[2] + ylim[0];
          ptcl[2] = uniform(mtp) * zlim[2] + zlim[0];
          ptcl[3] = normal(mtv) * vt;
          ptcl[4] = normal(mtv) * vt;
          ptcl[5] = normal(mtv) * vt;
          id64[6] = id + ip;
        }
      }

      // initial sort
      this->sort_particle(up);
    }
  }
};

class MainApplication : public PicApplication
{
public:
  using PicApplication::PicApplication; // inherit constructors

  std::unique_ptr<chunk_type> create_chunk(const int dims[], const bool has_dim[], int id) override
  {
    return std::make_unique<MainChunk>(dims, has_dim, id);
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
