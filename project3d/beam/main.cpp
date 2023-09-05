// -*- C++ -*-

#include "diagnoser.hpp"
#include "expic3d.hpp"

constexpr int order = 1;

class MainChunk;
class MainApplication;
using MainDiagnoser = Diagnoser;

class MainChunk : public ExChunk3D<order>
{
public:
  using ExChunk3D<order>::ExChunk3D; // inherit constructors

  virtual void setup(json& config) override
  {
    float64 delt;
    float64 delh;

    field_load = config.value("field_load", 1.0);

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
      float64 B0 = sqrt(Bx * Bx + By * By + Bz * Bz);

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
          tfm::format(std::cerr, "Error: invalid seed_type\n");
        }
      }

      json particle = config["particle"];

      mtv.seed(random_seed);
      up.resize(Ns);
      for (int is = 0; is < Ns; is++) {
        int     nz = dims[0] + 2 * Nb;
        int     ny = dims[1] + 2 * Nb;
        int     nx = dims[2] + 2 * Nb;
        int     np = particle[is]["np"].get<int>();
        int     mp = np * dims[0] * dims[1] * dims[2];
        int64   id = mp;
        float64 ro = particle[is]["ro"].get<float64>();
        float64 qm = particle[is]["qm"].get<float64>();
        float64 vt = particle[is]["vt"].get<float64>();
        float64 vx = particle[is]["vx"].get<float64>();
        float64 vy = particle[is]["vy"].get<float64>();
        float64 vz = particle[is]["vz"].get<float64>();

        id *= this->myid;

        up[is]     = std::make_shared<ParticleType>(2 * mp, nz * ny * nx);
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
          ptcl[3] = normal(mtv) * vt + vx;
          ptcl[4] = normal(mtv) * vt + vy;
          ptcl[5] = normal(mtv) * vt + vz;
          id64[6] = id + ip;
        }
      }

      // initial sort
      this->sort_particle(up);

      // use default MPI buffer allocator for particle
      float64 fraction = config.value("mpi_buffer_fraction", cc * delt / delh);
      setup_particle_mpi_buffer(fraction);
    }
  }
};

class MainApplication : public ExPIC3D<order, MainDiagnoser>
{
public:
  using ExPIC3D<order, MainDiagnoser>::ExPIC3D; // inherit constructors

  std::unique_ptr<ExChunk3D<order>> create_chunk(const int dims[], int id) override
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
