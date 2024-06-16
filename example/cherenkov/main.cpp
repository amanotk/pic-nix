// -*- C++ -*-

#include "diagnoser.hpp"
#include "expic3d.hpp"
#include "nix/random.hpp"

constexpr int order = PICNIX_SHAPE_ORDER;

class MainChunk;
class MainApplication;
using MainDiagnoser = Diagnoser;

class MainChunk : public ExChunk3D<order>
{
public:
  using ExChunk3D<order>::ExChunk3D; // inherit constructors

  virtual void setup(json& config) override
  {
    ExChunk3D<order>::setup(config);

    // check validity of assumptions
    {
      constexpr int Ns_mustbe = 2;

      Ns = config["Ns"].get<int>();

      if (Ns != Ns_mustbe) {
        ERROR << "Assumption of Ns = 2 is violated";
        exit(-1);
      }
    }

    // speed of light
    cc = config["cc"].get<float64>();

    int     nppc  = config["nppc"].get<int>();
    float64 delt  = config["delt"].get<float64>();
    float64 delh  = config["delh"].get<float64>();
    float64 wp    = config["wp"].get<float64>();
    float64 u0    = config["u0"].get<float64>();
    float64 vte   = config["vte"].get<float64>();
    float64 vti   = config["vti"].get<float64>();
    float64 mime  = config["mime"].get<float64>();
    float64 theta = config["theta"].get<float64>();
    float64 phi   = config["phi"].get<float64>();
    float64 sigma = config["sigma"].get<float64>();
    float64 gamma = sqrt(1.0 + (u0 * u0) / (cc * cc));
    float64 me    = 1.0 / nppc;
    float64 qe    = -wp / nppc * sqrt(gamma);
    float64 mi    = me * mime;
    float64 qi    = -qe;
    float64 b0    = wp * cc * std::abs((me / qe) * sigma);

    // set grid size and coordinate
    set_coordinate(delh, delh, delh);

    //
    // initialize field
    //
    {
      float64 Bx = b0 * cos(theta / 180 * nix::math::pi);
      float64 By = b0 * sin(theta / 180 * nix::math::pi) * cos(phi / 180 * nix::math::pi);
      float64 Bz = b0 * sin(theta / 180 * nix::math::pi) * sin(phi / 180 * nix::math::pi);

      // memory allocation
      allocate();

      for (int iz = Lbz; iz <= Ubz; iz++) {
        for (int iy = Lby; iy <= Uby; iy++) {
          for (int ix = Lbx; ix <= Ubx; ix++) {
            uf(iz, iy, ix, 0) = 0;
            uf(iz, iy, ix, 1) = +Bz * u0 / (gamma * cc);
            uf(iz, iy, ix, 2) = -By * u0 / (gamma * cc);
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

      // setup for Friedman filter
      this->setup_friedman_filter();
    }

    //
    // initialize particles
    //
    {
      float64             ratio       = this->get_buffer_ratio();
      int                 random_seed = option["random_seed"].get<int>();
      std::mt19937_64     mtp(random_seed);
      std::mt19937_64     mtv(random_seed);
      nix::rand_uniform   uniform(0.0, 1.0);
      nix::MaxwellJuttner mj_ele(vte * vte, u0);
      nix::MaxwellJuttner mj_ion(vti * vti, u0);

      {
        int   nz = dims[0] + 2 * Nb;
        int   ny = dims[1] + 2 * Nb;
        int   nx = dims[2] + 2 * Nb;
        int   mp = nppc * dims[0] * dims[1] * dims[2] * (1 + ratio);
        int64 id = static_cast<int64>(mp) * static_cast<int64>(this->myid);

        up.resize(Ns);

        // electron
        up[0]     = std::make_shared<ParticleType>(mp, nz * ny * nx);
        up[0]->m  = me;
        up[0]->q  = qe;
        up[0]->Np = mp;

        // ion
        up[1]     = std::make_shared<ParticleType>(mp, nz * ny * nx);
        up[1]->m  = mi;
        up[1]->q  = qi;
        up[1]->Np = mp;

        // initialize particle distribution
        for (int ip = 0; ip < mp; ip++) {
          // position: using these guarantees charge neutrality
          float64 x = uniform(mtp) * xlim[2] + xlim[0];
          float64 y = uniform(mtp) * ylim[2] + ylim[0];
          float64 z = uniform(mtp) * zlim[2] + zlim[0];

          // electrons
          {
            auto [ux, uy, uz] = mj_ele(mtv);

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

            up[1]->xu(ip, 0) = x;
            up[1]->xu(ip, 1) = y;
            up[1]->xu(ip, 2) = z;
            up[1]->xu(ip, 3) = ux;
            up[1]->xu(ip, 4) = uy;
            up[1]->xu(ip, 5) = uz;
          }

          // ID
          int64* ele_id64 = reinterpret_cast<int64*>(&up[0]->xu(ip, 0));
          int64* ion_id64 = reinterpret_cast<int64*>(&up[1]->xu(ip, 0));
          ele_id64[6]     = id + ip;
          ion_id64[6]     = id + ip;
        }
      }

      // initial sort
      this->sort_particle(up);
    }
  }
};

class MainApplication : public ExPIC3D<MainChunk, MainDiagnoser>
{
public:
  using ExPIC3D<MainChunk, MainDiagnoser>::ExPIC3D; // inherit constructors

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
