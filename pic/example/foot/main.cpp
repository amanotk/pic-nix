// -*- C++ -*-

#include "nix/random.hpp"
#include "pic_application.hpp"
#include "pic_chunk.hpp"
#include "pic_diag.hpp"

using MainApplication = PicApplication;

class MainChunk : public PicChunk
{
public:
  using PicChunk::PicChunk; // inherit constructors

  virtual void setup(json& config) override
  {
    PicChunk::setup(config);

    // check validity of assumptions
    {
      constexpr int Ns_mustbe = 3;

      Ns = config["Ns"].get<int>();

      if (Ns != Ns_mustbe) {
        ERROR << "Assumption of Ns = 3 is violated";
        exit(-1);
      }
    }

    // speed of light
    cc = config["cc"].get<float64>();

    int     nppc  = config["nppc"].get<int>();
    float64 wp    = config["wp"].get<float64>();
    float64 delt  = config["delt"].get<float64>();
    float64 delh  = config["delh"].get<float64>();
    float64 mime  = config["mime"].get<float64>();
    float64 ush   = config["ush"].get<float64>();
    float64 theta = config["theta"].get<float64>();
    float64 phi   = config["phi"].get<float64>();
    float64 sigma = config["sigma"].get<float64>();
    float64 alpha = config["alpha"].get<float64>();
    float64 betae = config["betae"].get<float64>();
    float64 betai = config["betai"].get<float64>();
    float64 betar = config["betar"].get<float64>();
    float64 gamsh = sqrt(1.0 + (ush * ush) / (cc * cc));
    float64 me    = 1.0 / nppc;
    float64 qe    = -wp / nppc;
    float64 mi    = me * mime;
    float64 qi    = -qe;
    float64 b0    = cc * wp * sqrt(sigma) / std::abs(qe / me);
    float64 vae   = cc * sqrt(sigma);
    float64 vai   = cc * sqrt(sigma / mime);
    float64 vte   = vae * sqrt(0.5 * betae);
    float64 vti   = vai * sqrt(0.5 * betai);
    float64 vtr   = vai * sqrt(0.5 * betar);

    // quantities in the simulation frame
    float64 vsh  = ush / gamsh;
    float64 vdi  = -2 * alpha * vsh / (1 - (1 - 2 * alpha) * vsh * vsh);
    float64 vdr  = +2 * (1 - alpha) * vsh / (1 + (1 - 2 * alpha) * vsh * vsh);
    float64 udi  = vdi / sqrt(1 - vdi * vdi / (cc * cc));
    float64 udr  = vdr / sqrt(1 - vdr * vdr / (cc * cc));
    float64 nref = alpha * (1 + (1 - 2 * alpha) * vsh * vsh) /
                   (1 - (1 - 2 * alpha) * (1 - 2 * alpha) * vsh * vsh);

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
            uf(iz, iy, ix, 1) = 0;
            uf(iz, iy, ix, 2) = 0;
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
      int                              random_seed = option["random_seed"].get<int>();
      std::mt19937_64                  mtp(random_seed);
      std::mt19937_64                  mtv(random_seed);
      nix::rand_uniform                uniform(0.0, 1.0);
      nix::MaxwellJuttner              mj_ele(vte * vte, 0.0);
      std::vector<nix::MaxwellJuttner> mj_ion;
      mj_ion.push_back(nix::MaxwellJuttner(vti * vti, udi));
      mj_ion.push_back(nix::MaxwellJuttner(vtr * vtr, udr));

      {
        int   mp  = nppc * dims[0] * dims[1] * dims[2];
        int   mp1 = mp * (1 - nref);
        int   mp2 = mp - mp1;
        int64 id  = static_cast<int64>(mp) * static_cast<int64>(this->myid);

        up.resize(Ns);

        // electron
        up[0]     = std::make_shared<ParticleType>(2 * mp, *this);
        up[0]->m  = me;
        up[0]->q  = qe;
        up[0]->Np = mp;

        // incoming ion
        up[1]     = std::make_shared<ParticleType>(2 * mp1, *this);
        up[1]->m  = mi;
        up[1]->q  = qi;
        up[1]->Np = mp1;

        // reflected ion
        up[2]     = std::make_shared<ParticleType>(2 * mp2, *this);
        up[2]->m  = mi;
        up[2]->q  = qi;
        up[2]->Np = mp2;

        // initialize particle distribution
        std::vector<int> mp_ele{0, mp1};
        std::vector<int> mp_ion{mp1, mp2};

        for (int is = 0; is < 2; is++) {
          const int is_ele      = 0;
          const int is_ion      = is + 1;
          const int ip_ele_zero = mp_ele[is];
          const int ip_ion_zero = 0;

          for (int ip = 0; ip < mp_ion[is]; ip++) {
            const int ip_ele = ip + ip_ele_zero;
            const int ip_ion = ip + ip_ion_zero;

            // position: using these guarantees charge neutrality
            float64 x = uniform(mtp) * xlim[2] + xlim[0];
            float64 y = uniform(mtp) * ylim[2] + ylim[0];
            float64 z = uniform(mtp) * zlim[2] + zlim[0];

            // electrons
            {
              auto [ux, uy, uz] = mj_ele(mtv);

              up[is_ele]->xu(ip_ele, 0) = x;
              up[is_ele]->xu(ip_ele, 1) = y;
              up[is_ele]->xu(ip_ele, 2) = z;
              up[is_ele]->xu(ip_ele, 3) = ux;
              up[is_ele]->xu(ip_ele, 4) = uy;
              up[is_ele]->xu(ip_ele, 5) = uz;
            }

            // ions
            {
              auto [ux, uy, uz] = mj_ion[is](mtv);

              up[is_ion]->xu(ip_ion, 0) = x;
              up[is_ion]->xu(ip_ion, 1) = y;
              up[is_ion]->xu(ip_ion, 2) = z;
              up[is_ion]->xu(ip_ion, 3) = ux;
              up[is_ion]->xu(ip_ion, 4) = uy;
              up[is_ion]->xu(ip_ion, 5) = uz;
            }

            // ID
            int64* ele_id64 = reinterpret_cast<int64*>(&up[is_ele]->xu(ip_ele, 0));
            int64* ion_id64 = reinterpret_cast<int64*>(&up[is_ion]->xu(ip_ion, 0));
            ele_id64[6]     = id + ip_ele;
            ion_id64[6]     = id + ip_ele;
          }
        }
      }

      // initial sort
      this->sort_particle(up);
    }
  }
};

class MainInterface : public PicApplicationInterface
{
public:
  virtual PtrChunk create_chunk(nix::Dims3D dims, nix::Bool3D has_dim, int id) override
  {
    return std::make_unique<MainChunk>(dims, has_dim, id);
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
