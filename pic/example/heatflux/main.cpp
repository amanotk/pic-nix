// -*- C++ -*-

#include "nix/random.hpp"
#include "pic_application.hpp"
#include "pic_chunk.hpp"
#include "pic_diag.hpp"

using MainApplication = PicApplication;

class MainChunk : public PicChunk
{
  static void set_field_aligned_coord(float64 th, float64 ph, float64 e1[3], float64 e2[3],
                                      float64 e3[3])
  {
    float64 ct = std::cos(th);
    float64 st = std::sin(th);
    float64 cp = std::cos(ph);
    float64 sp = std::sin(ph);

    e1[0] = ct;
    e1[1] = st * cp;
    e1[2] = st * sp;
    e2[0] = -st;
    e2[1] = ct * cp;
    e2[2] = ct * sp;
    e3[0] = 0.0;
    e3[1] = -sp;
    e3[2] = cp;
  }

public:
  using PicChunk::PicChunk;

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
    float64 sigma = config["sigma"].get<float64>();
    float64 theta = config["theta"].get<float64>();
    float64 phi   = config["phi"].get<float64>();
    float64 vti   = config["vti"].get<float64>();
    float64 vte   = config["vte"].get<float64>();

    // beam parameters
    float64 nb       = config["nb"].get<float64>();
    float64 vdb_para = config["vdb_para"].get<float64>();
    float64 vdb_perp = config["vdb_perp"].get<float64>();
    float64 vtb_para = config["vtb_para"].get<float64>();
    float64 vtb_perp = config["vtb_perp"].get<float64>();

    if (vdb_perp < 0) {
      ERROR << "vdb_perp must be >= 0";
      exit(-1);
    }

    float64 me = 1.0 / nppc;
    float64 qe = -wp / nppc;
    float64 mi = me * mime;
    float64 qi = -qe;

    float64 vdc_para = -nb * vdb_para / (1.0 - nb); // core drift velocity for current neutrality
    float64 b0       = cc * std::sqrt(sigma);       // magnetic field

    float64 tr = theta / 180.0 * nix::math::pi;
    float64 pr = phi / 180.0 * nix::math::pi;
    float64 e1vec[3];
    float64 e2vec[3];
    float64 e3vec[3];
    set_field_aligned_coord(tr, pr, e1vec, e2vec, e3vec);

    // set grid size and coordinate
    set_coordinate(delh, delh, delh);

    //
    // initialize field
    //
    {
      // memory allocation
      allocate();

      for (int iz = Lbz; iz <= Ubz; iz++) {
        for (int iy = Lby; iy <= Uby; iy++) {
          for (int ix = Lbx; ix <= Ubx; ix++) {
            uf(iz, iy, ix, 0) = 0;
            uf(iz, iy, ix, 1) = 0;
            uf(iz, iy, ix, 2) = 0;
            uf(iz, iy, ix, 3) = b0 * e1vec[0];
            uf(iz, iy, ix, 4) = b0 * e1vec[1];
            uf(iz, iy, ix, 5) = b0 * e1vec[2];
          }
        }
      }

      // allocate MPI buffer for field
      this->allocate_mpi_buffers();

      // setup for Friedman filter
      this->init_friedman();
    }

    //
    // initialize particles
    //
    {
      int                 random_seed = option["random_seed"].get<int>();
      std::mt19937_64     mtp(random_seed);
      std::mt19937_64     mtv(random_seed);
      nix::rand_uniform   uniform(0.0, 1.0);
      nix::rand_normal    normal(0.0, 1.0);
      nix::MaxwellianRing ring(vdb_perp, vtb_perp);

      {
        int   mp      = nppc * dims[0] * dims[1] * dims[2];
        int   mp_core = static_cast<int>(mp * (1.0 - nb));
        int   mp_beam = mp - mp_core;
        int64 id      = static_cast<int64>(mp) * static_cast<int64>(this->myid);

        up.resize(Ns);

        // core electron
        up[0]     = std::make_shared<ParticleType>(2 * mp_core, *this);
        up[0]->m  = me;
        up[0]->q  = qe;
        up[0]->Np = mp_core;

        // beam electron
        up[1]     = std::make_shared<ParticleType>(2 * mp_beam, *this);
        up[1]->m  = me;
        up[1]->q  = qe;
        up[1]->Np = mp_beam;

        // ion
        up[2]     = std::make_shared<ParticleType>(2 * mp, *this);
        up[2]->m  = mi;
        up[2]->q  = qi;
        up[2]->Np = mp;

        // initialize core electrons and paired ions
        for (int ip = 0; ip < mp_core; ip++) {
          float64 x = uniform(mtp) * xlim[2] + xlim[0];
          float64 y = uniform(mtp) * ylim[2] + ylim[0];
          float64 z = uniform(mtp) * zlim[2] + zlim[0];

          // core electron (isotropic Maxwellian)
          {
            up[0]->xu(ip, 0) = x;
            up[0]->xu(ip, 1) = y;
            up[0]->xu(ip, 2) = z;
            up[0]->xu(ip, 3) = normal(mtv) * vte + vdc_para * e1vec[0];
            up[0]->xu(ip, 4) = normal(mtv) * vte + vdc_para * e1vec[1];
            up[0]->xu(ip, 5) = normal(mtv) * vte + vdc_para * e1vec[2];
          }

          // ion
          {
            up[2]->xu(ip, 0) = x;
            up[2]->xu(ip, 1) = y;
            up[2]->xu(ip, 2) = z;
            up[2]->xu(ip, 3) = normal(mtv) * vti;
            up[2]->xu(ip, 4) = normal(mtv) * vti;
            up[2]->xu(ip, 5) = normal(mtv) * vti;
          }

          // ID
          {
            int64* ele_id64 = reinterpret_cast<int64*>(&up[0]->xu(ip, 0));
            int64* ion_id64 = reinterpret_cast<int64*>(&up[2]->xu(ip, 0));
            ele_id64[6]     = id + ip;
            ion_id64[6]     = id + ip;
          }
        }

        // initialize beam electrons and paired ions
        for (int ib = 0; ib < mp_beam; ib++) {
          int     ip = mp_core + ib;
          float64 x  = uniform(mtp) * xlim[2] + xlim[0];
          float64 y  = uniform(mtp) * ylim[2] + ylim[0];
          float64 z  = uniform(mtp) * zlim[2] + zlim[0];

          // beam electron
          {
            float64 v_para     = normal(mtv) * vtb_para + vdb_para;
            float64 v_perp_mag = ring(mtv);
            float64 theta_perp = uniform(mtv) * nix::math::pi * 2;
            float64 v_perp_a   = v_perp_mag * std::cos(theta_perp);
            float64 v_perp_b   = v_perp_mag * std::sin(theta_perp);

            up[1]->xu(ib, 0) = x;
            up[1]->xu(ib, 1) = y;
            up[1]->xu(ib, 2) = z;
            up[1]->xu(ib, 3) = v_para * e1vec[0] + v_perp_a * e2vec[0] + v_perp_b * e3vec[0];
            up[1]->xu(ib, 4) = v_para * e1vec[1] + v_perp_a * e2vec[1] + v_perp_b * e3vec[1];
            up[1]->xu(ib, 5) = v_para * e1vec[2] + v_perp_a * e2vec[2] + v_perp_b * e3vec[2];
          }

          // ion
          {
            up[2]->xu(ip, 0) = x;
            up[2]->xu(ip, 1) = y;
            up[2]->xu(ip, 2) = z;
            up[2]->xu(ip, 3) = normal(mtv) * vti;
            up[2]->xu(ip, 4) = normal(mtv) * vti;
            up[2]->xu(ip, 5) = normal(mtv) * vti;
          }

          // ID
          {
            int64* beam_id64 = reinterpret_cast<int64*>(&up[1]->xu(ib, 0));
            int64* ion_id64  = reinterpret_cast<int64*>(&up[2]->xu(ip, 0));
            beam_id64[6]     = id + ip;
            ion_id64[6]      = id + ip;
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
