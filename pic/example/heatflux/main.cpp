// -*- C++ -*-

#include "nix/random.hpp"
#include "pic_application.hpp"
#include "pic_chunk.hpp"
#include "pic_diag.hpp"

using MainApplication = PicApplication;

class MainChunk : public PicChunk
{
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

    std::string distribution_type = config["distribution_type"].get<std::string>();

    float64 fbeam = config["fbeam"].get<float64>();
    float64 vbeam = config["vbeam"].get<float64>();

    float64 vtec_para = config["vtec_para"].get<float64>();
    float64 vtec_perp = config["vtec_perp"].get<float64>();
    float64 vteb_para = config["vteb_para"].get<float64>();
    float64 vteb_perp = config["vteb_perp"].get<float64>();
    float64 vring     = config["vring"].get<float64>();
    float64 vti       = config["vti"].get<float64>();

    float64 me = 1.0 / nppc;
    float64 qe = -wp / nppc;
    float64 mi = me * mime;
    float64 qi = -qe;

    // magnetic field
    float64 b0 = cc * std::sqrt(sigma);
    float64 B_hat[3];

    if (sigma > 0) {
      float64 tr = theta / 180.0 * nix::math::pi;
      float64 pr = phi / 180.0 * nix::math::pi;

      B_hat[0] = std::cos(tr);
      B_hat[1] = std::sin(tr) * std::cos(pr);
      B_hat[2] = std::sin(tr) * std::sin(pr);
    } else {
      B_hat[0] = 1.0;
      B_hat[1] = 0.0;
      B_hat[2] = 0.0;
    }

    // perpendicular basis vectors (B-aligned coordinate system)
    float64 e_perp1_x, e_perp1_y, e_perp1_z;
    float64 e_perp2_x, e_perp2_y, e_perp2_z;

    {
      // e_perp1 = normalize(B_hat × x_hat)
      float64 cx = 0;
      float64 cy = B_hat[2];
      float64 cz = -B_hat[1];
      float64 cn = std::sqrt(cx * cx + cy * cy + cz * cz);

      if (cn > 1.0e-12) {
        e_perp1_x = cx / cn;
        e_perp1_y = cy / cn;
        e_perp1_z = cz / cn;
      } else {
        // fallback: B_hat × y_hat
        float64 fx = -B_hat[2];
        float64 fy = 0;
        float64 fz = B_hat[0];
        float64 fn = std::sqrt(fx * fx + fy * fy + fz * fz);

        e_perp1_x = fx / fn;
        e_perp1_y = fy / fn;
        e_perp1_z = fz / fn;
      }

      // e_perp2 = B_hat × e_perp1
      e_perp2_x = B_hat[1] * e_perp1_z - B_hat[2] * e_perp1_y;
      e_perp2_y = B_hat[2] * e_perp1_x - B_hat[0] * e_perp1_z;
      e_perp2_z = B_hat[0] * e_perp1_y - B_hat[1] * e_perp1_x;
    }

    // core drift velocity for current neutrality
    float64 v_core = -fbeam * vbeam / (1.0 - fbeam);

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
            uf(iz, iy, ix, 3) = b0 * B_hat[0];
            uf(iz, iy, ix, 4) = b0 * B_hat[1];
            uf(iz, iy, ix, 5) = b0 * B_hat[2];
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
      int                 random_seed = option["random_seed"].get<int>();
      std::mt19937_64     mtp(random_seed);
      std::mt19937_64     mtv(random_seed);
      nix::rand_uniform   uniform(0.0, 1.0);
      nix::rand_normal    normal(0.0, 1.0);
      nix::MaxwellianRing ring(vring, vteb_perp);

      {
        int   mp      = nppc * dims[0] * dims[1] * dims[2];
        int   mp_core = static_cast<int>(mp * (1.0 - fbeam));
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

        // pair electrons with ions for charge neutrality
        for (int ip = 0; ip < mp; ip++) {
          float64 x = uniform(mtp) * xlim[2] + xlim[0];
          float64 y = uniform(mtp) * ylim[2] + ylim[0];
          float64 z = uniform(mtp) * zlim[2] + zlim[0];

          // electron velocity in B-aligned frame
          float64 v_para, v_perp_mag, theta_perp;

          if (ip < mp_core) {
            // core electron
            v_para     = normal(mtv) * vtec_para + v_core;
            v_perp_mag = normal(mtv) * vtec_perp;
            theta_perp = uniform(mtv) * nix::math::pi * 2;

            up[0]->xu(ip, 0) = x;
            up[0]->xu(ip, 1) = y;
            up[0]->xu(ip, 2) = z;

            int64* id64 = reinterpret_cast<int64*>(&up[0]->xu(ip, 0));

            id64[6] = id + ip;
          } else {
            // beam electron
            int ib = ip - mp_core;

            if (distribution_type == "ring") {
              v_para     = normal(mtv) * vteb_para + vbeam;
              v_perp_mag = ring(mtv);
            } else {
              v_para     = normal(mtv) * vteb_para + vbeam;
              v_perp_mag = normal(mtv) * vteb_perp;
            }
            theta_perp = uniform(mtv) * nix::math::pi * 2;

            up[1]->xu(ib, 0) = x;
            up[1]->xu(ib, 1) = y;
            up[1]->xu(ib, 2) = z;

            int64* id64 = reinterpret_cast<int64*>(&up[1]->xu(ib, 0));

            id64[6] = id + ip;
          }

          // rotate velocity from B-aligned frame to simulation frame
          float64 v_perp_a = v_perp_mag * std::cos(theta_perp);
          float64 v_perp_b = v_perp_mag * std::sin(theta_perp);

          float64 vx = v_para * B_hat[0] + v_perp_a * e_perp1_x + v_perp_b * e_perp2_x;
          float64 vy = v_para * B_hat[1] + v_perp_a * e_perp1_y + v_perp_b * e_perp2_y;
          float64 vz = v_para * B_hat[2] + v_perp_a * e_perp1_z + v_perp_b * e_perp2_z;

          // ion
          up[2]->xu(ip, 0) = x;
          up[2]->xu(ip, 1) = y;
          up[2]->xu(ip, 2) = z;
          up[2]->xu(ip, 3) = normal(mtv) * vti;
          up[2]->xu(ip, 4) = normal(mtv) * vti;
          up[2]->xu(ip, 5) = normal(mtv) * vti;

          int64* ion_id64 = reinterpret_cast<int64*>(&up[2]->xu(ip, 0));

          ion_id64[6] = id + ip;

          // assign electron velocity (the velocity RNG was consumed above)
          if (ip < mp_core) {
            up[0]->xu(ip, 3) = vx;
            up[0]->xu(ip, 4) = vy;
            up[0]->xu(ip, 5) = vz;
          } else {
            int ib = ip - mp_core;

            up[1]->xu(ib, 3) = vx;
            up[1]->xu(ib, 4) = vy;
            up[1]->xu(ib, 5) = vz;
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
