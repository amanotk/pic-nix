// -*- C++ -*-

#include "nix/random.hpp"
#include "pic_application.hpp"
#include "pic_chunk.hpp"
#include "pic_diag.hpp"

// return true if the point (x, y, z) is inside the fireball
static bool is_inside_fireball(float64 z, float64 y, float64 x, float64 r)
{
  return x * x + y * y < r * r;
}

// count the number of cells within fireball
template <typename Range>
static int count_cell_within_fireball(Range zrange, Range yrange, Range xrange, float64 dz,
                                      float64 dy, float64 dx, float64 z0, float64 y0, float64 x0,
                                      float64 radius)
{
  int count = 0;

  for (int iz = zrange[0]; iz < zrange[1]; iz++) {
    for (int iy = yrange[0]; iy < yrange[1]; iy++) {
      for (int ix = xrange[0]; ix < xrange[1]; ix++) {
        float64 zz = (iz + 0.5) * dz - z0;
        float64 yy = (iy + 0.5) * dy - y0;
        float64 xx = (ix + 0.5) * dx - x0;
        if (is_inside_fireball(zz, yy, xx, radius)) {
          count++;
        }
      }
    }
  }

  return count;
}

class MainChunk : public PicChunk
{
public:
  using PicChunk::PicChunk; // inherit constructors

  virtual void setup(json& config) override
  {
    PicChunk::setup(config);

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

    int     nbg    = config["nbg"].get<int>();
    int     nfb    = config["nfb"].get<int>();
    float64 Tbg    = config["Tbg"].get<float64>();
    float64 Tfb    = config["Tfb"].get<float64>();
    float64 vtbg   = cc * sqrt(Tbg);
    float64 vtfb   = cc * sqrt(Tfb);
    float64 radius = config["radius"].get<float64>();
    float64 wp     = config["wp"].get<float64>();
    float64 delt   = config["delt"].get<float64>();
    float64 delh   = config["delh"].get<float64>();
    float64 mime   = config["mime"].get<float64>();
    float64 theta  = config["theta"].get<float64>();
    float64 phi    = config["phi"].get<float64>();
    float64 sigma  = config["sigma"].get<float64>();
    float64 me     = 1.0 / (sigma * nfb);
    float64 qe     = -wp * sqrt(sigma) * me;
    float64 mi     = me * mime;
    float64 qi     = -qe;
    float64 b0     = cc * sqrt(sigma) / std::abs(qe / me);

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
      float64           target      = 1 + this->get_buffer_ratio();
      int               random_seed = option["random_seed"].get<int>();
      std::mt19937_64   mtp(random_seed);
      std::mt19937_64   mtv(random_seed);
      nix::rand_uniform uniform(0.0, 1.0);
      nix::rand_normal  normal(0.0, 1.0);

      {
        std::array<int, 2> zr = {offset[0], offset[0] + dims[0]};
        std::array<int, 2> yr = {offset[1], offset[1] + dims[1]};
        std::array<int, 2> xr = {offset[2], offset[2] + dims[2]};
        float64            z0 = 0.5 * (gzlim[0] + gzlim[1]);
        float64            y0 = 0.5 * (gylim[0] + gylim[1]);
        float64            x0 = 0.5 * (gxlim[0] + gxlim[1]);

        int   count = count_cell_within_fireball(zr, yr, xr, delz, dely, delx, z0, y0, x0, radius);
        int   mp    = count * nfb + (dims[0] * dims[1] * dims[2] - count) * nbg;
        int64 id    = dims[0] * dims[1] * dims[2] * nfb * this->myid;

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

        // initialize particle distribution
        int ip = 0;
        for (int iz = Lbz; iz <= Ubz; iz++) {
          for (int iy = Lby; iy <= Uby; iy++) {
            for (int ix = Lbx; ix <= Ubx; ix++) {
              int     nppc;
              float64 vte;
              float64 vti;

              float64 zz = (iz - Lbz + 0.5) * delz + zlim[0] - z0;
              float64 yy = (iy - Lby + 0.5) * dely + ylim[0] - y0;
              float64 xx = (ix - Lbx + 0.5) * delx + xlim[0] - x0;

              if (is_inside_fireball(zz, yy, xx, radius)) {
                // fireball
                nppc = nfb;
                vte  = vtfb;
                vti  = vtfb / sqrt(mime);
              } else {
                // background
                nppc = nbg;
                vte  = vtbg;
                vti  = vtbg / sqrt(mime);
              }

              for (int jp = 0; jp < nppc; jp++) {
                float64 z = (uniform(mtp) + iz - Lbz) * delz + zlim[0];
                float64 y = (uniform(mtp) + iy - Lby) * dely + ylim[0];
                float64 x = (uniform(mtp) + ix - Lbx) * delx + xlim[0];

                // electrons
                up[0]->xu(ip, 0) = x;
                up[0]->xu(ip, 1) = y;
                up[0]->xu(ip, 2) = z;
                up[0]->xu(ip, 3) = normal(mtv) * vte;
                up[0]->xu(ip, 4) = normal(mtv) * vte;
                up[0]->xu(ip, 5) = normal(mtv) * vte;

                // ions
                up[1]->xu(ip, 0) = x;
                up[1]->xu(ip, 1) = y;
                up[1]->xu(ip, 2) = z;
                up[1]->xu(ip, 3) = normal(mtv) * vti;
                up[1]->xu(ip, 4) = normal(mtv) * vti;
                up[1]->xu(ip, 5) = normal(mtv) * vti;

                // particle ID
                int64* ele_id64 = reinterpret_cast<int64*>(&up[0]->xu(ip, 0));
                int64* ion_id64 = reinterpret_cast<int64*>(&up[1]->xu(ip, 0));
                ele_id64[6]     = id;
                ion_id64[6]     = id;

                ip++;
                id++;
              }
            }
          }
        }
        assert(ip == mp);
      }

      // initial sort
      this->sort_particle(up);
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

  void initialize_workload() override
  {
    const int nchunk_global = cdims[3];

    json    parameter = cfgparser->get_parameter();
    json    option    = cfgparser->get_application()["option"];
    int     nbg       = parameter["nbg"].get<float64>();
    int     nfb       = parameter["nfb"].get<float64>();
    float64 radius    = parameter["radius"].get<float64>();
    float64 cell_load = option.value("cell_load", 1.0);

    int     dims[3]     = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};
    int     ncell_chunk = dims[0] * dims[1] * dims[2];
    float64 dx          = cfgparser->get_delx();
    float64 dy          = cfgparser->get_dely();
    float64 dz          = cfgparser->get_delz();
    float64 x0          = 0.5 * dx * ndims[2];
    float64 y0          = 0.5 * dy * ndims[1];
    float64 z0          = 0.5 * dz * ndims[0];

    for (int i = 0; i < nchunk_global; i++) {
      auto [cz, cy, cx] = chunkmap->get_coordinate(i);

      std::array<int, 2> zr = {cz * dims[0], (cz + 1) * dims[0]};
      std::array<int, 2> yr = {cy * dims[1], (cy + 1) * dims[1]};
      std::array<int, 2> xr = {cx * dims[2], (cx + 1) * dims[2]};

      int     count = count_cell_within_fireball(zr, yr, xr, dz, dy, dx, z0, y0, x0, radius);
      float64 a     = 2.0 * count / ncell_chunk;
      float64 b     = 2.0 * (ncell_chunk - count) / ncell_chunk;

      balancer->load(i) = a * nfb + b * nbg + cell_load;
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
