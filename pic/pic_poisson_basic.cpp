// -*- C++ -*-
#include "pic_poisson_basic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <experimental/mdspan>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <vector>

namespace stdex = std::experimental;

struct PicPoissonBasic::Impl {
  using Span3D       = stdex::mdspan<float64, stdex::dextents<size_t, 3>>;
  using ExtentsArray = std::array<size_t, 3>;

  struct ChunkWork {
    PicChunk*            chunk;
    std::vector<float64> x;
    std::vector<float64> r;
    std::vector<float64> p;
    std::vector<float64> z;
    std::vector<float64> Ap;
    std::vector<float64> phi_backup;
    int                  nz;
    int                  ny;
    int                  nx;
    int                  stride_y;
    int                  stride_z;
    int                  Lbx;
    int                  Ubx;
    int                  Lby;
    int                  Uby;
    int                  Lbz;
    int                  Ubz;
    float64              delx;
    float64              dely;
    float64              delz;
    bool                 has_x;
    bool                 has_y;
    bool                 has_z;

    size_t size() const
    {
      return static_cast<size_t>(nz * ny * nx);
    }

    int index(int iz, int iy, int ix) const
    {
      return iz * stride_z + iy * stride_y + ix;
    }
  };

  Impl(const nix::Dims3D& global_dims_in, float64 delh_in)
      : global_dims(global_dims_in), delh(delh_in)
  {
  }

  int update_mapping(elliptic::ChunkAccessor& accessor)
  {
    (void)accessor;
    return 0;
  }

  int copy_chunk_to_src(elliptic::ChunkAccessor& accessor)
  {
    return accessor.get_num_grids_total();
  }

  int copy_sol_to_chunk(elliptic::ChunkAccessor& accessor)
  {
    return accessor.get_num_grids_total();
  }

  int set_option(const nlohmann::json& config)
  {
    const auto& solver_config = config.value("poisson_basic", nlohmann::json::object());
    max_iter                  = solver_config.value("max_iter", max_iter);
    tol                       = solver_config.value("tol", tol);
    omega                     = solver_config.value("omega", omega);

    if (omega <= 0.0 || omega >= 2.0) {
      ERROR << "omega must be in range (0, 2) for SSOR convergence" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    return 0;
  }

  int solve(elliptic::ChunkAccessor& accessor)
  {
    auto* pic_accessor = dynamic_cast<PicPoisson::PicChunkAccessor*>(&accessor);
    if (pic_accessor == nullptr) {
      ERROR << "PicPoissonBasic requires PicChunkAccessor" << std::endl;
      return 1;
    }

    prepare_chunks(*pic_accessor);

    exchange_boundary(&ChunkWork::x);
    apply_operator(&ChunkWork::x, &ChunkWork::Ap);
    build_rhs();
    compute_residual();

    double norm_b = rhs_norm2();
    double rz     = apply_preconditioner();
    double norm_r = dot_global(&ChunkWork::r, &ChunkWork::r);

    if (norm_b == 0.0) {
      norm_b = 1.0;
    }

    if (std::sqrt(norm_r / norm_b) < tol) {
      copy_solution_back();
      return 0;
    }

    for (auto& cw : chunk_work) {
      std::copy(cw.z.begin(), cw.z.end(), cw.p.begin());
    }

    for (int iter = 0; iter < max_iter; ++iter) {
      exchange_boundary(&ChunkWork::p);
      apply_operator(&ChunkWork::p, &ChunkWork::Ap);
      const double pAp   = dot_global(&ChunkWork::p, &ChunkWork::Ap);
      const double alpha = (pAp == 0.0) ? 0.0 : rz / pAp;

      axpy(alpha, &ChunkWork::p, &ChunkWork::x);
      axpy(-alpha, &ChunkWork::Ap, &ChunkWork::r);

      norm_r = dot_global(&ChunkWork::r, &ChunkWork::r);
      if (std::sqrt(norm_r / norm_b) < tol) {
        copy_solution_back();
        return 0;
      }

      const double rz_new = apply_preconditioner();
      const double beta   = (rz == 0.0) ? 0.0 : rz_new / rz;
      rz                  = rz_new;
      for (auto& cw : chunk_work) {
        for (size_t j = 0; j < cw.p.size(); ++j) {
          cw.p[j] = cw.z[j] + beta * cw.p[j];
        }
      }
    }

    copy_solution_back();
    return 0;
  }

  int scatter_forward()
  {
    return 0;
  }

  int scatter_reverse()
  {
    return 0;
  }

  void prepare_chunks(PicPoisson::PicChunkAccessor& accessor)
  {
    const auto& chunks = accessor.get_chunks();
    chunk_work.clear();
    chunk_work.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
      auto*     chunk = chunks[i];
      auto      data  = chunk->get_internal_data();
      ChunkWork cw;
      cw.chunk     = chunk;
      cw.Lbx       = data.Lbx;
      cw.Ubx       = data.Ubx;
      cw.Lby       = data.Lby;
      cw.Uby       = data.Uby;
      cw.Lbz       = data.Lbz;
      cw.Ubz       = data.Ubz;
      cw.delx      = data.delx;
      cw.dely      = data.dely;
      cw.delz      = data.delz;
      cw.has_x     = chunk->has_xdim();
      cw.has_y     = chunk->has_ydim();
      cw.has_z     = chunk->has_zdim();
      const int bm = data.boundary_margin;

      auto dims   = chunk->get_dims();
      cw.nz       = dims[0] + 2 * bm;
      cw.ny       = dims[1] + 2 * bm;
      cw.nx       = dims[2] + 2 * bm;
      cw.stride_y = cw.nx;
      cw.stride_z = cw.ny * cw.nx;

      cw.x.resize(cw.size(), 0.0);
      cw.r.resize(cw.size(), 0.0);
      cw.p.resize(cw.size(), 0.0);
      cw.z.resize(cw.size(), 0.0);
      cw.Ap.resize(cw.size(), 0.0);
      cw.phi_backup.resize(cw.size(), 0.0);

      const float64* phi_ptr = data.phi.data();
      std::copy(phi_ptr, phi_ptr + cw.size(), cw.x.begin());

      chunk_work.push_back(std::move(cw));
    }

    if (chunk_work.empty()) {
      ERROR << "No chunks provided" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    for (size_t i = 0; i < chunk_work.size(); ++i) {
      auto& cw        = chunk_work[i];
      int   this_dims = (cw.has_x ? 1 : 0) + (cw.has_y ? 1 : 0) + (cw.has_z ? 1 : 0);

      if (this_dims == 0) {
        ERROR << "Chunks must have at least one active dimension" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
      }

      if (i == 0) {
        num_dims = this_dims;
      } else if (this_dims != num_dims) {
        ERROR << "All chunks must have the same dimensionality" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }
  }

  void build_rhs()
  {
    rhs_storage.clear();
    for (auto& cw : chunk_work) {
      auto  data = cw.chunk->get_internal_data();
      auto& rhs  = rhs_storage[cw.chunk];
      rhs.assign(cw.size(), 0.0);

      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            const int idx = cw.index(iz, iy, ix);
            rhs[idx]      = data.uj(iz, iy, ix, 0);
          }
        }
      }
    }
  }

  void compute_residual()
  {
    for (size_t i = 0; i < chunk_work.size(); ++i) {
      auto& cw  = chunk_work[i];
      auto& rhs = rhs_storage[cw.chunk];
      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            const int idx = cw.index(iz, iy, ix);
            cw.r[idx]     = rhs[idx] - cw.Ap[idx];
            cw.z[idx]     = 0.0;
            cw.p[idx]     = 0.0;
          }
        }
      }
    }
  }

  void exchange_boundary(std::vector<float64> ChunkWork::*field)
  {
    // backup phi
    for (auto& cw : chunk_work) {
      auto   data     = cw.chunk->get_internal_data();
      auto*  phi_ptr  = data.phi.data();
      size_t phi_size = cw.size();
      std::copy(phi_ptr, phi_ptr + phi_size, cw.phi_backup.begin());
    }

    // copy vector to phi
    for (auto& cw : chunk_work) {
      auto  data   = cw.chunk->get_internal_data();
      auto* phi    = data.phi.data();
      auto& buffer = cw.*field;
      std::copy(buffer.begin(), buffer.end(), phi);
    }

    for (auto& cw : chunk_work) {
      cw.chunk->set_boundary_pack(BoundaryPhi);
      cw.chunk->set_boundary_begin(BoundaryPhi);
    }
    for (auto& cw : chunk_work) {
      cw.chunk->set_boundary_end(BoundaryPhi);
      cw.chunk->set_boundary_unpack(BoundaryPhi);
    }

    // copy back to vectors and restore phi
    for (auto& cw : chunk_work) {
      auto  data   = cw.chunk->get_internal_data();
      auto* phi    = data.phi.data();
      auto& buffer = cw.*field;
      std::copy(phi, phi + cw.size(), buffer.begin());
      std::copy(cw.phi_backup.begin(), cw.phi_backup.end(), phi);
    }
  }

  void apply_operator(std::vector<float64> ChunkWork::*src, std::vector<float64> ChunkWork::*dst)
  {
    if (num_dims == 3) {
      apply_operator_3d(src, dst);
    } else if (num_dims == 2) {
      apply_operator_2d(src, dst);
    } else {
      apply_operator_1d(src, dst);
    }
  }

  double apply_preconditioner()
  {
    for (auto& cw : chunk_work) {
      std::fill(cw.z.begin(), cw.z.end(), 0.0);
    }

    if (num_dims == 3) {
      preconditioner_forward_3d(&ChunkWork::r, &ChunkWork::z);
    } else if (num_dims == 2) {
      preconditioner_forward_2d(&ChunkWork::r, &ChunkWork::z);
    } else {
      preconditioner_forward_1d(&ChunkWork::r, &ChunkWork::z);
    }

    exchange_boundary(&ChunkWork::z);

    if (num_dims == 3) {
      preconditioner_backward_3d(&ChunkWork::r, &ChunkWork::z);
    } else if (num_dims == 2) {
      preconditioner_backward_2d(&ChunkWork::r, &ChunkWork::z);
    } else {
      preconditioner_backward_1d(&ChunkWork::r, &ChunkWork::z);
    }

    return dot_global(&ChunkWork::r, &ChunkWork::z);
  }

  void axpy(float64 alpha, std::vector<float64> ChunkWork::*xvec,
            std::vector<float64> ChunkWork::*yvec)
  {
    for (auto& cw : chunk_work) {
      auto& x = cw.*xvec;
      auto& y = cw.*yvec;
      for (size_t j = 0; j < x.size(); ++j) {
        y[j] += alpha * x[j];
      }
    }
  }

  double dot_global(std::vector<float64> ChunkWork::*a, std::vector<float64> ChunkWork::*b)
  {
    double local = 0.0;
    for (auto& cw : chunk_work) {
      auto a_view = make_view(cw.*a, cw.nz, cw.ny, cw.nx);
      auto b_view = make_view(cw.*b, cw.nz, cw.ny, cw.nx);
      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            local += a_view(iz, iy, ix) * b_view(iz, iy, ix);
          }
        }
      }
    }
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
  }

  double rhs_norm2()
  {
    double local = 0.0;
    for (auto& cw : chunk_work) {
      auto& rhs = rhs_storage[cw.chunk];
      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            const int idx = cw.index(iz, iy, ix);
            local += rhs[idx] * rhs[idx];
          }
        }
      }
    }

    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
  }

  void copy_solution_back()
  {
    // copy interior to chunk->phi
    for (auto& cw : chunk_work) {
      auto data = cw.chunk->get_internal_data();
      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            const int idx        = cw.index(iz, iy, ix);
            data.phi(iz, iy, ix) = cw.x[idx];
          }
        }
      }
    }

    // update phi halos for consistency
    for (auto& cw : chunk_work) {
      cw.chunk->set_boundary_pack(BoundaryPhi);
      cw.chunk->set_boundary_begin(BoundaryPhi);
    }
    for (auto& cw : chunk_work) {
      cw.chunk->set_boundary_end(BoundaryPhi);
      cw.chunk->set_boundary_unpack(BoundaryPhi);
    }
  }

  nix::Dims3D                                         global_dims;
  float64                                             delh;
  int                                                 max_iter = 200;
  double                                              tol      = 1.0e-12;
  float64                                             omega    = 1.0;
  std::vector<ChunkWork>                              chunk_work;
  std::unordered_map<PicChunk*, std::vector<float64>> rhs_storage;
  int                                                 num_dims = 0;

  static Span3D make_view(std::vector<float64>& buffer, int nz, int ny, int nx)
  {
    return Span3D(buffer.data(), ExtentsArray{static_cast<size_t>(nz), static_cast<size_t>(ny),
                                              static_cast<size_t>(nx)});
  }

  void apply_operator_1d(std::vector<float64> ChunkWork::*src, std::vector<float64> ChunkWork::*dst)
  {
    for (auto& cw : chunk_work) {
      auto& in       = cw.*src;
      auto& out      = cw.*dst;
      auto  in_view  = make_view(in, cw.nz, cw.ny, cw.nx);
      auto  out_view = make_view(out, cw.nz, cw.ny, cw.nx);

      const float64 dx2  = 1.0 / (cw.delx * cw.delx);
      const float64 ofdx = -dx2;

      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            const float64 diag   = 2.0 * dx2;
            const float64 sum    = ofdx * (in_view(iz, iy, ix - 1) + in_view(iz, iy, ix + 1));
            out_view(iz, iy, ix) = diag * in_view(iz, iy, ix) + sum;
          }
        }
      }
    }
  }

  void apply_operator_2d(std::vector<float64> ChunkWork::*src, std::vector<float64> ChunkWork::*dst)
  {
    for (auto& cw : chunk_work) {
      auto& in       = cw.*src;
      auto& out      = cw.*dst;
      auto  in_view  = make_view(in, cw.nz, cw.ny, cw.nx);
      auto  out_view = make_view(out, cw.nz, cw.ny, cw.nx);

      const float64 dx2  = 1.0 / (cw.delx * cw.delx);
      const float64 dy2  = 1.0 / (cw.dely * cw.dely);
      const float64 ofdx = -dx2;
      const float64 ofdy = -dy2;

      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            const float64 diag = 2.0 * dx2 + 2.0 * dy2;
            const float64 sum  = ofdx * (in_view(iz, iy, ix - 1) + in_view(iz, iy, ix + 1)) +
                                ofdy * (in_view(iz, iy - 1, ix) + in_view(iz, iy + 1, ix));
            out_view(iz, iy, ix) = diag * in_view(iz, iy, ix) + sum;
          }
        }
      }
    }
  }

  void apply_operator_3d(std::vector<float64> ChunkWork::*src, std::vector<float64> ChunkWork::*dst)
  {
    for (auto& cw : chunk_work) {
      auto& in       = cw.*src;
      auto& out      = cw.*dst;
      auto  in_view  = make_view(in, cw.nz, cw.ny, cw.nx);
      auto  out_view = make_view(out, cw.nz, cw.ny, cw.nx);

      const float64 dx2  = 1.0 / (cw.delx * cw.delx);
      const float64 dy2  = 1.0 / (cw.dely * cw.dely);
      const float64 dz2  = 1.0 / (cw.delz * cw.delz);
      const float64 ofdx = -dx2;
      const float64 ofdy = -dy2;
      const float64 ofdz = -dz2;

      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            const float64 diag = 2.0 * dx2 + 2.0 * dy2 + 2.0 * dz2;
            const float64 sum  = ofdx * (in_view(iz, iy, ix - 1) + in_view(iz, iy, ix + 1)) +
                                ofdy * (in_view(iz, iy - 1, ix) + in_view(iz, iy + 1, ix)) +
                                ofdz * (in_view(iz - 1, iy, ix) + in_view(iz + 1, iy, ix));
            out_view(iz, iy, ix) = diag * in_view(iz, iy, ix) + sum;
          }
        }
      }
    }
  }

  void preconditioner_forward_1d(std::vector<float64> ChunkWork::*src,
                                 std::vector<float64> ChunkWork::*dst)
  {
    for (auto& cw : chunk_work) {
      auto& r      = cw.*src;
      auto& z      = cw.*dst;
      auto  r_view = make_view(r, cw.nz, cw.ny, cw.nx);
      auto  z_view = make_view(z, cw.nz, cw.ny, cw.nx);

      const float64 dx2  = 1.0 / (cw.delx * cw.delx);
      const float64 ofdx = -dx2;

      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            const float64 diag = 2.0 * dx2;
            float64       sum  = r_view(iz, iy, ix);
            sum -= ofdx * (z_view(iz, iy, ix - 1) + z_view(iz, iy, ix + 1));
            float64 z_new      = (diag > 0.0) ? sum / diag : r_view(iz, iy, ix);
            z_view(iz, iy, ix) = (1.0 - omega) * z_view(iz, iy, ix) + omega * z_new;
          }
        }
      }
    }
  }

  void preconditioner_forward_2d(std::vector<float64> ChunkWork::*src,
                                 std::vector<float64> ChunkWork::*dst)
  {
    for (auto& cw : chunk_work) {
      auto& r      = cw.*src;
      auto& z      = cw.*dst;
      auto  r_view = make_view(r, cw.nz, cw.ny, cw.nx);
      auto  z_view = make_view(z, cw.nz, cw.ny, cw.nx);

      const float64 dx2  = 1.0 / (cw.delx * cw.delx);
      const float64 dy2  = 1.0 / (cw.dely * cw.dely);
      const float64 ofdx = -dx2;
      const float64 ofdy = -dy2;

      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            const float64 diag = 2.0 * dx2 + 2.0 * dy2;
            float64       sum  = r_view(iz, iy, ix);
            sum -= ofdx * (z_view(iz, iy, ix - 1) + z_view(iz, iy, ix + 1));
            sum -= ofdy * (z_view(iz, iy - 1, ix) + z_view(iz, iy + 1, ix));
            float64 z_new      = (diag > 0.0) ? sum / diag : r_view(iz, iy, ix);
            z_view(iz, iy, ix) = (1.0 - omega) * z_view(iz, iy, ix) + omega * z_new;
          }
        }
      }
    }
  }

  void preconditioner_forward_3d(std::vector<float64> ChunkWork::*src,
                                 std::vector<float64> ChunkWork::*dst)
  {
    for (auto& cw : chunk_work) {
      auto& r      = cw.*src;
      auto& z      = cw.*dst;
      auto  r_view = make_view(r, cw.nz, cw.ny, cw.nx);
      auto  z_view = make_view(z, cw.nz, cw.ny, cw.nx);

      const float64 dx2  = 1.0 / (cw.delx * cw.delx);
      const float64 dy2  = 1.0 / (cw.dely * cw.dely);
      const float64 dz2  = 1.0 / (cw.delz * cw.delz);
      const float64 ofdx = -dx2;
      const float64 ofdy = -dy2;
      const float64 ofdz = -dz2;

      for (int iz = cw.Lbz; iz <= cw.Ubz; ++iz) {
        for (int iy = cw.Lby; iy <= cw.Uby; ++iy) {
          for (int ix = cw.Lbx; ix <= cw.Ubx; ++ix) {
            const float64 diag = 2.0 * dx2 + 2.0 * dy2 + 2.0 * dz2;
            float64       sum  = r_view(iz, iy, ix);
            sum -= ofdx * (z_view(iz, iy, ix - 1) + z_view(iz, iy, ix + 1));
            sum -= ofdy * (z_view(iz, iy - 1, ix) + z_view(iz, iy + 1, ix));
            sum -= ofdz * (z_view(iz - 1, iy, ix) + z_view(iz + 1, iy, ix));
            float64 z_new      = (diag > 0.0) ? sum / diag : r_view(iz, iy, ix);
            z_view(iz, iy, ix) = (1.0 - omega) * z_view(iz, iy, ix) + omega * z_new;
          }
        }
      }
    }
  }

  void preconditioner_backward_1d(std::vector<float64> ChunkWork::*src,
                                  std::vector<float64> ChunkWork::*dst)
  {
    for (auto& cw : chunk_work) {
      auto& r      = cw.*src;
      auto& z      = cw.*dst;
      auto  r_view = make_view(r, cw.nz, cw.ny, cw.nx);
      auto  z_view = make_view(z, cw.nz, cw.ny, cw.nx);

      const float64 dx2  = 1.0 / (cw.delx * cw.delx);
      const float64 ofdx = -dx2;

      for (int iz = cw.Ubz; iz >= cw.Lbz; --iz) {
        for (int iy = cw.Uby; iy >= cw.Lby; --iy) {
          for (int ix = cw.Ubx; ix >= cw.Lbx; --ix) {
            const float64 diag = 2.0 * dx2;
            float64       sum  = r_view(iz, iy, ix);
            sum -= ofdx * (z_view(iz, iy, ix - 1) + z_view(iz, iy, ix + 1));
            float64 z_new      = (diag > 0.0) ? sum / diag : r_view(iz, iy, ix);
            z_view(iz, iy, ix) = (1.0 - omega) * z_view(iz, iy, ix) + omega * z_new;
          }
        }
      }
    }
  }

  void preconditioner_backward_2d(std::vector<float64> ChunkWork::*src,
                                  std::vector<float64> ChunkWork::*dst)
  {
    for (auto& cw : chunk_work) {
      auto& r      = cw.*src;
      auto& z      = cw.*dst;
      auto  r_view = make_view(r, cw.nz, cw.ny, cw.nx);
      auto  z_view = make_view(z, cw.nz, cw.ny, cw.nx);

      const float64 dx2  = 1.0 / (cw.delx * cw.delx);
      const float64 dy2  = 1.0 / (cw.dely * cw.dely);
      const float64 ofdx = -dx2;
      const float64 ofdy = -dy2;

      for (int iz = cw.Ubz; iz >= cw.Lbz; --iz) {
        for (int iy = cw.Uby; iy >= cw.Lby; --iy) {
          for (int ix = cw.Ubx; ix >= cw.Lbx; --ix) {
            const float64 diag = 2.0 * dx2 + 2.0 * dy2;
            float64       sum  = r_view(iz, iy, ix);
            sum -= ofdx * (z_view(iz, iy, ix - 1) + z_view(iz, iy, ix + 1));
            sum -= ofdy * (z_view(iz, iy - 1, ix) + z_view(iz, iy + 1, ix));
            float64 z_new      = (diag > 0.0) ? sum / diag : r_view(iz, iy, ix);
            z_view(iz, iy, ix) = (1.0 - omega) * z_view(iz, iy, ix) + omega * z_new;
          }
        }
      }
    }
  }

  void preconditioner_backward_3d(std::vector<float64> ChunkWork::*src,
                                  std::vector<float64> ChunkWork::*dst)
  {
    for (auto& cw : chunk_work) {
      auto& r      = cw.*src;
      auto& z      = cw.*dst;
      auto  r_view = make_view(r, cw.nz, cw.ny, cw.nx);
      auto  z_view = make_view(z, cw.nz, cw.ny, cw.nx);

      const float64 dx2  = 1.0 / (cw.delx * cw.delx);
      const float64 dy2  = 1.0 / (cw.dely * cw.dely);
      const float64 dz2  = 1.0 / (cw.delz * cw.delz);
      const float64 ofdx = -dx2;
      const float64 ofdy = -dy2;
      const float64 ofdz = -dz2;

      for (int iz = cw.Ubz; iz >= cw.Lbz; --iz) {
        for (int iy = cw.Uby; iy >= cw.Lby; --iy) {
          for (int ix = cw.Ubx; ix >= cw.Lbx; --ix) {
            const float64 diag = 2.0 * dx2 + 2.0 * dy2 + 2.0 * dz2;
            float64       sum  = r_view(iz, iy, ix);
            sum -= ofdx * (z_view(iz, iy, ix - 1) + z_view(iz, iy, ix + 1));
            sum -= ofdy * (z_view(iz, iy - 1, ix) + z_view(iz, iy + 1, ix));
            sum -= ofdz * (z_view(iz - 1, iy, ix) + z_view(iz + 1, iy, ix));
            float64 z_new      = (diag > 0.0) ? sum / diag : r_view(iz, iy, ix);
            z_view(iz, iy, ix) = (1.0 - omega) * z_view(iz, iy, ix) + omega * z_new;
          }
        }
      }
    }
  }
};

PicPoissonBasic::PicPoissonBasic(const nix::Dims3D& global_dims, float64 delh)
    : PicPoisson(global_dims, delh), impl(std::make_unique<Impl>(global_dims, delh))
{
}

PicPoissonBasic::~PicPoissonBasic() = default;

int PicPoissonBasic::update_mapping(elliptic::ChunkAccessor& accessor)
{
  return impl->update_mapping(accessor);
}

int PicPoissonBasic::copy_chunk_to_src(elliptic::ChunkAccessor& accessor)
{
  return impl->copy_chunk_to_src(accessor);
}

int PicPoissonBasic::copy_sol_to_chunk(elliptic::ChunkAccessor& accessor)
{
  return impl->copy_sol_to_chunk(accessor);
}

int PicPoissonBasic::set_option(const nlohmann::json& config)
{
  return impl->set_option(config);
}

int PicPoissonBasic::solve(elliptic::ChunkAccessor& accessor)
{
  return impl->solve(accessor);
}

int PicPoissonBasic::scatter_forward()
{
  return impl->scatter_forward();
}

int PicPoissonBasic::scatter_reverse()
{
  return impl->scatter_reverse();
}
