// -*- C++ -*-

///
/// Implementation of Space-Filing Curve
///
#include "sfc.hpp"

#include <algorithm>
#include <cstdlib>

namespace sfc
{
//
// internal helpers
//
void get_map1d(size_t Nx, int ncol, std::vector<int>& index, std::vector<int>& coord);

void gilbert2d(std::vector<int>& index, int Nx, int& id, int x, int y, int ax, int ay, int bx,
               int by);

void gilbert3d(std::vector<int>& index, int Ny, int Nx, int& id, int x, int y, int z, int ax,
               int ay, int az, int bx, int by, int bz, int cx, int cy, int cz);

inline int sign(const int x)
{
  return x == 0 ? 0 : (x > 0 ? +1 : -1);
}

inline void forward_id_2d(std::vector<int>& index, int Nx, int& id, int x, int y)
{
  index[y * Nx + x] = id;
  id++;
}

inline void forward_id_3d(std::vector<int>& index, int Ny, int Nx, int& id, int x, int y, int z)
{
  index[z * Ny * Nx + y * Nx + x] = id;
  id++;
}

//
// 1D SFC
//
void get_map1d(size_t Nx, int ncol, std::vector<int>& index, std::vector<int>& coord)
{
  for (size_t ix = 0; ix < Nx; ix++) {
    index[ix]        = static_cast<int>(ix);
    coord[ix * ncol] = static_cast<int>(ix);
  }
}

//
// 2D SFC
//
void get_map2d(size_t Ny, size_t Nx, int ncol, std::vector<int>& index, std::vector<int>& coord)
{
  if (Ny != 1 && Nx != 1) {
    int id = 0;
    int x  = 0;
    int y  = 0;

    if (Nx >= Ny) {
      gilbert2d(index, static_cast<int>(Nx), id, x, y, static_cast<int>(Nx), 0, 0,
                static_cast<int>(Ny));
    } else {
      gilbert2d(index, static_cast<int>(Nx), id, x, y, 0, static_cast<int>(Ny),
                static_cast<int>(Nx), 0);
    }
  } else if (Ny == 1 && Nx != 1) {
    std::vector<int> index1d(Nx, 0);
    get_map1d(Nx, ncol, index1d, coord);
    for (size_t ix = 0; ix < Nx; ix++) {
      index[0 * Nx + ix] = index1d[ix];
    }
  } else if (Ny != 1 && Nx == 1) {
    std::vector<int> index1d(Ny, 0);
    get_map1d(Ny, ncol, index1d, coord);
    for (size_t iy = 0; iy < Ny; iy++) {
      index[iy * Nx + 0] = index1d[iy];
    }
  } else {
    index[0] = 0;
  }

  for (size_t iy = 0; iy < Ny; iy++) {
    for (size_t ix = 0; ix < Nx; ix++) {
      int id               = index[iy * Nx + ix];
      coord[id * ncol + 0] = static_cast<int>(ix);
      coord[id * ncol + 1] = static_cast<int>(iy);
    }
  }
}

//
// 3D SFC
//
void get_map3d(size_t Nz, size_t Ny, size_t Nx, std::vector<int>& index, std::vector<int>& coord)
{
  if (Nz != 1 && Ny != 1 && Nx != 1) {
    int id = 0;
    int x = 0, y = 0, z = 0;

    if (Nx >= Ny && Nx >= Nz) {
      gilbert3d(index, static_cast<int>(Ny), static_cast<int>(Nx), id, x, y, z,
                static_cast<int>(Nx), 0, 0, 0, static_cast<int>(Ny), 0, 0, 0, static_cast<int>(Nz));
    } else if (Ny >= Nx && Ny >= Nz) {
      gilbert3d(index, static_cast<int>(Ny), static_cast<int>(Nx), id, x, y, z, 0,
                static_cast<int>(Ny), 0, static_cast<int>(Nx), 0, 0, 0, 0, static_cast<int>(Nz));
    } else if (Nz >= Nx && Nz >= Ny) {
      gilbert3d(index, static_cast<int>(Ny), static_cast<int>(Nx), id, x, y, z, 0, 0,
                static_cast<int>(Nz), static_cast<int>(Nx), 0, 0, 0, static_cast<int>(Ny), 0);
    }
  } else if (Nz == 1 && Ny == 1 && Nx != 1) {
    std::vector<int> index1d(Nx, 0);
    get_map1d(Nx, 3, index1d, coord);
    for (size_t ix = 0; ix < Nx; ix++) {
      index[ix] = index1d[ix];
    }
  } else if (Nz == 1 && Ny != 1 && Nx == 1) {
    std::vector<int> index1d(Ny, 0);
    get_map1d(Ny, 3, index1d, coord);
    for (size_t iy = 0; iy < Ny; iy++) {
      index[iy * Nx] = index1d[iy];
    }
  } else if (Nz != 1 && Ny == 1 && Nx == 1) {
    std::vector<int> index1d(Nz, 0);
    get_map1d(Nz, 3, index1d, coord);
    for (size_t iz = 0; iz < Nz; iz++) {
      index[iz * Ny * Nx] = index1d[iz];
    }
  } else if (Nz == 1 && Ny != 1 && Nx != 1) {
    std::vector<int> index2d(Ny * Nx, 0);
    get_map2d(Ny, Nx, 3, index2d, coord);
    for (size_t iy = 0; iy < Ny; iy++) {
      for (size_t ix = 0; ix < Nx; ix++) {
        index[iy * Nx + ix] = index2d[iy * Nx + ix];
      }
    }
  } else if (Nz != 1 && Ny == 1 && Nx != 1) {
    std::vector<int> index2d(Nz * Nx, 0);
    get_map2d(Nz, Nx, 3, index2d, coord);
    for (size_t iz = 0; iz < Nz; iz++) {
      for (size_t ix = 0; ix < Nx; ix++) {
        index[iz * Ny * Nx + ix] = index2d[iz * Nx + ix];
      }
    }
  } else if (Nz != 1 && Ny != 1 && Nx == 1) {
    std::vector<int> index2d(Nz * Ny, 0);
    get_map2d(Nz, Ny, 3, index2d, coord);
    for (size_t iz = 0; iz < Nz; iz++) {
      for (size_t iy = 0; iy < Ny; iy++) {
        index[iz * Ny * Nx + iy * Nx] = index2d[iz * Ny + iy];
      }
    }
  } else {
    index[0] = 0;
  }

  for (size_t iz = 0; iz < Nz; iz++) {
    for (size_t iy = 0; iy < Ny; iy++) {
      for (size_t ix = 0; ix < Nx; ix++) {
        int id            = index[iz * Ny * Nx + iy * Nx + ix];
        coord[id * 3 + 0] = static_cast<int>(ix);
        coord[id * 3 + 1] = static_cast<int>(iy);
        coord[id * 3 + 2] = static_cast<int>(iz);
      }
    }
  }
}

//
// validation
//
bool check_index(const std::vector<int>& index)
{
  bool status = true;

  std::vector<int> flat(index.begin(), index.end());
  std::sort(flat.begin(), flat.end());
  for (size_t id = 0; id < flat.size(); id++) {
    status = status & (static_cast<int>(id) == flat[id]);
  }

  return status;
}

bool check_locality2d(const std::vector<int>& coord, size_t N, int distmax2)
{
  bool status = true;

  int dx = coord[0 * 2 + 0];
  int dy = coord[0 * 2 + 1];
  for (size_t id = 1; id < N; id++) {
    int ix = coord[id * 2 + 0];
    int iy = coord[id * 2 + 1];
    dx     = dx - ix;
    dy     = dy - iy;
    status = status & (dx * dx + dy * dy <= distmax2);
    dx     = ix;
    dy     = iy;
  }

  return status;
}

bool check_locality3d(const std::vector<int>& coord, size_t N, int distmax2)
{
  bool status = true;

  int dx = coord[0 * 3 + 0];
  int dy = coord[0 * 3 + 1];
  int dz = coord[0 * 3 + 2];
  for (size_t id = 1; id < N; id++) {
    int ix = coord[id * 3 + 0];
    int iy = coord[id * 3 + 1];
    int iz = coord[id * 3 + 2];
    dx     = dx - ix;
    dy     = dy - iy;
    dz     = dz - iz;
    status = status & (dx * dx + dy * dy + dz * dz <= distmax2);
    dx     = ix;
    dy     = iy;
    dz     = iz;
  }

  return status;
}

//
// Gilbert curve recursion (2D)
//
void gilbert2d(std::vector<int>& index, int Nx, int& id, int x, int y, int ax, int ay, int bx,
               int by)
{
  int w   = std::abs(ax + ay);
  int h   = std::abs(bx + by);
  int dax = sign(ax);
  int day = sign(ay);
  int dbx = sign(bx);
  int dby = sign(by);

  //
  // trivial path
  //
  {
    if (h == 1) {
      for (int i = 0; i < w; i++) {
        forward_id_2d(index, Nx, id, x, y);
        x += dax;
        y += day;
      }
      return;
    }

    if (w == 1) {
      for (int i = 0; i < h; i++) {
        forward_id_2d(index, Nx, id, x, y);
        x += dbx;
        y += dby;
      }
      return;
    }
  }

  //
  // recursive call
  //
  {
    int ax2 = ax / 2;
    int ay2 = ay / 2;
    int bx2 = bx / 2;
    int by2 = by / 2;
    int w2  = abs(ax2 + ay2);
    int h2  = abs(bx2 + by2);

    if (2 * w > 3 * h) {
      if ((w2 % 2) && (w > 2)) {
        ax2 = ax2 + dax;
        ay2 = ay2 + day;
      }

      int ax3 = ax - ax2;
      int ay3 = ay - ay2;

      gilbert2d(index, Nx, id, x, y, +ax2, +ay2, +bx, +by);
      x += ax2;
      y += ay2;

      gilbert2d(index, Nx, id, x, y, +ax3, +ay3, +bx, +by);
    } else {
      if ((h2 % 2) && (h > 2)) {
        bx2 = bx2 + dbx;
        by2 = by2 + dby;
      }

      int ax3 = ax - ax2;
      int ay3 = ay - ay2;
      int bx3 = bx - bx2;
      int by3 = by - by2;

      gilbert2d(index, Nx, id, x, y, +bx2, +by2, +ax2, +ay2);
      x += bx2;
      y += by2;

      gilbert2d(index, Nx, id, x, y, +ax, +ay, +bx3, +by3);
      x += ax;
      y += ay;

      x -= dax + dbx;
      y -= day + dby;
      gilbert2d(index, Nx, id, x, y, -bx2, -by2, -ax3, -ay3);
    }
  }
}

//
// Gilbert curve recursion (3D)
//
void gilbert3d(std::vector<int>& index, int Ny, int Nx, int& id, int x, int y, int z, int ax,
               int ay, int az, int bx, int by, int bz, int cx, int cy, int cz)
{
  int w   = std::abs(ax + ay + az);
  int h   = std::abs(bx + by + bz);
  int d   = std::abs(cx + cy + cz);
  int dax = sign(ax);
  int day = sign(ay);
  int daz = sign(az);
  int dbx = sign(bx);
  int dby = sign(by);
  int dbz = sign(bz);
  int dcx = sign(cx);
  int dcy = sign(cy);
  int dcz = sign(cz);

  //
  // trivial path
  //
  {
    if (h == 1 && d == 1) {
      for (int i = 0; i < w; i++) {
        forward_id_3d(index, Ny, Nx, id, x, y, z);
        x += dax;
        y += day;
        z += daz;
      }
      return;
    }

    if (d == 1 && w == 1) {
      for (int i = 0; i < h; i++) {
        forward_id_3d(index, Ny, Nx, id, x, y, z);
        x += dbx;
        y += dby;
        z += dbz;
      }
      return;
    }

    if (w == 1 && h == 1) {
      for (int i = 0; i < d; i++) {
        forward_id_3d(index, Ny, Nx, id, x, y, z);
        x += dcx;
        y += dcy;
        z += dcz;
      }
      return;
    }
  }

  //
  // recursive call
  //
  {
    int ax2 = ax / 2;
    int ay2 = ay / 2;
    int az2 = az / 2;
    int bx2 = bx / 2;
    int by2 = by / 2;
    int bz2 = bz / 2;
    int cx2 = cx / 2;
    int cy2 = cy / 2;
    int cz2 = cz / 2;
    int w2  = std::abs(ax2 + ay2 + az2);
    int h2  = std::abs(bx2 + by2 + bz2);
    int d2  = std::abs(cx2 + cy2 + cz2);

    if ((w2 % 2) && (w > 2)) {
      ax2 = ax2 + dax;
      ay2 = ay2 + day;
      az2 = az2 + daz;
    }

    if ((h2 % 2) && (h > 2)) {
      bx2 = bx2 + dbx;
      by2 = by2 + dby;
      bz2 = bz2 + dbz;
    }

    if ((d2 % 2) && (d > 2)) {
      cx2 = cx2 + dcx;
      cy2 = cy2 + dcy;
      cz2 = cz2 + dcz;
    }

    if ((2 * w > 3 * h) && (2 * w > 3 * d)) {
      int ax3 = ax - ax2;
      int ay3 = ay - ay2;
      int az3 = az - az2;

      gilbert3d(index, Ny, Nx, id, x, y, z, +ax2, +ay2, +az2, +bx, +by, +bz, +cx, +cy, +cz);
      x += ax2;
      y += ay2;
      z += az2;

      gilbert3d(index, Ny, Nx, id, x, y, z, +ax3, +ay3, +az3, +bx, +by, +bz, +cx, +cy, +cz);
    } else if (3 * h > 4 * d) {
      int ax3 = ax - ax2;
      int ay3 = ay - ay2;
      int az3 = az - az2;
      int bx3 = bx - bx2;
      int by3 = by - by2;
      int bz3 = bz - bz2;

      gilbert3d(index, Ny, Nx, id, x, y, z, +bx2, +by2, +bz2, +cx, +cy, +cz, +ax2, +ay2, +az2);
      x += bx2;
      y += by2;
      z += bz2;

      gilbert3d(index, Ny, Nx, id, x, y, z, +ax, +ay, +az, +bx3, +by3, +bz3, +cx, +cy, +cz);
      x += ax;
      y += ay;
      z += az;

      x -= dax + dbx;
      y -= day + dby;
      z -= daz + dbz;
      gilbert3d(index, Ny, Nx, id, x, y, z, -bx2, -by2, -bz2, +cx, +cy, +cz, -ax3, -ay3, -az3);
    } else if (3 * d > 4 * h) {
      int ax3 = ax - ax2;
      int ay3 = ay - ay2;
      int az3 = az - az2;
      int cx3 = cx - cx2;
      int cy3 = cy - cy2;
      int cz3 = cz - cz2;

      gilbert3d(index, Ny, Nx, id, x, y, z, +cx2, +cy2, +cz2, +ax2, +ay2, +az2, +bx, +by, +bz);
      x += cx2;
      y += cy2;
      z += cz2;

      gilbert3d(index, Ny, Nx, id, x, y, z, +ax, +ay, +az, +bx, +by, +bz, +cx3, +cy3, +cz3);
      x += ax;
      y += ay;
      z += az;

      x -= dax + dcx;
      y -= day + dcy;
      z -= daz + dcz;
      gilbert3d(index, Ny, Nx, id, x, y, z, -cx2, -cy2, -cz2, -ax3, -ay3, -az3, +bx, +by, +bz);
    } else {
      int ax3 = ax - ax2;
      int ay3 = ay - ay2;
      int az3 = az - az2;
      int bx3 = bx - bx2;
      int by3 = by - by2;
      int bz3 = bz - bz2;
      int cx3 = cx - cx2;
      int cy3 = cy - cy2;
      int cz3 = cz - cz2;

      gilbert3d(index, Ny, Nx, id, x, y, z, +bx2, +by2, +bz2, +cx2, +cy2, +cz2, +ax2, +ay2, +az2);
      x += bx2;
      y += by2;
      z += bz2;

      gilbert3d(index, Ny, Nx, id, x, y, z, +cx, +cy, +cz, +ax2, +ay2, +az2, +bx3, +by3, +bz3);
      x += cx;
      y += cy;
      z += cz;

      x -= dbx + dcx;
      y -= dby + dcy;
      z -= dbz + dcz;
      gilbert3d(index, Ny, Nx, id, x, y, z, +ax, +ay, +az, -bx2, -by2, -bz2, -cx3, -cy3, -cz3);
      x += ax;
      y += ay;
      z += az;

      x -= dax - dbx;
      y -= day - dby;
      z -= daz - dbz;
      gilbert3d(index, Ny, Nx, id, x, y, z, -cx, -cy, -cz, -ax3, -ay3, -az3, +bx3, +by3, +bz3);
      x -= cx;
      y -= cy;
      z -= cz;

      x -= dbx - dcx;
      y -= dby - dcy;
      z -= dbz - dcz;
      gilbert3d(index, Ny, Nx, id, x, y, z, -bx2, -by2, -bz2, +cx2, +cy2, +cz2, -ax3, -ay3, -az3);
    }
  }
}

} // namespace sfc
