// -*- C++ -*-

///
/// Implementation of Space-Filing Curve
///
#include "sfc.hpp"

namespace sfc
{
void gilbert2d(array2d& index, int& id, int x, int y, int ax, int ay, int bx, int by);
void gilbert3d(array3d& index, int& id, int x, int y, int z, int ax, int ay, int az, int bx, int by,
               int bz, int cx, int cy, int cz);

inline int sign(const int x)
{
  return x == 0 ? 0 : (x > 0 ? +1 : -1);
}

inline void forward_id_2d(array2d& index, int& id, int x, int y)
{
  index.at(y, x) = id;
  id++;
}

inline void forward_id_3d(array3d& index, int& id, int x, int y, int z)
{
  index.at(z, y, x) = id;
  id++;
}

void get_map1d(size_t Nx, array1d& index, array2d& coord)
{
  for (int ix = 0; ix < Nx; ix++) {
    index(ix)    = ix;
    coord(ix, 0) = ix;
  }
}

void get_map2d(size_t Ny, size_t Nx, array2d& index, array2d& coord)
{
  // calculate coordiante to ID map
  if (Ny != 1 && Nx != 1) {
    // 2D mapping
    int id = 0;
    int x  = 0;
    int y  = 0;

    if (Nx >= Ny) {
      gilbert2d(index, id, x, y, Nx, 0, 0, Ny);
    } else {
      gilbert2d(index, id, x, y, 0, Ny, Nx, 0);
    }
  } else if (Ny == 1 && Nx != 1) {
    // 1D mapping along x
    array1d index1d = xt::zeros<int>({Nx});
    get_map1d(Nx, index1d, coord);
    xt::view(index, 1, xt::all()) = index1d;
  } else if (Ny != 1 && Nx == 1) {
    // 1D mapping along y
    array1d index1d = xt::zeros<int>({Ny});
    get_map1d(Ny, index1d, coord);
    xt::view(index, xt::all(), 1) = index1d;
  } else {
    // Nx = Ny = 1
    index(0, 0) = 0;
  }

  // calculate ID to coordiante map
  for (int iy = 0; iy < Ny; iy++) {
    for (int ix = 0; ix < Nx; ix++) {
      int id          = index(iy, ix);
      coord.at(id, 0) = ix;
      coord.at(id, 1) = iy;
    }
  }
}

void get_map3d(size_t Nz, size_t Ny, size_t Nx, array3d& index, array2d& coord)
{
  // calculate coordiante to ID map
  if (Nz != 1 && Ny != 1 && Nx != 1) {
    // 3D mapping
    int id = 0;
    int x  = 0;
    int y  = 0;
    int z  = 0;

    if (Nx >= Ny && Nx >= Nz) {
      gilbert3d(index, id, x, y, z, Nx, 0, 0, 0, Ny, 0, 0, 0, Nz);
    } else if (Ny >= Nx && Ny >= Nz) {
      gilbert3d(index, id, x, y, z, 0, Ny, 0, Nx, 0, 0, 0, 0, Nz);
    } else if (Nz >= Nx && Nz >= Ny) {
      gilbert3d(index, id, x, y, z, 0, 0, Nz, Nx, 0, 0, 0, Ny, 0);
    }
  } else if (Nz == 1 && Ny == 1 && Nx != 1) {
    // 1D mapping along x
    array1d index1d = xt::zeros<int>({Nx});
    get_map1d(Nx, index1d, coord);
    xt::view(index, 1, 1, xt::all()) = index1d;
  } else if (Nz == 1 && Ny != 1 && Nx == 1) {
    // 1D mapping along y
    array1d index1d = xt::zeros<int>({Ny});
    get_map1d(Ny, index1d, coord);
    xt::view(index, 1, xt::all(), 1) = index1d;
  } else if (Nz != 1 && Ny == 1 && Nx == 1) {
    // 1D mapping along z
    array1d index1d = xt::zeros<int>({Nz});
    get_map1d(Nz, index1d, coord);
    xt::view(index, xt::all(), 1, 1) = index1d;
  } else if (Nz == 1 && Ny != 1 && Nx != 1) {
    // 2D mapping along x, y
    array2d index2d = xt::zeros<int>({Ny, Nx});
    get_map2d(Ny, Nx, index2d, coord);
    xt::view(index, 1, xt::all(), xt::all()) = index2d;
  } else if (Nz != 1 && Ny == 1 && Nx != 1) {
    // 2D mapping along x, z
    array2d index2d = xt::zeros<int>({Nz, Nx});
    get_map2d(Nz, Nx, index2d, coord);
    xt::view(index, xt::all(), 1, xt::all()) = index2d;
  } else if (Nz != 1 && Ny != 1 && Nx == 1) {
    // 2D mapping along y, z
    array2d index2d = xt::zeros<int>({Nz, Ny});
    get_map2d(Nz, Ny, index2d, coord);
    xt::view(index, xt::all(), xt::all(), 1) = index2d;
  } else {
    // Nx = Ny = Nz = 1
    index(0, 0, 0) = 0;
  }

  // calculate ID to coordinate map
  for (int iz = 0; iz < Nz; iz++) {
    for (int iy = 0; iy < Ny; iy++) {
      for (int ix = 0; ix < Nx; ix++) {
        int id          = index.at(iz, iy, ix);
        coord.at(id, 0) = ix;
        coord.at(id, 1) = iy;
        coord.at(id, 2) = iz;
      }
    }
  }
}

template <typename T>
bool check_index(T& index)
{
  bool status = true;

  auto flatindex = xt::sort(xt::flatten(index));
  for (int id = 0; id < flatindex.size(); id++) {
    status = status & (id == flatindex(id));
  }

  return status;
}

bool check_locality2d(array2d& coord, const int distmax2)
{
  bool status = true;

  int dx, dy;
  dx = coord(0, 0);
  dy = coord(0, 1);
  for (int id = 1; id < coord.shape(0); id++) {
    int ix = coord(id, 0);
    int iy = coord(id, 1);
    dx     = dx - ix;
    dy     = dy - iy;
    status = status & (dx * dx + dy * dy <= distmax2);
    dx     = ix;
    dy     = iy;
  }

  return status;
}

bool check_locality3d(array2d& coord, const int distmax2)
{
  bool status = true;

  int dx, dy, dz;
  dx = coord(0, 0);
  dy = coord(0, 1);
  dz = coord(0, 2);
  for (int id = 1; id < coord.shape(0); id++) {
    int ix = coord(id, 0);
    int iy = coord(id, 1);
    int iz = coord(id, 2);
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

void gilbert2d(array2d& index, int& id, int x, int y, int ax, int ay, int bx, int by)
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
    // straight segment in x
    if (h == 1) {
      for (int i = 0; i < w; i++) {
        forward_id_2d(index, id, x, y);
        x += dax;
        y += day;
      }
      return;
    }

    // straight segment in y
    if (w == 1) {
      for (int i = 0; i < h; i++) {
        forward_id_2d(index, id, x, y);
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
      //
      // one-dimensional spliting in x
      //
      if ((w2 % 2) && (w > 2)) {
        ax2 = ax2 + dax;
        ay2 = ay2 + day;
      }

      int ax3 = ax - ax2;
      int ay3 = ay - ay2;

      // first step
      gilbert2d(index, id, x, y, +ax2, +ay2, +bx, +by);
      x += ax2;
      y += ay2;

      // second step
      gilbert2d(index, id, x, y, +ax3, +ay3, +bx, +by);
    } else {
      //
      // two-dimensional spliting
      //
      if ((h2 % 2) && (h > 2)) {
        bx2 = bx2 + dbx;
        by2 = by2 + dby;
      }

      int ax3 = ax - ax2;
      int ay3 = ay - ay2;
      int bx3 = bx - bx2;
      int by3 = by - by2;

      // first step
      gilbert2d(index, id, x, y, +bx2, +by2, +ax2, +ay2);
      x += bx2;
      y += by2;

      // second step
      gilbert2d(index, id, x, y, +ax, +ay, +bx3, +by3);
      x += ax;
      y += ay;

      // third step
      x -= dax + dbx;
      y -= day + dby;
      gilbert2d(index, id, x, y, -bx2, -by2, -ax3, -ay3);
    }
  }
}

void gilbert3d(array3d& index, int& id, int x, int y, int z, int ax, int ay, int az, int bx, int by,
               int bz, int cx, int cy, int cz)
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
    // straight segment in x
    if (h == 1 && d == 1) {
      for (int i = 0; i < w; i++) {
        forward_id_3d(index, id, x, y, z);
        x += dax;
        y += day;
        z += daz;
      }
      return;
    }

    // straight segment in y
    if (d == 1 && w == 1) {
      for (int i = 0; i < h; i++) {
        forward_id_3d(index, id, x, y, z);
        x += dbx;
        y += dby;
        z += dbz;
      }
      return;
    }

    // straight segment in z
    if (w == 1 && h == 1) {
      for (int i = 0; i < d; i++) {
        forward_id_3d(index, id, x, y, z);
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
      //
      // one-dimensional spliting in x
      //
      int ax3 = ax - ax2;
      int ay3 = ay - ay2;
      int az3 = az - az2;

      // first step
      gilbert3d(index, id, x, y, z, +ax2, +ay2, +az2, +bx, +by, +bz, +cx, +cy, +cz);
      x += ax2;
      y += ay2;
      z += az2;

      // second step
      gilbert3d(index, id, x, y, z, +ax3, +ay3, +az3, +bx, +by, +bz, +cx, +cy, +cz);
    } else if (3 * h > 4 * d) {
      //
      // two-dimensional spliting in x-y
      //
      int ax3 = ax - ax2;
      int ay3 = ay - ay2;
      int az3 = az - az2;
      int bx3 = bx - bx2;
      int by3 = by - by2;
      int bz3 = bz - bz2;

      // first step
      gilbert3d(index, id, x, y, z, +bx2, +by2, +bz2, +cx, +cy, +cz, +ax2, +ay2, +az2);
      x += bx2;
      y += by2;
      z += bz2;

      // second step
      gilbert3d(index, id, x, y, z, +ax, +ay, +az, +bx3, +by3, +bz3, +cx, +cy, +cz);
      x += ax;
      y += ay;
      z += az;

      // third step
      x -= dax + dbx;
      y -= day + dby;
      z -= daz + dbz;
      gilbert3d(index, id, x, y, z, -bx2, -by2, -bz2, +cx, +cy, +cz, -ax3, -ay3, -az3);
    } else if (3 * d > 4 * h) {
      //
      // two-dimensional spliting in x-z
      //
      int ax3 = ax - ax2;
      int ay3 = ay - ay2;
      int az3 = az - az2;
      int cx3 = cx - cx2;
      int cy3 = cy - cy2;
      int cz3 = cz - cz2;

      // first step
      gilbert3d(index, id, x, y, z, +cx2, +cy2, +cz2, +ax2, +ay2, +az2, +bx, +by, +bz);
      x += cx2;
      y += cy2;
      z += cz2;

      // second step
      gilbert3d(index, id, x, y, z, +ax, +ay, +az, +bx, +by, +bz, +cx3, +cy3, +cz3);
      x += ax;
      y += ay;
      z += az;

      // third step
      x -= dax + dcx;
      y -= day + dcy;
      z -= daz + dcz;
      gilbert3d(index, id, x, y, z, -cx2, -cy2, -cz2, -ax3, -ay3, -az3, +bx, +by, +bz);
    } else {
      //
      // fully three-dimensional spliting
      //
      int ax3 = ax - ax2;
      int ay3 = ay - ay2;
      int az3 = az - az2;
      int bx3 = bx - bx2;
      int by3 = by - by2;
      int bz3 = bz - bz2;
      int cx3 = cx - cx2;
      int cy3 = cy - cy2;
      int cz3 = cz - cz2;

      // first step
      gilbert3d(index, id, x, y, z, +bx2, +by2, +bz2, +cx2, +cy2, +cz2, +ax2, +ay2, +az2);
      x += bx2;
      y += by2;
      z += bz2;

      // second step
      gilbert3d(index, id, x, y, z, +cx, +cy, +cz, +ax2, +ay2, +az2, +bx3, +by3, +bz3);
      x += cx;
      y += cy;
      z += cz;

      // third step
      x -= dbx + dcx;
      y -= dby + dcy;
      z -= dbz + dcz;
      gilbert3d(index, id, x, y, z, +ax, +ay, +az, -bx2, -by2, -bz2, -cx3, -cy3, -cz3);
      x += ax;
      y += ay;
      z += az;

      // fourth step
      x -= dax - dbx;
      y -= day - dby;
      z -= daz - dbz;
      gilbert3d(index, id, x, y, z, -cx, -cy, -cz, -ax3, -ay3, -az3, +bx3, +by3, +bz3);
      x -= cx;
      y -= cy;
      z -= cz;

      // fifth step
      x -= dbx - dcx;
      y -= dby - dcy;
      z -= dbz - dcz;
      gilbert3d(index, id, x, y, z, -bx2, -by2, -bz2, +cx2, +cy2, +cz2, -ax3, -ay3, -az3);
    }
  }
}

template bool check_index(array1d& index);
template bool check_index(array2d& index);
template bool check_index(array3d& index);

} // namespace sfc

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
