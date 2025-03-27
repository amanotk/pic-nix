// -*- C++ -*-

#include "nixio.hpp"
#include "debug.hpp"

namespace nixio
{
template <typename T>
T get_size(const int32_t ndim, const T shape[])
{
  T size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  return size;
}

// calculate offset
void calculate_global_offset(size_t lsize, size_t* offset, size_t* gsize)
{
  int nprocess;
  int thisrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
  MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);

  std::unique_ptr<size_t[]> buffer = std::make_unique<size_t[]>(nprocess);

  MPI_Allgather(&lsize, 1, MPI_INT64_T, buffer.get(), 1, MPI_INT64_T, MPI_COMM_WORLD);

  *offset = 0;
  for (int i = 1; i <= thisrank; i++) {
    *offset += buffer[i - 1];
  }

  *gsize = 0;
  for (int i = 0; i < nprocess; i++) {
    *gsize += buffer[i];
  }
}

void open_file(const char* filename, MPI_File* fh, size_t* disp, const char* mode)
{
  int status;

  switch (mode[0]) {
  case 'r':
    // read only
    status = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, fh);
    if (status != MPI_SUCCESS) {
      ERROR << tfm::format("Failed to open file: %s", filename);
    }

    // set pointer to the beginning
    *disp = 0;
    MPI_File_seek(*fh, *disp, MPI_SEEK_SET);

    break;
  case 'w':
    // write only
    status = MPI_File_delete(filename, MPI_INFO_NULL);
    status = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                           MPI_INFO_NULL, fh);
    if (status != MPI_SUCCESS) {
      ERROR << tfm::format("Failed to open file: %s", filename);
    }

    // set pointer to the beginning
    *disp = 0;
    MPI_File_seek(*fh, *disp, MPI_SEEK_SET);

    break;
  case 'a':
    // append
    status = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                           MPI_INFO_NULL, fh);
    if (status != MPI_SUCCESS) {
      ERROR << tfm::format("Failed to open file: %s", filename);
    }

    //
    // <<< NOTE >>>
    // This implementation for setting the file pointer to the end of the file avoids an erroneous
    // behavior of MPI_File_seek() provided by OpenMPI.
    // See: https://github.com/open-mpi/ompi/issues/8266
    //
    MPI_Offset pos;
    MPI_File_get_size(*fh, &pos);
    MPI_File_seek(*fh, pos, MPI_SEEK_SET);
    *disp = static_cast<size_t>(pos);

    break;
  default:
    ERROR << tfm::format("No such mode available");
  }
}

void close_file(MPI_File* fh)
{
  MPI_File_close(fh);
}

// collective read/write with hindexed type
size_t readwrite_contiguous(MPI_File* fh, size_t* disp, void* data, const size_t offset,
                            const size_t size, const int32_t elembyte, const int32_t packbyte,
                            MPI_Request* req, const int mode)
{
  MPI_Datatype ptype, ftype;
  MPI_Aint     packed_offset[1];
  int32_t      packed_size[1];

  packed_offset[0] = elembyte * offset;
  packed_size[0]   = static_cast<int32_t>(size * elembyte / packbyte);

  MPI_Type_contiguous(packbyte, MPI_BYTE, &ptype);
  MPI_Type_commit(&ptype);

  MPI_Type_create_hindexed(1, packed_size, packed_offset, ptype, &ftype);
  MPI_Type_commit(&ftype);

  MPI_File_set_view(*fh, *disp, ptype, ftype, "native", MPI_INFO_NULL);

  switch (mode) {
  case +1:
    // read
    MPI_File_iread_all(*fh, data, packed_size[0], ptype, req);
    break;
  case -1:
    // write
    MPI_File_iwrite_all(*fh, data, packed_size[0], ptype, req);
    break;
  default:
    ERROR << tfm::format("No such mode available");
  }

  MPI_Type_free(&ptype);
  MPI_Type_free(&ftype);

  return size * elembyte;
}

// non-collective read/write
size_t readwrite_contiguous_at(MPI_File* fh, size_t* disp, void* data, const size_t size,
                               const int32_t elembyte, const int32_t packbyte, MPI_Request* req,
                               const int mode)
{
  MPI_Offset   pos;
  MPI_Datatype ptype;
  int32_t      psize;

  // seek to given position
  MPI_File_seek(*fh, *disp, MPI_SEEK_SET);
  MPI_File_get_position(*fh, &pos);

  psize = static_cast<int32_t>(size * elembyte / packbyte);
  MPI_Type_contiguous(packbyte, MPI_BYTE, &ptype);
  MPI_Type_commit(&ptype);

  switch (mode) {
  case +1:
    // read
    MPI_File_iread_at(*fh, pos, data, psize, ptype, req);
    break;
  case -1:
    // write
    MPI_File_iwrite_at(*fh, pos, data, psize, ptype, req);
    break;
  default:
    ERROR << tfm::format("No such mode available");
  }

  MPI_Type_free(&ptype);

  return size * elembyte;
}

// collective read/write with subarray type
size_t readwrite_subarray(MPI_File* fh, size_t* disp, void* data, const int32_t ndim,
                          const int32_t gshape[], const int32_t lshape[], const int32_t offset[],
                          const int32_t elembyte, MPI_Request* req, const int mode, const int order)
{
  MPI_Datatype ptype, ftype;
  int          count = get_size(ndim, lshape);

  MPI_Type_contiguous(elembyte, MPI_BYTE, &ptype);
  MPI_Type_commit(&ptype);

  MPI_Type_create_subarray(ndim, gshape, lshape, offset, order, ptype, &ftype);
  MPI_Type_commit(&ftype);

  MPI_File_set_view(*fh, *disp, ptype, ftype, "native", MPI_INFO_NULL);

  switch (mode) {
  case +1:
    // read
    MPI_File_iread_all(*fh, data, count, ptype, req);
    break;
  case -1:
    // write
    MPI_File_iwrite_all(*fh, data, count, ptype, req);
    break;
  default:
    ERROR << tfm::format("No such mode available");
  }

  MPI_Type_free(&ptype);
  MPI_Type_free(&ftype);

  return count * elembyte;
}

size_t read_single(MPI_File* fh, size_t* disp, void* data, const size_t size, MPI_Request* req)
{
  MPI_File_iread_at(*fh, *disp, data, size, MPI_BYTE, req);

  *disp += size;

  return size;
}

size_t write_single(MPI_File* fh, size_t* disp, void* data, const size_t size, MPI_Request* req)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    MPI_Status status;
    MPI_File_iwrite_at(*fh, *disp, data, size, MPI_BYTE, req);
  } else {
    *req = MPI_REQUEST_NULL;
  }

  *disp += size;

  return size;
}

size_t read_contiguous(MPI_File* fh, size_t* disp, void* data, const size_t size,
                       const int32_t elembyte, const int32_t packbyte, MPI_Request* req)
{
  size_t  gsize, offset;
  int32_t pbyte;

  if (packbyte < 0) {
    pbyte = elembyte;
  } else {
    pbyte = packbyte;
  }

  // calculate offset
  calculate_global_offset(size, &offset, &gsize);

  // read from disk
  readwrite_contiguous(fh, disp, data, offset, size, elembyte, pbyte, req, +1);

  *disp += gsize * elembyte;

  return gsize * elembyte;
}

size_t write_contiguous(MPI_File* fh, size_t* disp, void* data, const size_t size,
                        const int32_t elembyte, const int32_t packbyte, MPI_Request* req)
{
  size_t  gsize, offset;
  int32_t pbyte;

  if (packbyte < 0) {
    pbyte = elembyte;
  } else {
    pbyte = packbyte;
  }

  // calculate offset
  calculate_global_offset(size, &offset, &gsize);

  // write to disk
  readwrite_contiguous(fh, disp, data, offset, size, elembyte, pbyte, req, -1);

  *disp += gsize * elembyte;

  return gsize * elembyte;
}

size_t read_contiguous_at(MPI_File* fh, size_t* disp, void* data, const size_t size,
                          const int32_t elembyte, MPI_Request* req)
{
  return readwrite_contiguous_at(fh, disp, data, size, elembyte, elembyte, req, +1);
}

size_t write_contiguous_at(MPI_File* fh, size_t* disp, void* data, const size_t size,
                           const int32_t elembyte, MPI_Request* req)
{
  return readwrite_contiguous_at(fh, disp, data, size, elembyte, elembyte, req, -1);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
size_t read_subarray(MPI_File* fh, size_t* disp, void* data, const T1 ndim, const T2 gshape[],
                     const T3 lshape[], const T4 offset[], const T5 elembyte, MPI_Request* req,
                     const int order)
{
  if (order != MPI_ORDER_C && order != MPI_ORDER_FORTRAN) {
    ERROR << tfm::format(" No such order available");
  }

  // convert to int32_t for MPI call
  int32_t nd = static_cast<int32_t>(ndim);
  int32_t eb = static_cast<int32_t>(elembyte);
  int32_t gs[nd], ls[nd], os[nd];

  size_t size = 1;
  for (int i = 0; i < nd; i++) {
    gs[i] = static_cast<int32_t>(gshape[i]);
    ls[i] = static_cast<int32_t>(lshape[i]);
    os[i] = static_cast<int32_t>(offset[i]);
    size *= gshape[i];
  }

  readwrite_subarray(fh, disp, data, nd, gs, ls, os, eb, req, +1, order);
  *disp += size * elembyte;

  return size * elembyte;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
size_t write_subarray(MPI_File* fh, size_t* disp, void* data, const T1 ndim, const T2 gshape[],
                      const T3 lshape[], const T4 offset[], const T5 elembyte, MPI_Request* req,
                      const int order)
{
  if (order != MPI_ORDER_C && order != MPI_ORDER_FORTRAN) {
    ERROR << tfm::format("No such order available");
  }

  // convert to int32_t for MPI call
  int32_t nd = static_cast<int32_t>(ndim);
  int32_t eb = static_cast<int32_t>(elembyte);
  int32_t gs[nd], ls[nd], os[nd];

  size_t size = 1;
  for (int i = 0; i < nd; i++) {
    gs[i] = static_cast<int32_t>(gshape[i]);
    ls[i] = static_cast<int32_t>(lshape[i]);
    os[i] = static_cast<int32_t>(offset[i]);
    size *= gshape[i];
  }

  readwrite_subarray(fh, disp, data, nd, gs, ls, os, eb, req, -1, order);
  *disp += size * elembyte;

  return size * elembyte;
}

void put_metadata(json& obj, string name, string dtype, string desc, const size_t disp,
                  const size_t size, const int32_t ndim, const int32_t dims[])
{
  obj[name]["datatype"]    = dtype;
  obj[name]["description"] = desc;
  obj[name]["offset"]      = disp;
  obj[name]["size"]        = size;
  obj[name]["ndim"]        = ndim;
  obj[name]["shape"]       = json::array();

  for (int i = 0; i < ndim; i++) {
    obj[name]["shape"].push_back(dims[i]);
  }
}

void put_metadata(json& obj, string name, string dtype, string desc, const size_t disp,
                  const size_t size)
{
  const int ndim    = 1;
  const int dims[1] = {1};
  put_metadata(obj, name, dtype, desc, disp, size, ndim, dims);
}

void get_metadata(json& obj, string name, string& dtype, string& desc, size_t& disp, size_t& size,
                  int32_t& ndim, int32_t dims[])
{
  dtype = obj[name]["datatype"].get<string>();
  desc  = obj[name]["description"].get<string>();
  disp  = obj[name]["offset"].get<size_t>();
  size  = obj[name]["size"].get<size_t>();
  ndim  = obj[name]["ndim"].get<int>();

  auto v = obj[name]["shape"];
  for (int i = 0; i < ndim; i++) {
    dims[i] = v[i].get<int>();
  }
}

void get_metadata(json& obj, string name, string& dtype, string& desc, size_t& disp, size_t& size)
{
  int32_t ndim;
  int32_t dims[1];
  get_metadata(obj, name, dtype, desc, disp, size, ndim, dims);
}

template <typename T>
void get_attribute(json& obj, string name, size_t& disp, T& data)
{
  string dtype;
  string desc;
  size_t size;
  get_metadata(obj, name, dtype, desc, disp, size);
  data = obj[name]["data"].get<T>();
}

template <typename T>
void get_attribute(json& obj, string name, size_t& disp, int32_t length, T* data)
{
  string dtype;
  string desc;
  size_t size;
  get_metadata(obj, name, dtype, desc, disp, size);

  std::vector<T> vec = obj[name]["data"].get<std::vector<T>>();
  for (int i = 0; i < length; i++) {
    data[i] = vec[i];
  }
}

void put_attribute(json& obj, string name, const size_t disp, const int32_t data)
{
  put_metadata(obj, name, "i4", "", disp, sizeof(int32_t));
  obj[name]["data"] = data;
}

void put_attribute(json& obj, string name, const size_t disp, const int64_t data)
{
  put_metadata(obj, name, "i8", "", disp, sizeof(size_t));
  obj[name]["data"] = data;
}

void put_attribute(json& obj, string name, const size_t disp, const float32 data)
{
  put_metadata(obj, name, "f4", "", disp, sizeof(float32));
  obj[name]["data"] = data;
}

void put_attribute(json& obj, string name, const size_t disp, const float64 data)
{
  put_metadata(obj, name, "f8", "", disp, sizeof(float64));
  obj[name]["data"] = data;
}

void put_attribute(json& obj, string name, const size_t disp, const int32_t length,
                   const int32_t* data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "i4", "", disp, sizeof(int32_t) * length, 1, dims);
  obj[name]["data"] = std::vector<int32_t>(&data[0], &data[length]);
}

void put_attribute(json& obj, string name, const size_t disp, const int32_t length,
                   const int64_t* data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "i8", "", disp, sizeof(size_t) * length, 1, dims);
  obj[name]["data"] = std::vector<size_t>(&data[0], &data[length]);
}

void put_attribute(json& obj, string name, const size_t disp, const int32_t length,
                   const float32* data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "f4", "", disp, sizeof(float32) * length, 1, dims);
  obj[name]["data"] = std::vector<float32>(&data[0], &data[length]);
}

void put_attribute(json& obj, string name, const size_t disp, const int32_t length,
                   const float64* data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "f8", "", disp, sizeof(float64) * length, 1, dims);
  obj[name]["data"] = std::vector<float64>(&data[0], &data[length]);
}

template size_t read_subarray(MPI_File* fh, size_t* disp, void* data, const int32_t ndim,
                              const int32_t gshape[], const int32_t lshape[],
                              const int32_t offset[], const int32_t elembyte, MPI_Request* req,
                              const int order);
template size_t read_subarray(MPI_File* fh, size_t* disp, void* data, const size_t ndim,
                              const size_t gshape[], const size_t lshape[], const size_t offset[],
                              const size_t elembyte, MPI_Request* req, const int order);
template size_t write_subarray(MPI_File* fh, size_t* disp, void* data, const int32_t ndim,
                               const int32_t gshape[], const int32_t lshape[],
                               const int32_t offset[], const int32_t elembyte, MPI_Request* req,
                               const int order);
template size_t write_subarray(MPI_File* fh, size_t* disp, void* data, const size_t ndim,
                               const size_t gshape[], const size_t lshape[], const size_t offset[],
                               const size_t elembyte, MPI_Request* req, const int order);
template void   get_attribute(json& obj, string name, size_t& disp, int32_t& data);
template void   get_attribute(json& obj, string name, size_t& disp, int64_t& data);
template void   get_attribute(json& obj, string name, size_t& disp, float32& data);
template void   get_attribute(json& obj, string name, size_t& disp, float64& data);
template void   get_attribute(json& obj, string name, size_t& disp, int32_t length, int32_t* data);
template void   get_attribute(json& obj, string name, size_t& disp, int32_t length, int64_t* data);
template void   get_attribute(json& obj, string name, size_t& disp, int32_t length, float32* data);
template void   get_attribute(json& obj, string name, size_t& disp, int32_t length, float64* data);

} // namespace nixio

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
