// -*- C++ -*-
#ifndef _JSONIO_HPP_
#define _JSONIO_HPP_

#include <mpi.h>
#include <nlohmann/json.hpp>

///
/// @brief NIX-IO utility module
///
namespace nixio
{
using json    = nlohmann::ordered_json;
using float32 = float;
using float64 = double;
using std::string;

///
/// @brief collective open file with MPI-IO
/// @param filename
/// @param fh
/// @param disp
/// @param mode
///
void open_file(const char* filename, MPI_File* fh, size_t* disp, const char* mode);

///
/// @brief collective close file with MPI-IO
/// @param fh
///
void close_file(MPI_File* fh);

///
/// @brief primitive collective contiguous read/write
/// @param fh
/// @param disp
/// @param data
/// @param offset
/// @param size
/// @param elembyte
/// @param packbyte
/// @param mode
///
size_t readwrite_contiguous(MPI_File* fh, size_t* disp, void* data, const size_t offset,
                            const size_t size, const int32_t elembyte, const int32_t packbyte,
                            MPI_Request* req, const int mode);

///
/// @brief primitive collective contiguous read/write with explicit offset
/// @param fh
/// @param disp
/// @param data
/// @param size
/// @param elembyte
/// @param packbyte
/// @param req
/// @param mode
///
size_t readwrite_contiguous_at(MPI_File* fh, size_t* disp, void* data, const size_t size,
                               const int32_t elembyte, const int32_t packbyte, MPI_Request* req,
                               const int mode);

///
/// @brief primitive collective read/write of subarray with MPI-IO
/// @param fh
/// @param disp
/// @param data
/// @param ndim
/// @param gshape
/// @param lshape
/// @param offset
/// @param elembyte
/// @param mode
/// @param order
///
size_t readwrite_subarray(MPI_File* fh, size_t* disp, void* data, const int32_t ndim,
                          const int32_t gshape[], const int32_t lshape[], const int32_t offset[],
                          const int32_t elembyte, MPI_Request* req, const int mode,
                          const int order);

///
/// @brief non-collective read
/// @param fh
/// @param disp
/// @param data
/// @param size
///
size_t read_single(MPI_File* fh, size_t* disp, void* data, const size_t size, MPI_Request* req);

///
/// @brief non-collective write
/// @param fh
/// @param disp
/// @param data
/// @param size
///
size_t write_single(MPI_File* fh, size_t* disp, void* data, const size_t size, MPI_Request* req);

///
/// @brief collective and contiguous read
/// @param fh
/// @param disp
/// @param data
/// @param size
/// @param elembyte
/// @param packbyte
///
size_t read_contiguous(MPI_File* fh, size_t* disp, void* data, const size_t size,
                       const int32_t elembyte, const int32_t packbyte, MPI_Request* req);

///
/// @brief collective and contiguous write
/// @param fh
/// @param disp
/// @param data
/// @param size
/// @param elembyte
/// @param packbyte
///
size_t write_contiguous(MPI_File* fh, size_t* disp, void* data, const size_t size,
                        const int32_t elembyte, const int32_t packbyte, MPI_Request* req);

///
/// @brief collective and contiguous read with explicit offset
/// @param fh
/// @param disp
/// @param data
/// @param size
/// @param elembyte
/// @param req
///
size_t read_contiguous_at(MPI_File* fh, size_t* disp, void* data, const size_t size,
                          const int32_t elembyte, MPI_Request* req);

///
/// @brief collective and contiguous write with explicit offset
/// @param fh
/// @param disp
/// @param data
/// @param size
/// @param elembyte
/// @param req
///
size_t write_contiguous_at(MPI_File* fh, size_t* disp, void* data, const size_t size,
                           const int32_t elembyte, MPI_Request* req);

///
/// @brief collective read of subarray
/// @tparam T1
/// @tparam T2
/// @tparam T3
/// @tparam T4
/// @tparam T5
/// @param fh
/// @param disp
/// @param data
/// @param ndim
/// @param gshape
/// @param lshape
/// @param offset
/// @param elembyte
/// @param order
///
template <typename T1, typename T2, typename T3, typename T4, typename T5>
size_t read_subarray(MPI_File* fh, size_t* disp, void* data, const T1 ndim, const T2 gshape[],
                     const T3 lshape[], const T4 offset[], const T5 elembyte, MPI_Request* req,
                     const int order = MPI_ORDER_C);

///
/// @brief collective write of subarray
/// @tparam T1
/// @tparam T2
/// @tparam T3
/// @tparam T4
/// @tparam T5
/// @param fh
/// @param disp
/// @param data
/// @param ndim
/// @param gshape
/// @param lshape
/// @param offset
/// @param elembyte
/// @param order
///
template <typename T1, typename T2, typename T3, typename T4, typename T5>
size_t write_subarray(MPI_File* fh, size_t* disp, void* data, const T1 ndim, const T2 gshape[],
                      const T3 lshape[], const T4 offset[], const T5 elembyte, MPI_Request* req,
                      const int order = MPI_ORDER_C);

///
/// @brief put metadata of array to json object
/// @param obj
/// @param name
/// @param dtype
/// @param desc
/// @param disp
/// @param size
/// @param ndim
/// @param dims
///
void put_metadata(json& obj, string name, string dtype, string desc, const size_t disp,
                  const size_t size, const int32_t ndim, const int32_t dims[]);

///
/// @brief put metadata of scalar to json object
/// @param obj
/// @param name
/// @param dtype
/// @param desc
/// @param disp
/// @param size
///
void put_metadata(json& obj, string name, string dtype, string desc, const size_t disp,
                  const size_t size);

///
/// @brief get metadata of array from json object
/// @param obj
/// @param name
/// @param dtype
/// @param desc
/// @param disp
/// @param size
/// @param ndim
/// @param dims
///
void get_metadata(json& obj, string name, string& dtype, string& desc, size_t& disp, size_t& size,
                  int32_t& ndim, int32_t dims[]);

///
/// @brief get metadata of scalar from json object
/// @param obj
/// @param name
/// @param dtype
/// @param desc
/// @param disp
/// @param size
///
void get_metadata(json& obj, string name, string& dtype, string& desc, size_t& disp, size_t& size);

///
/// @brief get scalar attribute from json object
/// @tparam T
/// @param obj
/// @param name
/// @param disp
/// @param data
///
template <typename T>
void get_attribute(json& obj, string name, size_t& disp, T& data);

///
/// @brief get array attribute from json object
/// @tparam T
/// @param obj
/// @param name
/// @param disp
/// @param length
/// @param data
///
template <typename T>
void get_attribute(json& obj, string name, size_t& disp, int32_t length, T* data);

///
/// @brief put 32bit integer scalar attribute
/// @param obj
/// @param name
/// @param disp
/// @param data
///
void put_attribute(json& obj, string name, const size_t disp, const int32_t data);

///
/// @brief put 64bit integer scalar attribute
/// @param obj
/// @param name
/// @param disp
/// @param data
///
void put_attribute(json& obj, string name, const size_t disp, const int64_t data);

///
/// @brief put 32bit real scalar attribute
/// @param obj
/// @param name
/// @param disp
/// @param data
///
void put_attribute(json& obj, string name, const size_t disp, const float32 data);

///
/// @brief put 64bit real scalar attribute
/// @param obj
/// @param name
/// @param disp
/// @param data
///
void put_attribute(json& obj, string name, const size_t disp, const float64 data);

///
/// @brief put 32bit integer array attribute
/// @param obj
/// @param name
/// @param disp
/// @param length
/// @param data
///
void put_attribute(json& obj, string name, const size_t disp, const int32_t length,
                   const int32_t* data);

///
/// @brief put 64bit integer array attribute
/// @param obj
/// @param name
/// @param disp
/// @param length
/// @param data
///
void put_attribute(json& obj, string name, const size_t disp, const int32_t length,
                   const int64_t* data);

///
/// @brief put 32bit real array attribute
/// @param obj
/// @param name
/// @param disp
/// @param length
/// @param data
///
void put_attribute(json& obj, string name, const size_t disp, const int32_t length,
                   const float32* data);

///
/// @brief put 64bit real array attribute
/// @param obj
/// @param name
/// @param disp
/// @param length
/// @param data
///
void put_attribute(json& obj, string name, const size_t disp, const int32_t length,
                   const float64* data);

} // namespace nixio

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
