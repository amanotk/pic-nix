// -*- C++ -*-
#ifndef _PIC_ASCENT_FIELD_SCHEMA_HPP_
#define _PIC_ASCENT_FIELD_SCHEMA_HPP_

#include <array>
#include <string_view>

namespace pic_ascent
{
inline constexpr int schema_version = 1;

inline constexpr std::array<std::string_view, 6> uf_components = {
    "Ex", "Ey", "Ez", "Bx", "By", "Bz",
};

inline constexpr std::array<std::string_view, 4> uj_components = {
    "rho",
    "Jx",
    "Jy",
    "Jz",
};

inline constexpr std::array<std::string_view, 14> moment_components = {
    "m00", "m01", "m02", "m03", "m04", "m05", "m06",
    "m07", "m08", "m09", "m10", "m11", "m12", "m13",
};
} // namespace pic_ascent

#endif
