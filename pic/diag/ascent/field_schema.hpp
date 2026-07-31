// -*- C++ -*-
#ifndef _PIC_ASCENT_FIELD_SCHEMA_HPP_
#define _PIC_ASCENT_FIELD_SCHEMA_HPP_

#include <array>
#include <cstddef>
#include <string_view>

namespace pic_ascent
{
inline constexpr std::array<std::string_view, 14> moment_components = {
    "M0", "Mx", "My", "Mz", "Ttt", "Txx", "Tyy", "Tzz", "Ttx", "Tty", "Ttz", "Txy", "Tyz", "Tzx",
};

inline constexpr std::size_t mass_current_component_count = 4;
} // namespace pic_ascent

#endif
