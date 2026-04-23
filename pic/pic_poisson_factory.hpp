// -*- C++ -*-
#ifndef _PIC_POISSON_FACTORY_HPP_
#define _PIC_POISSON_FACTORY_HPP_

#include "pic_poisson.hpp"

#include <memory>

std::shared_ptr<PicPoisson> make_poisson_solver(const nix::Dims3D& global_dims, float64 delh);

#endif //_PIC_POISSON_FACTORY_HPP_
