// -*- C++ -*-
#ifndef NIX_UNITTEST_TEST_PARALLEL_HPP
#define NIX_UNITTEST_TEST_PARALLEL_HPP

extern int options_mpi_decomposition[3];

int get_mpi_size();
int get_mpi_rank();
bool require_mpi_size(int expected);

#endif
