# -*- Makefile -*-

# include directories
XTENSOR_FLAGS = $(shell pkg-config --cflags xtensor xtl xsimd)

# compilers and arguments
AR       = ar
CXX      = mpicxx
CXXFLAGS = -std=c++14 -O2 -MMD $(XTENSOR_FLAGS)
LDFLAGS  =
