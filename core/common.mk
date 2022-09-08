# -*- Makefile -*-

# base directory
BASEDIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

# include compilers
include $(BASEDIR)/compiler.mk

# default
.PHONY : all
.PHONY : clean

.SUFFIXES :
.SUFFIXES : .o .cpp

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -I$(BASEDIR) -I.. $< -o $@
