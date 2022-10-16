# -*- Makefile -*-

# base directory
BASEDIR  := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
NIXDIR   := $(BASEDIR)/nix
EXPICDIR := $(BASEDIR)/expic

# system specific compiler setting
include $(BASEDIR)/compiler.mk

# add options
CXXFLAGS += -I$(BASEDIR) -I$(NIXDIR) -I$(NIXDIR)/thirdparty -I$(EXPICDIR) 
LDFLAGS  += -L$(NIXDIR) -L$(EXPICDIR) -lnix -lexpic

# default
.PHONY : all
.PHONY : clean

.SUFFIXES :
.SUFFIXES : .o .cpp

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@
