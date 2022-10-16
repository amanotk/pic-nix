# -*- Makefile -*-
include ../common.mk

SRCS    = exchunk3d.cpp expic3d.cpp
OBJS    = $(SRCS:%.cpp=%.o)
DEPS    = $(SRCS:%.cpp=%.d)

default: libexpic.a

### library
libexpic.a: $(OBJS)
	$(AR) r $@ $(OBJS)

### clean
clean :
	rm -f $(OBJS) *.a *.d

### dependency
-include $(DEPS)
