# -*- Makefile -*-
include common.mk

SUBDIRS =
#SRCS    = base_pic3d.cpp expic3d1.cpp
SRCS    = exchunk3d.cpp
OBJS    = $(SRCS:%.cpp=%.o)
DEPS    = $(SRCS:%.cpp=%.d)

default: $(OBJS)

### clean
clean :
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
	done

### dependency
-include $(DEPS)
