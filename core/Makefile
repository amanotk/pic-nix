# -*- Makefile -*-
include common.mk

SUBDIRS =
SRCS    = exchunk3d.cpp expic3d.cpp main.cpp
OBJS    = $(SRCS:%.cpp=%.o)
DEPS    = $(SRCS:%.cpp=%.d)

default: $(OBJS)

# main
main.out: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) -L../nix -lnix

### clean
clean :
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
	done

### dependency
-include $(DEPS)
