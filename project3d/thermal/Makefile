# -*- Makefile -*-
include ../../common.mk

SRCS    = main.cpp
OBJS    = $(SRCS:%.cpp=%.o)
DEPS    = $(SRCS:%.cpp=%.d)

# main
main.out: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

### clean
clean :
	rm -f $(OBJS) *.a *.d

### dependency
-include $(DEPS)
