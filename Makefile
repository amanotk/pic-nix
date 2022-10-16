# -*- Makefile -*-

# base directory
BASEDIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

# include compilers
include $(BASEDIR)/common.mk

# target directories
SUBDIRS = nix expic

### default target
default:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done

### clean
clean:
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
	done

### cleanall
cleanall:
	rm -f $(OBJS) *.a *.d
	# clean subdirectories
	for dir in $(SUBDIRS); do \
		$(MAKE) cleanall -C $$dir; \
	done
