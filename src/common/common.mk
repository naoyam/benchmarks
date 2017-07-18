ifneq ($(USE_PGI),)
include $(INCLUDEPATH)pgi.mk
else ifneq ($(USE_INTEL),)
include $(INCLUDEPATH)intel.mk
else
# use GCC by default
include $(INCLUDEPATH)gcc.mk 
endif

UNAME := $(shell uname -s)

ifneq ($(DEBUG),)
DEBUG_FLAGS += -DDEBUG
endif

CFLAGS = -I.. $(DEBUG_FLAGS) $(OPT_FLAGS) $(WARNING_FLAGS)
CXXFLAGS = --std=c++11 -I.. $(DEBUG_FLAGS) $(OPT_FLAGS) $(WARNING_FLAGS)
FFLAGS =  -I.. $(DEBUG_FLAGS) $(OPT_FLAGS) $(WARNING_FLAGS)
LDFLAGS = -lm -m64

.SUFFIXES: .f90
.SUFFIXES: .F90
.F90.o:
	$(FC) -c $(FFLAGS) $<
.f90.o:
	$(FC) -c $(FFLAGS) $<

# MPI
MPICC = mpicc
MPICXX = mpicxx
MPI_INCLUDE = $(shell mpicc -show | sed 's/.*-I\([\/a-zA-Z0-9_\-]*\).*/\1/g')

# CUDA
NVCC = nvcc
NVCC_CFLAGS = --std=c++11 -m64 -I.. -Xcompiler -Wall -Xptxas -v # -keep
NVCC_ARCH = -arch sm_35
ifneq ($(DEBUG),)
NVCC_CFLAGS += -g -G -DDEBUG
else
NVCC_CFLAGS += -O3
endif
CUDA_INC = $(patsubst %bin/nvcc,%include, $(shell which $(NVCC)))
ifeq (,$(findstring Darwin,$(shell uname)))
	CUDA_LDFLAGS = -lcudart -L$(patsubst %bin/nvcc,%lib64, \
		$(shell which $(NVCC)))
else
	CUDA_LDFLAGS = -lcudart -L$(patsubst %bin/nvcc,%lib, \
		$(shell which $(NVCC)))
endif

ifneq ($(NVCC_USE_FAST_MATH),)
	NVCC_CFLAGS += -use_fast_math
endif


ifeq ($(UNAME),Darwin)
	NVCC_CFLAGS += -ccbin=llvm-g++
endif

.SUFFIXES: .cu

.cu.o:
	$(NVCC) -c $< $(NVCC_CFLAGS) $(NVCC_ARCH) -o $@

# Physis
PHYSISC_CONFIG ?= /dev/null
PHYSISC_CONFIG_KEY = $(shell basename $(PHYSISC_CONFIG))
PHYSISC_REF = $(PHYSIS_DIR)/bin/physisc-ref --config $(realpath $(PHYSISC_CONFIG)) -I$(realpath ..)
PHYSISC_CUDA = $(PHYSIS_DIR)/bin/physisc-cuda --config $(realpath $(PHYSISC_CONFIG)) -I$(realpath ..)
PHYSISC_MPI = $(PHYSIS_DIR)/bin/physisc-mpi --config $(realpath $(PHYSISC_CONFIG)) -I$(realpath ..)
PHYSISC_MPI_CUDA = $(PHYSIS_DIR)/bin/physisc-mpi-cuda --config $(realpath $(PHYSISC_CONFIG)) -I$(realpath ..)
PHYSIS_BUILD_DIR_TOP = physis_build
PHYSIS_BUILD_DIR = physis_build/$(PHYSISC_CONFIG_KEY)

