INCLUDEPATH=../common/
include $(INCLUDEPATH)common.mk

# Minimal configuration
SRC = 
OBJ = $(filter %.o,$(SRC:%.cc=%.o)) $(filter %.o,$(SRC:%.c=%.o))

EXE = diffusion_baseline diffusion_openmp diffusion_cuda

	# diffusion_openmp_temporal_blocking \
	# diffusion_cuda_zblock \
	# diffusion_cuda_opt0 \
	# diffusion_cuda_opt1 \
	# diffusion_cuda_opt2 \
	# diffusion_cuda_opt3 \
	# diffusion_cuda_shared \
	# diffusion_cuda_shared1 \
	# diffusion_cuda_shared2 \
	# diffusion_cuda_shared3 \
	# diffusion_cuda_temporal_blocking \
	# diffusion_mic \
	# diffusion_fortran \
	# diffusion_fortran_acc

ifneq ($(BLOCK_X),)
	CXXFLAGS += -DBLOCK_X=$(BLOCK_X)
	NVCC_CFLAGS += -DBLOCK_X=$(BLOCK_X)
endif	
ifneq ($(BLOCK_Y),)
	CXXFLAGS += -DBLOCK_Y=$(BLOCK_Y)
	NVCC_CFLAGS += -DBLOCK_Y=$(BLOCK_Y)
endif	
ifneq ($(GRID_Z),)
	CXXFLAGS += -DGRID_Z=$(GRID_Z)
	NVCC_CFLAGS += -DGRID_Z=$(GRID_Z)
endif	
ifneq ($(UNROLL),)
	CXXFLAGS += -DUNROLL=$(UNROLL)
	NVCC_CFLAGS += -DUNROLL=$(UNROLL)
endif	

##################################################

all: $(EXE)

.PHONY: baseline
baseline: diffusion_baseline

diffusion_baseline: $(OBJ) main_baseline.o
	$(CXX) -o $@ $^ $(LDFLAGS)

main_baseline.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS)

##################################################
# .PHONY: fortran
# fortran: diffusion_fortran

# main_fortran.o: main.cc diffusion_fortran.h
# 	$(CXX) -o $@ -c $< $(CXXFLAGS) -DFORTRAN
# diffusion_fortran: $(OBJ) main_fortran.o diffusion_fortran.o
# 	$(CXX) -o $@ $^ $(LDFLAGS)
##################################################
.PHONY: openmp
openmp: diffusion_openmp

diffusion_openmp.o: CXXFLAGS += $(OPENMP_CFLAGS)
diffusion_openmp: $(OBJ) main_openmp.o diffusion_openmp.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENMP_LDFLAGS)

main_openmp.o: main.cc
	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DOPENMP

# openmp_temporal_blocking: diffusion_openmp_temporal_blocking
# diffusion_openmp_temporal_blocking: $(OBJ) main_openmp_temporal_blocking.o \
# 	diffusion_openmp_temporal_blocking.o diffusion_openmp.o
# 	$(CXX) -o $@ $^ $(LDFLAGS) $(OPENMP_LDFLAGS)

# main_openmp_temporal_blocking.o: main.cc
# 	$(CXX) -o $@ -c $^ $(CXXFLAGS) -DOPENMP_TEMPORAL_BLOCKING

# diffusion_openmp_temporal_blocking.o: CXXFLAGS += $(OPENMP_CFLAGS)
##################################################
.PHONY: cuda
cuda: diffusion_cuda \
	diffusion_cuda_roc \
	diffusion_cuda_l1

# diffusion_cuda_zblock \
# 	diffusion_cuda_opt0 \
# 	diffusion_cuda_opt1 \
# 	diffusion_cuda_opt2 \
# 	diffusion_cuda_opt3 \
# 	diffusion_cuda_shared \
# 	diffusion_cuda_shared1 \
# 	diffusion_cuda_shared2 \
# 	diffusion_cuda_shared3 \
# 	diffusion_cuda_shared4 \
# 	diffusion_cuda_shared5 \
# 	diffusion_cuda_shared6 \
# 	diffusion_cuda_xy \
# 	diffusion_cuda_temporal_blocking
########################################
# diffusion_cuda (baseline)
diffusion_cuda: $(OBJ) main_cuda.o diffusion_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
diffusion_cuda.o: diffusion_cuda.h
main_cuda.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA -I$(CUDA_INC)
########################################
# diffusion_cuda_roc
diffusion_cuda_roc: $(OBJ) main_cuda_roc.o diffusion_cuda_roc.o diffusion_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
diffusion_cuda_roc.o: diffusion_cuda_roc.h
main_cuda_roc.o: main.cc diffusion_cuda.h diffusion_cuda_roc.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_ROC -I$(CUDA_INC)
########################################
# diffusion_cuda_cache
diffusion_cuda_l1: $(OBJ) main_cuda_l1.o diffusion_cuda_l1.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
diffusion_cuda_l1.cu: diffusion_cuda.cu
diffusion_cuda_l1.o: NVCC_CFLAGS += -Xptxas -dlcm=ca
main_cuda_l1.o: main.cc diffusion_cuda.h 
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_L1 -I$(CUDA_INC)
########################################
# diffusion_cuda_zblock
diffusion_cuda_zblock: $(OBJ) main_cuda_zblock.o diffusion_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_zblock.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_ZBLOCK -I$(CUDA_INC)
########################################
# diffusion_cuda_opt1
diffusion_cuda_opt1: $(OBJ) main_cuda_opt1.o diffusion_cuda.o \
	diffusion_cuda_opt.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_opt1.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_OPT1 -I$(CUDA_INC)
########################################
# diffusion_cuda_opt2
diffusion_cuda_opt2: $(OBJ) main_cuda_opt2.o diffusion_cuda.o \
	diffusion_cuda_opt.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_opt2.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_OPT2 -I$(CUDA_INC)
########################################
# diffusion_cuda_shared
diffusion_cuda_shared: $(OBJ) main_cuda_shared.o diffusion_cuda.o \
	diffusion_cuda_shared.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED -I$(CUDA_INC)
diffusion_cuda_shared.o: diffusion_cuda.h
# DISABLE L1 load caching
diffusion_cuda_shared.o: NVCC_CFLAGS += -Xptxas -dlcm=cg
########################################
# diffusion_cuda_shared1
diffusion_cuda_shared1: $(OBJ) main_cuda_shared1.o diffusion_cuda.o \
	diffusion_cuda_shared.o diffusion_cuda_shared1.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared1.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED1 -I$(CUDA_INC)
diffusion_cuda_shared1.o: diffusion_cuda.h
########################################
# diffusion_cuda_shared2
diffusion_cuda_shared2: $(OBJ) main_cuda_shared2.o diffusion_cuda.o \
	diffusion_cuda_shared.o diffusion_cuda_shared2.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared2.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED2 -I$(CUDA_INC)
diffusion_cuda_shared2.o: diffusion_cuda.h
########################################
# diffusion_cuda_shared3
diffusion_cuda_shared3: $(OBJ) main_cuda_shared3.o diffusion_cuda.o \
	diffusion_cuda_shared.o diffusion_cuda_shared3.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared3.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED3 -I$(CUDA_INC)
diffusion_cuda_shared3.o: diffusion_cuda.h
########################################
# diffusion_cuda_shared4
diffusion_cuda_shared4: $(OBJ) main_cuda_shared4.o diffusion_cuda.o \
	diffusion_cuda_shared.o diffusion_cuda_shared4.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared4.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED4 -I$(CUDA_INC)
diffusion_cuda_shared4.o: diffusion_cuda.h
########################################
# diffusion_cuda_shared5
diffusion_cuda_shared5: $(OBJ) main_cuda_shared5.o diffusion_cuda.o \
	diffusion_cuda_shared.o diffusion_cuda_shared5.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared5.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED5 -I$(CUDA_INC)
diffusion_cuda_shared5.o: diffusion_cuda.h
########################################
# diffusion_cuda_shared6
diffusion_cuda_shared6: $(OBJ) main_cuda_shared6.o diffusion_cuda.o \
	diffusion_cuda_shared.o diffusion_cuda_shared6.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_shared6.o: main.cc diffusion_cuda.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_SHARED6 -I$(CUDA_INC)
diffusion_cuda_shared6.o: diffusion_cuda.h
########################################
# diffusion_cuda_xy
# XY multi processing
diffusion_cuda_xy: $(OBJ) main_cuda_xy.o diffusion_cuda.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_xy.o: main.cc
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_XY -I$(CUDA_INC)
# Temporal blokcing
diffusion_cuda_temporal_blocking.o: diffusion_cuda.h diffusion_cuda_temporal_blocking.h
diffusion_cuda_temporal_blocking: $(OBJ) main_cuda_temporal_blocking.o diffusion_cuda.o diffusion_cuda_temporal_blocking.o
	$(CXX) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
main_cuda_temporal_blocking.o: main.cc diffusion_cuda_temporal_blocking.h
	$(CXX) -o $@ -c $< $(CXXFLAGS) -DCUDA_TEMPORAL_BLOCKING -I$(CUDA_INC)
##################################################
# Fortran OpenACC
# .PHONY: fortran-acc
# fortran-acc: diffusion_fortran_acc
# main_fortran_acc.o: main.cc diffusion_fortran.h
# 	$(CXX) -o $@ -c $< $(CXXFLAGS) -DFORTRAN_ACC
# diffusion_fortran_acc.o: diffusion_fortran.F90
# 	$(FC) -o $@ -c $< $(FFLAS) $(ACC_FFLAGS) -DFORTRAN_ACC
# diffusion_fortran_acc: $(OBJ) main_fortran_acc.o diffusion_fortran_acc.o
# 	$(CXX) -o $@ $^ $(LDFLAGS) $(ACC_LDFLAGS) $(F90_LIBS)
##################################################


# include physis.mk

clean:
	-$(RM) *.o $(EXE)
	-$(RM) diffusion_result.*.out
	-$(RM) *.cudafe* *.gpu *.stub.c *.pptx *.cubin *.i *.ii *.fatbin *.fatbin.c
	-$(RM) *_physis.ref.* *_physis.cuda.* *_physis.mpi.* \
		*_physis.mpi-cuda.*
	-$(RM) -rf $(PHYSIS_BUILD_DIR_TOP)
