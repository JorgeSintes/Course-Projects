TARGET	= libmatmult.so
LIBSRCS	= 	matmult_nmk.cu matmult_nkm.cu matmult_mnk.cu matmult_mkn.cu matmult_knm.cu matmult_kmn.cu  matmult_nat.cu matmult_lib.cu \
			matmult_gpu1.cu matmult_gpu2.cu matmult_gpu3.cu matmult_gpu4_6.cu matmult_gpu5.cu matmult_gpulib.cu
LIBOBJS	= $(LIBSRCS:.cu=.o)

OPT	= -g -O3 
PIC = -fpic
OMP   = -fopenmp
XPIC  = -Xcompiler -fpic
XOPT  = -Xptxas=-v -lineinfo #-G # only use -G for debugging / profiler
XARCH = -arch=sm_80

CXX	= nvcc
CXXFLAGS = --compiler-options "$(OPT) $(PIC) $(OMP)" $(XARCH) $(XOPT) $(XPIC)

CUDA_PATH ?= /appl/cuda/11.1
INCLUDES = -I$(CUDA_PATH)/include \
	       -I$(CUDA_PATH)/samples/common/inc

SOFLAGS = -shared
XLIBS	= -lcublas

$(TARGET): $(LIBOBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(SOFLAGS) $(INCLUDES) $^ $(XLIBS)

.SUFFIXES: .cu
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

clean:
	/bin/rm -f $(TARGET) $(LIBOBJS) 
