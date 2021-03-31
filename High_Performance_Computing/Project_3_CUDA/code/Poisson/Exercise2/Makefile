TARGET	= jacobi_gpu2
OBJS	= main.o jacobi.o alloc3d.o alloc3d_gpu.o transfer3d_gpu.o

OPT	= -O3
PARA	= -fopenmp

XOPTS = -Xptxas=-v -arch=sm_80 -lineinfo

CC	= gcc

CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS) -I $(PNGWRTIPATH)

CFLAGS	= $(OPT) $(ISA) $(PARA) $(XOPT)

CUDA_PATH ?= /appl/cuda/11.1
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(SOFLAGS) $(INCLUDES) $^ $(XLIBS) 

.SUFFIXES: .cu
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f $(TARGET) $(OBJS) 


# dependencies
#
main.o  : main.cu jacobi.h alloc3d_gpu.h transfer3d_gpu.h alloc3d.h
mxv.o   : mxv.h
transfer3d_gpu.o : transfer3d_gpu.h
alloc3d_gpu.o: alloc3d_gpu.h
alloc3d.o: alloc3d.h
