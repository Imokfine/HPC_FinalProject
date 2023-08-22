# Compilers and commands
CC=             gcc
CXX=            g++
NVCC=           nvcc 
LINK=           nvcc
DEL_FILE=       rm -f
MPICC=          mpicc
MPICXX=         mpicxx

# Variables for the makefile
CFLAGS = -Wall -O2  -ffast-math
LDFLAGS = -lm -ffast-math

MPI_INCLUDES    = /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIBS        = /usr/lib/openmpi
CUDA_INCLUDES   = /usr/local/cuda-12.1/include/

#Flags
PARALLEL        = -fopenmp
CXXFLAGS        = -O4 -W -Wall $(PARALLEL) $(DEFINES)
NVCCFLAGS       = -O4 --use_fast_math --compiler-options -fno-inline -std=c++17 -Xcompiler -fopenmp

LIBS            = $(PARALLEL)
INCPATH         = /usr/include/ 

####### Files
SOURCES         = main_mpi.cc kmeans_mpi.cc main.cc kmeans.cu comp.cc init.cc
OBJECTS         = $(SOURCES:.cu=.o)
DEPENDS         = kmeans.h utils.h
TARGET          = main_mpi main comp init


all: $(TARGET)

# MPI
main_mpi : main_mpi.o utils.o kmeans_mpi.o
	$(MPICC) -L$(MPI_LIBS) -lmpi -o $@ *.o -lstdc++ -lm -lmpi_cxx 
main_mpi.o: main_mpi.cc kmeans_mpi.h utils.h
	$(MPICC) -I$(MPI_INCLUDES) -o $@ -c $< $(CFLAGS)
kmeans_mpi.o: kmeans_mpi.cc kmeans_mpi.h utils.h
	$(MPICC) -I$(MPI_INCLUDES) -o $@ -c $< $(CFLAGS)

main: main.o kmeans.o utils.o
	$(NVCC) -o $@ $^ $(NVCCFLAGS) -I$(CUDA_INCLUDES)
main.o: main.cc $(DEPENDS)
	$(NVCC) main.cc -c $(NVCCFLAGS) -I$(CUDA_INCLUDES)
kmeans.o: kmeans.cu $(DEPENDS)
	$(NVCC) kmeans.cu -c $(NVCCFLAGS) -I$(CUDA_INCLUDES)
comp: comp.o utils.o
	$(CXX) -o $@ $^ $(CXXFLAGS)
init: init.o utils.o
	$(CXX) -o $@ $^ $(CXXFLAGS)
utils.o : utils.cc utils.h
	$(CXX) utils.cc -c $(CXXFLAGS) 

.PHONY:
clean:
	-$(DEL_FILE) *.o $(TARGET)
cleanall:
	-$(DEL_FILE) *.o *.csv $(TARGET)

