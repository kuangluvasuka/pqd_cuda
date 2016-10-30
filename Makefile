#### Makefile for MPI+CUDA pqd1.c ####
#
# The procedure is equivalent to:
# $ nvcc -c gpu_calc_prop.cu -o gpu_calc_prop.o
# $ mpicc -c pqd1.c -o pqd1.o
# $ mpicc pqd1.o gpu_calc_prop.o -lm -L/usr/usc/cuda/default/lib64 -lcudart -o pqd
#
####


# Path to libraries
CUDA_DIR = /usr/usc/cuda/default
MPI_DIR = /usr/usc/openmpi/default

# Compiler and linker
MPICC = mpicc
NVCC = nvcc

# Objects
OBJ = pqd1.o gpu_calc_prop.o

# Compiler flags

# Linker flags
LD_FLAGS = -lm -L$(CUDA_DIR)/lib64 -lcudart

# Build targets
TARGETS = pqd
all: $(TARGETS)


###
### Build rules ###
###

%.o: %.cu
	$(NVCC) -c $^

%.o: %.c
	$(MPICC) -c $^

pqd: $(OBJ)
	$(MPICC) -o $@ $^ $(LD_FLAGS)

clean:
	rm -rf *.o *~ core.* $(TARGETS)


