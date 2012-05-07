# we have used ATI's SDK although nVidia's would be appropriate as well
ATISTREAMSDKROOT=/home/garykche/ati-stream-sdk-v2.2-lnx64
OPENCL_INC_FLAGS = -I$(ATISTREAMSDKROOT)/include -I/usr/local/cuda_sdk/shared/inc
OPENCL_LIB_FLAGS = -L$(ATISTREAMSDKROOT)/lib/x86_64 -lOpenCL

# please specify here the location of your Boost header files
BOOST_INC_FLAGS = -I/home/garykche/boost/include

# please specify here the location of your MPI header files (e.g. mpi.h)
MPI_INC_FLAGS = -I/home/garykche/mpich2

COMMONCFLAGS = -Wall  -g 
COMMONLINKFLAGS = -lm  -g
GSL_LIB_FLAGS = -lgsl -lgslcblas

COMMON_OBJS=analyzer.o main.o io.o

# compiler
CC = g++
LINKER = mpicxx
PREPROC=-Dlasso=1 -DUSE_GPU -Dpower=1 -Dstability=1
OBJS=$(COMMON_OBJS) utility.o power.o stability.o slave_lasso2.o master_lasso2.o clsafe.o
CFLAGS=$(COMMONCFLAGS) $(BOOST_INC_FLAGS) $(OPENCL_INC_FLAGS) $(MPI_INC_FLAGS)
LINKFLAGS=$(COMMONLINKFLAGS) $(GSL_LIB_FLAGS) $(OPENCL_LIB_FLAGS) $(MPI_LIB_FLAGS)

PROGRAM=../bin/analyzer

bvs: $(OBJS)
	$(LINKER) -o $(PROGRAM) $(OBJS) $(LINKFLAGS)

clsafe.o: clsafe.c clsafe.h
	$(CC) $(CFLAGS) $(PREPROC) -o clsafe.o -c $<

io.o: io.cpp io.hpp
	$(CC) $(CFLAGS) $(PREPROC) -o io.o -c $<

main.o: main.cpp main.hpp
	$(CC) $(CFLAGS) $(PREPROC)  -o main.o -c $<

analyzer.o: analyzer.cpp analyzer.hpp
	$(CC) $(CFLAGS) $(PREPROC) -o analyzer.o -c $<

slave_lasso2.o:  slave_lasso2.cpp lasso_mpi2.hpp analyzer.hpp main.hpp dimension2.h
	$(CC) $(CFLAGS) $(PREPROC)  -o slave_lasso2.o -c $<

master_lasso2.o: master_lasso2.cpp lasso_mpi2.hpp analyzer.hpp main.hpp dimension2.h
	$(CC) $(CFLAGS) $(PREPROC)  -o master_lasso2.o -c $<

power.o:  power.cpp power.hpp analyzer.hpp main.hpp dimension2.h
	$(CC) $(CFLAGS) $(PREPROC)  -o power.o -c $<

stability.o:  stability.cpp stability.hpp analyzer.hpp main.hpp dimension2.h
	$(CC) $(CFLAGS) $(PREPROC)  -o stability.o -c $<

utility.o: utility.cpp utility.hpp
	$(CC) $(CFLAGS) -c $<

clean :
	rm -fr *.o $(PROGRAM)
