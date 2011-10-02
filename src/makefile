#CFLAGS = -Wall -O3  -I/usr/include

COMMONCFLAGS = -Wall  -g 
COMMONLINKFLAGS = -lm  -g
ATISTREAMSDKROOT=/home/garykche/ati-stream-sdk-v2.2-lnx64
OPENCL_INC_FLAGS = -I$(ATISTREAMSDKROOT)/include -I/usr/local/cuda_sdk/shared/inc
OPENCL_LIB_FLAGS = -L$(ATISTREAMSDKROOT)/lib/x86_64 -lOpenCL

BOOST_INC_FLAGS = -I/home/garykche/boost/include

GSL_LIB_FLAGS = -lgsl -lgslcblas
MPI_INC_FLAGS = -I/home/garykche/mpich2
#MPI_INC_FLAGS = -I/usr/lib64/openmpi/1.4-gcc/include

MYSQL_INC_FLAGS = -I/usr/include/mysql
MYSQL_LIB_FLAGS = -L/usr/lib64/mysql -lmysqlclient



# program name
#LASSO=lasso
#STEPWISE=stepwise
#UNIVARIATE=univariate
#HMM=hmm_cnv
#HMMIMPUTE=hmm_impute

# environment
ENV=s3

COMMON_OBJS=analyzer.o main.o io.o

ifeq ($(ENV),s3)
  # compiler
  CC = g++
  #LINKER = g++
  #CC = mpicxx
  LINKER = mpicxx
  PREPROC=-Dlasso2=1 -DUSE_GPU
  OBJS=$(COMMON_OBJS) utility.o slave_lasso2.o master_lasso2.o clsafe.o
  CFLAGS=$(COMMONCFLAGS) $(BOOST_INC_FLAGS) $(OPENCL_INC_FLAGS) $(MPI_INC_FLAGS)
  LINKFLAGS=$(COMMONLINKFLAGS) $(GSL_LIB_FLAGS) $(OPENCL_LIB_FLAGS) $(MPI_LIB_FLAGS)
endif

PROGRAM=../bin/analyzer

# stepwise.o univariate.o
#MPI_OBJS=$(COMMON_OBJS) lasso_mpi.o

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

utility.o: utility.cpp utility.hpp
	$(CC) $(CFLAGS) -c $<

#all : $(BVS)
clean :
	rm -fr *.o $(PROGRAM)
