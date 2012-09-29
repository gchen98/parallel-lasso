# USER NEEDS TO SET THE PATHS HERE
ATI=/home/garykche/ati-stream-sdk-v2.2-lnx64
OPENCL_INC_FLAGS = -I$(ATI)/include -I/usr/local/cuda_sdk/shared/inc
OPENCL_LIB_FLAGS = -L$(ATI)/lib/x86_64 -lOpenCL
BOOST_INC_FLAGS = -I/usr/include/boost141
BOOST_LIB_FLAGS = -L/usr/lib/boost141
GSL_LIB_FLAGS = -lgsl -lgslcblas
MPI_INC_FLAGS = -I/home/garykche/mpich2
MYSQL_INC_FLAGS = -I/usr/include/mysql
MYSQL_LIB_FLAGS = -L/usr/lib64/mysql -lmysqlclient
