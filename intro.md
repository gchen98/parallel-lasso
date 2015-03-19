# Introduction #


Scalable Stability Selection is a distributed application that enables variable selection to be carried out on large datasets where the number of variables far exceeds the number of observations.  We designed it for genetic association studies, taking advantage of the compact properties of genotype data.  However it is compatible with continuous valued data as well.  The program uses OpenCL to carry out data parallel calculations on GPU devices, and MPI to distribute the computational and memory load across multiple hosts.  More information can be found here.


# Requirements #
Although in principle the following requirements are compatible across all environments, we have tested the following in a Linux environment:

  * MPI implementation
  * OpenCL implementation
  * C++ compiler

# Installation #

This is a work in progress.  Below are notes from a Fedora installation.

Installing MPICH2 (a popular implementation of MPI) can be fairly straightforward if you use a package repository such as yum.  As root, you can type

yum install mpich2

You can test that MPI is running by typing the command

/usr/lib64/mpich2/bin/mpiexec -np 2 echo Hello

which should output Hello twice (each processor prints it once).

You will next to install a device driver for the GPU.  For nVidia:

http://developer.nvidia.com/cuda-toolkit-40

You can install the OpenCL SDK from AMD from

http://developer.amd.com/sdks/AMDAPPSDK/downloads/Pages/default.aspx#one