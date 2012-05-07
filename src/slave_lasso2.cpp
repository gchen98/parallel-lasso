#include<cstdlib>
#include<cstring>
#include<iostream>
#include<sstream>
#include<fstream>
#include<math.h>

#include"mpi.h"
#ifdef USE_GPU
#include<CL/cl.hpp>
#include"clsafe.h"
#endif
#include"main.hpp"
#include"analyzer.hpp"
#include"io.hpp"
#include"dimension2.h"
#include"utility.hpp"
#include"lasso_mpi2.hpp"

using namespace std;
typedef unsigned int uint;

void MpiLasso2::init_slave_dim(){
  MPI_Type_contiguous(MPI_FLOAT_ARR,MPI_FLOAT,&floatParamArrayType);
  MPI_Type_commit(&floatParamArrayType);
  MPI_Type_contiguous(MPI_INT_ARR,MPI_INT,&intParamArrayType);
  MPI_Type_commit(&intParamArrayType);
  int dim[MPI_INT_ARR];
  rc = MPI_Recv(dim,MPI_INT_ARR,MPI_INT,source,TAG_INIT_DIM,MPI_COMM_WORLD,&stat);
  iter = 0;
  n = dim[0];
  //n_subset = n = dim[0];
  totalsnps = dim[1];
  genoveclen = dim[2];
  logistic = dim[3];
  env_covariates = dim[4];
  algorithm = dim[5];
  slave_offset = dim[6];
  submodelsize = dim[7];  // number of tasks
  genetic_tasks = dim[8];  // number of snps
  ofs<<"I am rank "<<mpi_rank<<" and have "<<n<<" observations with "<<totalsnps<<" total SNPs, "<<slave_offset<<" offset, "<<submodelsize<<" tasks, "<<env_covariates<<" covariates, "<<genetic_tasks<<" SNPs, "<<genoveclen<<" vec length"<<endl;
  score = new float[n];
  for(int i=0;i<n;++i) score[i] = 0;
  betas = new float[submodelsize];
  deltas = new delta_t[submodelsize];
  means = new float[submodelsize];
  sds = new float[submodelsize];
  varcounts = new int[submodelsize];
  mask = new int[n];
  for(int i=0;i<n;++i) mask[i] = 1;
  // receive the task list
  //tasklist = new int[submodelsize];
  //rc = MPI_Recv(tasklist,submodelsize,MPI_INT,source,TAG_INITTASK,MPI_COMM_WORLD,&stat);
  //cerr<<"First task is "<<tasklist[0]<<endl;
  //for(int i=0;i<submodelsize;++i) tasklist[i]-=slave_offset-env_covariates;
  // receive the affection vector
  disease_status = new int[n];
  rc = MPI_Recv(disease_status,n,MPI_INT,source,TAG_INITAFF,MPI_COMM_WORLD,&stat);
  // receive the covariates
  covariates = new float[env_covariates*n];
  rc = MPI_Recv(covariates,env_covariates*n,MPI_FLOAT,source,TAG_INIT_COV,MPI_COMM_WORLD,&stat);
  //int totalrows = submodelsize;
  unsigned int matsize = genetic_tasks*genoveclen;
  this->genocharmatrix = new char[matsize];
  ofs<<"Allocated a char matrix of dimension: "<<matsize<<endl;
  //for(int i=0;i<totalrows;++i){
  //unsigned int offset = i*genoveclen;
  rc = MPI_Recv(this->genocharmatrix, matsize,MPI_CHAR,source,TAG_INITDESIGN,MPI_COMM_WORLD,&stat);
  //}
  ofs<<"Received genotypes of "<<genetic_tasks<<" total rows and veclen "<<genoveclen<<".\n";
  if (algorithm==ALGORITHM_GREEDY ){
    // create the padded genomatrix for use in the GPU
    ofs<<"Packing genotypes into container.\n";
    packedstride =  (n/512+(n%512>0)) * 512 / 16; //PACKED_SUBJECT_STRIDE; // 7168 / 16
    ofs<<"Packed genotype stride is "<<packedstride<<"\n";
    int packedgenolen = genetic_tasks * packedstride;
    ofs<<"Total packed genolen is "<<packedgenolen<<endl;
    packedgeno_matrix = new packedgeno_t[packedgenolen];
    int totalcols = n;
    for(int row=0;row<genetic_tasks;++row){
      int col=0;
      int p=0;
      for(int chunk=0;chunk<packedstride;++chunk){
        for(int subchunk=0;subchunk<4;++subchunk){
          if (col<totalcols){
            unsigned int geno_index = row*genoveclen+p++;
            packedgeno_matrix[row*packedstride+chunk].geno[subchunk] = genocharmatrix[geno_index];
            col+=4;
          }
        }
      }
    }
  }
  if (settings->use_gpu){
    #ifdef USE_GPU
      init_gpu();
    #endif
  }
}


float MpiLasso2::getLambda(int index){
   //cerr<<"Entering getLambda for index "<<index<<"\n";
   float dotprod = 0;
   for(int i=0;i<z_rank;++i){
    //cerr<<"i "<<i<<endl;
     dotprod+=z_matrix[index*z_rank+i]*abs(pi_vec[i]);
   }
   float lambda = 1/exp(dotprod);
   //if (lambda<1) lambda = 0;
   //ofs<<"Lambda for index: "<<index<<" is "<<lambda<<endl;
   //return .01;
   //
   return lambda;
   
}

void MpiLasso2::listen(){
  bool done = false;
  while(!done){
    int rpc_code;
    MPI_Recv(&rpc_code,1,MPI_INT,source,TAG_RPC,MPI_COMM_WORLD,&stat);
    //ofs<<"Got RPC code "<<rpc_code<<"\n";
    switch(rpc_code){
      case RPC_GREEDY:
        greedy_slave();
        break;
      case RPC_INIT_DIM:
        init_slave_dim();
        break;
      case RPC_BETA_UPDATE:
        beta_update_slave();
        break;
      case RPC_INIT_GREEDY:
        init_greedy();
        break;
      case RPC_END:
        cleanup_slave();
        done = true;
        break;
    }
  }
  //cerr<<"I am done listening at "<<mpi_rank<<endl;
  return;
}


void MpiLasso2::init_gpu(){
#ifdef USE_GPU
  personchunks = n/BLOCK_WIDTH + (n%BLOCK_WIDTH!=0);
  cl_int err;
  std::vector<cl::Platform> platforms;
  err = cl::Platform::get(&platforms);
  // Iterate over platforms
  ofs<<"Available OpenCL platforms: " << platforms.size() << endl;
  for(uint i=0;i<platforms.size();++i){
    ofs<<"Platform ID "<<i<<" has name "<<platforms[i].getInfo<CL_PLATFORM_NAME>().c_str()<<endl;
    if (settings->platform_id==i) ofs<<"*** SELECTED PLATFORM ABOVE ***\n";
    vector<cl::Device> devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    ofs<<"with total devices: " << devices.size() << endl;
    for (uint j = 0; j <devices.size(); ++j) {
      ofs<<"Device "<<j<<" is type ";
      string deviceName = devices[j].getInfo<CL_DEVICE_NAME>();
      cl_device_type dtype = devices[j].getInfo<CL_DEVICE_TYPE>();
      switch(dtype){
        case CL_DEVICE_TYPE_GPU:
          ofs<<"GPU\n";
          break;
        case CL_DEVICE_TYPE_CPU:
          ofs<<"CPU\n";
          break;
      }
      if (settings->platform_id==i && settings->device_id==j) ofs<<"*** SELECTED DEVICE ABOVE ***\n";
    }
  }
  // create a context from specified platform
  cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)(platforms[settings->platform_id])(),0};
  //cl::Context context;
  opencl_info.context = cl::Context(CL_DEVICE_TYPE_GPU,cps);
  // extract the devices from this platform
  vector<cl::Device> devices = opencl_info.context.getInfo<CL_CONTEXT_DEVICES>();
  opencl_info.command_queue = cl::CommandQueue(opencl_info.context,devices[settings->device_id],0,&err);
  clSafe(err,"creating command queue");
  // Create a program from the kernel source
  // Load the kernel source code into the array source_str
  ifstream ifs(settings->kernel_path.data());
  if (!ifs.is_open()){
    cerr<<"Could not load the kernel path "<<settings->kernel_path<<endl;
    exit(1);
  }
  string source_str(istreambuf_iterator<char>(ifs),(istreambuf_iterator<char>()));
  cl::Program::Sources source(1,make_pair(source_str.c_str(),source_str.length()+1));
  cl::Program program(opencl_info.context,source);
  ostringstream oss_build_options;
  err = program.build(devices,oss_build_options.str().data(),NULL);
  if (err!=CL_SUCCESS) {
      cerr<<"Build failed:\n";
      string buffer;
      program.getBuildInfo(devices[0],CL_PROGRAM_BUILD_LOG,&buffer);
      cerr<<buffer<<endl;
  }
  // Create the OpenCL kernels
  opencl_info.kernel_compute_geno_sum = cl::Kernel(program, "compute_geno_sum", &err);
  clSafe(err,"created kernel geno sum");
  opencl_info.kernel_compute_geno_mean_sd = cl::Kernel(program, "compute_geno_mean_sd", &err);
  clSafe(err,"created kernel geno mean_sd");
  opencl_info.kernel_update_best_beta = cl::Kernel(program, "update_best_beta", &err);
  clSafe(err,"created kernel update_best_beta");
  opencl_info.kernel_update_score = cl::Kernel(program, "update_score", &err);
  clSafe(err,"created kernel update_score");
  opencl_info.kernel_compute_gradient_hessian = cl::Kernel(program, "compute_gradient_hessian", &err);
  clSafe(err,"create kernel gradient hessian");
  opencl_info.kernel_compute_delta_beta = cl::Kernel(program, "compute_delta_beta", &err);
  clSafe(err,"create kernel delta_beta");
  opencl_info.kernel_compute_likelihoods = cl::Kernel(program, "compute_likelihoods", &err);
  clSafe(err,"created kernel compute_likelihoods");
  opencl_info.kernel_reduce_likelihoods = cl::Kernel(program, "reduce_likelihoods", &err);
  clSafe(err,"created kernel reduce_likelihoods");
  opencl_info.kernel_best_delta = cl::Kernel(program, "best_delta", &err);
  clSafe(err,"created kernel best_delta");
  opencl_info.kernel_zero_beta = cl::Kernel(program, "zero_beta", &err);
  clSafe(err,"created kernel zero_beta");
  opencl_info.kernel_zero_score = cl::Kernel(program, "zero_score", &err);
  clSafe(err,"created kernel zero_score");

  // Create the OpenCL Buffers
  // Universal buffers

  opencl_info.betas_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE, submodelsize * sizeof(float), NULL, &err);
  clSafe(err, "create buffer for betas");
  opencl_info.mean_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE, submodelsize * sizeof(float), NULL, &err);
  clSafe(err, "create buffer for means");
  opencl_info.sd_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE, submodelsize * sizeof(float), NULL, &err);
  clSafe(err, "create buffer for sds");
  opencl_info.group_indices_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, submodelsize * sizeof(int), NULL, &err);
  clSafe(err, "create buffer for group_indices");
  opencl_info.l2_norms_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, groups * sizeof(float), NULL, &err);
  clSafe(err, "create buffer for L2 norms");
  //opencl_info.l2_norms_big_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, totaltasks * sizeof(float), NULL, &err);
  //clSafe(err, "create buffer for L2 norms big");
  opencl_info.mask_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, n*sizeof(int), NULL, &err);
  clSafe(err, "create buffer for mask");
  err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.mask_mem_obj, CL_TRUE, 0,  sizeof(int)*n, mask , NULL, NULL);
  clSafe(err,"write mask");
  opencl_info.n_subset_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
  clSafe(err, "create buffer for subset n");
  opencl_info.aff_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, n*sizeof(int), NULL, &err);
  clSafe(err, "create buffer for aff");
  opencl_info.cov_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, n*env_covariates*sizeof(float), NULL, &err);
  clSafe(err, "create buffer for cov");
  opencl_info.packedgeno_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, (genetic_tasks*packedstride)*sizeof(packedgeno_t), NULL, &err);
  clSafe(err, "create buffer for packed genotypes");

  // THE FOLLOWING ARE INITIALIZED
  //err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.aff_mem_obj, CL_TRUE, 0,  sizeof(int)*n, disease_status, NULL, NULL );
  //clSafe(err, "write buffer for aff");
  //err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.cov_mem_obj, CL_TRUE, 0,  sizeof(float)*n*env_covariates, covariates, NULL, NULL );
  //clSafe(err, "write buffer for cov");
  //cerr<<"Created buffer for packedgenotypes...\n";
  //err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.packedgeno_mem_obj, CL_TRUE, 0,  sizeof(packedgeno_t)*genetic_tasks*packedstride, packedgeno_matrix, NULL, NULL );
  //clSafe(err, "write buffer for packed genotypes");
  //cerr<<"Wrote data to buffer for packedgenotypes\n";

  opencl_info.mean_sd_flag_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, 1 * sizeof(int), NULL, &err); 
  clSafe(err, "create buffer for mean sd flag");
  opencl_info.mean_sd_chunks_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE, submodelsize * sizeof(float)*personchunks, NULL, &err);
  clSafe(err, "create buffer for mean sd chunks");
  opencl_info.score_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE, n*sizeof(float), NULL, &err);
  clSafe(err, "create buffer for score");
  opencl_info.score_num_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE, n*sizeof(float), NULL, &err);
  clSafe(err, "create buffer for scorenum");
  opencl_info.score_den_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE, n*sizeof(float), NULL, &err);
  clSafe(err, "create buffer for scoreden");
  opencl_info.taskoffset_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, 1 * sizeof(int), NULL, &err); 
  clSafe(err, "create buffer for task offset");
  opencl_info.gradient_chunks_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE, submodelsize * sizeof(float)*personchunks, NULL, &err);
  clSafe(err, "create buffer for gradient chunk");
  opencl_info.hessian_chunks_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE, submodelsize * sizeof(float)*personchunks, NULL, &err);
  clSafe(err, "create buffer for hessian chunk");
  opencl_info.currLL_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, 1 * sizeof(float), NULL, &err); 
  clSafe(err, "create buffer for current LL");
  opencl_info.delta_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_WRITE_ONLY, submodelsize*sizeof(delta_t), NULL, &err);
  clSafe(err, "create buffer for delta object");
  opencl_info.bestvar_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE,  sizeof(best_t), NULL, &err);
  clSafe(err, "create buffer for best variable object");
  opencl_info.bestLLDelta_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE,  sizeof(float), NULL, &err);
  clSafe(err, "create buffer for best delta LL");
  opencl_info.lasso_tuning_param_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_ONLY, 1 * sizeof(tuning_param_t), NULL, &err); 
  clSafe(err, "create buffer for LASSO tuning");
  opencl_info.loglike_chunks_mem_obj = cl::Buffer(opencl_info.context, CL_MEM_READ_WRITE, submodelsize * sizeof(float)*personchunks, NULL, &err);
  clSafe(err, "created buffer for loglike chunk");
  cerr<<"Completed creating buffers...\n";

  // Set the OpenCL Kernel arguments
  int arg;
  // Universal kernels
  cerr<<"Creating kernel args...\n";
  arg = 0;
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, mpi_rank);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, env_covariates);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, n);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, submodelsize);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, personchunks);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, packedstride);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, opencl_info.mean_sd_flag_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, opencl_info.taskoffset_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, opencl_info.packedgeno_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, opencl_info.cov_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, opencl_info.mean_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, opencl_info.mean_sd_chunks_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, cl::__local(sizeof(packedgeno_t)*SMALL_BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, cl::__local(sizeof(float)*BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_sum.setArg( arg++, cl::__local(sizeof(float)*BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg");
  
  arg = 0;
  err = opencl_info.kernel_compute_geno_mean_sd.setArg( arg++, n);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_mean_sd.setArg( arg++, submodelsize);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_mean_sd.setArg( arg++, personchunks);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_mean_sd.setArg( arg++, opencl_info.mean_sd_flag_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_mean_sd.setArg( arg++, opencl_info.taskoffset_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_mean_sd.setArg( arg++, opencl_info.mean_sd_chunks_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_mean_sd.setArg( arg++, opencl_info.mean_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_mean_sd.setArg( arg++, opencl_info.sd_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_compute_geno_mean_sd.setArg( arg++, cl::__local(sizeof(float)*SMALL_BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg");

  arg = 0;
  err = opencl_info.kernel_update_best_beta.setArg( arg++, opencl_info.bestvar_mem_obj);
  clSafe(err,"clSetKernelArg_update_betabeta0");
  err = opencl_info.kernel_update_best_beta.setArg( arg++, opencl_info.betas_mem_obj);
  clSafe(err,"clSetKernelArg_update_betabeta1");

  arg = 0;
  err = opencl_info.kernel_update_score.setArg( arg++, n);
  clSafe(err,"clSetKernelArg_update_score0");
  err = opencl_info.kernel_update_score.setArg( arg++, opencl_info.aff_mem_obj);
  clSafe(err,"clSetKernelArg_update_score1");
  err = opencl_info.kernel_update_score.setArg( arg++, opencl_info.score_mem_obj);
  clSafe(err,"clSetKernelArg_update_score2");
  err = opencl_info.kernel_update_score.setArg( arg++, opencl_info.score_num_mem_obj);
  clSafe(err,"clSetKernelArg_update_score3");
  err = opencl_info.kernel_update_score.setArg( arg++, opencl_info.score_den_mem_obj);
  clSafe(err,"clSetKernelArg_update_score4");

  arg = 0;
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++,mpi_rank); 
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++,env_covariates); 
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++,n); 
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++,submodelsize); 
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++,personchunks); 
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++,packedstride); 
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.taskoffset_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.packedgeno_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.cov_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.score_num_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.score_den_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.mean_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.sd_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
//  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.debug_mem_obj);
//  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.gradient_chunks_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.hessian_chunks_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.mask_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, opencl_info.delta_mem_obj);
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, cl::__local(sizeof(packedgeno_t)*SMALL_BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, cl::__local(sizeof(float)*BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, cl::__local(sizeof(float)*BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_compute_gradient");
  err = opencl_info.kernel_compute_gradient_hessian.setArg( arg++, cl::__local(sizeof(float)*BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_compute_gradient");
  int kernelWorkGroupSize = opencl_info.kernel_compute_gradient_hessian.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[settings->device_id], &err);
  clSafe(err,"get workgroup size kernel compute gradient");
  cerr<<"Kernel compute gradient  work group size is "<<kernelWorkGroupSize<<endl;

  arg = 0;
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++,mpi_rank); 
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++,env_covariates); 
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++,submodelsize); 
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  //err = opencl_info.kernel_compute_delta_beta.setArg( arg++,lambda);
  //clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++,personchunks); 
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++, opencl_info.lasso_tuning_param_mem_obj);
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++, opencl_info.taskoffset_mem_obj);
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++, opencl_info.l2_norms_mem_obj);
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++, opencl_info.betas_mem_obj);
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++, opencl_info.delta_mem_obj);
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++, opencl_info.gradient_chunks_mem_obj);
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++, opencl_info.hessian_chunks_mem_obj);
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++, opencl_info.group_indices_mem_obj);
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++, cl::__local(sizeof(float)*SMALL_BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_compute_delta_beta_");
  err = opencl_info.kernel_compute_delta_beta.setArg( arg++, cl::__local(sizeof(float)*SMALL_BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_compute_delta_beta_");

  // set some of the arguments in kernel proposeLL
  
  arg = 0;
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, mpi_rank);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, env_covariates);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, n);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, submodelsize);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, personchunks);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, logistic);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, packedstride);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.taskoffset_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.currLL_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.packedgeno_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.cov_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.aff_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.score_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.betas_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.mean_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.sd_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.delta_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.loglike_chunks_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, opencl_info.mask_mem_obj);
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, cl::__local(sizeof(packedgeno_t)*SMALL_BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, cl::__local(sizeof(float)*BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_compute_likelihoods");
  err = opencl_info.kernel_compute_likelihoods.setArg( arg++, cl::__local(sizeof(float)*BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_compute_likelihoods");

  arg = 0;
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, n);
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, submodelsize);
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, personchunks);
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, logistic);
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, opencl_info.taskoffset_mem_obj);
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, opencl_info.currLL_mem_obj);
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, opencl_info.betas_mem_obj);
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, opencl_info.delta_mem_obj);
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  //err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, opencl_info.tasklist_mem_obj);
  //clSafe(err,"clSetKernelArg_reduce_likelihoods");
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, opencl_info.loglike_chunks_mem_obj);
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, opencl_info.n_subset_mem_obj);
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  err = opencl_info.kernel_reduce_likelihoods.setArg( arg++, cl::__local(sizeof(float)*SMALL_BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_reduce_likelihoods");
  // SET KERNEL ARGUMENTS FOR BEST DELTA_BETA
  
  arg = 0;
  err = opencl_info.kernel_best_delta.setArg( arg++, n);
  clSafe(err,"clSetKernelArg_best_delta");
  err = opencl_info.kernel_best_delta.setArg( arg++, submodelsize);
  clSafe(err,"clSetKernelArg_best_delta");
  //err = opencl_info.kernel_best_delta.setArg( arg++, submodelsize/BLOCK_WIDTH+1);
  err = opencl_info.kernel_best_delta.setArg( arg++, submodelsize/SMALL_BLOCK_WIDTH+1);
  clSafe(err,"clSetKernelArg_best_delta");
  //err = opencl_info.kernel_best_delta.setArg( arg++, opencl_info.tasklist_mem_obj);
  //clSafe(err,"clSetKernelArg_best_delta");
  err = opencl_info.kernel_best_delta.setArg( arg++, opencl_info.delta_mem_obj);
  clSafe(err,"clSetKernelArg_best_delta");
  err = opencl_info.kernel_best_delta.setArg( arg++, opencl_info.bestvar_mem_obj);
  clSafe(err,"clSetKernelArg_best_delta");
  err = opencl_info.kernel_best_delta.setArg( arg++, opencl_info.bestLLDelta_mem_obj);
  clSafe(err,"clSetKernelArg_best_delta");
  err = opencl_info.kernel_best_delta.setArg( arg++, opencl_info.mean_mem_obj);
  clSafe(err,"clSetKernelArg_best_delta");
  err = opencl_info.kernel_best_delta.setArg( arg++, opencl_info.sd_mem_obj);
  clSafe(err,"clSetKernelArg_best_delta");
  err = opencl_info.kernel_best_delta.setArg( arg++, cl::__local(sizeof(delta_t)*BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_best_delta");
  err = opencl_info.kernel_best_delta.setArg( arg++, cl::__local(sizeof(int)*BLOCK_WIDTH));
  clSafe(err,"clSetKernelArg_best_delta");

  arg = 0;
  err = opencl_info.kernel_zero_beta.setArg( arg++, n);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_zero_beta.setArg( arg++, submodelsize);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_zero_beta.setArg( arg++, opencl_info.taskoffset_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_zero_beta.setArg( arg++, opencl_info.betas_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_zero_beta.setArg( arg++, opencl_info.delta_mem_obj);
  clSafe(err,"clSetKernelArg");

  arg = 0;
  err = opencl_info.kernel_zero_score.setArg( arg++, n);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_zero_score.setArg( arg++, submodelsize);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_zero_score.setArg( arg++, opencl_info.score_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_zero_score.setArg( arg++, opencl_info.score_num_mem_obj);
  clSafe(err,"clSetKernelArg");
  err = opencl_info.kernel_zero_score.setArg( arg++, opencl_info.score_den_mem_obj);
  clSafe(err,"clSetKernelArg");
  cerr<<"Done init gpu\n";
#endif
}


float MpiLasso2::c2g(char * genocharmatrix,int genoveclen,int snp,int person){
  float geno=9;
  int val = genocharmatrix[snp*genoveclen+person/4];
  int shifts =  (person % 4);
  for(int shift=0;shift<shifts;++shift) val = val>>2;
  val = val & 3;
  switch (val){
    case 0:
      geno = 0;
      break;
    case 2:
      geno = 1;
      break;
    case 3:
      geno = 2;
      break;
    case 1:
      geno = 9;
      break;
  }
  return geno;
}

void MpiLasso2::beta_update_slave(){
  //int index;
  int dim[MPI_INT_ARR];
  float deltabeta[MPI_FLOAT_ARR];
  rc = MPI_Recv(dim,MPI_INT_ARR,MPI_INT,source,TAG_BETA_UPDATE_INDEX,MPI_COMM_WORLD,&stat);
  rc = MPI_Recv(deltabeta,MPI_FLOAT_ARR,MPI_FLOAT,source,TAG_BETA_UPDATE_VAL,MPI_COMM_WORLD,&stat);
  rc = MPI_Recv(score,n,MPI_FLOAT,source,TAG_UPDATE_SCORE,MPI_COMM_WORLD,&stat);
  // IF NECESSARY UPDATE BETA
  // IF NECESSARY UPDATE BETA
  bestsubmodelindex = dim[0];
  bestfullmodelindex = dim[1];
  //bestfullmodelindex2 = dim[2];
  bestdeltabeta = deltabeta[0];
  currentLL+=deltabeta[1];
  bestmean = deltabeta[2];
  bestsd = deltabeta[3];
  ofs<<"Host announced best delta beta of "<<bestdeltabeta<<" for index "<<bestsubmodelindex<<" and full model index "<<bestfullmodelindex<<" and mean/sd "<<bestmean<<","<<bestsd<<endl;

#ifdef USE_GPU
  //ofs<<"Updating GPU with L2 norm "<<tuning_param.l2norm<<endl;
  //cl_int err;
    //err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.lasso_tuning_param_mem_obj, CL_TRUE, 0,  sizeof(tuning_param_t), &tuning_param , NULL, NULL );
    //clSafe(err,"CommandQueue::Update LASSO penalties");
/**
float executionTime(cl_event & event){
  cl_ulong start, end;
  clWaitForEvents(1,&event);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  return  (end - start) ;
}
**/
#endif
}

void printExecutionCode2(const char * mesg,cl::Event event){
  cl_int code = event.wait();
  //opencl_info.command_queue.enqueueBarrier();
  //cl_int code;
  event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS,&code);
  if (code==CL_COMPLETE){
    //cerr<<" complete\n"; 
  }else{
    cerr<<mesg<<" has error with code is "<<code<<endl;
    exit(1);
  }
}

void MpiLasso2::convertgeno(int arrdim,int genoindex,float * genovec){
  if (genoindex>=0){
    if (arrdim==1){
      //ofs<<"Converting genotypes for SNP "<<genoindex<<endl;
      io->fetchgeno(genocharmatrix,genoveclen,genoindex,genovec,n);
    }else if (arrdim==2){
      io->fetchgeno(data.genomatrix,genoveclen,genoindex,genovec,n);
    }else{
      throw "Invalid array dimension";
    }
  }else if (genoindex<0){
    for(int i=0;i<n;++i){
       genovec[i] = covariates[(abs(genoindex)-1)*n+i];
    }
  }
}

bool MpiLasso2::load_mean_sd(int suffix,int len,float * mean_vec,
float * sd_vec){
  ofs<<"Loading mean_vec/SD from file\n";
  ostringstream oss;
  oss<<"means_sd."<<suffix;
  ifstream ifs(oss.str().data());
  if (!ifs.is_open()) return false;
  for(int i=0;i<len;++i){
    string line;
    getline(ifs,line);
    istringstream iss(line);
    iss>>mean_vec[i]>>sd_vec[i];
    //cerr<<i<<"\t"<<mean_vec[i]<<"\t"<<sd_vec[i]<<endl;
  }
  ifs.close();
  if (mpi_rank && settings->use_gpu){
    #ifdef USE_GPU
    int err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.mean_mem_obj, CL_TRUE, 0,  sizeof(float)*len, mean_vec , NULL, NULL );
    clSafe(err,"CommandQueue::write_mean()");
    err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.sd_mem_obj, CL_TRUE, 0,  sizeof(float)*len, sd_vec , NULL, NULL );
    clSafe(err,"CommandQueue::write_sd()");
    #endif
  }
  return true;
}

void MpiLasso2::compute_mean_sd(int genomatrix_dim){
  if(mpi_rank==0){
    int rpc_code = RPC_STANDARDIZE;
    for (int dest=1;dest<mpi_numtasks;++dest){
      rc = MPI_Send(&rpc_code,1,MPI_INT,dest,TAG_RPC,MPI_COMM_WORLD);
    }
  }else{
    ofs<<"Computing means and SDs\n";
    if (settings->use_gpu){
      #ifdef USE_GPU
      ofs<<"Computing GPU means\n";
      ostringstream oss;
      oss<<"means_sd."<<mpi_rank;
      ofstream ofs2(oss.str().data());
      // We want to compute all the means for standardization
      int err;
      for(int mean_sd_flag = 0;mean_sd_flag<2;++mean_sd_flag){
        ofs<<"Writing sd flag "<<mean_sd_flag<<endl;
        err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.mean_sd_flag_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &mean_sd_flag , NULL, NULL );
        int snpchunksize = GRID_WIDTH/BLOCK_WIDTH;
        int griddepth = submodelsize/snpchunksize+(submodelsize%snpchunksize>0);
        for (int taskoffset=0;taskoffset<griddepth;++taskoffset){
          err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL , NULL );
          err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_compute_geno_sum,cl::NullRange,cl::NDRange(GRID_WIDTH,personchunks),cl::NDRange(BLOCK_WIDTH,1),NULL,NULL);
          clSafe(err,"compute_geno_sum kernel");
        }
        int smallsnpchunksize = GRID_WIDTH/SMALL_BLOCK_WIDTH+(GRID_WIDTH%SMALL_BLOCK_WIDTH>1);
        int smallgriddepth = submodelsize/smallsnpchunksize+(submodelsize%smallsnpchunksize>0);
        for (int taskoffset=0;taskoffset<smallgriddepth;++taskoffset){
          err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL, NULL );
          err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_compute_geno_mean_sd,cl::NullRange,cl::NDRange(GRID_WIDTH,1),cl::NDRange(SMALL_BLOCK_WIDTH,1),NULL,NULL);
          clSafe(err,"compute_mean_sd kernel");
        }
      }
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.mean_mem_obj, CL_TRUE, 0, sizeof(float)*submodelsize,means,NULL,NULL);
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.sd_mem_obj, CL_TRUE, 0, sizeof(float)*submodelsize,sds,NULL,NULL);
      for(int i=0;i<submodelsize;++i){
        ofs2<<means[i]<<" "<<sds[i]<<endl;
      }
      ofs2.close();
      #endif
    }else{
      ofs<<"Computing CPU means\n";
      ostringstream oss;
      oss<<"means_sd."<<mpi_rank;
      ofstream ofs1(oss.str().data());
      // Compute the means and SDs
      float genovec1[n];
      //float genovec2[n];
      for (int j=0;j<submodelsize;++j){
        int genoindex = mpi_rank>1?j:j-env_covariates;
        convertgeno(genomatrix_dim,genoindex,genovec1);
        means[j] = 0;
        for(int i=0;i<n;++i){
          means[j]+= genovec1[i];
        }
        means[j]/=n;
        ofs1<<means[j];
        sds[j] = 0;
        for(int i=0;i<n;++i){
          if (genovec1[i]!=IO::MISSING_GENO ){
            sds[j]+= pow(genovec1[i]-means[j],2);
          }
        }
        //ofs<<","<<n_subset;
        sds[j]=sds[j]==0?1:sqrt(sds[j]/n);
        ofs1<<" "<<sds[j];
        ofs1<<endl;
        //cerr<<" "<<j<<":"<<means[j]<<","<<sds[j];
      }
      ofs1.close();
    }
  }
}

void MpiLasso2::zero_beta(){
  for(int j=0;j<totaltasks;++j) {
    betas[j] = 0;
  }
  for(int j=0;j<groups;++j){
    l2_norms[j] = 0;
  }
  ofs<<"Zeroed l2 norms\n";
  if (mpi_rank){
    if(settings->use_gpu){
      #ifdef USE_GPU
      if (algorithm==ALGORITHM_GREEDY){
        int err;
        for (int taskoffset=0;taskoffset<(submodelsize/GRID_WIDTH)+(submodelsize%GRID_WIDTH>0);++taskoffset){
          err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL, NULL);
          err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_zero_beta,cl::NullRange,cl::NDRange(GRID_WIDTH,1),cl::NDRange(BLOCK_WIDTH,1),NULL,NULL);
          clSafe(err,"CommandQueue::enqueueNDRangeKernelZeroBeta()");
        }
        ofs<<"Zeroed beta\n";
        //err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.l2_norms_mem_obj, CL_TRUE, 0,  sizeof(float)*groups,l2_norms, NULL, NULL );
        //clSafe(err,"CommandQueue::write buffer for L2 norms");
        //err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.l2_norms_mem_obj, CL_TRUE, 0, sizeof(float)*groups,l2_norms);
        //clSafe(err,"CommandQueue::read buffer for L2 norms");
        for(int i=0;i<groups;++i){
          //ofs<<"Group:\t"<<i<<"\t"<<l2_norms[i]<<endl;
        }

        int scoreworksize = (n/BLOCK_WIDTH+1)*BLOCK_WIDTH;
        err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_zero_score,cl::NullRange,cl::NDRange(scoreworksize,1),cl::NDRange(BLOCK_WIDTH,1),NULL,NULL);
        clSafe(err,"CommandQueue::enqueueNDRangeKernelZeroScore()");
        ofs<<"Zeroed score\n";
        // NEED TO ADD KERNEL TO ZERO L2 NORMS
      }
      #endif
    }else{
      for(int j=0;j<submodelsize;++j) {
        betas[j] = 0.;
        if (algorithm==ALGORITHM_GREEDY){
          deltas[j].delta_beta = 0.;
          deltas[j].delta_LL = 0.;
        }
      }
      ofs<<"Zeroed beta\n";
      for(int i=0;i<n;++i) score[i] = 0;
      ofs<<"Zeroed score\n";
    }
  }
}

void MpiLasso2::fetchgeno_subjectmajor(int subjectindex, float * genovec){
  int j = 0;
  for(int b = 0;b<veclen_subjectmajor;++b){
    char c = genocharmatrix_subjectmajor[subjectindex*veclen_subjectmajor+b];
    for (int shift = 0;shift<4;++shift){
      int val = ((int)c)>>(2*shift) & 3;
      if (val==3) ++val;
      genovec[j++] = val;
    }
  }
}

void MpiLasso2::init_greedy(){
  if (mpi_rank==0){
  }else{
    ++iter;
    zero_beta();
    float tuning_arr[MPI_FLOAT_ARR];
    rc = MPI_Recv(tuning_arr,MPI_FLOAT_ARR,MPI_FLOAT,source,TAG_UPDATE_TUNING_PARAMS,MPI_COMM_WORLD,&stat);
    //float lambda = tuning_arr[0];
    rc = MPI_Recv(mask,n,MPI_INT,source,TAG_UPDATE_MASK,MPI_COMM_WORLD,&stat);
    int n_subset = 0;
    for(int i=0;i<n;++i) n_subset+=mask[i];
    // COMPUTE THE BASE LIKELIHOOD
    nullLL = 0;
    float l = 1;
    for(int i=0;i<n;++i){
      if (mask[i]){
        float pY = .5;
        float pY2 = (disease_status[i]==1)?pY:1-pY;
        if (l*pY2==0){
          nullLL+=log(l);
          l=pY2;
        }else{
          l*=pY2;
        }
        //if (disease_status[i]==1) nullLL += log(pY); else nullLL += log(1-pY);
      }
    }
    nullLL+=log(l);
    ofs<<"Subset sample size is "<<n_subset<<" and base LL: "<<nullLL<<endl;
    bestsubmodelindex = 0;
    bestfullmodelindex = 0;
    bestdeltabeta = 0.;
    bestdeltaLL = 0.;
    bestmean = 0;
    bestsd = 1;
    if (settings->use_gpu){
      #ifdef USE_GPU
      cl_int err;
      err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.mask_mem_obj, CL_TRUE, 0,  sizeof(int)*n, mask, NULL, NULL );
      clSafe(err, "write buffer for mask");
      err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.n_subset_mem_obj, CL_TRUE, 0,  sizeof(int), &n_subset, NULL, NULL );
      clSafe(err, "write buffer for subset len");
  
      eventList0.clear();
      bool debug1 = false;
      if (debug1){
        float tempscore[n];
        float tempscore1[n];
        float tempscore2[n];
        err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.score_mem_obj, CL_TRUE, 0, n*sizeof(float),tempscore);
        err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.score_num_mem_obj, CL_TRUE, 0, n*sizeof(float),tempscore1);
        err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.score_den_mem_obj, CL_TRUE, 0, n*sizeof(float),tempscore2);
        for(int i=0;i<n;++i){
          cerr<<i<<" init score "<<tempscore[i]<<" "<<tempscore1[i]<<" "<<tempscore2[i]<<endl;
        }
        cerr<<endl;
        exit(0);
      }
      //int snpchunksize = GRID_WIDTH/BLOCK_WIDTH;
      //for (int taskoffset=0;taskoffset<(submodelsize/snpchunksize)+(submodelsize%snpchunksize>0);++taskoffset){
      //  //cerr<<"Task offset: "<<taskoffset<<endl;
      //  cl::Event task_event;
      //  err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL, &task_event );
      //  cl::Event e_weight;
      //  err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_update_weights,cl::NullRange,cl::NDRange(GRID_WIDTH,1),cl::NDRange(BLOCK_WIDTH,1),NULL,&e_weight);
      //  //printExecutionCode("UpdateWeight",e_weight);
      //  clSafe(err,"CommandQueue::Update LASSO weights");
      // }
      bool debug2 = false;
      if (debug2){
        cerr<<"Weights:";
        float zscores[submodelsize];
        float ascores[submodelsize];
        err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.z_means_mem_obj, CL_TRUE, 0, submodelsize*sizeof(float),zscores);
        err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.a_means_mem_obj, CL_TRUE, 0, submodelsize*sizeof(float),ascores);
        for(int i=0;i<submodelsize;++i){
           cerr<<i<<"  zscore/ascore:"<<zscores[i]<<"/"<<ascores[i]<<endl;
        }
        cerr<<endl;
        exit(1);
      }
      #endif
    }
    //if (!load_mean_sd(mpi_rank,submodelsize,means,sds)) compute_mean_sd(1);
    for(int i=0;i<submodelsize;++i) varcounts[i] = 0;
    currentLL = nullLL;
    ofs<<"Done init slave greedy"<<endl;
    // end MPI SLAVE
  }
}

void MpiLasso2::greedy_slave(){
  ofs<<"Current LL is "<<currentLL<<endl;
  if (settings->use_gpu){
    #ifdef USE_GPU
    double start = clock();
    cl_int err;
    bool debug1a = false;
    if (debug1a){
      float tempscore[n];
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.score_mem_obj, CL_TRUE, 0, n*sizeof(float),tempscore);
      ofs<<"score before beta update: ";
      for(int i=0;i<n;++i){
        ofs<<" "<<i<<":"<<tempscore[i];
      }
      ofs<<endl;
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.betas_mem_obj, CL_TRUE, 0, submodelsize*sizeof(float),betas);
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.delta_mem_obj, CL_TRUE, 0, submodelsize*sizeof(delta_t),deltas);
      ofs<<"deltas before beta update: ";
      for(int i=0;i<submodelsize;++i){
        ofs<<" "<<i<<":"<<betas[i]<<","<<deltas[i].delta_beta<<","<<deltas[i].delta_LL;
      }
      ofs<<endl;
      //exit(0);
      if(bestdeltabeta!=0){
      //exit(0);
      }
    }

    int snpchunksize = GRID_WIDTH/BLOCK_WIDTH;
    best_t bestvar;
    bestvar.best_delta_beta = bestdeltabeta;
    bestvar.best_submodel_index = bestsubmodelindex;
    bestvar.best_genoindex_1 = bestfullmodelindex;
    bestvar.mean = bestmean;
    bestvar.sd = bestsd;
    ofs<<"Assigning index "<<bestsubmodelindex<<" with "<<bestdeltabeta<<endl;
    if (bestsubmodelindex>-1 && bestdeltabeta!=0){
      err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.bestvar_mem_obj, CL_TRUE, 0,  sizeof(best_t), &bestvar , NULL, NULL );
      // update the delta beta of the best SNP
      err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_update_best_beta,cl::NullRange,cl::NDRange(SMALL_BLOCK_WIDTH,1),cl::NDRange(SMALL_BLOCK_WIDTH,1),NULL,NULL);
      clSafe(err,"CommandQueue::enqueueNDRangeKernelUpdateBestBeta()");
    }
    err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.score_mem_obj, CL_TRUE, 0,  sizeof(float)*n,score, NULL, NULL );
    clSafe(err,"CommandQueue::write buffer for score");
    // launch the kernel for updating the score based on the new beta
    int scoreworksize = (n/BLOCK_WIDTH+1)*BLOCK_WIDTH;
    err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_update_score,cl::NullRange,cl::NDRange(scoreworksize,1),cl::NDRange(BLOCK_WIDTH,1),NULL,NULL);
    clSafe(err,"CommandQueue::enqueueNDRangeKernelUpdateScore()");


    bool debug1 = false;
    if (debug1){
      float tempscore[n];
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.score_mem_obj, CL_TRUE, 0, n*sizeof(float),tempscore);
      ofs<<"score after beta update: ";
      for(int i=0;i<n;++i){
        ofs<<" "<<i<<":"<<tempscore[i];
      }
      ofs<<endl;
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.betas_mem_obj, CL_TRUE, 0, submodelsize*sizeof(float),betas);
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.delta_mem_obj, CL_TRUE, 0, submodelsize*sizeof(delta_t),deltas);
      ofs<<"deltas after beta update: ";
      for(int i=0;i<submodelsize;++i){
        ofs<<" "<<i<<":"<<betas[i]<<","<<deltas[i].delta_beta<<","<<deltas[i].delta_LL;
      }
      ofs<<endl;
      //exit(0);
      if(bestdeltabeta!=0){
      //exit(0);
      }
    }
    err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.currLL_mem_obj, CL_TRUE, 0,  sizeof(float)*1,&currentLL, NULL, NULL );
    snpchunksize = GRID_WIDTH/BLOCK_WIDTH;
    // launch the CLG kernel that computes the delta beta at each SNP
    //for (int taskoffset=0;taskoffset<1;++taskoffset){
    for (int taskoffset=0;taskoffset<(submodelsize/snpchunksize)+(submodelsize%snpchunksize>0);++taskoffset){
      err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL, NULL );
      //cl::Event delta_beta_event;
      err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_compute_gradient_hessian,cl::NullRange,cl::NDRange(GRID_WIDTH,personchunks),cl::NDRange(BLOCK_WIDTH,1),NULL,NULL);
      //eventList2.push_back(delta_beta_event);
      //clSafe(err,"CommandQueue::enqueueNDRangeKernelCLG()");
    }
    bool debug2 = false;
    if (debug2){
      ofs<<"Gradient/Hessian\n";
      float * gradientchunks = new float[submodelsize*personchunks];
      float * hessianchunks = new float[submodelsize*personchunks];
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.gradient_chunks_mem_obj, CL_TRUE, 0, submodelsize*personchunks*sizeof(float),gradientchunks);
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.hessian_chunks_mem_obj, CL_TRUE, 0, submodelsize*personchunks*sizeof(float),hessianchunks);
      for(int i=0;i<submodelsize;++i){
        ofs<<"var:"<<i;
        float num = 0, den = 0;
        for(int j=0;j<personchunks;++j){
          ofs<<" "<<gradientchunks[i*personchunks+j]<<"/"<<hessianchunks[i*personchunks+j];
          num+=gradientchunks[i*personchunks+j];
          den+=hessianchunks[i*personchunks+j];
        }
        ofs<<" real delta: "<<(num/den);
        ofs<<endl;
      }
      delete[]gradientchunks;
      delete[]hessianchunks;
      ofs.close();
      exit(0);
    }
    int smallsnpchunksize = GRID_WIDTH/SMALL_BLOCK_WIDTH+1;
    for (int taskoffset=0;taskoffset<(submodelsize/smallsnpchunksize)+(submodelsize%smallsnpchunksize>0);++taskoffset){
      err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL, NULL );
      err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_compute_delta_beta,cl::NullRange,cl::NDRange(GRID_WIDTH,1),cl::NDRange(SMALL_BLOCK_WIDTH,1),NULL,NULL);
      clSafe(err,"CommandQueue::enqueueNDRangeKernel delta beta()");
    }
    bool debug2b = false;
    if (debug2b){
      ofs<<"Reduction for deltabeta\n";
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.delta_mem_obj, CL_TRUE, 0, submodelsize*sizeof(delta_t),deltas);
      int ms = 0;
      for(int i=0;i<submodelsize;++i){
        if (deltas[i].delta_beta!=0){
           ++varcounts[i];
           ++ms;
           ofs<<i<<":"<<deltas[i].delta_beta<<" "<<varcounts[i]<<endl;
        }
      }
      ofs<<"Modelsize: "<<ms<<endl;
      //exit(0);
    }
    // launch the kernel that computes the delta LL at each SNP
    for (int taskoffset=0;taskoffset<(submodelsize/snpchunksize)+(submodelsize%snpchunksize>0);++taskoffset){
      err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL, NULL );
      //cl::Event proposeLLevent;
      err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_compute_likelihoods,cl::NullRange,cl::NDRange(GRID_WIDTH,personchunks),cl::NDRange(BLOCK_WIDTH,1),NULL,NULL);
      //eventList3.push_back(proposeLLevent);
      clSafe(err,"CommandQueue::enqueueNDRangeKernelProposeLL()");
      //printExecutionCode("LogLike",proposeLLevent);
    }
    bool debug3 = false;
    if (debug3){
      ofs<<"Likelihood chunks\n";
      float * likelihood_chunks = new float[submodelsize*personchunks];
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.loglike_chunks_mem_obj, CL_TRUE, 0, submodelsize*personchunks*sizeof(float),likelihood_chunks);
      for(int i=0;i<submodelsize;++i){
        ofs<<i;
        for(int j=0;j<personchunks;++j){
          ofs<<" "<<likelihood_chunks[i*personchunks+j];
        }
        ofs<<endl;
      }
      delete[] likelihood_chunks;
      ofs.close();
      exit(0);
    }
    for (int taskoffset=0;taskoffset<(submodelsize/smallsnpchunksize)+(submodelsize%smallsnpchunksize>0);++taskoffset){
      err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL, NULL );
      err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_reduce_likelihoods,cl::NullRange,cl::NDRange(GRID_WIDTH,1),cl::NDRange(SMALL_BLOCK_WIDTH,1),NULL,NULL);
      clSafe(err,"CommandQueue::enqueueNDRangeKernel reduce LL()");
    }
    bool debug4 = false;
    if (debug4){
      ofs<<"Reduction for deltabeta,deltaLL\n";
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.betas_mem_obj, CL_TRUE, 0, submodelsize*sizeof(float),betas);
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.delta_mem_obj, CL_TRUE, 0, submodelsize*sizeof(delta_t),deltas);
      //err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.debug_mem_obj, CL_TRUE, 0, 2*submodelsize*sizeof(float),debugvec);
      delta_t truemax;
      truemax.delta_LL = 0;
      int trueindex = 0;
      for(int i=0;i<submodelsize;++i){
         ofs<<" "<<i<<","<<betas[i]<<","<<deltas[i].delta_beta<<","<<deltas[i].delta_LL;
         if (deltas[i].delta_LL>truemax.delta_LL){
           truemax = deltas[i];
           trueindex = i;
         }
      }
      ofs<<endl;
      ofs<<"True max at index: "<<trueindex<<" with deltaLL: "<<truemax.delta_LL<<endl;
      ofs.close();
      exit(0);
    }
    float bestLLDelta = 0;
    err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.bestLLDelta_mem_obj, CL_TRUE, 0,  sizeof(float)*1, &bestLLDelta , NULL, NULL );
    //cl::Event bestEvent;
    err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_best_delta,cl::NullRange,cl::NDRange(SMALL_BLOCK_WIDTH,1),cl::NDRange(SMALL_BLOCK_WIDTH,1),NULL,NULL);
    clSafe(err,"CommandQueue::enqueueNDRangeKernelBestDelta()");
    //printExecutionCode("Best",bestEvent);
    //executionTime(events[eventid-1]);
    best_t bestvar_found;
    err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.bestvar_mem_obj, CL_TRUE, 0, sizeof(best_t),&bestvar_found,NULL,NULL);
    bestsubmodelindex = bestvar_found.best_submodel_index;
    bestdeltabeta = bestvar_found.best_delta_beta;
    bestmean = bestvar_found.mean;
    bestsd = bestvar_found.sd;
    err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.bestLLDelta_mem_obj, CL_TRUE, 0, sizeof(float),&bestdeltaLL,NULL,NULL);
    ofs<<"Elapsed time in seconds: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
    #endif
  }else{ // NON GPU
    double starttime = clock();
    float delta_beta[submodelsize];
    float deltaLL[submodelsize];
    for(int i=0;i<submodelsize;++i) delta_beta[i] = deltaLL[i]=0;
    // UPDATE Beta vector
    ofs<<"Assigning index "<<bestsubmodelindex<<" with "<<bestdeltabeta<<endl;
    if (bestsubmodelindex>=0){
      betas[bestsubmodelindex]+=bestdeltabeta;
    }
    float genovec1[n];
    float score_num[n];
    float score_den[n];
    for(int i=0;i<n;++i){
      if (mask[i]){
        score_num[i]=disease_status[i]/(1.+exp(score[i])) ;
        score_den[i]=exp(score[i])/pow(1.+exp(score[i]),2);
      }
    }

    float max = 0;
    bestdeltabeta = 0;
    bestsubmodelindex = 0;
    for(int j=0;j<submodelsize;++j){
      int genoindex = mpi_rank>1?j:j-env_covariates;
      float delta_beta=0;
      float deltaLL = 0;
      convertgeno(1,genoindex,genovec1);
      float gradient=0;
      float hessian=0;
      //int s = 0;
      for(int i=0;i<n;++i){
        if (mask[i]){
          gradient+=(genovec1[i]-means[j])/sds[j]*score_num[i];
          hessian+=pow((genovec1[i]-means[j])/sds[j],2)*score_den[i];
          //++s;
        }
      }
      //ofs<<"Subset size "<<s<<endl;
      float l1_penalty = genoindex<0?0:tuning_param.lambda;
      if (betas[j]>LAMBDA_EPSILON){
        delta_beta = (gradient-l1_penalty)/hessian;
        if (betas[j]-delta_beta<0) delta_beta = 0;
      }else if (betas[j]<-LAMBDA_EPSILON){
        delta_beta = (gradient+l1_penalty)/hessian;
        if (betas[j]-delta_beta>0) delta_beta = 0;
      }else{
        if (gradient>l1_penalty){
          delta_beta = (gradient-l1_penalty)/hessian;
        }else if (gradient<-l1_penalty){
          delta_beta = (gradient+l1_penalty)/hessian;
        }else{
          delta_beta = 0.;
        }
      }
      if (delta_beta==0){
        deltaLL = 0;
      }else{
        double llike2=0;
        for(int i=0;i<n;++i){
          if (mask[i]){
            float s = (genovec1[i]-means[j])/sds[j]; // standardize
            float pY = exp((score[i]+delta_beta*s*disease_status[i])*disease_status[i]);
            pY/=(1+pY);
        //if (j==3 && i<50) ofs<<" "<<i<<":"<<disease_status[i]<<","<<genovec1[i];
            float pY2=disease_status[i]==1?pY:1-pY;
            llike2+=log(pY2);
          }
        }
        deltaLL = llike2;
        deltaLL=(llike2-currentLL)>=LL_EPSILON?(llike2-currentLL):0;
      }
      if (deltaLL>max){
        bestsubmodelindex = j;
        bestdeltabeta = delta_beta;
        bestmean = means[j];
        bestsd = sds[j];
        max = deltaLL;
      }
    }
    ofs<<endl;
    bestdeltaLL = max;
    ofs<<"Elapsed time in seconds: "<<(clock()-starttime)/CLOCKS_PER_SEC<<endl;
  }
  rc = MPI_Send(&bestsubmodelindex,1,MPI_INT,source,TAG_BEST_INDEX,MPI_COMM_WORLD);
  float bestDelta[MPI_FLOAT_ARR];
  bestDelta[0] = bestdeltaLL;
  bestDelta[1] = bestdeltabeta;
  bestDelta[2] = bestmean;
  bestDelta[3] = bestsd;
  ofs<<"Notifying host of Best index: "<<bestsubmodelindex<<" of delta beta "<<bestdeltabeta<<" and deltaLL "<<bestdeltaLL<<" with means/sd: "<<bestmean<<"/"<<bestsd<<endl;
  rc = MPI_Send(bestDelta,1,floatParamArrayType,source,TAG_BEST_DELTA,MPI_COMM_WORLD);
}

void MpiLasso2::cleanup_slave(){
  if (algorithm==ALGORITHM_GRADIENT){
    if (settings->use_gpu){
      #ifdef USE_GPU
      int err;
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.betas_mem_obj, CL_TRUE, 0, sizeof(float)*submodelsize,betas,NULL,NULL);
      clSafe(err, "read buffer for betas");
      #endif
    }
    ofs<<"Final Betas:\n";
    int ms = 0;
    for(int j=0;j<submodelsize;++j){
      if (betas[j]!=0){
        ++ms;
        ofs<<j<<":"<<betas[j]<<endl;
      }
    }
    ofs<<"Modelsize:"<<ms<<endl;
    //ofs<<endl;
  }
  MPI_Finalize();
}
