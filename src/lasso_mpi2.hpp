#include<mpi.h>
#include<fstream>
#ifdef USE_GPU
//  #include<CL/opencl.h>
#endif
using namespace std;


class lasso2_settings_t:public global_settings_t{
public:
  string pedfile;
  string snpfile;
  string genofile;
  string covariatesdatafile;
  string covariatesselectionfile;
  bool use_gpu;
  bool lasso_path;
  int subsamples;
  float lambda;
  uint platform_id;
  uint device_id;
  string kernel_path;
  string tasklist;

// defunct
  string algorithm;
  bool permutation_mode;
  string traitfile;
  string regression;
  string true_beta_path;
  string z_matrix;
  string a_matrix;
  string opencl_host;
};

//struct genotype_container{
//  char genos[4]; // stores 16 genotypes
//} __attribute__ ((aligned(4))); // 4 bytes, the minimum word

#ifdef USE_GPU

struct lasso2_opencl_info_t{
  // Create an OpenCL context
  cl::Context context;
  // Create a command queue
  cl::CommandQueue command_queue;


  //cl::Buffer tasklist_mem_obj; // stores interaction variables
  cl::Buffer mean_mem_obj; // stores interaction variables
  cl::Buffer sd_mem_obj; // stores interaction variables
  cl::Buffer packedgeno_mem_obj;
  cl::Buffer float_geno_mem_obj;
  cl::Buffer aff_mem_obj;
  cl::Buffer trait_mem_obj;
  cl::Buffer cov_mem_obj;
  cl::Buffer mask_mem_obj;
  cl::Buffer n_subset_mem_obj;
  cl::Buffer z_mem_obj; // stores z matrix
  cl::Buffer a_mem_obj; // stores a matrix

  cl::Buffer mean_sd_chunks_mem_obj;
  cl::Buffer gradient_chunks_mem_obj;
  cl::Buffer hessian_chunks_mem_obj;
  cl::Buffer loglike_chunks_mem_obj;
  cl::Buffer bestdelta_chunks_mem_obj;
  cl::Buffer score_mem_obj; // stores the score across persons

  cl::Buffer currLL_mem_obj; // updates kernel with latest likelihood 
  cl::Buffer score_num_mem_obj; // caches the constant of the 1st der
  cl::Buffer score_den_mem_obj; // caches the constant of the 2nd der
  cl::Buffer debug_mem_obj; // a sink for debugging global memory
  cl::Buffer taskoffset_mem_obj; // for iterating through chunks of variables
  cl::Buffer mean_sd_flag_mem_obj; // boolean flag
  cl::Buffer betas_mem_obj; // stores the Beta across SNPs

  cl::Buffer pi_mem_obj; // coefficients of Z matrix
  cl::Buffer rho_mem_obj; // coefficients of A matrix
  cl::Buffer lasso_tuning_param_mem_obj;  // contains LASSO related weights
  cl::Buffer delta_mem_obj; // stores all the delta LLs across SNPs
  cl::Buffer bestvar_mem_obj;
  cl::Buffer bestLLDelta_mem_obj;
  cl::Buffer z_means_mem_obj;
  cl::Buffer a_means_mem_obj;
  // for cyclic coordinate descent
  cl::Buffer deltabetas_mem_obj;
  cl::Buffer gradient_chunks_ccd_mem_obj;
  cl::Buffer hessian_chunks_ccd_mem_obj;
  // for gradient descent
  cl::Buffer eta_mem_obj;
  cl::Buffer x_beta_mem_obj;
  cl::Buffer gradient_mem_obj;
  cl::Buffer fitted_value_mem_obj;
  cl::Buffer current_person_mem_obj;
  cl::Buffer current_iteration_mem_obj;

  // for cyclic coordinate descent
  cl::Kernel kernel_compute_gradient_hessian_ccd;
  cl::Kernel kernel_compute_delta_beta_ccd;
  cl::Kernel kernel_update_score_ccd;

  // for gradient descent
  cl::Kernel kernel_estimate_fitted_value_genetic;
  cl::Kernel kernel_estimate_fitted_value_env;
  cl::Kernel kernel_reduce_x_beta;
  cl::Kernel kernel_new_gradient_genetic;
  cl::Kernel kernel_new_gradient_env;
  cl::Kernel kernel_gradient_shrink;

  cl::Kernel kernel_test;
  cl::Kernel kernel_update_weights;
  cl::Kernel kernel_update_best_beta;
  cl::Kernel kernel_update_score;
  cl::Kernel kernel_zero_score;
  cl::Kernel kernel_compute_geno_sum; // sum over genos in chunks
  cl::Kernel kernel_compute_geno_mean_sd; // reduction step

  cl::Kernel kernel_compute_gradient_hessian;
  cl::Kernel kernel_compute_delta_beta;
  cl::Kernel kernel_compute_likelihoods;
  cl::Kernel kernel_reduce_likelihoods;
  cl::Kernel kernel_best_delta;
  cl::Kernel kernel_zero_beta;
};
#endif

class MpiLasso2:public Analyzer{
public:
  void init(const ptree & pt);
  //MpiLasso(IO * io, lasso_settings_t * settings);
  MpiLasso2();
  ~MpiLasso2();
  void run();
  //virtual void lasso(double & logL);
private:
  #ifdef USE_GPU
  lasso2_opencl_info_t opencl_info;
  //master_opencl_info_t master_opencl_info;
  #endif
  lasso2_settings_t * settings;
  plink_data_t data;
  //global_settings_t *settings;
  // the following uniquely identify types of MPI messages
  // a remote procedure call
  static const int TAG_RPC  = 0;
  // init the Z matrix
  static const int TAG_INITZ  = 1;
  // samplesize, totalsnps, char_veclen, modelsize
  static const int TAG_INIT_DIM  = 2;
  // tasklist vector
  //static const int TAG_INITTASK  = 3;
  // disease status vector
  static const int TAG_INITAFF  = 4;
  // multiple genotype vectors used in design matrix
  static const int TAG_INITDESIGN  = 5;
  // length of char vector containing compressed genotypes
  static const int TAG_UPDATE_PI = 6;
  static const int TAG_UPDATE_TUNING_PARAMS = 7;
  static const int TAG_UPDATE_RHO = 8;
  static const int TAG_UPDATE_MASK = 9;
  static const int TAG_INIT_COV = 10;
  static const int TAG_INIT_CURRENT_PERSON = 11;
  static const int TAG_FITTED_VALUE = 12;
  // send new score to slaves
  static const int TAG_UPDATE_SCORE = 13;
  static const int TAG_INIT_CURRENT_GRADIENT = 14;
  // send best variable index to master
  static const int TAG_BEST_INDEX = 15;
  static const int TAG_BEST_DELTA = 16;
  // send log likelihood increase of best variable to master
  // send change in beta of best variable to master
  static const int TAG_UPDATE_ETA= 17;
  // notify slave of variable index to update beta
  static const int TAG_BETA_UPDATE_INDEX = 18;
  // notify slave of new value for beta
  static const int TAG_BETA_UPDATE_VAL = 19;
  
  static const int source = 0;
  // various remote procedure code IDs
  static const int RPC_INIT=0;
  static const int RPC_BETA_UPDATE=1;
  static const int RPC_GREEDY=2;
  static const int RPC_INIT_GREEDY=3;
  static const int RPC_END=4;
  static const int RPC_INIT_GRADIENT=5;
  static const int RPC_FITTED_VALUE=6;
  static const int RPC_NEW_GRADIENT=7;
  static const int RPC_UPDATE_ETA=8;
  static const int RPC_STANDARDIZE=9;
  static const int RPC_INIT_PARALLEL_GRADIENT=10;
  static const int RPC_PARALLEL_FITTED_VALUE=11;
  static const int RPC_PARALLEL_NEW_GRADIENT=12;
  static const int RPC_GRADIENT_SHRINK=13;
  static const int RPC_FETCH_FITTED_VALUE=14;
  // the LASSO tuning constant
  
  
  static const int ALGORITHM_GRADIENT=0;
  static const int ALGORITHM_GREEDY=1;
  static const int ALGORITHM_CCD=2;
  static const int ALGORITHM_STANDARDIZE=3;
  static const int ALGORITHM_SUBJECT_MAJOR=4;
  static const int ALGORITHM_PARALLEL_GRADIENT=5;
  int algorithm;
  int n;
  int n_subset;
  int genoveclen;
  int totalsnps;
  int totaltasks;
  int submodelsize;
  int bestsubmodelindex ;
  int bestfullmodelindex;
  float bestdeltabeta ;
  float bestdeltaLL;
  float bestmean;
  float bestsd;
  float nullLL,currentLL;
  int mpi_rank,mpi_numtasks,rc;
  int slaves;
  int replicates;
  int iter;
  MPI_Status stat;
  //MPI_Datatype * tasklistIntArrayType;
  MPI_Datatype * charArrayType;
  MPI_Datatype * taskFloatArrayType;
  MPI_Datatype * zFloatArrayType;
  MPI_Datatype * aFloatArrayType;
  MPI_Datatype subjectIntArrayType;
  MPI_Datatype subjectFloatArrayType;
  MPI_Datatype covArrayType;
  MPI_Datatype piVecArrayType;
  MPI_Datatype rhoVecArrayType;
  MPI_Datatype varSubsetParamArrayType;
  MPI_Datatype intParamArrayType;
  MPI_Datatype floatParamArrayType;

  MathUtils * math;
  vector<cl::Event> eventList0,eventList1,eventList2,eventList3;
  float * betas;
  float * debugvec;
  float * beta_hats;
  float * means;
  float * sds;
  int * mask;
  float lambda;
  float lasso_mixture;
  //meta_data_t meta_data;
  delta_t * deltas;
  int * disease_status;
  float * trait;
  float * covariates;
  string * covariate_names;
  int env_covariates;
  int genetic_tasks;
  int * tasks_by_slave;
  float * score;
  char * genocharmatrix;
  
  int * varcounts;
  //int * tasklist; // ordered pairs of SNPs for interaction testing
  int z_rank;
  // BEGIN GRADIENT VARIABLES
  int env_tasks;
  int current_person;
  float fitted_value; 
  float gradient;
  //float * genomatrix_subjectmajor;
  float * covmatrix_subjectmajor;
  float * beta_mirror;
  float beta_p;
  float beta_mirror_scale;
  char * genocharmatrix_subjectmajor;
  string * rslist;
  int veclen_subjectmajor;
  float eta;
  int x_beta_chunks;
  float xbeta;
  // END GRADIENT VARIABLES
  float * z_matrix;
  //float z_residual;
  float * zzz;
  float * pi_vec;
  int a_rank;
  float * a_matrix;
  float * aaa;
  float * rho_vec;
  int packedstride;
  int personchunks;
  packedgeno_t * packedgeno_matrix;
  int * paddedaffectionvector;
  ofstream ofs;
  IO * io;
  covar_map_t cmap;
  int logistic;
  int slave_offset;

  // universal functions
  float logL(float * score);
  int offset(int slave);
  float c2g(char * genocharmatrix,int veclen,int snp,int person);
  void convertgeno(int arrdim,int genoindex,float * genovec);
  string getname(int genoindex);
  void zero_beta();
  void compute_mean_sd(int genomatrix_dim);
  
  void init_standardize();
  void init_parallel_gradient();
  void init_gradient();
  void init_greedy();

  // master functions
  void init_master();
  void loadMatrix(const string & filename, int & rank, float * &  matrix, float * & coeff, float * & hatmat, float ridge);
  float getLambda(int index);
  void fitBetaHats();
  void fitLassoGreedy(double  & logL, int & modelsize, bool & terminate, int replicate);
  void fitLassoParallelGradient();
  void fitLassoGradient();
  void fitLassoCCD(double  & logL, int & modelsize);
  void sampleCoeff(float * hatmat, int rank, float * coeff, float * designMat, float & residual);
  void cleanup_master();

  // slave functions
  bool load_mean_sd();
  void listen();
  void init_slave();
  void cleanup_slave();
  void init_gpu();
  void beta_update_slave();
  void greedy_slave();
  void estimate_fitted_value();
  void estimate_parallel_fitted_value();
  void new_gradient();
  void parallel_new_gradient();
  void gradient_shrink();
  void cleanup_gpu();
  void fetchgeno_subjectmajor(int subjectindex, float * geno);
};
