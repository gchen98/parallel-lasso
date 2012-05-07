#include<cstdlib>
#include<cstring>
#include<iostream>
#include<sstream>
#include<fstream>
#include<math.h>
#include<boost/foreach.hpp>
#include"mpi.h"
#ifdef USE_GPU
#include<CL/cl.hpp>
#include"clsafe.h"
#endif
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_permutation.h>
#include"main.hpp"
#include"analyzer.hpp"
#include"io.hpp"
#include"dimension2.h"
#include"utility.hpp"
#include"lasso_mpi2.hpp"

using namespace std;
typedef unsigned int uint;

int MpiLasso2::get_rank(){
  return mpi_rank;
}


void MpiLasso2::cleanup(){
  MPI_Finalize();
  ofs.close();
}

MpiLasso2::~MpiLasso2(){
  cerr<<"MpiLasso2 deleted\n";
}

inline int MpiLasso2::offset(int slave){
  int offset=0;
  for(int i=0;i<slave;++i) offset+=tasks_by_slave[i];
  return offset;
}

lasso2_settings_t::lasso2_settings_t(const ptree &pt,int mpi_rank){
  string t = pt.get<string>("enable_opencl");
  use_gpu=!t.compare("true")?true:false;
  lambda=pt.get<float>("lambda");
  lasso_mixture=pt.get<float>("lasso_mixture");
  t = pt.get<string>("lasso_path");
  lasso_path=!t.compare("true")?true:false;
  tasklist = pt.get<string>("inputdata.tasklist");
  annotationfile = pt.get<string>("inputdata.annotation");
  genofile = pt.get<string>("inputdata.genofile");
  pedfile = pt.get<string>("inputdata.pedfile");
  snpfile = pt.get<string>("inputdata.snpfile");
  covariatedatafile = pt.get<string>("inputdata.covariates.datafile");
  covariateselectionfile = pt.get<string>("inputdata.covariates.selectionfile");
  pt.get_child("opencl_settings.host.<xmlattr>.rank");
  BOOST_FOREACH(boost::property_tree::ptree::value_type host_object, pt.get_child("opencl_settings")){
    boost::property_tree::ptree host_tree = (boost::property_tree::ptree)host_object.second;
    int hostid = host_tree.get<int>("<xmlattr>.rank");
    if (hostid==mpi_rank){
      platform_id = host_tree.get<int>("platform_id");
      device_id = host_tree.get<int>("device_id");
      kernel_path = host_tree.get<string>("kernel_path");
    }
  }
}

MpiLasso2::MpiLasso2(){
  mpi_struct_init = false;
  MPI_Init(0,NULL);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  is_master = mpi_rank==0;
  io = new IO();
  unsigned long int seed = 0;
  math = new MathUtils(seed);
  // Open the debugging files on master and slave
  if (is_master){
    cerr<<"Total Nodes: "<<mpi_numtasks<<endl;
    slaves = mpi_numtasks-1;
    ofs.open("debug_master");
  }else{
    ostringstream oss;
    oss<<"debug_slave."<<mpi_rank;
    ofs.open(oss.str().data());
  }
  covariates = NULL;
  covariate_names = NULL;
  disease_status = NULL;
  genocharmatrix = NULL;
  mask = NULL;
  rslist = NULL;
  means = NULL;
  sds = NULL;
  betas = NULL;
  group_indices = NULL;
  l2_norms = NULL;
  //l2_norms_big = NULL;
  score = NULL;
  deltas = NULL;
  varcounts = NULL;
  //remove later
  this->logistic = true;
  this->algorithm = ALGORITHM_GREEDY;
}

void MpiLasso2::read_data(const char * snpfile, const char * pedfile, const char * genofile, const char * covdata, const char * covselection, const char * maskfile){
  if(mpi_rank>0) return;
  io->readPlinkFiles(snpfile,pedfile,genofile,data);
  n = data.totalpersons;
  totalsnps = data.totalsnps;
  genoveclen = data.veclen;
  cerr<<"Persons: "<<n<<", SNPs: "<<totalsnps<<", GenoVeclen: "<<genoveclen<<endl;
  if (disease_status==NULL) disease_status = new int[n];
  for(int i=0;i<n;++i) disease_status[i] = data.affection[i];
  if (mask==NULL) mask = new int[n];
  for(int i=0;i<n;++i) mask[i] = 1;
  if (covdata==NULL||covselection==NULL){
    ofs<<"Skipping covariate files read\n";
  }else{
    io->readCovariates(covdata,covselection,cmap);
    if (cmap.size()==0) throw "No valid covariates found.";
    env_covariates = cmap.size();
    if (covariates==NULL) covariates = new float[n * env_covariates];
    if (covariate_names==NULL) covariate_names = new string[env_covariates];
    int c = 0;
    for(covar_map_t::iterator it = cmap.begin();it!=cmap.end();it++){
      string key = it->first;
      vector<float> vec = it->second;
      // store covariate values
      for(int i=0;i<n;++i){
        covariates[c*n+i] = vec[i];
      }
      // store covariate names
      covariate_names[c] = it->first;
      //cerr<<"Loaded covariate name: "<<it->first<<endl;
      ++c;
    }
  }
  // store affection status
  if (maskfile==NULL){
    ofs<<"Skipping mask file read\n";
  }else{
    ifstream ifs_mask(maskfile);
    if (!ifs_mask.is_open()){
      throw "Cannot open mask file\n";
    }
    string line;
    for(int i=0;i<n;++i){
      getline(ifs_mask,line);
      istringstream iss(line);
      iss>>mask[i]; 
    }
    ifs_mask.close();
  }
}

void MpiLasso2::read_tasks(const char * taskfile, const char * annotationfile){
  if(mpi_rank>0) return;
  ifstream ifs_task(taskfile);
  if (!ifs_task.is_open()){
    throw "Cannot open taskfile\n";
  }
  genetic_tasks = 0;
  string line;
  while(getline(ifs_task,line)) ++genetic_tasks;
  ifs_task.close();
  //totaltasks = totaltasks;
  cerr<<"Total SNPs selected: "<<genetic_tasks<<endl;
  if (genocharmatrix==NULL) 
  genocharmatrix = new char[genetic_tasks * genoveclen];
  if (rslist==NULL)  rslist = new string[genetic_tasks];
  ifs_task.open(taskfile);
  for (int i=0;i<genetic_tasks;++i){
    getline(ifs_task,line);
    istringstream iss(line);
    int genetic_task;
    iss>>genetic_task;
    if (genetic_task>=totalsnps){
      cerr<<"The task "<<line<<" exceeded the total snps: "<<totalsnps<<endl;
      throw "Bad input";
    }else{
      char * genovec = data.genomatrix[genetic_task];
      for (int j=0;j<genoveclen;++j){
        genocharmatrix[i*genoveclen+j] = genovec[j];
      }
      rslist[i] = data.rs_list[genetic_task];
    }
  }
  ifs_task.close();
  totaltasks = env_covariates + genetic_tasks;
  if (annotationfile==NULL){
    ofs<<"Skipping annotation file read\n";
  }else{
    if (group_indices==NULL) group_indices = new int[totaltasks];
    for(int i=0;i<totaltasks;++i) group_indices[i] = -1;
    map<string,int> group_id;
    groups = 0;
    ifstream ifs(annotationfile);
    if (!ifs.is_open()){
      throw "Cannot open annotation file";
    }
    while(getline(ifs,line)){
      istringstream iss(line);
      int var;
      string group;
      iss>>var>>group;
      map<string,int>::iterator it;
      if (group_id.find(group)==group_id.end()){
        //ofs<<"Adding "<<group<<" with index "<<groups<<endl;
        group_names.push_back(group);
        group_id[group] = groups++; 
      }
    }
    ifs.close();
    ifs.open(annotationfile);
    while(getline(ifs,line)){
      istringstream iss(line);
      int var;
      string group;
      iss>>var>>group;
      int group_index = group_id[group];
      group_indices[var] = group_index; 
      //ofs<<"Assigning task "<<var<<" with group "<<group_names[group_index]<<", index "<<group_indices[var]<<endl;
    }
    ifs.close();
    //exit(0);
  }
}



void MpiLasso2::allocate_datastructures(){
  if (mpi_struct_init) return;
  // pass an array of integer parameters
  MPI_Type_contiguous(MPI_INT_ARR,MPI_INT,&intParamArrayType);
  MPI_Type_commit(&intParamArrayType);
  // pass an array of float parameters
  MPI_Type_contiguous(MPI_FLOAT_ARR,MPI_FLOAT,&floatParamArrayType);
  MPI_Type_commit(&floatParamArrayType);
  if (is_master){
    cerr<<"Initializing "<<slaves<<" slaves\n";
    // BEGIN INITIALIZE MPI CONTIGUOUS DATA STRUCTURES
    int division = totaltasks/slaves;
    tasks_by_slave = new int[slaves];
    snps_by_slave = new int[slaves];
    // for array of affection status
    MPI_Type_contiguous(n,MPI_INT,&subjectIntArrayType);
    MPI_Type_commit(&subjectIntArrayType);
    // for array of trait values
    MPI_Type_contiguous(n,MPI_FLOAT,&subjectFloatArrayType);
    MPI_Type_commit(&subjectFloatArrayType);
    // for matrix of covariates
    MPI_Type_contiguous(env_covariates*n,MPI_FLOAT,&covArrayType);
    MPI_Type_commit(&covArrayType);
    // for the L2 norms
    MPI_Type_contiguous(groups,MPI_FLOAT,&l2NormsFloatArrayType);
    MPI_Type_commit(&l2NormsFloatArrayType);
    int dim[MPI_INT_ARR];
    dim[0] = n;
    dim[1] = totalsnps; // total SNPs
    dim[2] = genoveclen;
    dim[3] = groups;
    dim[4] = env_covariates;
    dim[5] = algorithm;
    // stores the array of tasks per slave
    taskIntArrayType = new MPI_Datatype[slaves];
    // stores the genotype chars
    charArrayType = new MPI_Datatype[slaves];
    for(int i=0;i<slaves;++i){
      tasks_by_slave[i] = (i<slaves-1)?division:(totaltasks-(slaves-1)*division);
      snps_by_slave[i] = (i==0)?tasks_by_slave[i]-env_covariates:tasks_by_slave[i];
      cerr<<"Slave "<<i<<" has "<<tasks_by_slave[i]<<" total tasks and "<<snps_by_slave[i]<<" SNPs.\n";
      MPI_Type_contiguous(tasks_by_slave[i],MPI_INT,taskIntArrayType+i);
      MPI_Type_commit(taskIntArrayType+i);
      MPI_Type_contiguous(snps_by_slave[i] * genoveclen,MPI_CHAR,charArrayType+i);
      MPI_Type_commit(charArrayType+i);
    }
    for(int slave=0;slave<slaves;++slave){
      int o = offset(slave);
      int dest = slave+1;
      // notify of data dimensions
      //cerr<<"Notifying data dimensions to slave "<<slave<<"\n";
      dim[6] = o;
      dim[7] = tasks_by_slave[slave];
      dim[8] = snps_by_slave[slave];
      rc = MPI_Send(dim ,1, intParamArrayType,dest,TAG_INIT_DIM,MPI_COMM_WORLD);
    }
    cerr<<"Initialized data structures\n";
    mpi_struct_init = true;
  }else{
    int dim[MPI_INT_ARR];
    rc = MPI_Recv(dim,MPI_INT_ARR,MPI_INT,source,TAG_INIT_DIM,MPI_COMM_WORLD,&stat);
    iter = 0;
    n = dim[0];
    totalsnps = dim[1];
    genoveclen = dim[2];
    groups = dim[3];
    env_covariates = dim[4];
    algorithm = dim[5];
    slave_offset = dim[6];
    submodelsize = totaltasks = dim[7];  // number of tasks
    genetic_tasks = dim[8];  // number of snps
    logistic = true;
    mask = new int[n];
    for(int i=0;i<n;++i) mask[i] = 1;
    disease_status = new int[n];
    covariates = new float[env_covariates*n];
    slave_matsize = genetic_tasks*genoveclen;
    genocharmatrix = new char[slave_matsize];
    // create the padded genomatrix for use in the GPU
    ofs<<"Packing genotypes into container.\n";
    packedstride =  (n/512+(n%512>0)) * 512 / 16; //PACKED_SUBJECT_STRIDE; 
    ofs<<"Packed genotype stride is "<<packedstride<<"\n";
    int packedgenolen = genetic_tasks * packedstride;
    if (settings->use_gpu){
      ofs<<"Total packed genolen is "<<packedgenolen<<endl;
      packedgeno_matrix = new packedgeno_t[packedgenolen];
    }else{
    }
    ofs<<"I am rank "<<mpi_rank<<" and have "<<n<<" observations with "<<totalsnps<<" total SNPs, "<<slave_offset<<" offset, "<<totaltasks<<" tasks, "<<env_covariates<<" covariates, "<<genetic_tasks<<" SNPs, "<<genoveclen<<" vec length"<<endl;
    if (settings->use_gpu){
      init_gpu();
    }
  }
  if (deltas == NULL) deltas = new delta_t[totaltasks];
  if (varcounts == NULL) varcounts = new int[totaltasks];
  if (means==NULL) means = new float[totaltasks];
  if (sds==NULL) sds = new float[totaltasks];
  if (betas==NULL) betas = new float[totaltasks];
  for(int i=0;i<totaltasks;++i) betas[i] = 0;
  if (score==NULL) score = new float[n];
  for(int i=0;i<n;++i)score[i]=0;
  if (l2_norms==NULL) l2_norms = new float[groups];
  //if (l2_norms_big==NULL) l2_norms_big = new float[totaltasks];
  for(int i=0;i<groups;++i) l2_norms[i]=0;
  //for(int i=0;i<totaltasks;++i) l2_norms_big[i]=0;
  if (group_indices==NULL) group_indices = new int[totaltasks];
}

void MpiLasso2::send_mask(){
  if (is_master){
    for(int slave=0;slave<slaves;++slave){
      //int o = offset(slave);
      int dest = slave+1;
      rc = MPI_Send(mask,1,subjectIntArrayType,dest,TAG_INITMASK,MPI_COMM_WORLD);
    }
  }else{
    rc = MPI_Recv(mask,n,MPI_INT,source,TAG_INITMASK,MPI_COMM_WORLD,&stat);
    if (settings->use_gpu){
#ifdef USE_GPU
      cl_int err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.mask_mem_obj, CL_TRUE, 0,  sizeof(int)*n, mask, NULL, NULL );
      clSafe(err, "write buffer for mask");
      int n_subset = 0;
      for(int i=0;i<n;++i) n_subset+=mask[i];
      err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.n_subset_mem_obj, CL_TRUE, 0,  sizeof(int), &n_subset, NULL, NULL );
      clSafe(err, "write buffer for subset len");
#endif
    }
  }
}

void MpiLasso2::send_phenotypes(){
  if (is_master){
    for(int slave=0;slave<slaves;++slave){
      //int o = offset(slave);
      int dest = slave+1;
      rc = MPI_Send(disease_status,1,subjectIntArrayType,dest,TAG_INITAFF,MPI_COMM_WORLD);
    }
  }else{
    rc = MPI_Recv(disease_status,n,MPI_INT,source,TAG_INITAFF,MPI_COMM_WORLD,&stat);
    if (settings->use_gpu){
#ifdef USE_GPU
      cl_int err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.aff_mem_obj, CL_TRUE, 0,  sizeof(int)*n, disease_status, NULL, NULL );
      clSafe(err, "write buffer for aff");
#endif
    }
  }
}

void MpiLasso2::send_covariates(){
  if (is_master){
    for(int slave=0;slave<slaves;++slave){
      //int o = offset(slave);
      int dest = slave+1;
      rc = MPI_Send(covariates,1,covArrayType,dest,TAG_INIT_COV,MPI_COMM_WORLD);
    }
  }else{
    rc = MPI_Recv(covariates,env_covariates*n,MPI_FLOAT,source,TAG_INIT_COV,MPI_COMM_WORLD,&stat);
    if (settings->use_gpu){
#ifdef USE_GPU
      cl_int err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.cov_mem_obj, CL_TRUE, 0,  sizeof(float)*n*env_covariates, covariates, NULL, NULL );
      clSafe(err, "write buffer for cov");
#endif
    }
  }
}

void MpiLasso2::send_genotypes(){
  if (is_master){
    int genorow_offset = 0;
    int task_offset = 0;
    for(int slave=0;slave<slaves;++slave){
      //int o = offset(slave);
      int dest = slave+1;
      rc = MPI_Send(genocharmatrix+genorow_offset*genoveclen,1,charArrayType[slave],dest,TAG_INITDESIGN,MPI_COMM_WORLD);
      rc = MPI_Send(group_indices+task_offset,1,taskIntArrayType[slave],dest,TAG_INIT_GROUP_INDICES,MPI_COMM_WORLD);
      genorow_offset+=snps_by_slave[slave]; 
      task_offset+=tasks_by_slave[slave]; 
    }
  }else{
    rc = MPI_Recv(this->genocharmatrix, slave_matsize,MPI_CHAR,source,TAG_INITDESIGN,MPI_COMM_WORLD,&stat);
    rc = MPI_Recv(this->group_indices, totaltasks,MPI_INT,source,TAG_INIT_GROUP_INDICES,MPI_COMM_WORLD,&stat);
    if (settings->use_gpu){
#ifdef USE_GPU
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
      ofs<<"Packed genotypes\n";
      cl_int err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.packedgeno_mem_obj, CL_TRUE, 0,  sizeof(packedgeno_t)*genetic_tasks*packedstride, packedgeno_matrix, NULL, NULL );
      clSafe(err, "write buffer for packed genotypes");
      ofs<<"Transferred packed genotypes to buffer\n";
      err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.group_indices_mem_obj, CL_TRUE, 0,  sizeof(int)*totaltasks, group_indices, NULL, NULL );
      clSafe(err, "write buffer for group_indices");
//          int temp_indices[totaltasks];
//          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.group_indices_mem_obj, CL_TRUE, 0, totaltasks*sizeof(int),temp_indices);
//          ofs<<"group indices\n";
//          for(int j=0;j<totaltasks;++j){
//            if(temp_indices[j]!=0) ofs<<j<<": "<<temp_indices[j]<<endl;
//          }
#endif
    }
  }
}

void MpiLasso2::send_tuning_params(){
  send_tuning_params(settings->lambda,settings->lasso_mixture);
}

void MpiLasso2::send_tuning_params(float lambda, float mixing){
  float tuning_params[MPI_FLOAT_ARR];
  if (is_master){
    tuning_params[0] = lambda;
    tuning_params[1] = mixing;
    for(int slave=0;slave<slaves;++slave){
      //int o = offset(slave);
      int dest = slave+1;
      rc = MPI_Send(tuning_params,1,floatParamArrayType,dest,TAG_UPDATE_TUNING_PARAMS,MPI_COMM_WORLD);
    }
  }else{
    rc = MPI_Recv(tuning_params,MPI_FLOAT_ARR,MPI_FLOAT,source,TAG_UPDATE_TUNING_PARAMS,MPI_COMM_WORLD,&stat);
    tuning_param.lambda = tuning_params[0];
    tuning_param.lasso_mixture = tuning_params[1];
    if (settings->use_gpu){
#ifdef USE_GPU
      cl_int err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.lasso_tuning_param_mem_obj, CL_TRUE, 0,  sizeof(tuning_param_t), &tuning_param, NULL, NULL );
      clSafe(err, "write buffer for tuning_param");
      err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.lasso_tuning_param_mem_obj, CL_TRUE, 0, sizeof(tuning_param_t),&tuning_param,NULL,NULL);
#endif
      clSafe(err, "read buffer for tuning_param");
      ofs<<"Tuning parameters are lambda: "<<tuning_param.lambda<<" and mixture: "<<tuning_param.lasso_mixture<<endl;
    }
  }
}

vector<string> MpiLasso2::get_tasknames(){
  vector<string> names;
  if (mpi_rank==0){
    for(int i=0;i<env_covariates;++i){
      names.push_back(covariate_names[i]);
    }
    for(int i=0;i<genetic_tasks;++i){
      names.push_back(rslist[i]);
    }
  }
  return names;
}

string MpiLasso2::getname(int genoindex){
  ostringstream oss;
  if (genoindex<0){
    //cerr<<"Grabbing covindex: "<<abs(genoindex)-2<<endl;
    //return "hellocov";
    oss<<"COVARIATE\t"<<covariate_names[abs(genoindex)-1];
  }else if (genoindex>-1){
  //cerr<<"Grabbing index: "<<genoindex<<endl;
    //return "geno";
    oss<<"SNP\t"<<rslist[genoindex];
  }else{
    oss<<"NOT\tAVAILABLE";
  }
  return oss.str();
}


void MpiLasso2::testfit(vector<modelvariable_t>  & modelvariables, int & mislabels,int & correctlabels){
  float * mean_vec = new float[totaltasks];
  float * sd_vec = new float[totaltasks];
  int offset = 0;
  //ofstream ofsdebug("debug.meansd");
  for(int dest = 1;dest<mpi_numtasks;++dest){
    int slave = dest-1;
    //ofsdebug<<"Tasks by slave: "<<tasks_by_slave[slave]<<endl;
    load_mean_sd(dest,tasks_by_slave[slave],mean_vec+offset,sd_vec+offset);
    offset+=tasks_by_slave[slave];
  }
  //cerr<<"Loaded means and SDs\n";
  for(int i=0;i<totaltasks;++i){
    //ofsdebug<<i<<":"<<mean_vec[i]<<","<<sd_vec[i]<<endl;
  }
  //ofsdebug<<endl;
  //ofsdebug.close();
  float genovec[n];
  float fittedvalues[n];
  memset(fittedvalues,0,sizeof(float)*n);
  //ofsdebug.open("debug.geno");
  for(vector<modelvariable_t>::iterator it = modelvariables.begin();
  it!=modelvariables.end();it++){
    int genoindex = it->index-env_covariates;
    convertgeno(1,genoindex,genovec);
    for(int i=0;i<n;++i){
      //ofsdebug<<genovec[i];
      if (mask[i]){
        fittedvalues[i]+=it->beta*((genovec[i]-mean_vec[it->index])/sd_vec[it->index]);
      }
    }
    //ofsdebug<<endl;
  }
  //ofsdebug.close();
  //MPI_Finalize();
  for(int i=0;i<n;++i){
    if (mask[i]){
      float prob = 1./(1.+exp(-fittedvalues[i]));
      //cerr<<"Prob for person "<<i<<" is "<<prob<<endl;
      if (prob>=.5 && disease_status[i]==1 || prob<.5 && disease_status[i]==-1){
        ++correctlabels;
      }else{
        ++mislabels;
      }
    }
  }
  delete[] mean_vec;
  delete[] sd_vec;
  //exit(1);
}

bool MpiLasso2::fitLassoGreedy(int replicate, double & logL, vector<modelvariable_t>  & modelvariables){
//void MpiLasso2::fitLassoGreedy(double & logL, int & modelsize, bool & terminate, int replicate){
  modelvariables.clear();
  int modelsize = modelvariables.size();
  
  bool terminate = false;
  zero_beta();
  if(mpi_rank){
    if (!load_mean_sd(mpi_rank,submodelsize,means,sds)) compute_mean_sd(1);
  }
  for(int i=0;i<totaltasks;++i) betas[i] = 0;
  int n_subset = 0;
  currentLL = 0;
  ofs<<"computing base LL\n";
  for(int i=0;i<n;++i){
    score[i] = 0;
    n_subset+=mask[i];
    if(mask[i]){
      double pY = exp(score[i]*disease_status[i]);
      pY/=(1+pY);
      currentLL+= (disease_status[i]==1) ? log(pY) : log(1-pY);
    }
  }
  // BEGIN LASSO LOOP
  iter=0;
  double start = clock();
  double tolerance;
  bestsubmodelindex = 0;
  bestdeltabeta = 0;
  int best_indices[MPI_INT_ARR];
  float bestvar[MPI_FLOAT_ARR];
  converged = false;
  do{
    //double start = clock();
    ++iter;
    int bestslaveindex = 0;
    if (!is_master){
      if (settings->use_gpu){
#ifdef USE_GPU
        double gpustart = clock();
        cl_int err;
        int snpchunksize = GRID_WIDTH/BLOCK_WIDTH;
        best_t bestvar;
        bestvar.best_genoindex_1 = bestfullmodelindex;
        bestvar.best_submodel_index = bestsubmodelindex;
        bestvar.best_delta_beta = bestdeltabeta;
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
        for(int i=0;i<totaltasks;++i){
           //int groupindex = group_indices[i];
           //l2_norms_big[i] = groupindex>-1?l2_norms[groupindex]:0;
        }
   
        err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.l2_norms_mem_obj, CL_TRUE, 0,  sizeof(float)*groups,l2_norms, NULL, NULL );
        //err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.l2_norms_big_mem_obj, CL_TRUE, 0,  sizeof(float)*totaltasks,l2_norms_big, NULL, NULL );
        clSafe(err,"CommandQueue::write buffer for L2 norms");
        int scoreworksize = (n/BLOCK_WIDTH+1)*BLOCK_WIDTH;
        err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_update_score,cl::NullRange,cl::NDRange(scoreworksize,1),cl::NDRange(BLOCK_WIDTH,1),NULL,NULL);
        clSafe(err,"CommandQueue::enqueueNDRangeKernelUpdateScore()");
        bool debug1 = false;
        if (debug1){
          float tempscore[n];
          float tempscore1[n];
          float tempscore2[n];
          //int temp_indices[totaltasks];
          //err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.group_indices_mem_obj, CL_TRUE, 0, totaltasks*sizeof(int),temp_indices);
          //ofs<<"group indices\n";
          //for(int j=0;j<totaltasks;++j){
          //  if(temp_indices[j]!=-1) ofs<<j<<": "<<temp_indices[j]<<endl;
          //}
          float temp_norms[groups];
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.l2_norms_mem_obj, CL_TRUE, 0, groups*sizeof(float),temp_norms);
          ofs<<"L2 norms\n";
          for(int j=0;j<groups;++j){
           if(temp_norms[j]!=0) ofs<<j<<": "<<temp_norms[j]<<endl;
          }
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.score_mem_obj, CL_TRUE, 0, n*sizeof(float),tempscore);
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.score_num_mem_obj, CL_TRUE, 0, n*sizeof(float),tempscore1);
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.score_den_mem_obj, CL_TRUE, 0, n*sizeof(float),tempscore2);
          ofs<<"score after beta update: ";
          for(int i=0;i<n;++i){
            ofs<<i<<":"<<tempscore[i]<<","<<tempscore1[i]<<","<<tempscore2[i]<<endl;
          }
          ofs<<"currentLL: "<<currentLL<<endl;
          ofs.close(); 
          exit(0);
        }
        err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.currLL_mem_obj, CL_TRUE, 0,  sizeof(float)*1,&currentLL, NULL, NULL );
        snpchunksize = GRID_WIDTH/BLOCK_WIDTH;
        // launch the CLG kernel that computes the delta beta at each SNP
        //for (int taskoffset=0;taskoffset<1;++taskoffset){
//cerr<<"computing gradient across "<<(totaltasks/snpchunksize)+(totaltasks%snpchunksize>0) <<" person chunks: "<<personchunks<<"\n";
        //for (int taskoffset=0;taskoffset<1;++taskoffset){
        for (int taskoffset=0;taskoffset<(totaltasks/snpchunksize)+(totaltasks%snpchunksize>0);++taskoffset){
          //cerr<<"Writing task offset "<<taskoffset<<endl;
          err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL, NULL );
          //cerr<<"Wrote task offset "<<taskoffset<<endl;
          //cl::Event delta_beta_event;
          err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_compute_gradient_hessian,cl::NullRange,cl::NDRange(GRID_WIDTH,personchunks),cl::NDRange(BLOCK_WIDTH,1),NULL,NULL);
          clSafe(err,"CommandQueue::enqueueNDRangeKernelComputeGradient()");
          //cerr<<"Launched gradient hessian\n";
        }
//cerr<<"computed gradient\n";
        bool debug2 = false;
        if (debug2){
          ofs<<"Gradient/Hessian chunks\n";
          float * gradientchunks = new float[submodelsize*personchunks];
          float * hessianchunks = new float[submodelsize*personchunks];
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.gradient_chunks_mem_obj, CL_TRUE, 0, submodelsize*personchunks*sizeof(float),gradientchunks);
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.hessian_chunks_mem_obj, CL_TRUE, 0, submodelsize*personchunks*sizeof(float),hessianchunks);
          //for(int i=0;i<500;++i){
          for(int i=0;i<submodelsize;++i){
            ofs<<"var:"<<i;
            float num = 0, den = 0;
            for(int j=0;j<personchunks;++j){
              ofs<<" "<<gradientchunks[i*personchunks+j]<<"/"<<hessianchunks[i*personchunks+j];
              num+=gradientchunks[i*personchunks+j];
              den+=hessianchunks[i*personchunks+j];
            }
            ofs<<"GPUDEBUG:"<<i<<",Gradient:"<<num<<"Hessian:"<<den<<endl;
            ofs<<endl;
          }
          delete[]gradientchunks;
          delete[]hessianchunks;
          ofs.close();
          exit(0);
        }
        int smallsnpchunksize = GRID_WIDTH/SMALL_BLOCK_WIDTH+1;
        for (int taskoffset=0;taskoffset<(totaltasks/smallsnpchunksize)+(totaltasks%smallsnpchunksize>0);++taskoffset){
          err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL, NULL );
          err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_compute_delta_beta,cl::NullRange,cl::NDRange(GRID_WIDTH,1),cl::NDRange(SMALL_BLOCK_WIDTH,1),NULL,NULL);
          clSafe(err,"CommandQueue::enqueueNDRangeKernel delta beta()");
        }
//cerr<<"reduced gradient\n";
        bool debug2b = false;
        if (debug2b){
          float tempgroups[groups];
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.l2_norms_mem_obj, CL_TRUE, 0, sizeof(float)*groups,tempgroups);
          clSafe(err,"CommandQueue::read buffer for L2 norms");

          int temp_indices[totaltasks];
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.group_indices_mem_obj, CL_TRUE, 0, sizeof(int)*totaltasks,temp_indices);
          clSafe(err,"CommandQueue::read buffer for L2 norms");
           
          //for(int i=0;i<groups;++i){
            //ofs<<"Group:\t"<<i<<"\t"<<l2_norms[i]<<endl;
          //}
          ofs<<"Reduction for deltabeta\n";
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.delta_mem_obj, CL_TRUE, 0, submodelsize*sizeof(delta_t),deltas);
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.betas_mem_obj, CL_TRUE, 0, submodelsize*sizeof(float),betas);
          int ms = 0;
          //for(int i=0;i<500;++i){
          //float l1_penalty = (tuning_param.lasso_mixture)*tuning_param.lambda;
          //float l2_penalty = (1.-tuning_param.lasso_mixture)*tuning_param.lambda;
          for(int i=0;i<submodelsize;++i){
            
            float l2norm = 0;
            l2norm = tempgroups[temp_indices[i]];
            //if (deltas[i].delta_beta!=0 || betas[i]!=0 || l2norm!=0){
               ++varcounts[i];
               ++ms;
               ofs<<"GPUDEBUG:"<<i<<",DeltaBeta:"<<deltas[i].delta_beta<<",OrigBeta:"<<betas[i]<<",L2 norm:"<<l2norm<<",group:"<<temp_indices[i]<<endl;
            //}
          }
          ofs<<"Modelsize: "<<ms<<endl;
          ofs.close();
          exit(0);
        }
       // launch the kernel that computes the delta LL at each SNP
//cerr<<"launching likelihood\n";
        for (int taskoffset=0;taskoffset<(totaltasks/snpchunksize)+(totaltasks%snpchunksize>0);++taskoffset){
          err = opencl_info.command_queue.enqueueWriteBuffer(opencl_info.taskoffset_mem_obj, CL_TRUE, 0,  sizeof(int)*1, &taskoffset , NULL, NULL );
          //cl::Event proposeLLevent;
          err = opencl_info.command_queue.enqueueNDRangeKernel(opencl_info.kernel_compute_likelihoods,cl::NullRange,cl::NDRange(GRID_WIDTH,personchunks),cl::NDRange(BLOCK_WIDTH,1),NULL,NULL);
          //eventList3.push_back(proposeLLevent);
          clSafe(err,"CommandQueue::enqueueNDRangeKernelProposeLL()");
          //printExecutionCode("LogLike",proposeLLevent);
        }
//cerr<<"launched likelihood\n";
        bool debug3 = false;
        if (debug3){
          ofs<<"Likelihood chunks\n";
          float * likelihood_chunks = new float[submodelsize*personchunks];
          err = opencl_info.command_queue.enqueueReadBuffer(opencl_info.loglike_chunks_mem_obj, CL_TRUE, 0, submodelsize*personchunks*sizeof(float),likelihood_chunks);
          //for(int i=0;i<500;++i){
          for(int i=0;i<submodelsize;++i){
            float sum=0;
            for(int j=0;j<personchunks;++j){
              sum+=likelihood_chunks[i*personchunks+j];
            }
            ofs<<"GPUDEBUG:"<<i<<",Likelihood:"<<sum<<endl;
          }
          delete[] likelihood_chunks;
          ofs.close();
          exit(0);
        }

        for (int taskoffset=0;taskoffset<(totaltasks/smallsnpchunksize)+(totaltasks%smallsnpchunksize>0);++taskoffset){
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
        ofs<<"GPU iteration: "<<(clock()-gpustart)/CLOCKS_PER_SEC<<endl;
#endif
      }else{ // non GPU version
        double cpustart = clock();
        float delta_beta[totaltasks];
        float deltaLL[totaltasks];
        for(int i=0;i<totaltasks;++i) delta_beta[i] = deltaLL[i]=0;
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
        //ofs<<"L2 norms: "<<endl;
        for(int k=0;k<groups;++k){
         // if(l2_norms[k]!=0) ofs<<k<<": "<<l2_norms[k]<<endl;
        }

        for(int j=0;j<totaltasks;++j){
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
          //ofs<<"CPUDEBUG:"<<j<<",Gradient:"<<gradient<<"Hessian:"<<hessian<<endl;
          int group_index = group_indices[j];
          float l1_penalty = 0,l2_penalty = 0;
          if (group_index==-1){ 
            l1_penalty = (mpi_rank>1 || j-env_covariates>=0) ? 
            tuning_param.lambda:0;
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
          }else{
            l1_penalty = tuning_param.lasso_mixture*tuning_param.lambda;
            l2_penalty = (1-tuning_param.lasso_mixture)*tuning_param.lambda;
            //l1_penalty = .5*45;
            //l2_penalty = (1-.5)*45;
            float l2norm = l2_norms[group_index];
            float full_penalty = l1_penalty;
            if (betas[j]>LAMBDA_EPSILON){
              float l2 = l2_penalty/sqrt(l2norm);
              full_penalty += l2 * betas[j];
              hessian+=l2*(1-betas[j]*betas[j]/l2norm);
              delta_beta = (gradient-full_penalty)/hessian;
              if (betas[j]-delta_beta<0) delta_beta = 0;
            }else if (betas[j]<-LAMBDA_EPSILON){
              float l2 = l2_penalty/sqrt(l2norm);
              full_penalty -= l2*betas[j];
              hessian+=l2*(1-betas[j]*betas[j]/l2norm);
              delta_beta = (gradient+full_penalty)/hessian;
              if (betas[j]-delta_beta>0) delta_beta = 0;
            }else{
              if (l2norm<LAMBDA_EPSILON){
                full_penalty += l2_penalty;
              }else{
                hessian+=l2_penalty/sqrt(l2norm);
              }
              if (gradient>full_penalty){
                delta_beta = (gradient-full_penalty)/hessian;
              }else if (gradient<-full_penalty){
                delta_beta = (gradient+full_penalty)/hessian;
              }else{
                delta_beta = 0;
              }
            }
          }
          float l2_norm = 0;
          if (group_indices[j]>-1) l2_norm = l2_norms[group_indices[j]];
          //ofs<<"CPUDEBUG:"<<j<<",DeltaBeta:"<<delta_beta<<",OrigBeta:"<<betas[j]<<",L2norm:"<<l2_norm<<endl;
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
            //ofs<<"CPUDEBUG:"<<j<<",Likelihood:"<<llike2<<endl;
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
        }  // end loop across variables
        //ofs<<endl;
        bestdeltaLL = max;
        //ofs<<"Elapsed CPU time in seconds: "<<(clock()-starttime)/CLOCKS_PER_SEC<<endl;
        ofs<<"CPU iteration: "<<(clock()-cpustart)/CLOCKS_PER_SEC<<endl;
      } // end CPU version
      float bestDelta[MPI_FLOAT_ARR];
      bestDelta[0] = bestdeltaLL;
      bestDelta[1] = bestdeltabeta;
      bestDelta[2] = bestmean;
      bestDelta[3] = bestsd;
      ofs<<"Notifying host of Best index: "<<bestsubmodelindex<<" of delta beta "<<bestdeltabeta<<" and deltaLL "<<bestdeltaLL<<" with means/sd: "<<bestmean<<"/"<<bestsd<<endl;
      rc = MPI_Send(&bestsubmodelindex,1,MPI_INT,source,TAG_BEST_INDEX,MPI_COMM_WORLD);
      rc = MPI_Send(bestDelta,1,floatParamArrayType,source,TAG_BEST_DELTA,MPI_COMM_WORLD);
    }else{  // master receives best variable found.
      cerr<<".";
      //cerr<<"master side collect best\n";
      // COLLECT THE BEST VARIABLE FROM EACH SLAVE
      struct containerBest{
        int submodelindex;
        int fullmodelindex;
        float deltaLL;
        float deltaBeta;
        float mean,sd;
      };
      containerBest best[slaves];
      containerBest bestslave = best[0];
      bestslave.deltaLL = 0;
      for (int dest=1;dest<mpi_numtasks;++dest){
        int slave = dest-1;
        rc = MPI_Recv(&best[slave].submodelindex,1,MPI_INT,dest,TAG_BEST_INDEX,MPI_COMM_WORLD,&stat);
        best[slave].fullmodelindex=best[slave].submodelindex+offset(slave);
        float bestdelta[MPI_FLOAT_ARR];
        rc = MPI_Recv(bestdelta,MPI_FLOAT_ARR,MPI_FLOAT,dest,TAG_BEST_DELTA,MPI_COMM_WORLD,&stat);
        best[slave].deltaLL = bestdelta[0];
        best[slave].deltaBeta = bestdelta[1]; 
        best[slave].mean = bestdelta[2]; 
        best[slave].sd = bestdelta[3]; 
        if (best[slave].deltaLL>=bestslave.deltaLL){
  	//cerr<<"Received greedy response "<<best[slave].deltaLL<<" "<<best[slave].deltaBeta<<"\n";
          bestslaveindex = slave;
          bestslave = best[slave];
        }
        // NOW DETERMINE THE VARIABLE THAT BEST IMPROVES THE LL
      }
      bestdeltabeta = bestslave.deltaBeta;
      terminate = isnan(bestdeltabeta);
      if (terminate) ofs <<"Best delta beta isnan\n";
      bestsubmodelindex = bestslave.submodelindex;
      bestfullmodelindex= bestslave.fullmodelindex;
      bestmean = bestslave.mean;
      bestsd = bestslave.sd;
      // UPDATE THE SCORE VECTOR HERE ON THE MASTER  
      int bestgenoindex = bestfullmodelindex - env_covariates;
      float g1[n];
      convertgeno(1,bestgenoindex,g1);
      double perturb = 0.;
      //double totalcorr = 0;
      tolerance = 0;
      //int s=0;
      for(int i=0;i<n;++i){
        if (mask[i]){
          //++s;
          perturb = bestdeltabeta * (g1[i]-bestmean)/bestsd * data.affection[i];
          score[i] += perturb;
        }
      }
      float old_beta = betas[bestfullmodelindex];
      float new_beta = old_beta+bestdeltabeta;
      int groupindex = group_indices[bestfullmodelindex];
      if (groupindex>-1){
        float l2_delta = (new_beta+old_beta)*(new_beta-old_beta);
        //for(int g=0;g<groups;++g) l2_norms[g]+=l2_delta;
        l2_norms[groupindex]+=l2_delta<0?0:l2_delta;
        //ofs<<"groupindex: "<<groupindex<<endl;
        //ofs<<"Best full model index: "<<bestfullmodelindex<<endl;
        //ofs<<"L2 norm for task "<<bestfullmodelindex<<" in group "<<group_names[groupindex]<<" now "<<l2_norms[groupindex]<<endl;
      }
      tolerance=bestslave.deltaLL;
      betas[bestfullmodelindex]+=bestdeltabeta;
      best_indices[0] = bestsubmodelindex;
      best_indices[1] = bestfullmodelindex;
      bestvar[0] = bestdeltabeta;
      bestvar[1] = bestslave.deltaLL;
      bestvar[2] = bestmean;
      bestvar[3] = bestsd;
    } // end master code
    if (is_master){
      for (int dest=1;dest<mpi_numtasks;++dest){
        int slave = dest-1;
        best_indices[0] = slave==bestslaveindex?bestsubmodelindex:-1;
        //rpc_code = RPC_BETA_UPDATE;
        //rc = MPI_Send(&rpc_code,1,MPI_INT,dest,TAG_RPC,MPI_COMM_WORLD);
        rc = MPI_Send(best_indices,1,intParamArrayType,dest,TAG_BETA_UPDATE_INDEX,MPI_COMM_WORLD);
        rc = MPI_Send(bestvar,1,floatParamArrayType,dest,TAG_BETA_UPDATE_VAL,MPI_COMM_WORLD);
        rc = MPI_Send(score,1,subjectFloatArrayType,dest,TAG_UPDATE_SCORE,MPI_COMM_WORLD);
        rc = MPI_Send(l2_norms,1,l2NormsFloatArrayType,dest,TAG_UPDATE_L2,MPI_COMM_WORLD);
      }
      //END CHECK FOR CONVERGENCE FOR VARIABLE J
      logL = 0.;
      for(int i=0;i<n;++i){
        if(mask[i]){
          double pY = exp(score[i]*data.affection[i]);
          pY/=(1+pY);
          if (data.affection[i]==1) logL += log(pY); else logL += log(1-pY);
        }
      }
      bool poll = true;
      if (poll){
        modelsize = 0;
        for(int j=0;j<totaltasks;++j){
          if (betas[j]!=0.) {
            ++modelsize;
            string name1,name2;
            name1 = getname(j-env_covariates);
            ofs<<j<<"\t"<<name1<<"\t"<<betas[j]<<endl;
          }
        }
        ofs<<"MS: "<<modelsize<<", currentLL: "<<(2.*currentLL)<<", newLL: "<<(2.*logL)<<", BIC: "<<-(2.*logL)+modelsize*log(n_subset) <<", Iterations: "<<iter<<endl;
        ofs<<endl;
      }
    }else{ // slave receives best indices
      rc = MPI_Recv(best_indices,MPI_INT_ARR,MPI_INT,source,TAG_BETA_UPDATE_INDEX,MPI_COMM_WORLD,&stat);
      rc = MPI_Recv(bestvar,MPI_FLOAT_ARR,MPI_FLOAT,source,TAG_BETA_UPDATE_VAL,MPI_COMM_WORLD,&stat);
      rc = MPI_Recv(score,n,MPI_FLOAT,source,TAG_UPDATE_SCORE,MPI_COMM_WORLD,&stat);
      rc = MPI_Recv(l2_norms,groups,MPI_FLOAT,source,TAG_UPDATE_L2,MPI_COMM_WORLD,&stat);
      // IF NECESSARY UPDATE BETA
      bestsubmodelindex = best_indices[0];
      bestfullmodelindex = best_indices[1];
      bestdeltabeta = bestvar[0];
      //terminate = isnan(bestdeltabeta);
      //if (terminate) ofs <<"Best delta beta isnan\n";
      ofs<<"Before currentLL: "<<(2.*currentLL)<<endl;
      currentLL+=bestvar[1];
      bestmean = bestvar[2];
      bestsd = bestvar[3];
      ofs<<"After currentLL: "<<(2.*currentLL)<<endl;
      ofs<<"Host announced best delta beta of "<<bestdeltabeta<<" for index "<<bestsubmodelindex<<" and full model index "<<bestfullmodelindex<<" and mean/sd "<<bestmean<<","<<bestsd<<endl;
    }
    if (is_master){
      converged = (tolerance<.000000001 || terminate);
      //ofs<<"Master says converge is "<<converged<<" and terminate is "<<terminate<<endl;
      for (int dest=1;dest<mpi_numtasks;++dest){
        rc = MPI_Send(&converged,1,MPI_INT,dest,TAG_CONVERGE_FLAG,MPI_COMM_WORLD);
        rc = MPI_Send(&terminate,1,MPI_INT,dest,TAG_CONVERGE_FLAG,MPI_COMM_WORLD);
      }
    }else{
      rc = MPI_Recv(&converged,1,MPI_INT,source,TAG_CONVERGE_FLAG,MPI_COMM_WORLD,&stat);
      rc = MPI_Recv(&terminate,1,MPI_INT,source,TAG_CONVERGE_FLAG,MPI_COMM_WORLD,&stat);
      //ofs<<"Slave says converge is "<<converged<<" and terminate is "<<terminate<<endl;
    }
    //ofs<<"Elapsed time in seconds: "<<(clock()-start)/CLOCKS_PER_SEC<<endl;
  }while(!converged && !terminate); // NORMAL CASE
  //cerr<<"MPI RANK: "<<mpi_rank<<" converged!\n";
  if (is_master){
    cout<<endl;
    ofs<<"Model fit time: " <<(clock()-start)/CLOCKS_PER_SEC<<endl;
    modelsize = 0;
    for(int j=0;j<totaltasks;++j){
      if (betas[j]!=0.) {
        ++modelsize;
        string name1,name2;
        name1 = getname(j-env_covariates);
        modelvariable_t m;
        m.index = j;
        m.beta = betas[j];
        modelvariables.push_back(m);
        ofs<<j<<"\t"<<name1<<"\t"<<betas[j]<<endl;
      }
    }
    logL = 0.;
    for(int i=0;i<n;++i){
      if(mask[i]){
        double pY = exp(score[i]*data.affection[i]);
        pY/=(1+pY);
        if (data.affection[i]==1) logL += log(pY); else logL += log(1-pY);
      }
    }
    ofs<<"MS: "<<modelsize<<", currentLL: "<<(2.*currentLL)<<", newLL: "<<(2.*logL)<<", BIC: "<<-(2.*logL)+modelsize*log(n_subset) <<", Iterations: "<<iter<<endl;
    ofs<<endl;
  }
  return terminate;
}
