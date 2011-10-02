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

MpiLasso2::MpiLasso2(){}


void MpiLasso2::init(const ptree &pt){
  MPI_Init(0,NULL);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  settings = new lasso2_settings_t;
  string t = pt.get<string>("enable_opencl");
  settings->use_gpu=!t.compare("true")?true:false;
  settings->lambda=pt.get<float>("lambda");
  t = pt.get<string>("lasso_path");
  settings->lasso_path=!t.compare("true")?true:false;
  algorithm = ALGORITHM_GREEDY;
  settings->tasklist = pt.get<string>("inputdata.tasklist");
  settings->genofile = pt.get<string>("inputdata.genofile");
  settings->pedfile = pt.get<string>("inputdata.pedfile");
  settings->snpfile = pt.get<string>("inputdata.snpfile");
  settings->covariatedatafile = pt.get<string>("inputdata.covariates.datafile");
  settings->covariateselectionfile = pt.get<string>("inputdata.covariates.selectionfile");
  settings->subsamples = pt.get<int>("subsamples");
  settings->use_gpu=!t.compare("true")?true:false;
  pt.get_child("opencl_settings.host.<xmlattr>.rank");
  BOOST_FOREACH(boost::property_tree::ptree::value_type host_object, pt.get_child("opencl_settings")){
    boost::property_tree::ptree host_tree = (boost::property_tree::ptree)host_object.second;
    int hostid = host_tree.get<int>("<xmlattr>.rank");
    if (hostid==mpi_rank){
      settings->platform_id = host_tree.get<int>("platform_id");
      settings->device_id = host_tree.get<int>("device_id");
      settings->kernel_path = host_tree.get<string>("kernel_path");
    }
  } 
  if (mpi_rank==0){
    ofs.open("debug_master");
  }else{
    ostringstream oss;
    string proc = settings->use_gpu?"gpu":"cpu";
    oss<<"debug_"<<proc<<"_"<<mpi_rank;
    ofs.open(oss.str().data());
  }
  ofs<<"Platform ID: "<<settings->platform_id<<endl;
  ofs<<"Device ID: "<<settings->device_id<<endl;
  ofs<<"Kernel Path: "<<settings->kernel_path<<endl;
  this->logistic = true;
  io = new IO();
  unsigned long int seed = 0;
  math = new MathUtils(seed);
  if (mpi_rank==0){
    cerr<<"Reading in files "<<settings->snpfile.data()<<" "<<settings->pedfile.data()<<" "<<settings->genofile.data()<<" \n";
    io->readPlinkFiles(settings->snpfile.data(),settings->pedfile.data(),settings->genofile.data(),data);
    cerr<<"Read in files "<<settings->snpfile.data()<<" "<<settings->pedfile.data()<<" "<<settings->genofile.data()<<" \n";
    io->readCovariates(settings->covariatedatafile.data(),settings->covariateselectionfile.data(),cmap);
    cerr<<"got here\n";
    n = data.totalpersons;
    lambda = settings->lambda;
    totalsnps = data.totalsnps;
    genoveclen = data.veclen;
    if (cmap.size()==0) throw "No valid covariates found.";
    env_covariates = cmap.size();
    covariates = new float[n * env_covariates];
    covariate_names = new string[env_covariates];
    int c = 0;
    for(covar_map_t::iterator it = cmap.begin();it!=cmap.end();it++){
      string key = it->first;
      vector<float> vec = it->second;
      for(int i=0;i<n;++i){
        covariates[c*n+i] = vec[i];
      }
      covariate_names[c] = it->first;
      cerr<<"Loaded covariate name: "<<it->first<<endl;
      ++c;
    }
    disease_status = new int[n];
    for(int i=0;i<n;++i) disease_status[i] = data.affection[i];
    cerr<<"Total Nodes: "<<mpi_numtasks<<", Persons: "<<n<<", SNPs: "<<totalsnps<<endl;
    replicates = settings->subsamples?settings->subsamples:1;
    cerr<<"Beginning LASSO with "<<totalsnps<<" SNPs\n";
    // load task list
    ifstream ifs_task(settings->tasklist.data());
    if (!ifs_task.is_open()){
      cerr<<"Cannot find "<<settings->tasklist<<endl;
      exit(1);
    }
    genetic_tasks = 0;
    string line;
    while(getline(ifs_task,line)) ++genetic_tasks;
    ifs_task.close();
    //submodelsize = totaltasks;
    cerr<<"Total SNPs selected: "<<genetic_tasks<<endl;
    int * genetic_tasklist = new int[genetic_tasks];
    genocharmatrix = new char[genetic_tasks * genoveclen];
    rslist = new string[genetic_tasks];
    ifs_task.open(settings->tasklist.data());
    for (int i=0;i<genetic_tasks;++i){
      getline(ifs_task,line);
      istringstream iss(line);
      iss>>genetic_tasklist[i];
      if (genetic_tasklist[i]>=totalsnps){
        cerr<<"The task "<<line<<" exceeded the total snps: "<<totalsnps<<endl;
        throw "Bad input";
      }else{
        char * genovec = data.genomatrix[genetic_tasklist[i]];
        for (int j=0;j<genoveclen;++j){
          genocharmatrix[i*genoveclen+j] = genovec[j];
        }
        rslist[i] = data.rs_list[genetic_tasklist[i]];
      }
    }
    ifs_task.close();
    totaltasks = env_covariates + genetic_tasks;
    //tasklist = new int[totaltasks];
    //int index = 0;
    //for(int i=0;i<env_covariates;++i){
    //  tasklist[index++] = (i+2)*-1;
   // }
    //for(int i=0;i<genetic_tasks;++i){
    //  tasklist[index++] = i;
   // }
    means = new float[totaltasks];
    sds = new float[totaltasks];
    betas = new float[totaltasks];
    for(int i=0;i<totaltasks;++i) betas[i] = 0;
    score = new float[n];
    for(int i=0;i<n;++i)score[i]=0;
    slaves = mpi_numtasks-1;
    if (slaves) init_master();
    //cerr<<"Exiting\n";
    //exit(1);
  }else{
    //cerr<<"Exiting\n";
    //ostringstream oss;
    //string proc = settings->use_gpu?"gpu":"cpu";
    //oss<<"debug_"<<proc<<"_"<<mpi_rank;
    //ofs.open(oss.str().data());
  }
}

MpiLasso2::~MpiLasso2(){
  //delete settings;
  //if (mpi_rank == 0) delete io;
  //else{
  if (mpi_rank){
    //delete[] betas;
    //delete[]disease_status;
    //delete[] score;
    //for(uint j=0;<ms;++j){
    //  delete[] genomatrix[j];
   // }
    //delete[] charmatrix;
    //delete[]trust_slave;
    cerr<<"MPI rank "<<mpi_rank<<" deleted\n";
  }else{
    delete settings;
    delete io;
    delete math;
    //delete [] covariates;
    //delete [] covariate_names;
    if (trait!=NULL) delete [] trait;
    //delete [] tasklist;
    if (means!=NULL) delete[]means;
    if (sds!=NULL) delete[]sds;
    delete[]betas;
    delete[]score;
    cerr<<"Master node delete\n";
  }
}


int MpiLasso2::offset(int slave){
  int offset=0;
  for(int i=0;i<slave;++i) offset+=tasks_by_slave[i];
  return offset;
}


float MpiLasso2::logL(float * score){
//double MpiLasso2::logL(float * score){
  //double logL = 0.;
  float f_L=1;
  float f_logL = 0.;
  for(int i=0;i<n;++i){
    //float pY = exp(score[i]);
    float pY = exp(score[i]*disease_status[i]);
    pY/=(1+pY);
    float pY2=disease_status[i]==1?pY:1-pY;
    if (f_L*pY2==0){
      f_logL+=log(f_L);
      f_L = pY2;
    }else{
      f_L*=pY2;
    }
  }
  f_logL+=log(f_L);
  //cout <<"double: "<<logL<<"float: "<<f_logL<<endl;
  //return f_logL;
  return f_logL;
}

void MpiLasso2::loadMatrix(const string & filename, int & rank, float * & matrix, float * & coeff, float * & hatmat, float ridge){
}

void MpiLasso2::fitBetaHats(){
}


void MpiLasso2::init_master(){
  cerr<<"Initializing "<<slaves<<" slaves\n";
  mask = new int[replicates * n];
  for(int i=0;i<replicates*n;++i) mask[i] = 1;
  if (settings->subsamples){ 
    // we want to do random subsampling
    for(int i=0;i<replicates*n;++i) mask[i] = math->RandomUniform()>.5?1:0;
  }
  bestdeltabeta = 0;
  bestsubmodelindex = 0;
  bestfullmodelindex = 0;
  // BEGIN INITIALIZE MPI CONTIGUOUS DATA STRUCTURES
  int division = totaltasks/slaves;
  tasks_by_slave = new int[slaves];
  int * snps_by_slave = new int[slaves];
  MPI_Type_contiguous(MPI_INT_ARR,MPI_INT,&intParamArrayType);
  MPI_Type_commit(&intParamArrayType);
  MPI_Type_contiguous(MPI_FLOAT_ARR,MPI_FLOAT,&floatParamArrayType);
  MPI_Type_commit(&floatParamArrayType);
  MPI_Type_contiguous(n,MPI_INT,&subjectIntArrayType);
  MPI_Type_commit(&subjectIntArrayType);
  MPI_Type_contiguous(n,MPI_FLOAT,&subjectFloatArrayType);
  MPI_Type_commit(&subjectFloatArrayType);
  MPI_Type_contiguous(env_covariates*n,MPI_FLOAT,&covArrayType);
  MPI_Type_commit(&covArrayType);
  int dim[MPI_INT_ARR];
  dim[0] = n;
  dim[1] = totalsnps; // total SNPs
  dim[2] = genoveclen;
  dim[3] = logistic;
  dim[4] = env_covariates;
  dim[5] = algorithm;
  taskFloatArrayType = new MPI_Datatype[slaves];
  charArrayType = new MPI_Datatype[slaves];
  for(int i=0;i<slaves;++i){
    tasks_by_slave[i] = (i<slaves-1)?division:(totaltasks-(slaves-1)*division);
    snps_by_slave[i] = (i==0)?tasks_by_slave[i]-env_covariates:tasks_by_slave[i];
    cerr<<"Slave "<<i<<" has "<<tasks_by_slave[i]<<" total tasks and "<<snps_by_slave[i]<<" SNPs.\n";
    // ALLOCATE APPROPRIATE # OF BYTES FOR EACH SLAVE FOR THEIR TASKS
    MPI_Type_contiguous(tasks_by_slave[i],MPI_FLOAT,taskFloatArrayType+i);
    MPI_Type_commit(taskFloatArrayType+i);
    MPI_Type_contiguous(snps_by_slave[i] * genoveclen,MPI_CHAR,charArrayType+i);
    MPI_Type_commit(charArrayType+i);
  }
  // DONE INITIALIZE MPI CONTIGUOUS DATA STRUCTURES
  int rpc_code = RPC_INIT;
  for (int dest=1;dest<mpi_numtasks;++dest){
    rc = MPI_Send(&rpc_code,1,MPI_INT,dest,TAG_RPC,MPI_COMM_WORLD);
  }
  //int totalrows = n;
  int genorow_offset = 0;
  for(int slave=0;slave<slaves;++slave){
    int o = offset(slave);
    int dest = slave+1;
    // notify of data dimensions
    //cerr<<"Notifying data dimensions to slave "<<slave<<"\n";
    dim[6] = o;
    dim[7] = tasks_by_slave[slave];
    dim[8] = snps_by_slave[slave];
    rc = MPI_Send(dim ,1, intParamArrayType,dest,TAG_INIT_DIM,MPI_COMM_WORLD);
    // move the data over
    //cerr<<"Sending affection status to slave "<<slave<<endl;
    rc = MPI_Send(data.affection,1,subjectIntArrayType,dest,TAG_INITAFF,MPI_COMM_WORLD);
    rc = MPI_Send(covariates,1,covArrayType,dest,TAG_INIT_COV,MPI_COMM_WORLD);
    //cerr<<"Sending task list to "<<dest<<"\n";
    rc = MPI_Send(genocharmatrix+genorow_offset*genoveclen,1,charArrayType[slave],dest,TAG_INITDESIGN,MPI_COMM_WORLD);
    genorow_offset+=snps_by_slave[slave]; 
  }
  cerr<<"Initialized data structures\n";
}

void MpiLasso2::cleanup_master(){
    // CLEANUP MEMORY
    int rpc_code = RPC_END;
    for (int dest=1;dest<mpi_numtasks;++dest){
      rc = MPI_Send(&rpc_code,1,MPI_INT,dest,TAG_RPC,MPI_COMM_WORLD);
    }
    MPI_Finalize();
    ofs.close();
/**
    MPI_Type_free(&intParamArrayType);
    MPI_Type_free(&floatParamArrayType);
    if (algorithm==ALGORITHM_GREEDY){
      MPI_Type_free(&piVecArrayType);
      MPI_Type_free(&rhoVecArrayType);
    }
    MPI_Type_free(&subjectIntArrayType);
    MPI_Type_free(&charArrayType);
    for(int i=0;i<slaves;++i){
      MPI_Type_free(taskFloatArrayType+i);
      if (algorithm==ALGORITHM_GREEDY){
        MPI_Type_free(zFloatArrayType+i);
        MPI_Type_free(aFloatArrayType+i);
      }
    }
    delete[] tasks_by_slave;
    delete[] betas;
**/
}


float compute_fitted2(float * aff,float * betas,float * x_row){
  //float meanLL = 0;
  //float fitted[n];
  return 0;
}

float pnorm_func2(float * theta, int len,int index, float p){
  if (theta[index]==0) return 0;
  float num = theta[index]/abs(theta[index]) * pow(theta[index],p-1);
  float den = 0;
  for(int i=0;i<len;++i){
    den+=pow(abs(theta[i]),p);
  }
  den=pow(den,1/p);
  den=pow(den,p-2);
  return num/den;
  
}


void MpiLasso2::fitLassoCCD(double & logL, int & modelsize){
}

void MpiLasso2::fitLassoParallelGradient(){
}

void MpiLasso2::fitLassoGradient(){
}

void MpiLasso2::fitLassoGreedy(double & logL, int & modelsize, bool & terminate, int replicate){
  int * current_mask = mask+replicate*n;
  int rpc_code;
  rpc_code = RPC_INIT_GREEDY;
  float tuning_params[4];
  tuning_params[0] = lambda;
  for (int dest=1;dest<mpi_numtasks;++dest){
    rc = MPI_Send(&rpc_code,1,MPI_INT,dest,TAG_RPC,MPI_COMM_WORLD);
    rc = MPI_Send(tuning_params,1,floatParamArrayType,dest,TAG_UPDATE_TUNING_PARAMS,MPI_COMM_WORLD);
    rc = MPI_Send(current_mask,1,subjectIntArrayType,dest,TAG_UPDATE_MASK,MPI_COMM_WORLD);
  }
  //return;
  for(int i=0;i<totaltasks;++i){
    betas[i] = 0;
  }
  n_subset = 0;
  double oldLL = 0;
  for(int i=0;i<n;++i){
    score[i] = 0;
    n_subset+=current_mask[i];
    if(current_mask[i]){
      double pY = exp(score[i]*data.affection[i]);
      pY/=(1+pY);
      oldLL+= (data.affection[i]==1) ? log(pY) : log(1-pY);
    }
  }
  // BEGIN LASSO LOOP
  iter=0;
  double start = clock();
  double tolerance;
  do{
    cerr<<".";
    ++iter;
    // INVOKE DISTRIBUTED GREEDY COORDINATE ASCENT
    rpc_code = RPC_GREEDY;
    for (int dest=1;dest<mpi_numtasks;++dest){
      rc = MPI_Send(&rpc_code,1,MPI_INT,dest,TAG_RPC,MPI_COMM_WORLD);
    }
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
    int bestslaveindex = 0;
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
      if (current_mask[i]){
        //++s;
        perturb = bestdeltabeta * (g1[i]-bestmean)/bestsd * data.affection[i];
        score[i] += perturb;
      }
    }
    //cerr<<"subset: "<<s<<" bestdeltabeta "<<bestdeltabeta<<" index "<<bestgenoindex<<" bestmean/sd "<<bestmean<<"/"<<bestsd<<endl;
  //double newLL=0;
  //for(int i=0;i<n;++i){
  //  if(current_mask[i]){
  //    float pY = exp(score[i]*data.affection[i]);
  //    pY/=(1+pY);
  //   //if (i<50) ofs<<" "<<i<<":"<<data.affection[i]<<","<<g1[i];
  //    newLL+= (data.affection[i]==1) ? log(pY) : log(1-pY);
  //  }
  //}
  //cerr<< "LLs are "<<newLL<<" and "<<oldLL<<endl;
  //oldLL = newLL;
  //exit(0);
  

    tolerance=bestslave.deltaLL;
    betas[bestfullmodelindex]+=bestdeltabeta;
    int dim[MPI_INT_ARR];
    dim[0] = bestsubmodelindex;
    dim[1] = bestfullmodelindex;
    float bestvar[MPI_FLOAT_ARR];
    bestvar[0] = bestdeltabeta;
    bestvar[1] = bestslave.deltaLL;
    bestvar[2] = bestmean;
    bestvar[3] = bestsd;
    for (int dest=1;dest<mpi_numtasks;++dest){
      int slave = dest-1;
      dim[0] = slave==bestslaveindex?bestsubmodelindex:-1;
      rpc_code = RPC_BETA_UPDATE;
      rc = MPI_Send(&rpc_code,1,MPI_INT,dest,TAG_RPC,MPI_COMM_WORLD);
      rc = MPI_Send(dim,1,intParamArrayType,dest,TAG_BETA_UPDATE_INDEX,MPI_COMM_WORLD);
      rc = MPI_Send(bestvar,1,floatParamArrayType,dest,TAG_BETA_UPDATE_VAL,MPI_COMM_WORLD);
      rc = MPI_Send(score,1,subjectFloatArrayType,dest,TAG_UPDATE_SCORE,MPI_COMM_WORLD);
    }
    //END CHECK FOR CONVERGENCE FOR VARIABLE J
    }while(tolerance>.000000001  ); // NORMAL CASE
    cout<<endl;
    ofs<<"Time: " <<(clock()-start)/CLOCKS_PER_SEC<<endl;
    modelsize = 0;
    for(int j=0;j<totaltasks;++j){
      if (betas[j]!=0.) {
        ++modelsize;
        string name1,name2;
        name1 = getname(j-2);
        ofs<<j<<"\t"<<name1<<"\t"<<betas[j]<<endl;
      }
    }
    logL = 0.;
    for(int i=0;i<n;++i){
      if(current_mask[i]){
        double pY = exp(score[i]*data.affection[i]);
        pY/=(1+pY);
        if (data.affection[i]==1) logL += log(pY); else logL += log(1-pY);
      }
    }
    ofs<<"MS: "<<modelsize<<", oldLL: "<<(2.*oldLL)<<", newLL: "<<(2.*logL)<<", BIC: "<<-(2.*logL)+modelsize*log(n_subset) <<", Iterations: "<<iter<<endl;
    ofs<<endl;
    terminate = false;
}

void MpiLasso2::sampleCoeff(float * hatmat, int rank, float * coeff, float * designmat, float & residual){
  for(int i=0;i<rank;++i){
    coeff[i] = 0;
    if (hatmat!=NULL){
      for(int j=0;j<totaltasks;++j){
        coeff[i]+=hatmat[i*totaltasks+j] * beta_hats[j];
        //if (betas[j]>0) cerr<<"hatmat: "<<hatmat[i*ms+j]<<" beta: "<<betas[j]<<endl;
      }
    }else{
      cerr<<"Hat matrix null, assuming coeff of zero\n";
    }
  }
  //coeff[0] = settings->lambda;
  ofs<<"Coeff hat:\n";
  for(int i=0;i<rank;++i){
    ofs<<i<<": "<<coeff[i]<<endl;
  }
  
  ofs<<endl;
  residual = 0;  
  for(int j=0;j<totaltasks;++j){
    float fitted = 0;
    for(int i=0;i<rank;++i){
      fitted+=coeff[i] * designmat[j*rank+i];
    }
    residual+=1.*abs(fitted - beta_hats[j]);
  }
  residual/=totaltasks;
  if (residual==0) residual = 1;
}

string MpiLasso2::getname(int genoindex){
  if (genoindex<0){
    //cerr<<"Grabbing covindex: "<<abs(genoindex)-2<<endl;
    //return "hellocov";
    return covariate_names[abs(genoindex)-1];
  }else if (genoindex>-1){
  //cerr<<"Grabbing index: "<<genoindex<<endl;
    //return "geno";
    return rslist[genoindex];
  }
  return "N/A";
}

void MpiLasso2::run(){
  //double start = clock();
  if (mpi_rank>0){
    listen();
    ofs<<"Loop complete!\n";
    ofs.close();
  }else{
    bool terminate = false;
    do{
      //for(int i=0;i<totaltasks;++i) varcounts[i] = 0;
      for(int replicate = 0;replicate<replicates;++replicate){
        // BEGIN ZERO OUT BETA AND SCORE AND UPDATE NEW PI 
        //float pi = modelindex;
        double logL;
        int modelsize;
        cerr<<"REPLICATE:\t"<<replicate<<endl;
        cerr<<"LAMBDA:\t"<<lambda<<endl;
        ofs<<"REPLICATE:\t"<<replicate<<endl;
        ofs<<"LAMBDA:\t"<<lambda<<endl;
        if(algorithm==ALGORITHM_GREEDY){
          fitLassoGreedy(logL,modelsize,terminate,replicate);
        }else if(algorithm==ALGORITHM_CCD){
          fitLassoCCD(logL,totaltasks);
        }else if(algorithm==ALGORITHM_STANDARDIZE){
          compute_mean_sd(1);
        }
      }
      if (settings->lasso_path ){
        lambda = terminate?-1:lambda-10;
      }else{
        lambda=-1;
      }
      if (algorithm==ALGORITHM_STANDARDIZE) lambda=-1;
    } while(lambda>=0);
 // END LOOP FOR MODEL INDEX
    cleanup_master();
  }
}


int main_lasso2(int argc,char* argv[]){
  if (argc<2){
    cerr<<"Arguments: <fileprefix>\n";
    exit(1);
  }
  return 0;
}
