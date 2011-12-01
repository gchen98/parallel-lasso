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
#include"power.hpp"


using namespace std;
typedef unsigned int uint;

power_settings_t::power_settings_t(const ptree & pt,int mpi_rank):
lasso2_settings_t(pt,mpi_rank){
  subsamples = pt.get<int>("subsamples"); 
  affection_basepath = pt.get<string>("inputdata.affection_basepath"); 
}

Power::Power():MpiLasso2(){
//  cerr<<"Initialized constructor\n";
}

Power::~Power(){}

void Power::init(const ptree & pt){
//  cerr<<"Initializing from xml\n";
  settings = new power_settings_t(pt,this->get_rank());
  ofs<<"Platform ID: "<<settings->platform_id<<endl;
  ofs<<"Device ID: "<<settings->device_id<<endl;
  ofs<<"Kernel Path: "<<settings->kernel_path<<endl;
  read_data(settings->snpfile.data(),settings->pedfile.data(),settings->genofile.data(),settings->covariatedatafile.data(),settings->covariateselectionfile.data(),NULL);
  read_tasks(settings->tasklist.data(),settings->annotationfile.data());
  varnames = get_tasknames();
  totaltasks = varnames.size();
  varcounts = new int[totaltasks];
  ofs<<"There are "<<totaltasks<<" total tasks\n";
  power_settings_t * settings = static_cast<power_settings_t *>(this->settings);
  replicates = settings->subsamples?settings->subsamples:1;
  affection_basepath = settings->affection_basepath;
  allocate_datastructures();
  //send_phenotypes();
  send_covariates();
  send_genotypes();
}

void Power::run(){
  //cerr<<"Rank is "<<get_rank()<<endl;
  //for(int replicate = 1;replicate>=0;){
  for(int i=0;i<totaltasks;++i) varcounts[i] = 0;
  for(int replicate = 0;replicate<replicates;++replicate){
    ostringstream oss;
    oss<<affection_basepath<<"."<<replicate;
    read_data(NULL,oss.str().data(),NULL,NULL,NULL,NULL);
    send_phenotypes();
    float lambdastart = settings->lambda;
    float lambdaend = settings->lasso_path?0:settings->lambda;
    float lambda = lambdastart;
    bool terminate  = false;
    while(lambda>=lambdaend && !terminate){
      send_tuning_params(lambda,settings->lasso_mixture);
      double logL;
      ofs<<"REPLICATE:\t"<<replicate<<endl;
      ofs<<"LAMBDA:\t"<<lambda<<endl;
      vector<modelvariable_t> modelvariables;
      terminate = fitLassoGreedy(replicate, logL,modelvariables);
      if (is_master){
        int modelsize = modelvariables.size();
        cerr<<"REPLICATE:\t"<<replicate<<endl;
        cerr<<"LAMBDA:\t"<<lambda<<endl;
        cerr<<"MODELSIZE:\t"<<modelsize<<endl;
        for(int i=0;i<modelsize;++i){
          ++varcounts[modelvariables[i].index];
        }
      }
      --lambda;
      if (terminate) ofs<<"Model terminated as problem is underdetermined\n";
    }
    cerr<<"Terminated\n";
  }
  if (is_master){
    for(int i=0;i<totaltasks;++i){
      ofs<<"Power:\n";
      if (varcounts[i]>0){
        ofs<<varnames[i]<<":\t"<<1.*varcounts[i]/replicates<<endl;
      }
    }
  }
  cerr<<"Cleaning up\n";
  cleanup();
}

