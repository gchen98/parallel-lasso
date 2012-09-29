#include<cstdlib>
#include<cstring>
#include<iostream>
#include<sstream>
#include<fstream>
#include<math.h>
#ifdef USE_GPU
#include<CL/cl.hpp>
#include"../common/clsafe.h"
#endif
#include"../common/main.hpp"
#include"../common/analyzer.hpp"
#include"../common/io.hpp"
#include"../common/utility.hpp"
#include"dimension2.h"
#include"lasso_mpi2.hpp"
#include"cross_validation.hpp"


using namespace std;
typedef unsigned int uint;

// extends the XML configuration file container for LASSO settings 
// specific to CrossValidation Selection

cross_validation_settings_t::cross_validation_settings_t(const ptree & pt,int mpi_rank):
lasso2_settings_t(pt,mpi_rank){
  replicates = pt.get<int>("subsamples"); 
  training_mask_basepath = pt.get<string>("inputdata.training_mask_basepath"); 
  testing_mask_basepath = pt.get<string>("inputdata.testing_mask_basepath"); 
}

CrossValidation::CrossValidation():MpiLasso2(){}

CrossValidation::~CrossValidation(){
  delete settings;
}

void CrossValidation::init(const ptree & pt){
  // For loading configuration settings
  settings = new cross_validation_settings_t(pt,this->get_rank());
  ofs<<"Platform ID: "<<settings->platform_id<<endl;
  ofs<<"Device ID: "<<settings->device_id<<endl;
  ofs<<"Kernel Path: "<<settings->kernel_path<<endl;
  // Read data files
  read_data(settings->snpfile.data(),settings->pedfile.data(),settings->genofile.data(),settings->covariatedatafile.data(),settings->covariateselectionfile.data(),NULL);
  read_tasks(settings->tasklist.data(),settings->annotationfile.data());
  varnames = get_tasknames();
  totaltasks = varnames.size();
  ofs<<"There are "<<totaltasks<<" total tasks\n";
  allocate_datastructures();
  send_phenotypes();
  send_covariates();
  send_genotypes();
  send_tuning_params();
}

void CrossValidation::run(){
  cross_validation_settings_t * settings = 
  static_cast<cross_validation_settings_t * >(this->settings);
  for(int replicate = 0;replicate<settings->replicates;++replicate){
    ostringstream oss;
    oss<<settings->training_mask_basepath<<replicate;
    read_data(NULL,NULL,NULL,NULL,NULL,oss.str().data());
    send_mask();
    double logL;
    vector<modelvariable_t> modelvariables;
    fitLassoGreedy(replicate, logL,modelvariables);
    if (is_master){
      int modelsize = modelvariables.size();
      for(int i=0;i<modelsize;++i){
        //ofs<<"model var "<<i<<"has index "<<modelvariables[i].index<<endl;
      }
      ostringstream oss2;
      oss2<<settings->testing_mask_basepath<<replicate;
      read_data(NULL,NULL,NULL,NULL,NULL,oss2.str().data());
      int mislabels = 0;
      int correctlabels = 0;
      //testfit(modelvariables,mislabels,correctlabels);
      ofs<<"REPLICATE:\t"<<replicate<<endl;
      ofs<<"LAMBDA:\t"<<settings->lambda<<endl;
      ofs<<"MISLABELS:\t"<<mislabels<<endl;
      ofs<<"CORRECTLABELS:\t"<<correctlabels<<endl;
    }
  }
  cleanup();
}

