#include<cstdlib>
#include<cstring>
#include<iostream>
#include<sstream>
#include<fstream>
#include<math.h>
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
#include"stability.hpp"


using namespace std;
typedef unsigned int uint;

// extends the XML configuration file container for LASSO settings 
// specific to Stability Selection

stability_settings_t::stability_settings_t(const ptree & pt,int mpi_rank):
lasso2_settings_t(pt,mpi_rank){
  replicates = pt.get<int>("subsamples"); 
  mask_basepath = pt.get<string>("inputdata.mask_basepath"); 
}

Stability::Stability():MpiLasso2(){}

Stability::~Stability(){
  delete settings;
}

void Stability::init(const ptree & pt){
  // For loading configuration settings
  settings = new stability_settings_t(pt,this->get_rank());
  ofs<<"Platform ID: "<<settings->platform_id<<endl;
  ofs<<"Device ID: "<<settings->device_id<<endl;
  ofs<<"Kernel Path: "<<settings->kernel_path<<endl;
  // Read data files
  read_data(settings->snpfile.data(),settings->pedfile.data(),settings->genofile.data(),settings->covariatedatafile.data(),settings->covariateselectionfile.data(),NULL);
  read_tasks(settings->tasklist.data(),settings->annotationfile.data());
  varnames = get_tasknames();
  totaltasks = varnames.size();
  varcounts = new int[totaltasks];
  ofs<<"There are "<<totaltasks<<" total tasks\n";
  allocate_datastructures();
  send_phenotypes();
  send_covariates();
  send_genotypes();
  send_tuning_params();
}

void Stability::run(){
  stability_settings_t * settings = 
  static_cast<stability_settings_t * >(this->settings);
  for(int i=0;i<totaltasks;++i) varcounts[i] = 0;
  for(int replicate = 0;replicate<settings->replicates;++replicate){
    ostringstream oss;
    oss<<settings->mask_basepath<<"."<<replicate;
    read_data(NULL,NULL,NULL,NULL,NULL,oss.str().data());
    send_mask();
    double logL;
    vector<modelvariable_t> modelvariables;
    fitLassoGreedy(replicate, logL,modelvariables);
    if (is_master){
      ofs<<"REPLICATE:\t"<<replicate<<endl;
      ofs<<"LAMBDA:\t"<<settings->lambda<<endl;
      int modelsize = modelvariables.size();
      for(int i=0;i<modelsize;++i){
        ++varcounts[modelvariables[i].index];
      }
    }
  }
  if(is_master){
    for(int i=0;i<totaltasks;++i){
      ofs<<"Selection probabilities:\n";
      if (varcounts[i]>0){
        ofs<<varnames[i]<<":\t"<<1.*varcounts[i]/settings->replicates<<endl;
      }
    }
  }
  cleanup();
}

