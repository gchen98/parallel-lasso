#include<iostream>
#include<fstream>
#ifdef USE_GPU
#include<CL/cl.hpp>
#include"clsafe.h"
#endif

//#include<boost/property_tree/exceptions.hpp>
#include"dimension2.h"
#include"io.hpp"
#include"main.hpp"
#include"analyzer.hpp"
#include"utility.hpp"
#include"lasso_mpi2.hpp"
#include"stability.hpp"
#include"power.hpp"

int main(int argc,char* argv[]){
  if (argc<2){
    cout<<"<analysis>"<<endl;
    exit(0);
  }
  string selected_analysis(argv[1]);
  ostringstream oss;
  oss<<selected_analysis<<".xml";
  string config_file = oss.str();
  ifstream ifs(config_file.data());
  if (!ifs.is_open()){
    cout<<"Configuration file "<<config_file<<" not found.\n";
    exit(0);
  }
  ptree pt;
  read_xml(config_file, pt);
  ifs.close();
  Analyzer * analyzer = NULL;
  //cerr<<"Selected analysis: "<<selected_analysis<<endl;
  try{
    if (!selected_analysis.compare("power")){
      #if defined power
      analyzer = new Power();
      #endif
    }else if (!selected_analysis.compare("stability")){
      #if defined stability
      analyzer = new Stability();
      #endif
    }else if (!selected_analysis.compare("univariate")){
      #if defined univariate
      analyzer = new Univariate();
      #endif
    }else if (!selected_analysis.compare("hmm")){
      #if defined hmm
      analyzer = new HMM();
      #endif
    }else if (!selected_analysis.compare("hmm_cnv")){
      #if defined hmm_cnv
      analyzer = new HMM_CNV();
      #endif
    }else if (!selected_analysis.compare("hmm_impute")){
      #if defined hmm_impute
      analyzer = new HMM_impute();
      #endif
    }else if (!selected_analysis.compare("stepwise")){
      #if defined stepwise
      analyzer = new Stepwise();
      #endif
    }
  }catch(const char * & mesg){
    cerr<<"Loader aborted with message: "<<mesg<<endl;
  }
  if (analyzer!=NULL){
    try{
      analyzer->init(pt);
      analyzer->run();
    }catch(const char * & mesg){
      cerr<<"Analyzer aborted with message: "<<mesg<<endl;
    }
    delete(analyzer);
  }else{
    cerr<<"Did not find an analyzer. Exiting\n";
  }
  return 0;
}
