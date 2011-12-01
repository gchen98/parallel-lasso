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

//using boost::property_tree::ptree;

//Analyzer::~Analyzer(){
//    cerr<<"Analyzer destructed.\n";
//}

void global_settings_t::load(const string & filename, Analyzer * & analyzer){
  using boost::property_tree::ptree;
  ptree pt;
  std::ifstream ifstest(filename.data());
  if (!ifstest.is_open()){
     cerr<<"Cannot find "<<filename<<endl;
     exit(0);
  }
  ifstest.close();
  read_xml(filename, pt);
  try{
    pedfile = pt.get<string>("settings.io.pedfile");
    snpfile = pt.get<string>("settings.io.snpfile");
    genofile = pt.get<string>("settings.io.genofile");
    covariatedatafile = pt.get<string>("settings.io.covariates.datafile");
    covariateselectionfile = pt.get<string>("settings.io.covariates.selectionfile");
    string enable = pt.get<string>("settings.io.db_enable");
    use_db = !enable.compare("true")?true:false;
    db_host = pt.get<string>("settings.io.db_host");
    db_user = pt.get<string>("settings.io.db_user");
    db_pw = pt.get<string>("settings.io.db_pw");
    db_db = pt.get<string>("settings.io.db_db");
    db_table = pt.get<string>("settings.io.db_table");
    if (selected_analysis.length()<1){
      selected_analysis = pt.get<string>("settings.selected_analysis");
    }
    cerr<<"Selected analysis is "<<selected_analysis<<endl;
    if (!selected_analysis.compare("lasso")){
       #if defined lasso
       lasso_settings_t * lasso_settings = new lasso_settings_t;
       lasso_settings->lambda = pt.get<float>("settings.analyses.lasso.lambda");
       string t =  pt.get<string>("settings.analyses.lasso.use_gpu");
       lasso_settings->use_gpu = !t.compare("true")?true:false;
       cerr<<"Assigned lambda of "<<lasso_settings->lambda<<endl;
       //analyzer = new MpiLasso( new IO(this, true),lasso_settings);
       #endif
    }else if (!selected_analysis.compare("hmm")){
      #if defined hmm
       #endif
    }else if (!selected_analysis.compare("univariate")){
      #if defined univariate
      univariate_settings_t * u = new univariate_settings_t;
      u->phenofile = pt.get<string>("settings.analyses.univariate.phenofile");
      u->maskfile = pt.get<string>("settings.analyses.univariate.maskfile");
      u->annotationfile = pt.get<string>("settings.analyses.univariate.annotationfile");
      u->type = pt.get<string>("settings.analyses.univariate.type");
      u->regression = pt.get<string>("settings.analyses.univariate.regression");
      u->geno_format = pt.get<char>("settings.analyses.univariate.geno_format");
      cerr<<"Loading Univariate"<<endl;
      //analyzer = new Univariate(new IO(this,false),u);
      #endif
    }else if (!selected_analysis.compare("stepwise")){
      #if defined stepwise
       cerr<<"Initializing stepwise\n";
      stepwise_settings_t * stepwise_settings = new stepwise_settings_t;
      string t =  pt.get<string>("settings.analyses.stepwise.impute_missing");
      stepwise_settings->impute_missing = !t.compare("true")?true:false;
      stepwise_settings->penalty_type = pt.get<string>("settings.analyses.stepwise.penalty_type");
      stepwise_settings->penalty_file = pt.get<string>("settings.analyses.stepwise.penalty_file");
      //analyzer = new Stepwise(new IO(this,true),stepwise_settings);
      #endif
    }
  }catch(boost::property_tree::ptree_bad_data & p){
    cerr<<p.what()<<": ";//<<p.data()<<endl;
  }
  cerr<<"Configuration file loaded\n";
  if (analyzer!=NULL){
    cerr<<"Analyzer loaded successfully\n";
  }
}


// to do list:
// - verify that if mask file not provided, genostring len = persons
// - if mask file provided, sum(1's) = persons
// - move getDouble() into IO

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
    if (!selected_analysis.compare("lasso")){
      #if defined lasso
      analyzer = new MpiLasso();
      #endif
    }else if (!selected_analysis.compare("power")){
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
