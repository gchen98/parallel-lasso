#include<mpi.h>
#include<fstream>
#ifdef USE_GPU
//  #include<CL/opencl.h>
#endif
using namespace std;

class power_settings_t:public lasso2_settings_t{
public:
  power_settings_t(const ptree & pt,int mpi_rank);
  int subsamples;
  string affection_basepath;
};


class Power:public MpiLasso2{
public:
  Power();
  ~Power();
  void init(const ptree & pt);
  void run();
private:
  const char * tpstr;
  const char * tnstr;
  int * varcounts;
  int totaltasks;
  vector<string> varnames;
  int replicates;
  string affection_basepath;
  int max_tp;
  int max_tn;
};
