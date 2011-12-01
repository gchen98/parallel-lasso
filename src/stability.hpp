class stability_settings_t:public lasso2_settings_t{
public:
  stability_settings_t(const ptree & pt,int mpi_rank);
  int replicates;
  string mask_basepath;
};

class Stability:public MpiLasso2{
public:
  Stability();
  ~Stability();
  void init(const ptree & pt);
  void run();
private:
  int * varcounts;
  int totaltasks;
  vector<string> varnames;
  int * mask;
};
