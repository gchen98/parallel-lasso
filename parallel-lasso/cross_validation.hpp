class cross_validation_settings_t:public lasso2_settings_t{
public:
  cross_validation_settings_t(const ptree & pt,int mpi_rank);
  int replicates;
  string training_mask_basepath;
  string testing_mask_basepath;
};

class CrossValidation:public MpiLasso2{
public:
  CrossValidation();
  ~CrossValidation();
  void init(const ptree & pt);
  void run();
private:
  int * varcounts;
  int totaltasks;
  vector<string> varnames;
  int * mask;
};
