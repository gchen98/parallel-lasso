#define GRID_WIDTH 524288
//#define GRID_WIDTH 65536
#define GRID_HEIGHT 1
#define GRID_HEIGHT_GRADIENT 16
#define BLOCK_WIDTH 512
#define SMALL_BLOCK_WIDTH 32
#define BLOCK_HEIGHT 1
//#define PACKED_SUBJECT_STRIDE 448
//#define PACKED_SNP_STRIDE 65504
#define MPI_FLOAT_ARR 6
#define MPI_INT_ARR 10
#define LL_EPSILON .00001
#define LAMBDA_EPSILON 0.0001
//#define SAMPLESIZE 697
//#define SAMPLESIZE 1000
//#define SAMPLESIZE 5761
//#define SAMPLESIZE 6806
//#define PADDED_SAMPLESIZE 7168
//#define TOTALSNPS 10 
//#define TOTALSNPS 100000
//#define taskindex (GRID_WIDTH/BLOCK_WIDTH)* *taskoffset + get_group_id(1) * GRID_WIDTH  + get_group_id(0)

#define MAPPING  __local float mapping[4]; mapping[0]=0; mapping[1]=9; mapping[2]=1; mapping[3]=2;

#define MAPPING2  __local float mapping2[4]; mapping2[0]=0; mapping2[1]=1; mapping2[2]=2; mapping2[3]=4;

typedef struct {
  char geno[4];
}__attribute__((aligned(4))) packedgeno_t;

typedef struct {
  float delta_beta;
  float delta_LL;
}__attribute__((aligned(8))) delta_t; //read/write

typedef struct {
  float best_delta_beta;
  int best_submodel_index;
  int best_genoindex_1;
  int best_genoindex_2;
  float mean;
  float sd;
} best_t; //read/write

/**
typedef struct {
  int totalpersons;
  int totaltasks;
} meta_data_t; //read only
**/

typedef struct {
  float lambda;
//  float lasso_mixture;
  float l2norm;
//  float l1norm;
//  int a_rank;
  float a_residual;
//  int z_rank;
  float z_residual;
} tuning_param_t;  //read only
