#include"dimension2.h"

//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline float c2g(char c,int shifts, __local float * mapping){
  float geno = 0;
  int val = ((int)c)>>(2*shifts) & 3;
  return mapping[val];
  switch (val){
    case 0:
      geno = 0;
      break;
    case 2:
      geno = 1;
      break;
    case 3:
      geno = 2;
      break;
    case 1:
      geno = 9;
      break;
    default:
      geno=9;
      break;
  }
  return geno;
}

inline void convertcov(int genoindex,
int threadindex,int index,int n,
__global const float * cov,
__local float * subset_geno
){
  subset_geno[threadindex] = cov[n*(-1*genoindex-1)+index];
}

inline void convertgeno(int genoindex,
int threadindex,int chunk,int packedstride,
__global const packedgeno_t * packedgeno_matrix,
__local packedgeno_t * geno,
__local float * subset_geno,
__local float * mapping
){
  geno[threadindex] = packedgeno_matrix[genoindex*packedstride+chunk*32+threadindex];
  int t = 0;
  for(int b=0;b<4;++b){
    for(int c=0;c<4;++c){
      //subset_geno[threadindex * 16 + t] = 1;
      subset_geno[threadindex * 16 + t] = c2g(geno[threadindex].geno[b],c,mapping);
      ++t;
    }
  }
}

__kernel void zero_beta(
const unsigned int n,
const unsigned int totaltasks,
__constant int * taskoffset,
__global float * betas,
__global delta_t * deltas
){
  int threadindex = get_local_id(0);
  int taskindex = *taskoffset * GRID_WIDTH + get_group_id(0) * BLOCK_WIDTH + threadindex;
  if (taskindex>=totaltasks) return;
  betas[taskindex] = 0.0;
  deltas[taskindex].delta_LL = 0.0;
  deltas[taskindex].delta_beta = 1.0;
}

__kernel void zero_score(
//__constant meta_data_t * meta_data,
const unsigned int n,
const unsigned int totaltasks,
__global float * score,
__global float * score_num,
__global float * score_den
){
  int threadindex = get_local_id(0);
  int chunk = get_group_id(0);
  int index = chunk*BLOCK_WIDTH+threadindex;
  if (index<n){
    score[index] = 0.00;
    score_num[index] = 0.00;
    score_den[index] = 0.00;
  }
}

__kernel void update_best_beta(
__global const best_t * best,
__global float * betas
){
  if (get_global_id(0)==0 && best->best_submodel_index>-1){
    betas[best->best_submodel_index]+=best->best_delta_beta;
  }
}

__kernel void compute_geno_sum(
const unsigned int mpi_rank,
const unsigned int env_covariates,
const unsigned int n,
const unsigned int totaltasks,
const unsigned int chunks,
const unsigned int packedstride,
__constant int * mean_sd_flag,
__constant int * taskoffset,
__global const packedgeno_t * packedgeno_matrix,
__global float * cov,
__global float * means,
__global float * mean_sd_chunks,
__local packedgeno_t * geno_1,
__local float * local_mean_sd,
__local float * subset_geno1
){
  MAPPING
  int mean_sd = *mean_sd_flag;
  int taskindex = (GRID_WIDTH/BLOCK_WIDTH)* *taskoffset + get_group_id(0);
  int chunk = get_group_id(1);
  if (taskindex>=totaltasks) return;
  float mu = means[taskindex];
  int threadindex = get_local_id(1) * BLOCK_WIDTH + get_local_id(0);
  int snp1 = mpi_rank>1?taskindex:taskindex-env_covariates;
  local_mean_sd[threadindex] = 0;
  subset_geno1[threadindex] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  int index = chunk*BLOCK_WIDTH+threadindex;
  if (index<n){
    // LOAD ALL THE COMPRESSED GENOTYPES INTO LOCAL MEMORY
    if (snp1<0) convertcov(snp1,threadindex,index,n,cov,subset_geno1);
    else{
      if (threadindex<32) convertgeno(snp1,threadindex,chunk,packedstride,packedgeno_matrix,geno_1,subset_geno1,mapping);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    local_mean_sd[threadindex]= mean_sd?pow(subset_geno1[threadindex]-mu,2):subset_geno1[threadindex]; 
    barrier(CLK_LOCAL_MEM_FENCE); // ALL WARPS HAVE UPDATED TEMP1,2
  }
  for(unsigned int s=BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex < s) {
      local_mean_sd[threadindex] += local_mean_sd[threadindex + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  mean_sd_chunks[taskindex*chunks+chunk] = local_mean_sd[0];
  return;
}

__kernel void compute_geno_mean_sd(
const unsigned int n,
const unsigned int totaltasks,
const unsigned int chunks,
__constant int * mean_sd_flag,
__constant int * taskoffset,
__global float * mean_sd_chunks,
__global float * means,
__global float * sds,
__local float * local_chunk
){
  //int taskindex = get_group_id(2) + get_group_id(0);
  int taskindex = (GRID_WIDTH/SMALL_BLOCK_WIDTH)* *taskoffset + get_group_id(0);
  int chunk = get_group_id(1);
  if (taskindex>=totaltasks) return;
  //means[taskindex] = mean_sd_chunks[taskindex];
  //sds[taskindex] = mean_sd_chunks[taskindex];
  //return;
  int threadindex = get_local_id(1) * SMALL_BLOCK_WIDTH + get_local_id(0);
  local_chunk[threadindex] = 0;
  //barrier(CLK_LOCAL_MEM_FENCE);
  if(threadindex<chunks){
    local_chunk[threadindex] = mean_sd_chunks[taskindex*chunks+threadindex];
  }
  for(unsigned int s=SMALL_BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex < s) {
        local_chunk[threadindex] += local_chunk[threadindex + s];
    }
   // barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (threadindex==0){
    if (*mean_sd_flag){
     //sds[taskindex] = taskindex+1;
     sds[taskindex] = fabs(local_chunk[0])<LAMBDA_EPSILON?1:sqrt(local_chunk[0]/ n);
    }else{
     //means[taskindex] = local_chunk[0];
     //means[taskindex] = taskindex;
     means[taskindex] = local_chunk[0] / n;
    }
  }
  return;
}

__kernel void update_score(
const unsigned int n,
__global const int * disease_status,
__global float * score,
__global float * score_num,
__global float * score_den
){
  MAPPING
  int personchunk = get_group_id(1) * GRID_WIDTH  + get_group_id(0);
  int threadindex = get_local_id(1) * BLOCK_WIDTH + get_local_id(0);
  int index = personchunk*BLOCK_WIDTH+threadindex;
  if (index<n){
    float aff =  disease_status[index];
    float score_new = score[index];
//    if (1/(1+exp(score_new))<LAMBDA_EPSILON){
//      score_num[index] = 0;
//      score_den[index] = LAMBDA_LARGE;
//    }else{
      score_num[index] = aff/(1+exp(score_new));
      score_den[index] = exp(score_new)/((1+exp(score_new))*(1+exp(score_new)));
//    }
  }
}

__kernel void compute_gradient_hessian(
const unsigned int mpi_rank,
const unsigned int env_covariates,
const unsigned int n,
const unsigned int totaltasks,
const unsigned int chunks,
const unsigned int packedstride,
__constant int * taskoffset,
__global const packedgeno_t * packedgeno_matrix,
__global float * cov,
__global float * score_num,
__global float * score_den,
__global float * means,
__global float * sds,
//__global float * debugvec,
__global float * gradient_chunks,
__global float * hessian_chunks,
__global int * mask,
__global delta_t * deltas,
__local packedgeno_t * geno_1,
__local float * local_gradient,
__local float * local_hessian,
__local float * subset_geno1
){
  MAPPING
  int taskindex = (GRID_WIDTH/BLOCK_WIDTH)* *taskoffset + get_group_id(0);
  int chunk = get_group_id(1);
  if (taskindex>=totaltasks) return;
  //if (taskindex>=totaltasks||deltas[taskindex].delta_beta==0) return;
  float mean = means[taskindex];
  float sd = sds[taskindex];
  int threadindex = get_local_id(1) * BLOCK_WIDTH + get_local_id(0);
  int snp1 = mpi_rank>1?taskindex:taskindex-env_covariates;
  // LOAD ALL THE COMPRESSED GENOTYPES INTO LOCAL MEMORY
  local_gradient[threadindex] = 0;
  local_hessian[threadindex] = 0;
  subset_geno1[threadindex] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  int index = chunk*BLOCK_WIDTH+threadindex;
  if (index<n){
    if (snp1<0) convertcov(snp1,threadindex,index,n,cov,subset_geno1);
    else{
      if (threadindex<32) convertgeno(snp1,threadindex,chunk,packedstride,packedgeno_matrix,geno_1,subset_geno1,mapping);
    }
    barrier(CLK_LOCAL_MEM_FENCE);  // WAIT UNTIL ALL GENOTYPES ARE CONVERTED
    local_gradient[threadindex]= (subset_geno1[threadindex]-mean)/sd * score_num[index] * mask[index];
    //local_hessian[threadindex]=pow((subset_geno1[threadindex]-mean)/sd,2);
    local_hessian[threadindex]=pow((subset_geno1[threadindex]-mean)/sd,2)*score_den[index] *mask[index];
    barrier(CLK_LOCAL_MEM_FENCE); // ALL WARPS HAVE UPDATED TEMP1,2
  }
  for(unsigned int s=BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex < s) {
      local_gradient[threadindex] += local_gradient[threadindex + s];
      local_hessian[threadindex] += local_hessian[threadindex + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  //gradient_chunks[taskindex*chunks+chunk] = 8;
  //hessian_chunks[taskindex*chunks+chunk] = 9;
  gradient_chunks[taskindex*chunks+chunk] = local_gradient[0];
  hessian_chunks[taskindex*chunks+chunk] = local_hessian[0];
  return;
}

__kernel void compute_delta_beta(
const unsigned int mpi_rank,
const unsigned int env_covariates,
const unsigned int totaltasks,
//const float lambda,
const unsigned int chunks,
__constant tuning_param_t * tuning_param,
__constant int * taskoffset,
__constant float * l2_norms,
__global float * betas,
__global delta_t * deltas,
__global float * gradient_chunks,
__global float * hessian_chunks,
__global int * group_indices,
__local float * local_gradient,
__local float * local_hessian
){
  float lambda = tuning_param->lambda;
  int taskindex = (GRID_WIDTH/SMALL_BLOCK_WIDTH)* *taskoffset + get_group_id(0);
  if (taskindex>=totaltasks ) return;
  //if (taskindex>=totaltasks || deltas[taskindex].delta_beta==0) return;
  int threadindex = get_local_id(1) * SMALL_BLOCK_WIDTH + get_local_id(0);
  local_gradient[threadindex] = 0;
  local_hessian[threadindex] = 0;
  if (threadindex<chunks){
    local_gradient[threadindex] = gradient_chunks[taskindex*chunks+threadindex];
    local_hessian[threadindex] = hessian_chunks[taskindex*chunks+threadindex];
  }
  for(unsigned int s=SMALL_BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex < s) {
        local_gradient[threadindex] += local_gradient[threadindex + s];
        local_hessian[threadindex] += local_hessian[threadindex + s];
    }
  }
  if(threadindex==0){
    if (taskindex==4787) {
      //currdeltabeta = l2_norms[group_index];
      //deltas[taskindex].delta_beta = l2_norms[497];
      //return;
    }
    int group_index = group_indices[taskindex];
    float origbeta = betas[taskindex];
    float currdeltabeta = 0;
    float l1_penalty =  0,l2_penalty = 0;
    //group_index = -1;
    if (group_index==-1){  // for majority of SNPs, apply standard LASSO
      l1_penalty = (mpi_rank>1 || taskindex-env_covariates>=0)?                       tuning_param->lambda:0;  // penalize only genetic predictors 
      //l1_penalty = 10000;
      if (origbeta>LAMBDA_EPSILON){
        currdeltabeta = (local_gradient[0]-l1_penalty)/
        local_hessian[0];
        if (origbeta-currdeltabeta<0) currdeltabeta = 0;
      }else if (origbeta<-LAMBDA_EPSILON){
        currdeltabeta = (local_gradient[0]+l1_penalty)/
        local_hessian[0];
        if (origbeta-currdeltabeta>0) currdeltabeta = 0;
      }else{
        if (local_gradient[0]>l1_penalty){
          currdeltabeta = (local_gradient[0]-l1_penalty)/
          local_hessian[0];
        }else if (local_gradient[0]<-l1_penalty){
          currdeltabeta = (local_gradient[0]+l1_penalty)/
          local_hessian[0];
        }else{
          currdeltabeta = 0;
        }
      }
    }else{  // for SNPs in groups, apply mixed penalty of Zhou and Lange
      l1_penalty = tuning_param->lasso_mixture*tuning_param->lambda;
      l2_penalty = (1-tuning_param->lasso_mixture)*tuning_param->lambda;
      if (taskindex==4787){
        //deltas[taskindex].delta_beta = group_indices[taskindex];
        //deltas[taskindex].delta_beta = l2_norms_big[taskindex];
        //return;
      }
      float l2norm = l2_norms[group_index];
      float full_penalty = l1_penalty;
      if (origbeta>LAMBDA_EPSILON){
        float l2 = l2_penalty/sqrt(l2norm);
        full_penalty += l2 * origbeta;
        local_hessian[0]+=l2*(1-origbeta*origbeta/l2norm);
        currdeltabeta = (local_gradient[0]-full_penalty)/local_hessian[0];
        if (origbeta-currdeltabeta<0) currdeltabeta = 0;
      }else if (origbeta<-LAMBDA_EPSILON){
        float l2 = l2_penalty/sqrt(l2norm);
        full_penalty -= l2*origbeta;
        local_hessian[0]+=l2*(1-origbeta*origbeta/l2norm);
        currdeltabeta = (local_gradient[0]+full_penalty)/local_hessian[0];
        if (origbeta-currdeltabeta>0) currdeltabeta = 0;
      }else{
        if (l2norm<LAMBDA_EPSILON){
          if (taskindex==4787) full_penalty += l2_penalty;
          else full_penalty += l2_penalty;
        }else{
          full_penalty = taskindex==4787?l1_penalty:l1_penalty;
          local_hessian[0]+=l2_penalty/sqrt(l2norm);
        }
        if (local_gradient[0]>full_penalty){
          currdeltabeta = (local_gradient[0]-full_penalty)/local_hessian[0];
        }else if (local_gradient[0]<-full_penalty){
          currdeltabeta = (local_gradient[0]+full_penalty)/local_hessian[0];
        }else{
          currdeltabeta = (taskindex==4787)?0:0;
        }
      }
    }
    deltas[taskindex].delta_beta = currdeltabeta;
      //deltas[taskindex].delta_beta = l2_norms[group_indices[taskindex]];
      //return;
  }
  return;
}

__kernel void compute_likelihoods(
const unsigned int mpi_rank,
const unsigned int env_covariates,
const unsigned int n,
const unsigned int totaltasks,
const unsigned int chunks,
const unsigned int logistic,
const unsigned int packedstride,
__constant int * taskoffset,
__constant float * currentLL,
__global const packedgeno_t * packedgeno_matrix,
__global const float * cov,
__global const int * aff,
__global float * score,
__global float * betas,
__global float * means,
__global float * sds,
__global delta_t * deltas,
__global float * likelihood_chunks,
__global int * mask,
__local packedgeno_t * geno_1,
__local float * subset_geno1,
__local float * likelihood
){
  MAPPING
  int taskindex = (GRID_WIDTH/BLOCK_WIDTH)* *taskoffset + get_group_id(0);
  int chunk = get_group_id(1);
  //if (taskindex>=totaltasks) return;
  float currdeltabeta = deltas[taskindex].delta_beta;
  if (taskindex>=totaltasks||currdeltabeta==0) return;
  float mean = means[taskindex];
  float sd = sds[taskindex];
  int snp1 = mpi_rank>1?taskindex:taskindex-env_covariates;
  int threadindex = get_local_id(1) * BLOCK_WIDTH + get_local_id(0);
 
  likelihood[threadindex] = logistic?1:0;
  subset_geno1[threadindex] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  int index = chunk*BLOCK_WIDTH+threadindex;
  if (index<n){
    float subset_aff = aff[index];
    float subset_score = score[index];
    if (snp1<0) convertcov(snp1,threadindex,index,n,cov,subset_geno1);
    else{
      if (threadindex<32) convertgeno(snp1,threadindex,chunk,packedstride,packedgeno_matrix,geno_1,subset_geno1,mapping);
    }
    barrier(CLK_LOCAL_MEM_FENCE); // wait until all warps have computed LL
    if (mask[index]){
      if (logistic){
        float pY = exp((subset_score+currdeltabeta*(subset_geno1[threadindex]-mean)/sd * subset_aff)*subset_aff);
        pY/=(1+pY);
        pY = subset_aff==1?pY:1-pY;
        likelihood[threadindex]*=pY;
      }else{
        likelihood[threadindex]+=pow(subset_aff-(subset_score+currdeltabeta*(subset_geno1[threadindex]-mean)/sd),2);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // wait until all warps have computed LL
  }
  likelihood[threadindex] = log(likelihood[threadindex]);
  barrier(CLK_LOCAL_MEM_FENCE); // wait until all warps have computed LL
  for(unsigned int s=BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex < s) {
        likelihood[threadindex] += likelihood[threadindex + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  likelihood_chunks[taskindex*chunks+chunk] = likelihood[0];
  //likelihood_chunks[taskindex*chunks+chunk] = 99;
  return;
}


__kernel void reduce_likelihoods(
const unsigned int n,
const unsigned int totaltasks,
const unsigned int chunks,
const unsigned int logistic,
__constant int * taskoffset,
__constant float * currentLL,
__global float * betas,
__global delta_t * deltas,
__global float * loglike_chunks,
__constant int * n_subset,
__local float * likelihood
){

  int taskindex = (GRID_WIDTH/SMALL_BLOCK_WIDTH)* *taskoffset + get_group_id(0);
  int chunk = get_group_id(1);
  if (taskindex>=totaltasks) return;
  float currdeltabeta = deltas[taskindex].delta_beta;
  if (currdeltabeta==0) {
    deltas[taskindex].delta_LL = 0;
    return;
  }
  int threadindex = get_local_id(1) * SMALL_BLOCK_WIDTH + get_local_id(0);
  likelihood[threadindex] = 0;
  if(threadindex<chunks){
    likelihood[threadindex] = loglike_chunks[taskindex*chunks+threadindex];
  }
  for(unsigned int s=SMALL_BLOCK_WIDTH/2; s>0; s>>=1) {
    if (threadindex < s) {
        likelihood[threadindex] += likelihood[threadindex + s];
    }
  }
  if (threadindex==0) {
    if (logistic){
      deltas[taskindex].delta_LL = (likelihood[0] - *currentLL)>=LL_EPSILON?(likelihood[0] - *currentLL):0;

    }else{
      deltas[taskindex].delta_LL  = (-.5* *n_subset *1.837877 - .5* *n_subset *log(likelihood[0])-.5* *n_subset) - *currentLL;
    }
  }
  return;
}

__kernel void best_delta(
//__constant meta_data_t * meta_data,
const unsigned int n,
const unsigned int totaltasks,
const unsigned int chunks,
__global const delta_t * deltas,
//__global int * best_index,
__global best_t * best,
__global float * best_LL_delta,
__global float * means,
__global float * sds,
__local delta_t * delta_chunk,
__local int * best_indices
){
  //__local int l_totaltasks;
//  __local int l_best_index;
//  __local delta_t l_best_delta;
  *best_LL_delta = -99999;
  float local_maxdelta_LL = *best_LL_delta;
  best_t local_best;
  
  //__local int chunks;
//  __local int offset;

  //if (get_global_id(0)>0) return;
  //chunks = totaltasks/BLOCK_WIDTH+1;
  //l_totaltasks = meta_data->totaltasks;
  //l_best_index = 0;
  //l_best_delta.delta_beta = 0;
  //l_best_delta.delta_LL = 0;
  //maxdelta_LL = 0;
  int threadindex = get_local_id(0);
  //delta_chunk[threadindex] = l_best_delta;
  //barrier(CLK_LOCAL_MEM_FENCE);
 /** 
//NAIVE APPROACH
  if (threadindex==0){
    //for(int i=0;i<1;++i){
    for(int i=0;i<totaltasks;++i){
      if (deltas[i].delta_LL>maxdelta_LL){
        maxdelta_LL = deltas[i].delta_LL;
        l_best_delta = deltas[i];
        //l_best_index = i==0?j:(i+1)*BLOCK_WIDTH+j;
        l_best_index = i;
        //l_best_index = taskindex;
      }
    }
    best->best_submodel_index = l_best_index;
    best->best_delta_beta = l_best_delta.delta_beta;
    *best_LL_delta = l_best_delta.delta_LL;
  }
  return;
**/

//OPTIMIZED 
    //best->best_delta_beta = l_best_delta.delta_beta;
    //best_LL_delta = l_best_delta.delta_LL;

  int taskindex;
  int chunk;
  for(chunk=0;chunk<chunks;++chunk){
    best_indices[threadindex] = 0;
    delta_chunk[threadindex].delta_LL = 0;
    delta_chunk[threadindex].delta_beta = 0;
    taskindex = chunk*SMALL_BLOCK_WIDTH+threadindex;
    //barrier(CLK_LOCAL_MEM_FENCE);
    if (taskindex<totaltasks){
      delta_chunk[threadindex] = deltas[taskindex];
      best_indices[threadindex] = taskindex;
      //barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int s=SMALL_BLOCK_WIDTH/2;s>0;s>>=1){
        if (threadindex<s){
          if (delta_chunk[threadindex].delta_LL<delta_chunk[threadindex+s].delta_LL){
            delta_chunk[threadindex] = delta_chunk[threadindex+s];
            best_indices[threadindex] = best_indices[threadindex+s];
            //barrier(CLK_LOCAL_MEM_FENCE);
          }
        }
      }
    }
    //barrier(CLK_LOCAL_MEM_FENCE);
    if (threadindex==0){
      if (delta_chunk[0].delta_LL>local_maxdelta_LL){
      //if (delta_chunk[0].delta_LL>*best_LL_delta){
        local_best.best_submodel_index = best_indices[0];
        local_best.best_delta_beta = delta_chunk[0].delta_beta;
        local_maxdelta_LL = delta_chunk[0].delta_LL;
        
        //best->best_submodel_index = best_indices[0];
        //best->best_delta_beta = delta_chunk[0].delta_beta;
        //*best_LL_delta = delta_chunk[0].delta_LL;
        
        //l_best_index = best_indices[0];
    //best->best_submodel_index = l_best_index;

    //best->best_delta_beta = l_best_delta.delta_beta;
    // *best_LL_delta = l_best_delta.delta_LL;
        //best->best_submodel_index = 1992;
        //return;
      }
    }
  }
  if (threadindex==0){
    *best_LL_delta = local_maxdelta_LL;
    best->best_submodel_index = local_best.best_submodel_index;
    best->best_delta_beta = local_best.best_delta_beta;
    best->mean = means[local_best.best_submodel_index];
    best->sd = sds[local_best.best_submodel_index];
  }
  return;
}


// BEGIN CYCLIC COORDINATE DESCENT GOES HERE
// END CYCLIC COORDINATE DESCENT GOES HERE
