#../bin/bvs

if [ $# -lt 2 ] ; then
  echo "<use opencl? [true|false]> <lambda> <lasso path? [true|false]"  
  exit 1
fi


enable_opencl=$1
lambda=$2
lasso_path=$3
processors=3
lasso_mixtures='1' 
analysis_name='power'
dataset='gaw17'
template_file=$analysis_name'.template.'$dataset
xml_file=$analysis_name'.xml'

for lasso_mixture in $lasso_mixtures
do
  echo Using lasso mixture $lasso_mixture
  sed "s/ENABLE_OPENCL/$enable_opencl/" $template_file | sed "s/LAMBDA/$lambda/" | sed "s/LASSO_PATH/$lasso_path/"  | sed "s/LASSO_MIXTURE/$lasso_mixture/" > $xml_file
  ln -fs ../bin/analyzer lasso2
  mpiexec -np $processors  ./lasso2 $analysis_name  </dev/null  2>&1
  mv debug_master 'debug_master.'$lasso_mixture
done
