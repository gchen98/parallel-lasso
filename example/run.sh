#../bin/bvs

if [ $# -lt 2 ] ; then
  echo "<use opencl? [true|false]> <lambda>" 
  exit 1
fi

enable_opencl=$1
lambda=$2

echo Using Lambda $lambda
sed "s/ENABLE_OPENCL/$enable_opencl/" lasso2.template | sed "s/LAMBDA/$lambda/"  > lasso2.xml
mpiexec -np 3  ../bin/analyzer lasso2 </dev/null  2>&1
