module purge
module load gcc/9.2.0
module load binutils/2.26
module load cmake-3.6.2 

source /home/yangjunjie/intel/oneapi/setvars.sh --force;
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/yangjunjie/packages/pyscf/pyscf-main:/home/yangjunjie/packages/qed-tddft-dev-opt:/home/yangjunjie/packages/geomeTRIC:/home/yangjunjie/packages/networkx
export PYSCF_TMPDIR=/scratch/global/yangjunjie/

export CC=icc
export FC=ifort
export CXX=icpc

