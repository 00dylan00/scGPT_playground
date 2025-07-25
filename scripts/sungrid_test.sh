 /bin/bash
#$ -N sample_job
#$ -e /aloy/home/ddalton/area_52/scripts/run_log
#$ -o /aloy/home/ddalton/area_52/scripts/run_log
#$ -pe make 16
#$ -r yes
#$ -j yes
#$ -l mem_free=100.0G,h_vmem=105.0G
#$ -cwd

# Loads default environment configuration 
if [[ -f $HOME/.bashrc ]]; then
  source $HOME/.bashrc
fi

# Set thread and memory-related environment variables
OMP_NUM_THREADS=16 
OPENBLAS_NUM_THREADS=16 
MKL_NUM_THREADS=16 
VECLIB_MAXIMUM_THREADS=16 
NUMEXPR_NUM_THREADS=16 
NUMEXPR_MAX_THREADS=16 

# Execute the Python script inside a Singularity container
singularity exec /aloy/home/ddalton/cc/artifacts/images/cc_py37.simg python $1
