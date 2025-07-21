#!/bin/bash

#SBATCH --job-name=happy_hour
#SBATCH --time=0-24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=25GB
#SBATCH --output=/aloy/home/ddalton/projects/scGPT_playground/scripts/logs/happy_hour.%j.out

#SBATCH --gpus=1




# Source LMOD
# Necessary for using `module` - this when using 
# paramiko is not loaded
source /etc/profile.d/z00-lmod.sh

# CUDA drivers
module load CUDA/12.0.0

cd /aloy/home/ddalton/projects/scGPT_playground/scripts

# for DB servers connection
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/home/sbnb:/aloy/home,/data/sbnb/data:/aloy/data,/data/sbnb/scratch:/aloy/scratch,/data/sbnb/chemicalchecker:/aloy/web_checker,/data/sbnb/web_updates:/aloy/web_repository"
singularity exec --cleanenv --nv /aloy/home/ddalton/singularity_images/scgpt.sif python "$@"
