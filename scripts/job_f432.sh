#!/bin/bash

#SBATCH --job-name=ft
#SBATCH --time=0-10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --output=/aloy/home/ddalton/projects/scGPT_playground/scripts/logs/ft.%j.out

#SBATCH --partition=sbnb-gpu
#SBATCH --gpus=1




# Source LMOD
# Necessary for using `module` - this when using 
# paramiko is not loaded
source /etc/profile.d/z00-lmod.sh

# CUDA drivers
module load CUDA/12.0.0

cd /aloy/home/ddalton/projects/scGPT/playground


# for DB servers connection
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/aloy/home"
singularity exec --cleanenv --nv /aloy/home/ddalton/singularity_images/scgpt.sif python $1
