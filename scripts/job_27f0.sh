#!/bin/bash

#SBATCH --job-name=bryan_come_cuy
#SBATCH --time=0-10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=60GB
#SBATCH --output=/aloy/home/ddalton/projects/scGPT_playground/scripts/logs/bryan_come_cuy.%j.out

#SBATCH --partition=sbnb_cpu_zen3

cd /aloy/home/ddalton/projects/scGPT_playground/scripts

# for DB servers connection
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/aloy/home"
singularity exec /aloy/home/ddalton/singularity_images/scgpt.sif python $1
