#!/bin/bash

#SBATCH --job-name=test_ft_0
#SBATCH --time=2-12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB
#SBATCH --output=/aloy/home/ddalton/projects/scGPT_playground/scripts/logs/test_ft_0.%j.out

#SBATCH --partition=sbnb_cpu_zen3





cd /aloy/home/ddalton/projects/scGPT_playground/scripts


# for DB servers connection
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/aloy/home"
singularity exec /aloy/home/ddalton/singularity_images/scgpt.sif python $1 $2 $3 $4 $5
