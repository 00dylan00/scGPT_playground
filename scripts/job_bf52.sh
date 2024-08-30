#!/bin/bash

#SBATCH --job-name=pp_annot
#SBATCH --time=0-1:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120GB
#SBATCH --output=/aloy/home/ddalton/projects/scGPT_playground/scripts/logs/pp_annot.%j.out






cd /aloy/home/ddalton/projects/scGPT_playground/scripts


# for DB servers connection
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/aloy/home"
singularity exec /aloy/home/ddalton/singularity_images/scgpt.sif python $1
