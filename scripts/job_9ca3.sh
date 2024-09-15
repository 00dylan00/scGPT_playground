#!/bin/bash

#SBATCH --job-name=ft
#SBATCH --time=6-10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=150GB
#SBATCH --output=/aloy/home/ddalton/projects/scGPT_playground/scripts/logs/ft.%j.out

#SBATCH --partition=sbnb-gpu





cd /aloy/home/ddalton/projects/scGPT_playground/scripts


# for DB servers connection
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/aloy/home"
singularity exec /aloy/home/ddalton/singularity_images/scgpt.sif python $1
