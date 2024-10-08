#!/bin/bash

#SBATCH --job-name=test_ft_0
#SBATCH --time=2-12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB
#SBATCH --output=/aloy/home/ddalton/projects/scGPT_playground/scripts/logs/test_ft_0.%j.out

#SBATCH --partition=sbnb_cpu_zen3





cd Tutorial_Annotation.nb_clean.py --data_path /aloy/home/ddalton/projects/scGPT_playground/data/pp_data-24-10-06-01


# for DB servers connection
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/aloy/home"
singularity exec /aloy/home/ddalton/singularity_images/scgpt.sif python $1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12 $13
