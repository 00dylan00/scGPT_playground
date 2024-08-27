#!/bin/bash
#
#

# Options for sbatch
#SBATCH --output=%x.%j.out
#SBATCH --array=1-1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH -t 7-00:00:00
#SBATCH --qos=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -J GpuTest
# End of sbatch

# Loads default environment configuration
D="/aloy/home/ddalton/projects/scGPT_playground/scripts/"
# IMG="${D}gpu_test.simg"
#IMG="/aloy/home/pmarcos/CC/IMAGE/cc.simg"
#IMG="/aloy/home/mbertoni/images/cc.simg"
#IMG="/aloy/home/bsaldivar/DrugGPT/druggpt_20230703.simg"
#IMG="/aloy/home/bsaldivar/gpu_test/gpu_test.simg"
# IMG="/aloy/home/bsaldivar/gpu_test/tf_test.simg"
IMG="/aloy/home/ddalton/singularity_images/scgpt.sif"


# JFILE="${D}gpu_test.sh"
PFILE="${D}test_gpu.py"
# source singularity
# source /apps/manual/software/Singularity/3.9.6/etc/profile
# CUDA drivers
#export LD_LIBRARY_PATH=/apps/manual/software/CUDA/11.6.1/lib64:/apps/manual/software/CUDA/11.6.1/targets/x86_64-linux/lib:/apps/manual/software/CUDA/11.6.1/extras/CUPTI/lib64/:/apps/manual/software/CUDA/11.6.1/nvvm/lib64/:/apps/manual/software/CUDNN/8.3.2/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/apps/manual/software/CUDA/11.6.1/lib64:/apps/manual/software/CUDA/11.6.1/targets/x86_64-linux/lib:/apps/manual/software/CUDA/11.6.1/extras/CUPTI/lib64/:/apps/manual/software/CUDA/11.6.1/nvvm/lib64/:$LD_LIBRARY_PATH
# export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH #/.singularity.d/libs
export SINGULARITY_BINDPATH="/aloy/home,/aloy/data,/aloy/scratch,/aloy/web_checker,/aloy/web_repository"

module load CUDA/12.0.0

#singularity exec --cleanenv --nv "${IMG}" /bin/bash "${JFILE}"  
singularity exec --cleanenv --nv "${IMG}" python "${PFILE}"  
#singularity exec "${IMG}" /bin/bash "${JFILE}"  



    