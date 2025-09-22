#!/bin/bash -l
# Standard output and error:
#SBATCH -o /viper/u/dvoss/al_pmssmwithgp/model/slurm/job-management/job_%j.out
#SBATCH -e /viper/u/dvoss/al_pmssmwithgp/model/slurm/job-management/job_%j.err
# Initial working directory:
#SBATCH -D /viper/u/dvoss/al_pmssmwithgp/model
#
# For CPU (uncomment)
# Number of nodes and MPI tasks per node:
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --mem=100000
#
# For GPU
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60000

module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0
module load python-waterboa/2024.06

pip install wheel
pip install --index-url https://download.pytorch.org/whl/rocm6.3 torch==2.7.0+rocm6.3 torchvision==0.22.0+rocm6.3 torchaudio==2.7.0+rocm6.3
pip install gpytorch 

source /mpcdf/soft/RHEL_9/packages/x86_64/python-waterboa/2024.06/etc/profile.d/conda.sh
conda activate /u/dvoss/conda-envs/ALenv

# Only set index, when SLURM_ARRAY_TASK_ID exists
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    INDEX=0
else
    INDEX=$SLURM_ARRAY_TASK_ID
fi

python -u gp_pipeline/main.py --config /u/dvoss/al_pmssmwithgp/model/gp_pipeline/config/config.yaml --index $INDEX
