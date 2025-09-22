#!/bin/bash -l
# Standard output and error:
#SBATCH -o /u/dvoss/al_pmssmwithgp/model/slurm/job-management/job_%j.out
#SBATCH -e /u/dvoss/al_pmssmwithgp/model/slurm/job-management/job_%j.err
# Initial working directory:
#SBATCH -D /u/dvoss/al_pmssmwithgp/model
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
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=125000

# Load compiler and MPI modules
module purge
module load anaconda/3/2021.11 # <-> python 3.9.
module load cuda/11.6
module load cudnn/8.8
module load pytorch/gpu-cuda-11.6/2.0.0
module load gpytorch/gpu-cuda-11.6/pytorch-1.11.0/1.8.0
pip install uproot
pip install argparse
pip install pyslha
pip install structlog
pip install imageio[ffmpeg] Pillow
pip install seaborn

# Only set index, when SLURM_ARRAY_TASK_ID exists
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    INDEX=0
else
    INDEX=$SLURM_ARRAY_TASK_ID
fi

python -u gp_pipeline/main.py --config /u/dvoss/al_pmssmwithgp/model/gp_pipeline/config/config.yaml --index $INDEX
