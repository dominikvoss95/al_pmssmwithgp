# Active Learning in pMSSM

## Table of Contents

- [Installation](#installation)
- [On startup](#on-startup)
- [Configuration](#configuration)
- [Running](#running)

## Installation

git clone --recurse-submodules git@github.com:dominikvoss95/active-learning-pmssm.git

## On Startup

INSTALL CONDA ENVIRONMENT:
conda env create -f alenvironment.yaml

activate environment:
conda activate ALenv

## Configuration
Use the config yamls to configure the training (There are three configs availaible to run seperate jobs)

CREATE TEST DATA for DMRD or CrossSection:
- First create true test data just configure 1 run per iteration, in which you want to train
- To create all: n_dim [1,2,3,4,5,6...,19]

EXACT GP Training:
- set all other models to False
- is_active: [False, True] -> number of runs for both active learning and random

DEEP GP:
- set Deep GP hyerparams

EVALUATION:
- set evaluation_mode=True
- you can evaluate it in debug mode for lower dimensions or fewer training points
- for more training points in high dimension use a job to evaluate
- GPs need training data to evaluate model

## Running

FOR DEBUGGING SESSION 15min  (viper)

for starting a debugging session:
srun -n 1 --mem=3G --partition=gpudev --time=00:15:00 --gres=gpu:1 --pty /bin/bash

loading environment:
module load gcc/14 rocm/6.3 openmpi_gpu/5.0 python-waterboa/2024.06

activate environment:
conda activate ALenv

and run script with:
python -u gp_pipeline/main.py --config /u/dvoss/al_pmssmwithgp/model/gp_pipeline/config/config.yaml

python -u gp_pipeline/main.py --config /u/dvoss/al_pmssmwithgp/model/gp_pipeline/config/config2.yaml

python -u gp_pipeline/main.py --config /u/dvoss/al_pmssmwithgp/model/gp_pipeline/config/config3.yaml

to exit: 
exit

FOR SLURM BATCH JOBS:

make folder executeable:
cd model/
chmod -R a=rwx slurm/

submit batch jobs:
/u/dvoss/al_pmssmwithgp/model/slurm/submit_jobs_viper.sh
/u/dvoss/al_pmssmwithgp/model/slurm/submit_jobs2.sh
/u/dvoss/al_pmssmwithgp/model/slurm/submit_jobs3.sh

## Authors and acknowledgment
Thanks to Jonas Wuerzinger and Lukas Heinrich.
