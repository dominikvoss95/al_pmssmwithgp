# Active Learning in pMSSM

## Table of Contents

- [Installation](#installation)
- [On startup](#on-startup)
- [Configuration](#configuration)
- [Running](#running)

## Installation

git clone --recurse-submodules https://github.com/dominikvoss95/al_pmssmwithgp.git

if you clone first, you have to execute git submodule update --init --recursive afterwards

## On Startup

INSTALL CONDA ENVIRONMENT:
```bash
conda env create -f alenvironment.yaml
```

SETUP RUN3MODELGEN:
```bash
pixi shell
cmake -S source -B build
cmake --build build -j8
source build/setup.sh
```

## Configuration
Use the config yamls to configure the training (There are three configs availaible to run seperate jobs)

ADJUST PATHS
change username

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
```bash
srun -n 1 --mem=3G --partition=gpudev --time=00:15:00 --gres=gpu:1 --pty /bin/bash
module load gcc/14 rocm/6.3 openmpi_gpu/5.0 python-waterboa/2024.06
conda activate ALenv
```

navigate to model directory and run script with:
```bash
python -u gp_pipeline/main.py --config gp_pipeline/config/config.yaml
python -u gp_pipeline/main.py --config gp_pipeline/config/config2.yaml
python -u gp_pipeline/main.py --config gp_pipeline/config/config3.yaml
```

to exit: 
exit

FOR SLURM BATCH JOBS:

make folder executeable:
```bash
chmod -R a=rwx slurm/
```

submit batch jobs:
```bash
slurm/submit_jobs_viper.sh
slurm/submit_jobs2.sh
slurm/submit_jobs3.sh
```

## Authors and acknowledgment
Thanks to Jonas Wuerzinger and Lukas Heinrich.
