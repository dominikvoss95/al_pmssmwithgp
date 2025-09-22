#!/bin/bash

JOB_NAME="CLs"
CONFIG_PATH="/u/dvoss/al_pmssmwithgp/model/gp_pipeline/config/config3.yaml"
SBATCH_SCRIPT="/u/dvoss/al_pmssmwithgp/model/slurm/run_jobs3.sbatch"
TIME="4:00:00"

TOTAL=$(python -c '
import yaml; from itertools import product
cfg = yaml.safe_load(open("'"$CONFIG_PATH"'"))
sweep = {k:v for k,v in cfg.items() if isinstance(v, list)}
print(len(list(product(*sweep.values()))))
')

echo "[INFO] Submitting array job with $TOTAL combinations"
sbatch --job-name=$JOB_NAME --time=$TIME --array=0-$(($TOTAL - 1)) $SBATCH_SCRIPT
