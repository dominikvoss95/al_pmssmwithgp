import argparse
import yaml
import sys
import os
from itertools import product

# Set up the path to the parent directory for slurm job
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gp_pipeline.pipeline import run_pipeline_iteration

class Config:
    '''Config class for loading parameters from YAML'''
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            # Cast strings to float, because i.e. 1e-3 is not recognized
            if isinstance(v, str):
                try:
                    # Try to convert string into float
                    float_val = float(v)
                    setattr(self, k, float_val)
                    continue
                except ValueError:
                    pass  # Keep string
            setattr(self, k, v)

def apply_sweep_combination(cfg, sweep_params: dict, index: int):
    '''Function, which applies a specific combination of sweep parameters based on the index.'''
    keys = list(sweep_params.keys())
    value_lists = list(sweep_params.values())

    combinations = list(product(*value_lists))
    total = len(combinations)

    # Check if the index is wrong if no sweep parameters or to many
    if index < 0 or index >= total:
        print(f"[WARN] Index {index} out of bounds (max {total-1}), skipping.")
        return
    
    selected_values = combinations[index]
    
    for key, value in zip(keys, selected_values):
        # Creating config objects like: setattr(cfg, "lr", 0.01) -> cfg.lr = 0.01
        setattr(cfg, key, value)
        print(f"[SWEEP] {key} = {value}")

    print(f"[INFO] Using sweep combination #{index} of {total}")
    

def main():
    '''Main function, which loads config parameters with sweep and runs pipeline iteratively'''

    # Read CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    print(f"[INFO] Current working directory: {os.getcwd()}")
    print(f"[INFO] Config file path: {args.config}")

    if not os.path.isfile(args.config):
        print(f"[ERROR] config file not found at: {args.config}")
        sys.exit(1)

    # Load config.yaml as object
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
        cfg = Config(cfg_dict)

    # Get sweep parameters
    param = "is_deep"
    # Saved here to generate name later
    cfg.__sweep_param__ = param

    if args.index is not None:
        # All sweep parameters
        sweep_params = {key: value for key, value in cfg_dict.items() if isinstance(value, list)}

        if not sweep_params:
            print("[WARN] No sweepable parameters found in config.")
        else:
            # Excluded parameters for name generation
            excluded_for_name = {"is_active", "is_deep", "run"}
            sweep_keys_for_name = [k for k in sweep_params if k not in excluded_for_name]

            # Save sweeped parameter in cfg for name generation 
            cfg.__sweep_params__ = sweep_keys_for_name

            print(f"[SWEEP] Applying sweep parameters: {list(sweep_params.keys())}")
            apply_sweep_combination(cfg, sweep_params, args.index)


    # Pipeline-Loop 
    if cfg.iteration == 0: # Set in config
        for i in range(1, cfg.max_iterations + 1):
            cfg.iteration = i
            run_pipeline_iteration(cfg)
    else: # Do only a specific iteration
        run_pipeline_iteration(cfg)

if __name__ == "__main__":
    main()

