import os
import subprocess
import time
import shutil
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

USER = os.environ.get("USER")

class Run3PhysicsInterface:
    '''Interface to control the Run3ModelGen and Prospino pipeline for generating physical models and ROOT files.'''
    def __init__(self,
                 working_dir=f"/u/{USER}/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen",
                 setup_script=f"/u/{USER}/al_pmssmwithgp/Run3ModelGen/build/setup.sh",
                 genmodels_script=f"/u/{USER}/al_pmssmwithgp/Run3ModelGen/source/Run3ModelGen/scripts/genModels.py",
                 prospino_script=f"/u/{USER}/al_pmssmwithgp/prospino-pmssm/prospino_all.sh",
                 timeout_sec=600):
        '''Initializes the physics interface with paths and timeouts.'''
        self.working_dir = working_dir
        self.setup_script = setup_script
        self.genmodels_script = genmodels_script
        self.prospino_script = prospino_script
        self.timeout_sec = timeout_sec

    def generate_targets(self, iteration, n_dim, total_name, target, seed, true=False, root_missing=True):
        '''Function to run model generation and optionally Prospino to produce ROOT files.'''
        if true:
            scan_dir = f"/ptmp/{USER}/al_pmssmwithgp/scans/{n_dim}D/scan_true"
            config_file = f"{self.working_dir}/data/true_config.yaml"
        else:
            scan_dir = f"/ptmp/{USER}/al_pmssmwithgp/scans/{n_dim}D/{total_name}/scan_{iteration}"
            if iteration == "start":
                config_file = f"{self.working_dir}/data/start_config_{total_name}.yaml"
            else:
                config_file = f"{self.working_dir}/data/new_config_{total_name}.yaml"
            print(f"[INFO] Name of config: {config_file}")
            
        # Set always a different seed (corresponding to run)
        #seed = secrets.randbits(32)
        #seed = 42
        seed = seed

        if root_missing:
            # Clean and recreate scan_dir
            if os.path.exists(scan_dir):
                print(f"[INFO] Removing old scan directory: {scan_dir}")
                shutil.rmtree(scan_dir)
            os.makedirs(scan_dir, exist_ok=True)

            # Step 1: Run model generation via genModels.py
            gen_cmd = (
                f"cd {self.working_dir} && "
                f"source {self.setup_script} && "
                f"export PYTHONPATH={os.path.dirname(self.working_dir)}:$PYTHONPATH && "
                f"python {self.genmodels_script} --config_file {config_file} --scan_dir {scan_dir} --seed {seed}"
            )

            self._run_bash_command(gen_cmd, "genModels.py")

            # Step 2: Wait until ROOT file is created
            root_file = f"{scan_dir}/ntuple.0.0.root"
            self._wait_for_file(root_file)
            print(f"[INFO] Target file ready: {root_file}")

        # Step 3: Optional Prospino call for cross-section generation 
        slha_dir = f"{scan_dir}/SPheno/"
        if target == "CrossSection":
            if not os.path.isdir(slha_dir):
                raise FileNotFoundError(f"SLHA directory not found: {slha_dir}")

            slha_files = sorted(glob.glob(os.path.join(slha_dir, "*.slha")))

            if not slha_files:
                raise FileNotFoundError(f"No .slha files found in: {slha_dir}")

            args_list = []
            for slha_file in slha_files:
                model_name = os.path.splitext(os.path.basename(slha_file))[0]
                step_name = f"{model_name}_{iteration}"
                args_list.append((self.prospino_script, step_name, model_name, slha_file))

            # Set number of workers: match SLURM_CPUS_ON_NODE or limit manually
            max_workers = int(os.environ.get("SLURM_CPUS_ON_NODE", 4))

            print(f"[INFO] Running Prospino on {len(args_list)} models using {max_workers} workers...")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._run_prospino_call, args) for args in args_list]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(str(e))

            # Step 4: Wait until csv file is created
            csv_file = f"{scan_dir}/SPheno/.prospino_temp/crosssections.csv"
            self._wait_for_file(csv_file)
            print(f"[INFO] Target file ready: {csv_file}")

        return csv_file if target == "CrossSection" else root_file

    def _run_bash_command(self, cmd, description):
        print(f"[INFO] Running {description}: {cmd}")
        subprocess.run(cmd, shell=True, executable="/bin/bash")
    
    def _run_prospino_call(self, args):
        script, step_name, model_name, slha_file = args
        cmd = f"bash {script} {step_name} {model_name} {slha_file}"
        subprocess.run(cmd, shell=True)

    def _wait_for_file(self, file_path):
        '''Waits for a file to appear, with a timeout.'''
        start = time.time()
        while not os.path.exists(file_path):
            if time.time() - start > self.timeout_sec:
                raise TimeoutError(f"[ERROR] Timeout waiting for file: {file_path}")
            time.sleep(5)

