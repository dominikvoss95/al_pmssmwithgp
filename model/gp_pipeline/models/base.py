import os
import torch
import time
import gpytorch
import numpy as np
import pandas as pd
import pickle
import uproot
import json
from torch import Tensor
from scipy.stats import qmc
from gpytorch.likelihoods import GaussianLikelihood
from sklearn.preprocessing import StandardScaler

from gp_pipeline.models.exact_gp import ExactGP
from gp_pipeline.models.sparse_gp import SparseGP
from gp_pipeline.models.deep_gp import DeepGP
from gp_pipeline.models.mlp import MLP
from gp_pipeline.utils.sampling import create_lhs_samples, create_random_samples
from gp_pipeline.utils.truth import create_two_gaussian_hyperspheres, create_two_different_circles
from gp_pipeline.utils.selection import EntropySelectionStrategy
from gp_pipeline.models.gp_model_config import GPModelConfig
from gp_pipeline.utils.physics_interface import Run3PhysicsInterface
from gp_pipeline.utils.create_config import create_config
from gp_pipeline.utils.plotting import plot_corner, plot_y_distribution

USER = os.environ.get("USER")

class GPModelPipeline:
    '''Class for loading data and training different models'''
    def __init__(self, config: GPModelConfig):
        '''Initialize model parameter and data'''
        self.config = config

        # Instantiate all config parameter as attributes
        for key in dir(config):
            if not key.startswith("_"):
                try:
                    setattr(self, key, getattr(config, key))
                except Exception:
                    pass  

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GP components
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = None
        self.best_model = None
        self.losses = None
        self.losses_valid = None

        self.selection_strategy = EntropySelectionStrategy(self.blur, self.beta, self.tolerance_sampling, self.proximity_sampling)

        self.truth_fn = create_two_gaussian_hyperspheres(self.n_dim)
        # self.truth_fn = create_two_different_circles(self.n_dim)
        if self.target == "Toy":
            self._load_or_compute_threshold(path=f"/u/{USER}/al_pmssmwithgp/model/training_data/threshold_{self.n_dim}.json")
        elif self.target == "CLs":
            self.thr = 0.05
        else:
            self.thr = 0.0

        # True value for log division of target
        if self.target == "DMRD":
            self.true_value = 0.12
        elif self.target == "CrossSection":
            self.true_value = 0.03
        elif self.target == "CLs":
            self.true_value = 0.05

        if self.target == "DMRD":
            base_order = ["IN_M_1", "IN_M_2", "IN_tanb", "IN_mu", "IN_M_3", "IN_At", "IN_Ab", "IN_Atau", 
                            "IN_mA", "IN_meL", "IN_mtauL", "IN_meR", "IN_mtauR", "IN_mqL1", "IN_mqL3", 
                            "IN_muR", "IN_mtR", "IN_mdR", "IN_mbR"]
        else:
            base_order = ["M_1", "M_2", "tan_beta", "mu", "M_3", "At", "Ab", "Atau", 
                            "mA", "mqL3", "mtR", "mbR", "meL", "mtauL", "meR", "mtauR", 
                            "mqL1", "muR", "mdR"]

        order = {i: base_order[:i] for i in range(1, len(base_order) + 1)}
        
        print("[INFO] Number of dimensions:", self.n_dim)

        # Get the parameters to include based on n
        self.selected_columns = order.get(self.n_dim, None)

        print("[INFO] Selected columns =", self.selected_columns)

        # Define data_min and data_max 
        base_order_data = ["M_1", "M_2", "tanb", "mu", "M_3", "AT", "Ab", "Atau",
                    "mA", "mqL3",  "mtR", "mbR", "meL", "mtauL", "meR", "mtauR", 
                    "mqL1", "muR", "mdR"]

        range_dict = {
            "M_1": [-2000, 2000], "M_2": [-2000, 2000], "tanb": [1, 60], "mu": [-2000, 2000],
            "M_3": [1000, 5000], "AT": [-8000, 8000], "Ab": [-2000, 2000], "Atau": [-2000, 2000],
            "mA": [0, 5000], "mqL3": [2000, 5000], "mtR": [2000, 5000], "mbR": [2000, 5000], 
            "meL": [0, 10000], "mtauL": [0, 10000], "meR": [0, 10000], "mtauR": [0, 10000], 
            "mqL1": [0, 10000], "muR": [0, 10000], "mdR": [0, 10000]
        }

        selected_keys = base_order_data[:self.n_dim]
        mins = [range_dict[k][0] for k in selected_keys]
        maxs = [range_dict[k][1] for k in selected_keys]

        self.data_min = torch.tensor(mins, dtype=torch.float32).to(self.device)
        self.data_max = torch.tensor(maxs, dtype=torch.float32).to(self.device)

        # Standard scaling
        self.y_norm = False
        if self.y_norm:
            self.scaler_y = StandardScaler()

        # Load true data for DMRD
        if self.target in ["DMRD", "CLs", "CrossSection"]:
            self.load_true_physical_data()
        else:
            self.x_true = torch.tensor(qmc.LatinHypercube(d=self.n_dim).random(n=10000), dtype=torch.float32).to(self.device)
            self.y_true = torch.tensor(self.truth_fn(self.x_true))

        self.x_test = torch.tensor(qmc.LatinHypercube(d=self.n_dim).random(n=10000), dtype=torch.float32).to(self.device)

        self.load_initial_data()

        # Define labales for plotting
        self.labels = base_order_data

    def _load_or_compute_threshold(self, path="threshold.json"):
        '''Function to load or compute threshold for Toy Problem'''
        if os.path.exists(path):
            with open(path, "r") as f:
                self.thr = json.load(f)["threshold"]
            print(f"[INFO] Loaded threshold from {path}: {self.thr:.4f}")
        else:
            train_dir = f'/u/{USER}/al_pmssmwithgp/model/training_data'
            if not os.path.exists(train_dir):
                os.makedirs(train_dir, exist_ok=True)
            self.thr = self._estimate_threshold()
            with open(path, "w") as f:
                json.dump({"threshold": self.thr}, f)
            print(f"[INFO] Computed and saved threshold to {path}: {self.thr:.4f}")

    def _estimate_threshold(self):
        '''Function to estimate threshold for Toy Problem'''
        points = create_lhs_samples(self.n_dim, 1000000)
        values = self.truth_fn(points)
        print(f"[INFO] Threshold: {values.mean()}")
        return values.mean()
    
    def load_initial_data(self, not_active=False):
        '''Function to decide to load either physical or toy data'''
        if self.target == "Toy":
            self._load_initial_toy_data()
        else:
            self._load_initial_physical_data(not_active)

    def _load_filtered_df(self, path):
        '''Function to create dataframe for different physical targets'''
        if self.target == "CrossSection":
            final_df = pd.read_csv(path)
            target = final_df['xsec_TOTAL']
            mask = (target > 0)
        elif self.target == "DMRD":
            file = uproot.open(path)
            tree = file["susy"]
            final_df = tree.arrays(library="pd")
            target = final_df['MO_Omega']
            mask = (target > 0) & (final_df['SP_m_h'] != -1)
        elif self.target == "CLs":
            final_df = pd.read_csv(path)
            target = final_df['Final__CLs']
            mask = (target >= 0)
        filtered_data = {param: final_df[f"{param}"][mask] for param in self.selected_columns}
        filtered_data[f'{self.target}'] = target[mask] 
        return pd.DataFrame(filtered_data)

    def _load_initial_physical_data(self, not_active):
        '''Function to create training and validation '''

        print(f"[INFO] Total name: {self.total_name}")

        is_initial_scan = self.iteration == 1 
        scan_name = "scan_start" if is_initial_scan else f"scan_{self.iteration}"
        config_file = f"{'start' if is_initial_scan and not not_active else 'new'}_config_{self.total_name}.yaml"

        # Generate training data for LH
        if self.target == "DMRD":
            train_root_file_path = (
                f"/ptmp/{USER}/al_pmssmwithgp/scans/"
                f"{self.n_dim}D/{self.total_name}/{scan_name}/ntuple.0.0.root"
            )
        elif self.target == "CrossSection":
            train_root_file_path = (
                f"/ptmp/{USER}/al_pmssmwithgp/scans/"
                f"{self.n_dim}D/{self.total_name}/{scan_name}/SPheno/.prospino_temp/crosssections.csv"
            )
        elif self.target == "CLs":
            train_root_file_path = f"/u/{USER}/al_pmssmwithgp/model/pmssm_data/EWKino.csv"

        
        # Only create inital data for other targets than CLs
        if self.target != "CLs":
            if is_initial_scan and not self.evaluation_mode: # or not_active:
                print("[INFO] Loading initial data from ROOT file...")
                total_points = self.initial_train_points + self.valid_points
                # if is_initial_scan:
                #     if not_active:
                #         total_points = self.initial_train_points
                #     else:
                #         total_points = self.initial_train_points + self.valid_points
                # else: 
                #     total_points = self.initial_train_points
                print(f"[INFO] Creating config with {total_points} points")
                create_config(
                    new_points=total_points,
                    n_dim=self.n_dim,
                    output_file=config_file,
                    prior_type="flat"
                )
                physics_interface = Run3PhysicsInterface()
                physics_interface.generate_targets(
                    'start' if is_initial_scan and not not_active else self.iteration,
                    self.n_dim,
                    self.total_name,
                    self.target,
                    self.run
                )
            else:
                print("[INFO] Skipping loading initial data")

        # Validation data always from scan_start
        base_valid_root_file_path = (
            f"/ptmp/{USER}/al_pmssmwithgp/scans/"
            f"{self.n_dim}D/{self.total_name}/scan_start/"
        )
        if self.target == "DMRD":
            valid_root_file_path = base_valid_root_file_path + f'ntuple.0.0.root'
        elif self.target == "CrossSection": 
            valid_root_file_path = base_valid_root_file_path + f'SPheno/.prospino_temp/crosssections.csv'
        elif self.target == "CLs":
            valid_root_file_path = f"/u/{USER}/al_pmssmwithgp/model/pmssm_data/EWKino.csv"


        # Only initialize, if iteration greater than 1, because training data is loaded
        if self.iteration > 1 and not not_active:
            # Dummy train set 
            train_df = self._load_filtered_df(valid_root_file_path)  
            self.x_train = torch.empty((0, self.n_dim), dtype=torch.float32).to(self.device)
            self.y_train = torch.empty((0,), dtype=torch.float32).to(self.device)
        else:
            # Load training data
            train_df = self._load_filtered_df(train_root_file_path)
            self.x_train = torch.stack([torch.tensor(train_df[param].values[:self.initial_train_points], dtype=torch.float32) for param in self.selected_columns], dim=1).to(self.device)
            if self.target == "CLs":
                print(f"[DEBUG] Number of initial training points {self.initial_train_points}")
                print(f"[DEBUG] Total number of points {train_df[f'{self.target}'].shape}")
                self.y_train = torch.tensor(train_df[f'{self.target}'].values[:self.initial_train_points], dtype=torch.float32).to(self.device)
            else:
                self.y_train = torch.log(torch.tensor(train_df[f'{self.target}'].values[:self.initial_train_points], dtype=torch.float32).to(self.device) / self.true_value)
            
            if self.y_norm:
                self.y_train = self.scaler_y.fit_transform(self.y_train.reshape(-1, 1)).squeeze()

        # Load validation data always from scan_start
        valid_df = self._load_filtered_df(valid_root_file_path)
        self.x_valid = torch.stack([torch.tensor(valid_df[param].values[-self.valid_points:], dtype=torch.float32) for param in self.selected_columns], dim=1).to(self.device)
        if self.target == "CLs":
            self.y_valid = torch.tensor(valid_df[f'{self.target}'].values[-self.valid_points:], dtype=torch.float32).to(self.device)
        else:
            self.y_valid = torch.log(torch.tensor(valid_df[f'{self.target}'].values[-self.valid_points:], dtype=torch.float32).to(self.device) / self.true_value)

        # Total data
        self.x_all = torch.stack([torch.tensor(train_df[param].values, dtype=torch.float32) for param in self.selected_columns], dim=1).to(self.device)
        self.y_all = torch.tensor(train_df[f'{self.target}'].values, dtype=torch.float32).to(self.device)

        # Normalization
        self.x_train = self._normalize(self.x_train)
        self.x_valid = self._normalize(self.x_valid)
        self.x_all = self._normalize(self.x_all)

        if self.y_norm:
            self.y_valid = self.scaler_y.transform(self.y_valid.reshape(-1, 1)).squeeze()
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
            self.y_valid = torch.tensor(self.y_valid, dtype=torch.float32).to(self.device)

        print(f"[INFO] X_train: {self.x_train}")
        print(f"[INFO] X_train Shape: {self.x_train.shape}")
        print(f"[INFO] Y_train: {self.y_train}")
        print(f"[INFO] Y_train Shape: {self.y_train.shape}")

    def _load_initial_toy_data(self):
        '''Function to create toy training and validation data'''
        if self.is_lh:
            x_train_np = create_lhs_samples(self.n_dim, self.initial_train_points)
            x_valid_np = create_lhs_samples(self.n_dim, self.valid_points)
        else:
            x_train_np = create_random_samples(self.n_dim, self.initial_train_points)
            x_valid_np = create_random_samples(self.n_dim, self.valid_points)
        self.x_train = torch.tensor(x_train_np, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.truth_fn(x_train_np), dtype=torch.float32).to(self.device)

        self.x_valid = torch.tensor(x_valid_np, dtype=torch.float32).to(self.device)
        self.y_valid = torch.tensor(self.truth_fn(x_valid_np), dtype=torch.float32).to(self.device)

        if self.y_norm:
            self.y_train = self.scaler_y.fit_transform(self.y_train.reshape(-1, 1)).squeeze()
            self.y_valid = self.scaler_y.transform(self.y_valid.reshape(-1, 1)).squeeze()
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
            self.y_valid = torch.tensor(self.y_valid, dtype=torch.float32).to(self.device)

    def load_true_physical_data(self):
        '''Function to load true data from physical targets to evaluate accuracy'''

        print("[INFO] Loading true data from ROOT file...")

        if self.target == "CrossSection":
            if self.n_dim == 12:
                true_file_path = f"/u/{USER}/al_pmssmwithgp/model/pmssm_data/EWKino.csv"
                final_df = pd.read_csv(true_file_path)
            else:
                true_root_file_path = f"/ptmp/{USER}/al_pmssmwithgp/scans/{self.n_dim}D/scan_true/ntuple.0.0.root"
                true_csv_file_path = f"/ptmp/{USER}/al_pmssmwithgp/scans/{self.n_dim}D/scan_true/SPheno/.prospino_temp/crosssections.csv"
                if not os.path.exists(true_root_file_path):
                    total_points = 10000
                    create_config(new_points=total_points, n_dim=self.n_dim, output_file =f'true_config.yaml', prior_type="flat")
                    physics_interface = Run3PhysicsInterface()
                    physics_interface.generate_targets('true', self.n_dim, self.run, target="CrossSection", run=self.run,  true=True, root_missing=True) 
                else:
                    if not os.path.exists(true_csv_file_path):
                        physics_interface = Run3PhysicsInterface()
                        physics_interface.generate_targets('true', self.n_dim, self.run, target="CrossSection", seed=self.run, true=True, root_missing=False) 
                    else:
                        print("[INFO] True csv file exists!")
                final_df = pd.read_csv(true_csv_file_path)
        elif self.target == "CLs":
            true_file_path = f"/u/{USER}/al_pmssmwithgp/model/pmssm_data/EWKino.csv"
            final_df = pd.read_csv(true_file_path)
            final_df = final_df[self.initial_train_points+self.valid_points:]
        else:
            true_root_file_path = f"/ptmp/{USER}/al_pmssmwithgp/scans/{self.n_dim}D/scan_true/ntuple.0.0.root"
            if not os.path.exists(true_root_file_path):
                total_points = 10000
                create_config(new_points=total_points, n_dim=self.n_dim, output_file =f'true_config.yaml', prior_type="flat")
                physics_interface = Run3PhysicsInterface()
                physics_interface.generate_targets('true', self.n_dim, self.run, target=self.target, seed=self.run, true=True)
            
            file = uproot.open(true_root_file_path)
            tree_name = "susy"
            tree = file[tree_name]
            final_df = tree.arrays(library="pd")

        # Apply the mask to filter only valid values
        if self.target == "DMRD":
            target = final_df['MO_Omega']
            mask = (target > 0) & (final_df['SP_m_h'] != -1)
        elif self.target == "CrossSection":
            target = final_df['xsec_TOTAL']
            mask = (target > 0)
        elif self.target == "CLs":
            target = final_df['Final__CLs']
            mask = (target >= 0) # Also include 0 values
        filtered_data = {param: final_df[f"{param}"][mask] for param in self.selected_columns}
        filtered_data[f'{self.target}'] = target[mask] 

        limited_df = pd.DataFrame(filtered_data)
        self.x_true = torch.stack([torch.tensor(limited_df[param].values, dtype=torch.float32) for param in self.selected_columns], dim=1).to(self.device)
        if self.target == "CLs": # CLs values are naturally normalized between [0,1]
            self.y_true = torch.tensor(limited_df[f'{self.target}'].values, dtype=torch.float32).to(self.device)
        else:
            self.y_true = torch.log(torch.tensor(limited_df[f'{self.target}'].values, dtype=torch.float32).to(self.device) / self.true_value)

        self.x_true = self._normalize(self.x_true)

        plot_y_distribution(self, save_path=f'/u/{USER}/al_pmssmwithgp/model/plots/y_true_{self.target}.png')
        #plot_corner(self, save_path=f'/u/{USER}/al_pmssmwithgp/model/plots/corner_{self.target}.png')


    def load_additional_data(self, new_x=None, new_root_file=None):
        '''Function to load new points selected by Active Learning Selection '''
        if self.target == "DMRD" or self.target == "CrossSection":
            self._load_additional_physical_data(new_root_file)
        else:
            self._load_additional_toy_data(new_x)

    def _load_additional_toy_data(self, new_x=None):
        '''Function to load new points selected by Active Learning for toy problem'''
        new_y = torch.tensor(self.truth_fn(new_x.cpu().numpy()), dtype=torch.float32).to(self.device)

        if self.y_norm:
            new_y = self.scaler_y.transform(new_y.reshape(-1, 1)).squeeze()
            new_y = torch.tensor(new_y, dtype=torch.float32).to(self.device)

        if new_y.ndim == 0:
            new_y = new_y.unsqueeze(0)
        self.x_train = torch.cat([self.x_train, new_x])
        self.y_train = torch.cat([self.y_train, new_y])

        # Remove duplicates
        data_set = {tuple(x.tolist()) + (float(y.item()),) for x, y in zip(self.x_train, self.y_train)}
        x_unique, y_unique = zip(*[(x[:-1], x[-1]) for x in data_set])
        self.x_train = torch.tensor(x_unique, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_unique, dtype=torch.float32).to(self.device)

    def _check_close_points_in_x_train(self, min_distance=1e-4):
        '''Function to check if any points in x_train are closer to each other than min_distance.'''

        n = self.x_train.shape[0]
        count_close = 0

        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(self.x_train[i] - self.x_train[j])
                if dist < min_distance:
                    count_close += 1
                    print(f"[WARNING] Point {i} and {j} are very close (distance = {dist:.2e})")

        if count_close == 0:
            print("[INFO] No close points found in x_train.")
        else:
            print(f"[INFO] Found {count_close} pairs of points with distance < {min_distance}")

    def _load_additional_physical_data(self, new_root_file=None):
        '''Function to load new points selected by Active Learning for physical targets'''

        limited_al_df = self._load_filtered_df(new_root_file)

        additional_x_train = torch.stack([torch.tensor(limited_al_df[param].values[:self.n_new_points], dtype=torch.float32) for param in self.selected_columns], dim=1).to(self.device)
        additional_x_train = self._normalize(additional_x_train)

        additional_y_train = torch.log(torch.tensor(limited_al_df[f'{self.target}'].values[:self.n_new_points], dtype=torch.float32).to(self.device) / self.true_value)

        print(f"[INFO] X_train shape: {self.x_train.shape}")  
        print(f"[INFO] Additional_x_train shape: {additional_x_train.shape}")
        print(f"[INFO] Additional_x_train: {additional_x_train}")

        # Ensure the additional_x_train has the same number of columns as self.x_train
        if additional_x_train.shape[1] != self.x_train.shape[1]:
            additional_x_train = additional_x_train[:, :self.x_train.shape[1]] 
        
        # Append the new points to the existing training data
        self.x_train = torch.cat((self.x_train, additional_x_train))
        self.y_train = torch.cat((self.y_train, additional_y_train))

        # Check if points are too close together
        self._check_close_points_in_x_train()

        combined_set = {tuple(float(x) for x in x_row) + (float(y.item()),) for x_row, y in zip(self.x_train, self.y_train)}
        x_train, y_train = zip(*[(x[:self.n_dim], x[self.n_dim]) for x in combined_set])
        self.x_train = torch.tensor(list(x_train), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(list(y_train), dtype=torch.float32).to(self.device)

        if self.y_norm:
            self.y_train = self.scaler_y.transform(self.y_train.reshape(-1, 1)).squeeze()
            self.y_train = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)

        print("[INFO] Training points after adding: ", self.x_train, self.x_train.shape)
        
    def _normalize(self, data):
        '''Function to normalize the data'''
        return (data - self.data_min) / (self.data_max - self.data_min)

    def _unnormalize(self, data):
        '''Function to unnormalize the data'''
        return data * (self.data_max - self.data_min) + self.data_min

    def initialize_model(self):
        '''Function to initialize the models'''

        print("[INFO] Initializing model with following points:")
        print(f"[INFO] X_train shape: {self.x_train.shape}")
        print(f"[INFO] Y_train shape: {self.y_train.shape}")
        print(f"[INFO] X_valid shape: {self.x_valid.shape}")
        print(f"[INFO] Y_valid shape: {self.y_valid.shape}")

        # Train with Multilayer Perceptron
        if self.is_mlp:
            self.model = MLP(self.x_train,
                             self.y_train, 
                             self.x_valid, 
                             self.y_valid, 
                             self.n_dim).to(self.device)
        # Train with Deep GP
        elif self.is_deep:
            self.model = DeepGP(
                self.x_train, 
                self.y_train, 
                self.x_valid, 
                self.y_valid, 
                self.n_dim, 
                num_hidden_dims=self.num_hidden_dims, 
                num_middle_dims=self.num_middle_dims, 
                num_samples=self.num_samples, 
                num_inducing_max=self.num_inducing_max,
                thr=self.thr,
                kernel=self.kernel,
                lengthscale=self.lengthscale,
                noise=self.noise,
                m_nu=self.m_nu,
                ).to(self.device)
        # Train with Sparse GP
        elif self.is_sparse:
            self.model = SparseGP(
                self.x_train, 
                self.y_train, 
                self.x_valid, 
                self.y_valid, 
                self.n_dim,
                num_inducing_max=self.num_inducing_max,
                thr=self.thr,
                lengthscale=self.lengthscale,
                kernel=self.kernel,
                m_nu=self.m_nu,
                noise=self.noise
            )
        # Train with Exact GP
        else:
            self.model = ExactGP(
                self.x_train,
                self.y_train,
                self.x_valid,
                self.y_valid,
                self.n_dim,
                lengthscale=self.lengthscale,
                use_ard=self.use_ard,
                noise=self.noise,
                kernel=self.kernel,
                m_nu=self.m_nu,
                num_mixtures=self.num_mixtures,
                use_dkl=self.use_dkl,
                feature_dim=self.feature_dim,
                thr=self.thr,
                epsilon=self.epsilon
                ).to(self.device)
            
        # Train a MLP with AL points
        if self.is_mlp_with_al:
            self.model_mlp = MLP(self.x_train,
                                self.y_train, 
                                self.x_valid, 
                                self.y_valid, 
                                self.n_dim).to(self.device)

    def train_model(self):
        '''Function to train the models'''
        start_time = time.time()
        if self.is_mlp:
            self.best_model, self.losses, self.losses_valid = self.model.do_train_loop(lr=self.learning_rate, 
                                                                                       iters=self.iterations)
        elif self.is_deep:
            self.best_model, self.losses, self.losses_valid = self.model.do_train_loop(lr=self.learning_rate,
                                                                                       iters=self.iterations, 
                                                                                       batch_size=self.batch_size, 
                                                                                       jitter=self.jitter)
        elif self.is_sparse:
            self.best_model, self.losses, self.losses_valid = self.model.do_train_loop(lr=self.learning_rate,
                                                                                       iters=self.iterations, 
                                                                                       batch_size=self.batch_size, 
                                                                                       jitter=self.jitter)
        else:
            self.best_model, self.losses, self.losses_valid = self.model.do_train_loop(lr=self.learning_rate, 
                                                                                       iters=self.iterations, 
                                                                                       jitter=self.jitter)
            
        end_time = time.time()
        duration = end_time - start_time
        print(f"[INFO] Training took {duration:.2f} seconds.")

    def train_mlp_parallel(self):
        '''Function to train mlp with AL points'''
        start_time = time.time()
        print(f"[DEBUG] Training with these points {self.x_train} and shape {self.x_train.shape}")
        self.best_model_mlp, self.losses_mlp, self.losses_valid_mlp = self.model_mlp.do_train_loop(lr=self.learning_rate, 
                                                                                                    iters=self.iterations)
        end_time = time.time()
        duration = end_time - start_time
        print(f"[INFO] Training took {duration:.2f} seconds.")

    def evaluate_model(self):
        '''Function to evaluate the model and the likelihood'''
        print("[INFO] Evaluating the model and the likelihood")
        self.model.eval()
        self.likelihood.eval()

    def select_new_points(self, N = 4):
        '''Function to select new points according to selection strategy'''
        if self.is_active:
            print(f"[INFO] Selecting {N} new points with Active Learning...")
            selection = self.selection_strategy.select_new_points(self, N=N)
        else:
            print(f"[INFO] Selecting {N} new points randomly...")
            selection = self.selection_strategy.select_randomly(self, N=N)
        return selection

    def save_training_data(self, filepath):
        '''Function to save the training data'''
        with open(filepath, 'wb') as f:
            pickle.dump((self.x_train, self.y_train), f)

    def load_training_data(self, filepath):
        '''Function to load the training data'''
        with open(filepath, 'rb') as f:
            self.x_train, self.y_train = pickle.load(f)

    def save_model(self, path):
        '''Function to save the model'''
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.model.likelihood.state_dict(),
        }, path)
        print(f"[INFO] Model and likelihood saved to {path}.")

    def load_model(self, path, ignore_dkl = True):
        '''Function to load the model'''
        self.initialize_model()
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=map_location)

        model_state = checkpoint['model_state_dict']
        likelihood_state = checkpoint['likelihood_state_dict']

        # Ignore Deep Kernel parameters
        if ignore_dkl:
            keys_to_delete = [k for k in model_state.keys() if k.startswith("feature_extractor")]
            for k in keys_to_delete:
                del model_state[k]
            print(f"[INFO] Ignored {len(keys_to_delete)} feature extractor parameters (DKL warm start disabled).")

        self.model.load_state_dict(model_state, strict=False)
        self.model.likelihood.load_state_dict(likelihood_state)
        print("[INFO] Model and likelihood loaded.")

    def save_lengthscale(self, path):
        '''Function to log lengthscales'''
        ls = (
            self.model.covar_module.base_kernel.lengthscale
            .detach().cpu().numpy().ravel()
        )

        row = pd.DataFrame([{
            "iteration": int(getattr(self, "iteration", 0)),
            **{f"ls_dim{j}": float(v) for j, v in enumerate(ls)}
        }])

        if os.path.exists(path):
            header = open(path, "r", encoding="utf-8").readline().strip().split(",")
            for c in row.columns:
                if c not in header:
                    header.append(c)
            row = row.reindex(columns=header)
            row.to_csv(path, mode="a", header=False, index=False)
        else:
            row.to_csv(path, mode="w", header=True, index=False)

        print(f"[INFO] Lengthscale saved to {path}.")
