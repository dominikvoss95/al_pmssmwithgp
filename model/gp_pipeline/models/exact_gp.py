import gpytorch
import torch
import numpy as np
import copy
import torch.nn as nn
from gpytorch.kernels import RBFKernel, MaternKernel, RQKernel, ScaleKernel, SpectralMixtureKernel
from gpytorch.priors import GammaPrior
from gp_pipeline.models.linearMean import LinearMean  

class FeatureExtractor(nn.Sequential):
    def __init__(self, input_dim, output_dim=8):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, output_dim))

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, x_valid, y_valid, n_dim, lengthscale, use_ard,
                 noise, kernel, m_nu, num_mixtures, use_dkl=False, feature_dim=8, thr=0, epsilon=0.2, seed=42):

        self.set_seed(seed)
  
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid

        self.thr = thr
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(x_train, y_train, likelihood)
        self.likelihood = likelihood

        # === FEATURE NET FOR DEEP KERNEL LEARNING ===
        self.use_dkl = use_dkl
        if use_dkl:
            self.feature_extractor = FeatureExtractor(input_dim=n_dim, output_dim=feature_dim)
            effective_dim = feature_dim
        else:
            effective_dim = n_dim

        self.mean_module = LinearMean(input_size=effective_dim)

        # === CHOOSE KERNEL ===
        if kernel == "RBF":
            base_kernel = RBFKernel(ard_num_dims=effective_dim if use_ard else None)
        elif kernel == "Matern":
            base_kernel = MaternKernel(nu=m_nu, ard_num_dims=effective_dim if use_ard else None)
        elif kernel == "RQK":
            base_kernel = RQKernel(ard_num_dims=effective_dim if use_ard else None)
        elif kernel == "SpectralMixture":
            base_kernel = SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=effective_dim if use_ard else None)
        elif kernel == "RBF+Matern":
            base_kernel = RBFKernel(ard_num_dims=effective_dim if use_ard else None) + \
                            MaternKernel(nu=m_nu, ard_num_dims=effective_dim if use_ard else None)
        elif kernel == "Additive":
            base_1d = gpytorch.kernels.RBFKernel(ard_num_dims=None)
            base_kernel = gpytorch.kernels.AdditiveStructureKernel(base_1d, num_dims=effective_dim)

        self.covar_module = ScaleKernel(base_kernel)

        # === LENGTHSCALES ===
        def set_lengthscale(kernel, lengthscale):
            # Skip SpectralMixtureKernel
            if isinstance(kernel, gpytorch.kernels.SpectralMixtureKernel):
                return
            
            # If kernel is additive, delegate to its internal base kernel
            if isinstance(kernel, gpytorch.kernels.AdditiveStructureKernel):
                set_lengthscale(kernel.base_kernel, lengthscale)
                return

            # If kernel is RBF + Matern, set lengthscale for subkernels
            if hasattr(kernel, "kernels"):
                for subkernel in kernel.kernels:
                    set_lengthscale(subkernel, lengthscale)
                return 

            # If the kernel directly has a lengthscale
            if hasattr(kernel, "lengthscale"):
                kernel.initialize(lengthscale=lengthscale)
            
        set_lengthscale(self.covar_module.base_kernel, lengthscale)
        
        if hasattr(self.covar_module.base_kernel, "lengthscale") and self.covar_module.base_kernel.lengthscale is not None:
            print("[INFO] Lengthscales per dimension:", self.covar_module.base_kernel.lengthscale.detach().cpu().numpy())
        else:
            print(f"[INFO] Kernel {type(self.covar_module.base_kernel).__name__} has no lengthscale.")

        # === NOISE ===
        self.likelihood.noise = noise
        try:
            print("[INFO] Noise level:", self.likelihood.noise.item())
        except:
            print("[INFO] Noise level (via noise_covar):", self.likelihood.noise_covar.noise.item())

        self.to(self.device)


    def set_seed(self, seed):
        '''Set the seed for reproducibility'''
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, x):
        if self.use_dkl:
            x = self.feature_extractor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def do_train_loop(self, lr=0.005, iters=1000, optimizer="Adam", jitter=1e-5):
        '''Train the ExactGP Model'''

        if optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0)
        else:
            if self.use_dkl:
                optimizer = torch.optim.Adam([
                    {'params': self.feature_extractor.parameters(), 'lr': 1e-3, 'weight_decay': 1e-3},
                    {'params': self.covar_module.parameters(), 'lr': 5e-3},
                    {'params': self.mean_module.parameters(), 'lr': 5e-3},
                ])
            else:
                optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        best_loss = 1e10
        best_model = None
        losses_train = []
        losses_valid = []

        with gpytorch.settings.cg_tolerance(1e-2), \
             gpytorch.settings.max_cg_iterations(100), \
             gpytorch.settings.cholesky_jitter(jitter):

            for i in range(iters):
                # Training
                self.train()
                self.likelihood.train()
                optimizer.zero_grad()
                output = self(self.x_train)

                loss = -mll(output, self.y_train.view(-1))
                #
                # if self.epsilon == 0:
                #     weights = 1
                # else:
                #     weights = torch.exp(-((self.y_train.view(-1) - self.thr) ** 2) / (2 * self.epsilon ** 2))
                # loss = (-mll(output, self.y_train.view(-1)) * weights).mean()

                loss.backward()
                optimizer.step()

                losses_train.append(loss.item())

                # Validation
                self.eval()
                self.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    output_valid = self(self.x_valid)
                    loss_valid = -mll(output_valid, self.y_valid.view(-1))
                    losses_valid.append(loss_valid.item())
                
                if loss < best_loss:
                    best_loss = loss.item()
                    best_model = self.state_dict()

                if i % 100 == 0:
                    print(f"[INFO] Iter {i + 1}/{iters} - Loss (Train): {loss.item():.3f} - Loss (Val): {loss_valid.item():.3f}")
                    if self.covar_module.base_kernel.lengthscale is not None:
                        print(f"[INFO] Lengthscales: {self.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}")

        if best_model is not None:
            self.load_state_dict(best_model)

        return self, losses_train, losses_valid
