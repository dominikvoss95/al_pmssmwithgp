import gpytorch
import torch
import numpy as np
import copy

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, RQKernel, SpectralMixtureKernel
from gpytorch.mlls import VariationalELBO
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans


class SparseGP(gpytorch.models.ApproximateGP):

    def __init__(self, x_train, y_train, x_valid, y_valid, n_dim, lengthscale, noise, num_inducing_max=200, 
                 seed=42, thr=0.05, kernel="RBF", m_nu=1.5, inducing_strategy="kmeans"):

        self.seed = seed
        self.set_seed(seed)

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.thr = thr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === INDUCING POINTS ===
        num_inducing = min(int(0.5 * len(x_train)), num_inducing_max)
        print(f"[INFO] Number of Inducing Points: {num_inducing}")
        print(f"[INFO] X_train shape: {x_train.shape}")

        # Inducing Points first part of the training data
        if inducing_strategy == 'vanilla':
            inducing_points = x_train[:num_inducing].clone().to(self.device)

        # KMeans clustering for inducing points
        elif inducing_strategy == 'kmeans':
            kmeans = KMeans(n_clusters=num_inducing, random_state=self.seed).fit(x_train.cpu().numpy())
            inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

        variational_distribution = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        # === KERNELS ===
        if kernel == "RBF":
            base_kernel = RBFKernel(ard_num_dims=n_dim)
        elif kernel == "Matern":
            base_kernel = MaternKernel(nu=m_nu, ard_num_dims=n_dim)
        elif kernel == "RQK":
            base_kernel = RQKernel(ard_num_dims=n_dim)
        # elif kernel == "SpectralMixture":
        #     base_kernel = SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=n_dim)
        elif kernel == "RBF+Matern":
            base_kernel = RBFKernel(ard_num_dims=n_dim) + \
                            MaternKernel(nu=m_nu, ard_num_dims=n_dim)
        
        self.covar_module = ScaleKernel(base_kernel)

        # === NOISE ===
        self.likelihood = GaussianLikelihood()
        self.likelihood.noise = noise

        # === LENGTHSCALE ===
        kernel = self.covar_module.base_kernel
        if hasattr(kernel, "kernels"):
                for k in kernel.kernels:
                    k.lengthscale = lengthscale
        else:
            kernel.lengthscale = lengthscale
        
        self.to(self.device)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def do_train_loop(self, lr=0.005, iters=1000, batch_size=512, jitter=1e-4):        
        '''Train the SparseGP Model'''
        train_dataset = TensorDataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        mll = VariationalELBO(self.likelihood, self, num_data=self.x_train.shape[0])

        best_loss = 1e10
        best_model = None
        losses_train = []
        losses_valid = []
        
        with gpytorch.settings.cholesky_jitter(jitter):
            for i in range(iters):
                # Training
                self.train()
                self.likelihood.train()
                epoch_loss = 0.0

                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    optimizer.zero_grad()
                    output = self(x_batch)
                    loss = -mll(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                epoch_avg = epoch_loss / len(train_loader)
                losses_train.append(epoch_avg)

                # Validation
                self.eval()
                self.likelihood.eval()
                with torch.no_grad():
                    output_valid = self(self.x_valid.to(self.device))
                    loss_valid = -mll(output_valid, self.y_valid.to(self.device).view(-1))
                    losses_valid.append(loss_valid.item())

                # Save best model
                    if loss_valid.item() < best_loss:
                        best_loss  = loss_valid.item()
                        best_model = copy.deepcopy(self.state_dict()) 

                if i % 100 == 0:
                    print(f"Iter {i} / {iters} - Loss (Train): {epoch_avg:.3f} - Loss (Val): {loss_valid.item():.3f}")

        if best_model is not None:
            self.load_state_dict(best_model)

        return self, losses_train, losses_valid
