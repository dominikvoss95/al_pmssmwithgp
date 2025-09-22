import torch
import gpytorch
import numpy as np
import copy

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, RQKernel
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

class DeepGPHiddenLayer(gpytorch.models.deep_gps.DeepGPLayer):
    '''A single deep hidden layer with variational inference approximation with inducing points'''
    def __init__(self, input_dims, output_dims, inducing_points, mean_type='constant', kernel='RBF', m_nu=1.5):

        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])

        # Cholesky decomposition of the covariance matrix
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[0],
            batch_shape=batch_shape
        )

        # Determines how inducing points are used in training
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True # allows optimization of inducing points positions
        )

        # Initialize baseclass -> DeepGPLayer from GPyTorch
        super(DeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        # Mean and covariance
        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        # === KERNELS ===
        if kernel == "RBF":
            base_kernel = RBFKernel(ard_num_dims=input_dims, batch_shape=batch_shape)
        elif kernel == "Matern":
            base_kernel = MaternKernel(nu=m_nu, ard_num_dims=input_dims, batch_shape=batch_shape)
        elif kernel == "RQK":
            base_kernel = RQKernel(ard_num_dims=input_dims, batch_shape=batch_shape)
        # elif kernel == "SpectralMixture":
        #     base_kernel = SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=input_dims, batch_shape=batch_shape)
        elif kernel == "RBF+Matern":
            base_kernel = RBFKernel(ard_num_dims=input_dims, batch_shape=batch_shape) + \
                            MaternKernel(nu=m_nu, ard_num_dims=input_dims, batch_shape=batch_shape)
        
        self.covar_module = ScaleKernel(base_kernel, batch_shape=batch_shape)

    def forward(self, x):
        '''Function returns multivariate distribution over mean and covariance'''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepGP(gpytorch.models.deep_gps.DeepGP):
    def __init__(self, x_train, y_train, x_valid, y_valid, n_dim, lengthscale, noise, num_hidden_dims=10, 
                 num_middle_dims=0, num_inducing_max=256, inducing_strategy='kmeans', thr=0, kernel='RBF', 
                 m_nu=1.5, num_samples=8, seed=42):
        super().__init__()

        self.seed = seed

        self.set_seed(self.seed)

        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid
        
        self.thr = thr

        self.num_middle_dims = num_middle_dims

        self.num_samples = num_samples
        self.num_inducing_max = num_inducing_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === INDUCING POINTS ===

        # Slowly increase inducing points to maximum to adjust to number of training points over the iterations
        num_inducing = min(int(0.5 * len(x_train)), num_inducing_max)
        print(f"[INFO] Number of Inducing Points: {num_inducing}")
        print(f"[INFO] X_train shape: {x_train.shape}")

        # Inducing Points first part of the training data
        if inducing_strategy == 'vanilla':
            inducing_points_hidden = x_train[:num_inducing].clone().to(self.device)

        # KMeans clustering for inducing points
        elif inducing_strategy == 'kmeans':
            kmeans = KMeans(n_clusters=num_inducing, random_state=self.seed).fit(x_train.cpu().numpy())
            inducing_points_hidden = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

        # === 1ST (AND 2ND) LAYERS ===

        # Hidden layer: input_dim -> num_hidden_dims
        self.hidden_layer = DeepGPHiddenLayer(
            input_dims=n_dim,
            output_dims=num_hidden_dims,
            inducing_points=inducing_points_hidden,
            mean_type='linear'
        )

        if num_middle_dims == 0:
            inducing_points_output = torch.randn(num_inducing, num_hidden_dims).to(self.device)
            # Output layer: num_hidden_dims -> 1
            self.output_layer = DeepGPHiddenLayer(
                input_dims=num_hidden_dims,
                output_dims=None,
                inducing_points=inducing_points_output,
                mean_type='constant', 
                kernel=kernel,
                m_nu=m_nu
            )
        
        else:
            # Middle layer: num_hidden_dims -> num_middle_dims
            inducing_points_middle = torch.randn(num_inducing, num_hidden_dims).to(self.device)
            self.middle_layer = DeepGPHiddenLayer(
                input_dims=num_hidden_dims,  
                output_dims=self.num_middle_dims,  
                inducing_points=inducing_points_middle,
                mean_type='linear',
                kernel=kernel,
                m_nu=m_nu
            )

            # Output Layer: num_middle_dims  -> 1
            inducing_points_output = torch.randn(num_inducing, self.num_middle_dims).to(self.device)
            self.output_layer = DeepGPHiddenLayer(
                input_dims=self.num_middle_dims, 
                output_dims=None,
                inducing_points=inducing_points_output,
                mean_type='constant',
                kernel=kernel,
                m_nu=m_nu
            )

        # === NOISE ===
        self.likelihood = GaussianLikelihood()
        self.likelihood.noise = noise

        # === LENGTHSCALES ===
        def set_lengthscales(layer, lengthscale):
            kernel = layer.covar_module.base_kernel
            if hasattr(kernel, "kernels"):
                for k in kernel.kernels: k.initialize(lengthscale=lengthscale)
            else:
                kernel.initialize(lengthscale=lengthscale)      

        set_lengthscales(self.hidden_layer, lengthscale)

        if self.num_middle_dims > 0:
            set_lengthscales(self.middle_layer, lengthscale)
        set_lengthscales(self.output_layer, lengthscale)

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
        
    def forward(self, inputs):
        if self.num_middle_dims == 0:
            hidden_rep = self.hidden_layer(inputs)
            output = self.output_layer(hidden_rep)
        else:
            hidden_rep = self.hidden_layer(inputs)
            middle_rep = self.middle_layer(hidden_rep)
            output = self.output_layer(middle_rep)
        return output

    def do_train_loop(self, lr=0.005, iters=1000, batch_size=512, jitter=1e-4):
        '''Train the DeepGP Model'''
        train_dataset = TensorDataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, self.x_train.shape[-2]))
        
        best_loss = 1e10
        best_model = None
        losses_train = []
        losses_valid = []

        with gpytorch.settings.cholesky_jitter(jitter), gpytorch.settings.fast_pred_var(False):
            
            for i in range(iters):
                # Training
                self.train()
                self.likelihood.train()
                epoch_loss = 0.0

                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    with gpytorch.settings.num_likelihood_samples(self.num_samples):
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
                with torch.no_grad(), gpytorch.settings.num_likelihood_samples(self.num_samples):
                    hidden_rep_valid = self.hidden_layer(self.x_valid.to(self.device))
                    if self.num_middle_dims > 0:
                        middle_rep_valid = self.middle_layer(hidden_rep_valid)
                        output_valid = self.output_layer(middle_rep_valid)
                    else:
                        output_valid = self.output_layer(hidden_rep_valid)
                    loss_valid = -mll(output_valid, self.y_valid.to(self.device))
                losses_valid.append(loss_valid.item())

                #Save best model
                if loss_valid.item() < best_loss:
                    best_loss  = loss_valid.item()
                    best_model = copy.deepcopy(self.state_dict()) 
                
                if i % 100 == 0:
                    print(f"Iter {i}/{iters} - Loss (Train): {epoch_avg:.3f} - Loss (Val): {loss_valid.item():.3f}")

        if best_model is not None:
            self.load_state_dict(best_model)

        return self, losses_train, losses_valid
