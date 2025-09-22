from matplotlib import pyplot as plt
import os
import corner
import torch
import numpy as np

from gp_pipeline.utils.selection import EntropySelectionStrategy
from gp_pipeline.utils.sampling import create_random_samples

def plot_corner(self, save_path=None, title="Corner Plot"):
    """Function for a corner plot of the normalized training data."""
    # Unnormalize data for readability
    data = self._unnormalize(self.x_true).cpu().numpy() 

    labels = [
        "M_1", "M_2", "tanb", "mu", "M_3", "AT", "Ab", "Atau",
        "mA", "meL", "mtauL", "meR", "mtauR", "mqL1", "mqL3",
        "muR", "mtR", "mdR", "mbR"
    ][:self.n_dim]

    fig = corner.corner(data, labels=labels, show_titles=True, title_fmt=".2f")

    if save_path:
        fig.savefig(save_path)
        print(f"[INFO] Corner plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_losses(losses, losses_valid, save_path=None, iteration=None):
    '''Function to plot training and validation loss curves.'''
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label='training loss')
    plt.plot(losses_valid, label='validation loss')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    if iteration is not None:
        plt.title(f"Log Training and Validation Loss - Iteration {iteration}")

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_y_distribution(self, bins=100, save_path=None):
    '''Function to plot the distribution of the target data.'''
    y_np = self.y_true.detach().cpu().numpy()
    plt.figure(figsize=(7, 4))
    plt.hist(y_np, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel('y value')
    plt.ylabel('Count')
    plt.title('Distribution of y-values')
    #plt.yscale('log') 
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] y_distribution saved under: {save_path}")
    else:
        plt.show()

def plotGPTrueDifference(self, slice_dim_x1=0, slice_dim_x2=1, new_x=None, save_path=None, iteration=None):
    '''Function to plot 3 subplots: GP prediction, True function and difference between them'''
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    x = self.x_true
    y = self.y_true

    with torch.no_grad():
        predictions = self.model(x)

    observed_pred = self.model.likelihood(predictions)
    if self.is_deep:
        mean = observed_pred.mean.cpu().numpy().mean(axis=0)
    else:
        mean = observed_pred.mean.cpu().numpy()

    diff = y - torch.tensor(mean).to(self.device)

    # Determine vmin and vmax
    vmin = min(y.min().cpu().numpy(), mean.min())
    vmax = max(y.max().cpu().numpy(), mean.max())

    # Plot True Function
    heatmap, xedges, yedges = np.histogram2d(
        x[:, slice_dim_x1].cpu().numpy(),
        x[:, slice_dim_x2].cpu().numpy(),
        bins=30, weights=y.cpu().numpy()
    )
    heatmap_counts, _, _ = np.histogram2d(
        x[:, slice_dim_x1].cpu().numpy(),
        x[:, slice_dim_x2].cpu().numpy(),
        bins=30
    )
    heatmap = heatmap / heatmap_counts

    im1 = axes[0].imshow(
        heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
        origin='lower', cmap='inferno', aspect='auto', vmin=vmin, vmax=vmax
    )
    axes[0].set_xlabel(self.labels[slice_dim_x1])
    axes[0].set_ylabel(self.labels[slice_dim_x2])
    axes[0].set_title("True Function")

    if self.target == "CLs":
        fig.colorbar(im1, ax=axes[0], label=f'CLs')
    elif self.target == "CrossSection":
        fig.colorbar(im1, ax=axes[0], label=f'log(XSec/0.3)')
    elif self.target == "DMRD":
        fig.colorbar(im1, ax=axes[0], label=f'log(DMRD/0.12)')
    elif self.target == "Toy":
        fig.colorbar(im1, ax=axes[0], label=f'y_true')

    # Plot GP Prediction
    heatmap, xedges, yedges = np.histogram2d(
        x[:, slice_dim_x1].cpu().numpy(),
        x[:, slice_dim_x2].cpu().numpy(),
        bins=30, weights=mean
    )
    heatmap_counts, _, _ = np.histogram2d(
        x[:, slice_dim_x1].cpu().numpy(),
        x[:, slice_dim_x2].cpu().numpy(),
        bins=30
    )
    heatmap = heatmap / heatmap_counts

    im2 = axes[1].imshow(
        heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower', cmap='inferno', aspect='auto', vmin=vmin, vmax=vmax
    )
    axes[1].set_xlabel(self.labels[slice_dim_x1])
    axes[1].set_ylabel(self.labels[slice_dim_x2])
    axes[1].set_title("GP Prediction Mean")
    fig.colorbar(im2, ax=axes[1], label='GP Prediction Mean')

    # Plot Difference between True and GP Prediction
    heatmap, xedges, yedges = np.histogram2d(
        x[:, slice_dim_x1].cpu().numpy(),
        x[:, slice_dim_x2].cpu().numpy(),
        bins=30, weights=diff.cpu().numpy()
    )
    heatmap_counts, _, _ = np.histogram2d(
        x[:, slice_dim_x1].cpu().numpy(),
        x[:, slice_dim_x2].cpu().numpy(),
        bins=30
    )
    heatmap = heatmap / heatmap_counts

    im3 = axes[2].imshow(
        heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower', cmap='inferno', aspect='auto'
    )
    axes[2].set_xlabel(self.labels[slice_dim_x1])
    axes[2].set_ylabel(self.labels[slice_dim_x2])
    axes[2].set_title("Difference (True - GP Prediction)")
    fig.colorbar(im3, ax=axes[2], label='Difference')

    if save_path is not None:
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
    else:
        plt.show()

def plotEntropyHistogram(self, slice_dim_x1=0, slice_dim_x2=1, new_x=None, save_path=None, iteration=None):
    '''Function to plot entropy with a histogram in n dimensions'''
    fig, ax = plt.subplots(figsize=(8, 6))

    x = self.x_true
    with torch.no_grad():
        predictions = self.model(x)                
    
    print("[DEBUG] Likelihood")
    observed_pred = self.likelihood(predictions)
    print("[DEBUG] Mean")
    if self.is_deep:
        mean = observed_pred.mean.cpu().numpy().mean(axis=0)
    else:
        mean = observed_pred.mean.cpu().numpy()
    print("[DEBUG] Covariance")

    cov = observed_pred.covariance_matrix.cpu().detach().numpy()

    if cov.ndim == 3: 
        cov = np.diagonal(cov, axis1=-2, axis2=-1).mean(axis=0)
    else:          
        cov = np.diagonal(cov, axis1=-2, axis2=-1)              

    # Ensure mean and cov are tensors
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(cov, torch.Tensor):
        cov = torch.tensor(cov)

    var = observed_pred.variance.cpu().detach().numpy().mean(axis=0)
    print(f"[DEBUG] Vairance Shape: {var.shape}")

    blur = 0.15
    mean_adjusted = mean - self.thr + blur * torch.sign(mean)
    strategy = EntropySelectionStrategy()
    print("[DEBUG] Entropy")
    entropy = strategy.approximate_batch_entropy(mean=mean_adjusted[:, None], var=torch.diag(cov)[:, None, None], device=self.device)
    print(f"[DEBUG] Entropy Shape: {entropy.shape}")

    # Plot Entropy
    heatmap, xedges, yedges = np.histogram2d(
        x[:, slice_dim_x1].cpu().numpy(),
        x[:, slice_dim_x2].cpu().numpy(),
        bins=30, weights=entropy
    )
    heatmap_counts, _, _ = np.histogram2d(
        x[:, slice_dim_x1].cpu().numpy(),
        x[:, slice_dim_x2].cpu().numpy(),
        bins=30
    )
    heatmap = heatmap / heatmap_counts

    im = ax.imshow(
        heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower', cmap='inferno', aspect='auto'
    )
    ax.set_xlabel(self.labels[slice_dim_x1])
    ax.set_ylabel(self.labels[slice_dim_x2])
    ax.set_title("Entropy")
    fig.colorbar(im, ax=ax, label='Entropy')

    if save_path is not None:
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
    else:
        plt.show()

def plotSlice4D(self, slice_dim_x1=0, slice_dim_x2=1, remaining_dims=[2,3], slice_value=0.5, tolerance=0.1, func="entropy", new_x=None, save_path=None, iteration=None):
    '''Function to plot a 4D slice of the GP prediction.'''
    grid_size = 50
    x1_range = np.linspace(0, 1, grid_size)
    x2_range = np.linspace(0, 1, grid_size)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    x1_grid_flat = x1_grid.flatten()
    x2_grid_flat = x2_grid.flatten()

    slice_values = [0.2, 0.4, 0.6, 0.8]
    fig, axes = plt.subplots(len(slice_values), len(slice_values), figsize=(15, 15))
    
    for i, slice_value_fixed in enumerate(slice_values):
        for j, slice_value_iter in enumerate(slice_values):
            x_test = np.zeros((grid_size * grid_size, self.n_dim))
            x_test[:, slice_dim_x1] = x1_grid_flat
            x_test[:, slice_dim_x2] = x2_grid_flat
            
            # Set one remaining dimension to a fixed slice value
            x_test[:, remaining_dims[0]] = slice_value_fixed
            x_test[:, remaining_dims[1]] = slice_value_iter
            
            x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(x_test)
                observed_pred = self.likelihood(predictions)
                if self.is_deep:
                    mean = observed_pred.mean.cpu().numpy().mean(axis=0)
                else:
                    mean = observed_pred.mean.cpu().numpy()
                cov = observed_pred.covariance_matrix.cpu().detach().numpy()

            if func == "mean":
                mean_grid = mean.reshape(grid_size, grid_size)
                vmin, vmax = np.min(mean), np.max(mean)
                
            elif func == "entropy":
                # Ensure mean and cov are tensors
                if not isinstance(mean, torch.Tensor):
                    mean = torch.tensor(mean)
                if not isinstance(cov, torch.Tensor):
                    cov = torch.tensor(cov)

                blur = 0.15
                mean_adjusted = mean - self.thr + blur * torch.sign(mean)
                strategy = EntropySelectionStrategy()
                entropy = strategy.approximate_batch_entropy(mean=mean_adjusted[:, None], cov=torch.diag(cov)[:, None, None], device=self.device)

                entropy_np = entropy.detach().cpu().numpy()
                vmin, vmax = entropy_np.min(), entropy_np.max()
                entropy_grid = entropy_np.reshape(grid_size, grid_size)

            ax = axes[i, j]
            im = ax.imshow(mean_grid.T if func == "mean" else entropy_grid.T, extent=[0, 1, 0, 1], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax, aspect='auto')

            ax.set_title(f"{self.labels[remaining_dims[1]]}={slice_value_iter:.1f}, {self.labels[remaining_dims[0]]}={slice_value_fixed:.1f}", fontsize=10)
            ax.set_xlabel(self.labels[slice_dim_x1], fontsize=10)
            ax.set_ylabel(self.labels[slice_dim_x2], fontsize=10)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)

            # Contour plot for threshold
            ax.contour(x1_grid, x2_grid, mean_grid if func == "mean" else entropy_grid, levels=[self.thr], colors='white', linewidths=2, linestyles='solid')

            # Plot training points
            x_train = self.x_train[:,slice_dim_x1]
            y_train = self.x_train[:,slice_dim_x2]
            ax.scatter(x_train, y_train, marker='o', s=120, c='red', edgecolor='black')

            # Plot new points
            if new_x is not None:
                new_x = np.array(new_x)

                # Filter points which fit to slice
                mask = (
                    np.isclose(new_x[:, remaining_dims[0]], slice_value_fixed, atol=tolerance) &
                    np.isclose(new_x[:, remaining_dims[1]], slice_value_iter, atol=tolerance)
                )
                if np.any(mask):
                    x_new = new_x[mask, slice_dim_x1]
                    y_new = new_x[mask, slice_dim_x2]
                    ax.scatter(x_new, y_new, marker='o', s=120, c='green', edgecolor='black')

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02]) 
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label='GP Prediction Mean' if func == "mean" else 'Entropy')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  
    
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
    else:
        plt.show()


def plotSlice2D(self, slice_dim_x1=0, slice_dim_x2=1, slice_value=0.5, tolerance=0.1, func="entropy", new_x=None, save_path=None, iteration=None):
    '''Function to plot a 2D slice of the truth function.'''
    # Determine the remaining dimensions to plot
    remaining_dims = [dim for dim in range(self.n_dim) if dim != slice_dim_x1 and dim != slice_dim_x2]

    # Create a grid over the remaining dimensions
    grid_size = 50
    x1_range = np.linspace(0, 1, grid_size)
    x2_range = np.linspace(0, 1, grid_size)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    x1_grid_flat = x1_grid.flatten()
    x2_grid_flat = x2_grid.flatten()

    x_test = np.zeros((grid_size * grid_size, self.n_dim))
    x_test[:, slice_dim_x1] = x1_grid_flat 
    x_test[:, slice_dim_x2] = x2_grid_flat 

    # Set the remaining dimensions to the slice value
    for dim in remaining_dims:
        x_test[:, dim] = slice_value
    x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)

    with torch.no_grad():
        predictions = self.model(x_test)
        observed_pred = self.likelihood(predictions)
        mean = observed_pred.mean.cpu().numpy()
        cov = observed_pred.covariance_matrix.cpu().detach().numpy()
    
    if func == "mean":
        mean_grid = mean.reshape(grid_size, grid_size)
        vmin, vmax = np.min(mean), np.max(mean)

    elif func == "entropy":
        # Ensure mean and cov are tensors
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(cov, torch.Tensor):
            cov = torch.tensor(cov)

        blur = 0.15
        mean_adjusted = mean - self.thr + blur * torch.sign(mean)
        strategy = EntropySelectionStrategy()
        entropy = strategy.approximate_batch_entropy(mean=mean_adjusted[:, None], cov=torch.diag(cov)[:, None, None], device=self.device)

        entropy_np = entropy.detach().cpu().numpy()
        vmin, vmax = entropy_np.min(), entropy_np.max()
        entropy_grid = entropy_np.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_grid.T if func == "mean" else entropy_grid.T,
                extent=[0, 1, 0, 1],
                origin='lower',
                cmap='inferno',
                vmin=vmin, vmax=vmax,
                aspect='auto')
    fig.colorbar(im, ax=ax, label='GP Prediction Mean' if func == "mean" else 'Entropy')

    ax.set_xlabel(self.labels[slice_dim_x1], fontsize=10)
    ax.set_ylabel(self.labels[slice_dim_x2], fontsize=10)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Contour 
    ax.contour(x1_grid, x2_grid,
            mean_grid if func == "mean" else entropy_grid,
            levels=[self.thr], colors='white', linewidths=2, linestyles='solid')
    

    # Plot points
    if new_x is not None:

        # Plot training points
        x_train = self.x_train[:,slice_dim_x1]
        y_train = self.x_train[:,slice_dim_x2]
        ax.scatter(x_train, y_train, marker='o', s=100, edgecolor='black', label="Training points")

        # Plot new points
        new_x = np.array(new_x)

        # Plot only the two slice dimensions
        x_coords = new_x[:, slice_dim_x1]
        y_coords = new_x[:, slice_dim_x2]

        ax.scatter(x_coords, y_coords, marker='o', s=100, c='cyan', edgecolor='black', label='New points')
        ax.legend(loc="upper right", fontsize=8)

    if iteration is not None:
        ax.set_title(f"GP Prediction - Iteration {iteration}")

    # Save the plot or show it
    if save_path is not None:
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
    else:
        plt.show()

