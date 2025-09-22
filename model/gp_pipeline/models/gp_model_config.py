from dataclasses import dataclass

@dataclass
class GPModelConfig:
    '''Class to initialize all config parameters'''
    output_dir: str = None
    start_root_file_path: str = None
    root_file_path: str = None
    initial_train_points: int = 10
    valid_points: int = 20
    additional_points_per_iter: int = 1
    n_dim: int = 4
    latent_dim: int = 2
    num_layers: int = 3
    num_hidden_dims: int = 10
    num_middle_dims: int = 5
    num_samples: int = 8
    num_inducing_max: int = 256
    batch_size: int = 256
    learning_rate: float = 0.01
    iterations: int = 1000
    lengthscale: float = 0.1
    noise: float = 0.1
    jitter: float = 1e-5
    use_ard: bool = True
    kernel: str = "RBF"
    m_nu: int = 2
    num_mixtures: int = 4
    use_dkl: bool = False
    feature_dim: int = 4
    epsilon: int = 0.2
    tolerance_sampling: int = 1.0 
    proximity_sampling: int = 0.1 
    beta: int = 50
    blur: int = 0.15

