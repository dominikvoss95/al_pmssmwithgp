import os 
from itertools import product

from helper import load_dataframes
from metrics import plot_results, compute_mean_std

USER = os.environ.get("USER")

# Adjust these paramters accordingly
n_dim = 19
options = [50,150,500,1000] #["RBF", "Matern", "RQK", "SpectralMixture", "RBF+Matern", "Additive"] #[0.1, 0.01, 0.001] #[True,False] #[0.1, 1.0, 5.0] #[True,False] #[0, 1.5, 3.0] #[2,4,5,6,8] #[True,False] # [2,4,5,6,8] #[True,False] #[1,15,40,100] #[4,8,12]  #[f'{n_dim}'] 
param = None #'initial_train_points' #kernel' #'noise' #'warm_starting' #'lengthscale'#'tolerance_sampling' #warm_starting'  #'num_hidden_dims' #'warm_starting' #None #'n_new_points' #'feature_dim'  #'n_dim' or None, if you dont want any parameter
n_new_points = 200
runs = 4
start = 300
target = 'DMRD' #'CrossSection'  #'DMRD' 
path = f'/u/{USER}/al_pmssmwithgp/model/goodness_of_fit/{n_dim}D'

random_mode = "random" #"LH"
learning_modes = [random_mode, "AL"]
model_types = ["standard", "sparse", "deep", "mlp"]
model_types = ["standard", "deep", "mlp"]

model_label  = {"standard": "Std GP", "sparse": "Sparse GP", "deep": "Deep GP", "mlp": "MLP"}
model_suffix = {"standard": "", "sparse": "_sparse_",  "deep": "_deep_",  "mlp": "_mlp_"}

method_paths = {}
x_multipliers = {}

for value, mode, model in product(options, learning_modes, model_types):
    is_al = "_AL_" if mode == "AL" else ""
    is_lh = f"_{random_mode}__" if mode == random_mode else ""
    suffix = model_suffix[model]

    if param == "n_new_points":
        n = value if is_al else value#*10
    else:
        n = n_new_points if is_al else n_new_points#*10

    folder_parts = [f'{target}_{n_dim}D']
    label_parts  = [mode, model_label[model]]

    if param:
        folder_parts.append(f'{param}_{value}')
        label_parts.append(f'{param}={value}')

    label = ' '.join(label_parts) + f' (n={n})'
    folder = '_'.join(folder_parts)

    method_paths[label] = [

        f"{path}/{folder}/gof_{target}_{n_dim}D{suffix}{is_al}{is_lh}run{i}.csv"
        for i in range(1, runs + 1)
    ]

    x_multipliers[label] = n

folder_path= f'/u/{USER}/al_pmssmwithgp/model/goodness_of_fit/plots/{n_dim}D/'
os.makedirs(folder_path, exist_ok=True)

metrics_to_plot = ["accuracy","weighted_acc_alpha_1.0","weighted_acc_alpha_2.0","weighted_acc_alpha_5.0","weighted_acc_alpha_10.0",
    "mse", "rmse", "r2", "chi2", "chi2_red", "mean_abs_pull", "rms_pull",
    "mse_w", "rmse_w", "r2_w", "chi2_w", "chi2_red_w", "mean_abs_pull_w", "rms_pull_w",
]
metrics_to_plot = ["accuracy"]

if target == "Toy":
    title = f"Toy Problem: Two Gaussian Hyperspheres ({n_dim}D)"
    name = f"two_circles{n_dim}D_np:{n}_{param}"
elif target == "DMRD":
    title = f"Dark Matter Relic Density ({n_dim}D)"
    name  = f"DMRD{n_dim}D_np:{n}_{param}"
elif target == "CrossSection":
    title = f"CrossSection ({n_dim}D)"
    name  = f"CrossSection{n_dim}D_np:{n}_{param}"

dfs = load_dataframes(method_paths)

for metric in metrics_to_plot:
    res_metric = compute_mean_std(dfs, metric)
    save_path   = os.path.join(folder_path, f"{name}__{metric}.png")
    plot_results(
        res_metric,
        metric=metric,
        x_multipliers=x_multipliers,
        title=title,
        save_path=save_path,
        start=start,
    )


