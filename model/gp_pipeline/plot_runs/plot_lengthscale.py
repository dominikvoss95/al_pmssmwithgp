from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt

USER = os.environ.get("USER")

target   = "DMRD" #"CrossSection"
n_dim    = 19
modes    = ("AL", "random_")
runs     = range(5, 9)
base_dir = Path(f"/u/{USER}/al_pmssmwithgp/model/lengthscales")

features = [
    "M_1", "M_2", "tan_beta", "mu", "M_3", "At", "Ab", "Atau",
    "mA", "mqL3", "mtR", "mbR", "meL", "mtauL", "meR", "mtauR",
    "mqL1", "muR", "mdR"
]

ls_cols_all = [f"ls_dim{i}" for i in range(n_dim)]
rename_map  = {f"ls_dim{i}": features[i] for i in range(n_dim)}

def read_last_iteration_row(csv_path):
    '''Function to read out lengthscales from last iteration row '''
    df = pd.read_csv(csv_path)
    last_iter = int(df["iteration"].max())
    row = df.loc[df["iteration"] == last_iter].iloc[0]
    ls = [c for c in ls_cols_all if c in row.index]
    lengthscales = row[ls].rename(rename_map)
    lengthscales = lengthscales.reindex(features)
    return lengthscales

rows = []

for al in modes:
    print(al)
    for run in runs:
        path = os.path.join(base_dir, f"lengthscales_{target}_{n_dim}D_{al}_run{run}.csv")
        if os.path.exists(path):
            lengthscales = read_last_iteration_row(path)
            lengthscales.name = f"{al}run{run}"
            rows.append(lengthscales)
        else:
            print(f"[WARN] File not found and skipped: {path}")


lengthscales_df = pd.DataFrame(rows)

folder_path = f"/u/{USER}/al_pmssmwithgp/model/lengthscales/plots/{n_dim}D/"
os.makedirs(folder_path, exist_ok=True)
save_path = os.path.join(folder_path, f"lengthscales_{target}.png") 

plt.figure(figsize=(14, 6))
plt.boxplot(
    [lengthscales_df[col] for col in features],
    labels=features,
    showmeans=True
)
plt.xticks(rotation=60, ha="right")
plt.ylabel("Lengthscale")
plt.title(f"Lengthscale Distribution {target} {n_dim}D")
plt.tight_layout()
plt.savefig(save_path)
print(f"Plot saved to: {save_path}")
plt.show()
