import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from itertools import cycle

from helper import normalize_label

def compute_mean_std(dfs, metric):
    '''Function to create average over the dataframes and compute std'''
    results = {}
    for method, frames in dfs.items():
        series_list = []
        for df in frames:
            series_list.append(df[metric])
        metrics = pd.concat(series_list, axis=1, join='outer') 
        x = np.arange(len(metrics), dtype=float)

        # Scale to percentage
        if metric == 'accuracy':
            metrics = metrics * 100.0
        results[method] = {
            'mean': metrics.mean(axis=1),
            'std':  metrics.std(axis=1),
            'x': x
        }
    return results

def _metric_style(metric):
    '''Function to provide y_label and plot_title'''

    metric_clean = metric.replace("_w", "")  # for the same style for weighted variants

    y_label = metric_clean
    # Prettier labels
    pretty = {
        "accuracy": "Accuracy",
        "weighted_acc_alpha_1.0": "Accuracy (weighted)",
        "weighted_acc_alpha_2.0": "Accuracy (weighted)",
        "weighted_acc_alpha_5.0": "Accuracy (weighted)",
        "weighted_acc_alpha_10.0": "Accuracy (weighted)",
        "mse": "MSE",
        "rmse": "RMSE",
        "r2": "R²",
        "chi2": "χ²",
        "chi2_red": "reduced χ²",
        "mean_abs_pull": "⟨|Pull|⟩",
        "rms_pull": "RMS(Pull)",
    }
    y_label = pretty.get(metric_clean.lower(), metric_clean)
    if metric.endswith("_w"):
        y_label += " (weighted)"

    return y_label


def plot_results(results, metric, x_multipliers,  title, save_path, start = 0):
    '''Function to plot mean of gof metric with standard deviation'''
    plt.figure(figsize=(10, 6))

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    base_color_map = {}
    color_cycle = iter(colors)

    for method in results:
        base_key = normalize_label(method)
        if base_key not in base_color_map:
            base_color_map[base_key] = next(color_cycle)

    print(f"Plotting {len(results)} methods with color mapping to base methods.")

    for method, stats in results.items():
        base_key = normalize_label(method)
        color = base_color_map[base_key]

        x = (stats['x'] * (x_multipliers.get(method, 1))) + start
        y = stats['mean'].to_numpy()
        s = stats['std'].to_numpy()

        linestyle = '--' if "AL" in method else '-'
        plt.plot(x, y, label=method, linestyle=linestyle, color=color, linewidth=2)
        plt.fill_between(x, y - s, y + s, color=color, alpha=0.2)

        # Set dotted borders for envelope of dotted mean
        if "AL" in method:
            plt.plot(x, y + s, color=color, linestyle=':', linewidth=1)
            plt.plot(x, y - s, color=color, linestyle=':', linewidth=1)
        # else:
        #     plt.plot(x, y + s, color=color, linestyle='-', linewidth=1)
        #     plt.plot(x, y - s, color=color, linestyle='-', linewidth=1)

    y_label = _metric_style(metric)
    title = f"{title} - {y_label}"
    plt.title(title)
    plt.xscale('log')
    plt.xlim(start, None)

    xticks = [300, 500, 1000, 2000, 5000]
    #xticks = [1000, 5000, 10000, 20000]
    plt.xticks(xticks, [str(x) for x in xticks])

    plt.xlabel('Number of Labeled Training Points')
    plt.ylabel(y_label)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}'))

    plt.xlim(300, 5000)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()

