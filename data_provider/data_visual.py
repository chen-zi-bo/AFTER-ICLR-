import argparse

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

from data_provider.data_loader import Dataset_Agriculture


def plot_multivariate_series(dataset, use_scaled=True, max_dims=5, save_path=None):
    data = dataset.data_raw  # (T, D)

    # ===== 标准化 =====
    # if use_scaled:
    #     scaler = StandardScaler()
    #     data = scaler.fit_transform(data)

    T, D = data.shape
    D_plot = min(D, max_dims)

    plt.figure(figsize=(12, 6))

    for i in range(D_plot):
        plt.plot(data[:, i], label=f'feature_{i}')

    plt.title("Agriculture")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # ===== 保存图像 =====
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

parser = argparse.ArgumentParser(description='MultiModal')
parser.add_argument('--text_len', type=int, default=4)
args = parser.parse_args()
dataset = Dataset_Agriculture(args, flag='train')

plot_multivariate_series(dataset, save_path="figs/multivariate_series.png")
