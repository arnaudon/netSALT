import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    n_modes = 200

    mat = pickle.load(open("buffon_control_optimized/out/single_mode_matrix.pkl", "rb"))

    #_mat = mat["spectra_matrix"]
    _mat = mat[:n_modes, :n_modes]
    df = pd.DataFrame(_mat)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        df,
        ax=plt.gca(),
        cmap="Blues",
        square=True,
        cbar_kws={"label": "modal amplitude", "shrink": 0.7},
        vmin=0,
    )
    plt.axis("equal")
    plt.savefig("control_matrix.pdf")

    mat = pickle.load(open("buffon_control_threshold/out/single_mode_matrix.pkl", "rb"))

    _mat = mat["spectra_matrix"]
    _mat = _mat[:n_modes, :n_modes]
    df = pd.DataFrame(_mat)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        df,
        ax=plt.gca(),
        cmap="Blues",
        square=True,
        cbar_kws={"label": "modal amplitude", "shrink": 0.7},
        vmin=0,
    )
    plt.axis("equal")
    plt.savefig("control_matrix_threshold.pdf")
