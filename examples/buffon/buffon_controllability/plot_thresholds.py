import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.figure(figsize=(4, 3))
    for i in range(200):
        df = pd.read_hdf(f"buffon_control_optimized/out/modal_intensities_{i}.h5")
        thresh = df["lasing_thresholds"]
        int_thresh = df["interacting_lasing_thresholds"]
        plt.scatter(thresh, int_thresh, s=1, marker=".", c="k")
    plt.axis([0.003, 0.01, 0.003, 0.01])
    plt.xlabel("threshold")
    plt.ylabel("interacting threshold")
    plt.tight_layout()
    plt.savefig("threshold.pdf")
