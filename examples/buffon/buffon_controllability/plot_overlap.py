import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
from netsalt.io import load_graph
from netsalt.modes import (
    compute_overlapping_factor,
    from_complex,
    compute_mode_competition_matrix,
    mode_on_nodes,
    mean_mode_on_edges,
    compute_mode_IPR,
)
from tqdm import tqdm

if __name__ == "__main__":
    df_uni = pd.read_hdf("../buffon_uniform/out/modal_intensities.h5")
    graph = load_graph("../buffon_uniform/out/quantum_graph.gpickle")
    plt.figure(figsize=(4, 3))
    counter = 0
    df = pd.DataFrame()
    for i in tqdm(range(50)):

        pump = yaml.safe_load(open(f"buffon_control_optimized/out/pump_profile_{i}.yaml"))
        df_opt = pd.read_hdf(f"buffon_control_optimized/out/modal_intensities_{i}.h5")
        graph.graph["params"]["pump"] = np.array(pump)

        for gid in df_uni.index:
            thresh = np.real(df_opt.loc[gid, "lasing_thresholds"].to_list()[0])
            if thresh < 0.1:
                diff = thresh - np.real(df_uni.loc[gid, "lasing_thresholds"].to_list()[0])
                df.loc[counter, "threshold"] = diff
                over = abs(np.real(
                    compute_overlapping_factor(df_uni.loc[gid, "passive"].to_list()[0], graph)
                ))
                df.loc[counter, "over"] = np.clip(over, 0, 0.3)
                counter += 1
    plt.scatter(df["threshold"], df["over"], s=2, c='k')
    #import seaborn as sns

    #sns.jointplot(data=df, x="threshold", y="over", ax=plt.gca(), kind="reg",  marker=".", color='k')
    plt.xlabel("lasing threshold increase wrt uniform")
    plt.ylabel("pump overlap")
    plt.tight_layout()
    plt.savefig("threshold_overlap.pdf")
