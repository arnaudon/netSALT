from netsalt.pump import make_threshold_pump
import yaml
import numpy as np
import matplotlib.pyplot as plt
from netsalt.plotting import plot_pump_profile, plot_single_mode

import pandas as pd
from netsalt.io import load_graph

if __name__ == "__main__":
    mode_id = 1
    threshold =  0.007
    graph = load_graph("../buffon_uniform_square/out/quantum_graph.gpickle")
    graph.graph["params"]["quality_threshold"] = 1e-3
    modes_df = pd.read_hdf("../buffon_uniform_square/out/passive_modes.h5")
    mode = modes_df.loc[mode_id, "passive"].to_list()[0]
    pump = np.array(make_threshold_pump(graph, mode, threshold))
    pump = 1 - pump
    yaml.dump(pump.tolist(), open('pump.yaml', 'w'))

    plt.figure()
    ax = plt.gca()
    plot_pump_profile(graph, pump, ax=ax)
    plot_single_mode(graph, modes_df, mode_id, ax=ax)
    plt.savefig("pump_profile.pdf")
    print(pump)
