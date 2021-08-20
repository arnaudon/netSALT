import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

import netsalt

if len(sys.argv) > 1:
    graph_tpe = sys.argv[-1]
else:
    print("give me a type of graph please!")

params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_tpe]

os.chdir(graph_tpe)

graph = netsalt.load_graph()
netsalt.update_parameters(graph, params)


def plot_ordered_sub_T_matrix(modes_df, sort_by):
    """Plot subset of ordered matrix ordered by values in column sort_by, only for < inf values."""
    modes_df = modes_df.sort_values(by=sort_by, axis=0)[modes_df[sort_by] < np.inf]
    mode_competition_matrix = netsalt.compute_mode_competition_matrix(graph, modes_df,
                                                                      with_gamma=False)
    df = pd.DataFrame(data=mode_competition_matrix, index=modes_df.index, columns=modes_df.index)

    plt.figure()
    sns.heatmap(df)


modes_df = netsalt.load_modes()
sort_by = 'interacting_lasing_thresholds'
plot_ordered_sub_T_matrix(modes_df, sort_by)
plt.show()
