"""input/output functions"""
import pickle
import pandas as pd
import numpy as np


def save_graph(graph, filename="graph.pkl"):
    """Save a the naq graph."""
    # pickle.dump([graph, params], open(filename, "wb"))
    pickle.dump(graph, open(filename, "wb"))


def load_graph(filename="graph.pkl"):
    """Load a the naq graph."""
    return pickle.load(open(filename, "rb"))


def save_modes_old(modes, lasing_thresholds=None, filename="passive_modes"):
    """save modes in a pickle"""
    if lasing_thresholds is None:
        pickle.dump(modes, open(filename + ".pkl", "wb"))
    else:
        pickle.dump(
            [np.array(modes), np.array(lasing_thresholds)],
            open(filename + ".pkl", "wb"),
        )


def save_modes(modes_df, filename="modes_results"):
    """Save modes dataframe into h5."""
    modes_df.to_hdf(filename + ".h5", key="modes")
    modes_df.to_csv(filename + ".csv")


def load_modes(filename="modes_results.h5"):
    """Return modes dataframe from hdf5"""
    return pd.read_hdf(filename, "modes")
