"""input/output functions"""
import pickle
from pathlib import Path

import pandas as pd


def save_graph(graph, filename="graph.pkl"):
    """Save a the quantum graph."""
    with open(filename, "wb") as pickle_file:
        pickle.dump(graph, pickle_file)


def load_graph(filename="graph.pkl"):
    """Load a the quantum graph."""
    with open(filename, "rb") as pickle_file:
        return pickle.load(pickle_file)


def save_modes(modes_df, filename="results.h5"):
    """Save modes dataframe into hdf5."""
    modes_df.to_hdf(filename, key="modes")
    modes_df.to_csv(Path(filename).with_suffix(".csv"))


def load_modes(filename="results.h5"):
    """Return modes dataframe from hdf5."""
    return pd.read_hdf(filename, "modes")


def save_qualities(qualities, filename="results.h5"):
    """Save qualities in the results hdf5 file."""
    pd.DataFrame(data=qualities, index=None, columns=None).to_hdf(filename, key="qualities")


def load_qualities(filename="results.h5"):
    """Load qualities from the results hdf5 file."""
    return pd.read_hdf(filename, "qualities").to_numpy()
