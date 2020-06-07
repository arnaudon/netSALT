"""input/output functions"""
import pickle

import h5py
import numpy as np
import pandas as pd


def save_graph(graph, filename="graph.pkl"):
    """Save a the quantum graph."""
    with open(filename, "wb") as pickle_file:
        pickle.dump(graph, pickle_file)


def load_graph(filename="graph.pkl"):
    """Load a the quantum graph."""
    with open(filename, "rb") as pickle_file:
        return pickle.load(pickle_file)


def save_modes(modes_df, filename="results"):
    """Save modes dataframe into h5."""
    modes_df.to_hdf(filename + ".h5", key="modes")
    modes_df.to_csv(filename + ".csv")


def load_modes(filename="results.h5"):
    """Return modes dataframe from hdf5."""
    return pd.read_hdf(filename, "modes")


def save_qualities(qualities, filename="results.h5"):
    """Save qualities in the results hdf5 file."""
    with h5py.File(filename, "a") as all_results:
        if "scan_qualities" in all_results:
            del all_results["scan_qualities"]
        dset = all_results.create_dataset("scan_qualities", data=qualities)


def load_qualities(filename="results.h5"):
    """Load qualities from the results hdf5 file."""
    with h5py.File(filename, "r") as all_results:
        return all_results["scan_qualities"][:]


def save_mode_competition_matrix(mode_competition_matrix, filename="results.h5"):
    """Save mode competitian matrix in the results hdf5 file."""
    with h5py.File(filename, "a") as all_results:
        if "mode_competition_matrix" in all_results:
            del all_results["mode_competition_matrix"]
        dset = all_results.create_dataset(
            "mode_competition_matrix", data=mode_competition_matrix
        )


def load_mode_competition_matrix(filename="results.h5"):
    """Load mode competitiaon matrixfrom the results hdf5 file."""
    with h5py.File(filename, "r") as all_results:
        return all_results["mode_competition_matrix"][:]
