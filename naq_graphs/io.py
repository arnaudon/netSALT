"""input/output functions"""
import pickle
import numpy as np


def save_graph(graph, params, filename="graph.pkl"):
    """save a the naq graph"""
    pickle.dump([graph, params], open(filename, "wb"))


def load_graph(filename="graph.pkl"):
    """load a the naq graph"""
    return pickle.load(open(filename, "rb"))


def save_modes(modes, lasing_thresholds=None, filename="passive_modes"):
    """save modes in a pickle"""
    if lasing_thresholds is None:
        pickle.dump(modes, open(filename + ".pkl", "wb"))
    else:
        pickle.dump(
            [np.array(modes), np.array(lasing_thresholds)],
            open(filename + ".pkl", "wb"),
        )


def load_modes(filename="passive_modes"):
    """return modes in a pickle"""
    return pickle.load(open(filename + ".pkl", "rb"))
