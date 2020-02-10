"""input/output functions"""
import pickle


def save_modes(modes, filename="passive_modes.pkl"):
    """save modes in a pickle"""
    pickle.dump(modes, open(filename, "wb"))


def load_modes(filename="passive_modes.pkl"):
    """return modes in a pickle"""
    return pickle.load(open(filename, "rb"))
