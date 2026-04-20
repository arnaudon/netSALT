"""Public API for netsalt."""

from .algorithm import refine_mode
from .contour import find_modes_contour, find_modes_contour_subdivided
from .io import (
    load_graph,
    load_modes,
    load_qualities,
    save_graph,
    save_modes,
    save_qualities,
)
from .modes import (
    compute_modal_intensities,
    compute_mode_competition_matrix,
    find_modes,
    find_passive_modes,
    find_threshold_lasing_modes,
    lasing_threshold_linear,
    mode_on_nodes,
    pump_trajectories,
    scan_frequencies,
)
from .params import NetSaltParams
from .physics import set_dielectric_constant, set_dispersion_relation
from .quantum_graph import (
    create_quantum_graph,
    oversample_graph,
    set_total_length,
    update_parameters,
)
from .utils import lorentzian

__all__ = [
    "NetSaltParams",
    "compute_modal_intensities",
    "compute_mode_competition_matrix",
    "create_quantum_graph",
    "find_modes",
    "find_modes_contour",
    "find_modes_contour_subdivided",
    "find_passive_modes",
    "find_threshold_lasing_modes",
    "lasing_threshold_linear",
    "load_graph",
    "load_modes",
    "load_qualities",
    "lorentzian",
    "mode_on_nodes",
    "oversample_graph",
    "pump_trajectories",
    "refine_mode",
    "save_graph",
    "save_modes",
    "save_qualities",
    "scan_frequencies",
    "set_dielectric_constant",
    "set_dispersion_relation",
    "set_total_length",
    "update_parameters",
]
