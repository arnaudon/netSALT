"""Import main functions."""

from .dispersion_relations import set_dielectric_constant, set_dispersion_relation
from .graph_construction import (
    create_naq_graph,
    oversample_graph,
    set_total_length,
    update_parameters,
)
from .io import load_graph, load_modes, save_graph, save_modes
from .modes import (
    compute_modal_intensities,
    compute_mode_competition_matrix,
    mean_mode_on_edges,
    mode_on_nodes,
)
from .naq_graphs import *

from .utils import lorentzian
