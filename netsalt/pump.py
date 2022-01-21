"""Pump optimisation module."""
import logging
import multiprocessing
from functools import partial

import numpy as np
from scipy import optimize
from tqdm import tqdm

from netsalt.quantum_graph import get_total_inner_length
from .modes import compute_overlapping_single_edges, mean_mode_on_edges
from .physics import gamma, q_value
from .utils import to_complex

L = logging.getLogger(__name__)


def pump_mapping(pump_thresholds, mean_edge_modes):
    """Map the mode profiles to pump profile with threshold."""
    return (pump_thresholds.dot(mean_edge_modes) > 1.0).astype(int)


def pump_cost(
    pump,
    modes_to_optimise,
    pump_overlapps,
    pump_min_size,
    mode="ratio",
    n_modes=20,
    pump_mapper=None,
):
    """Cost function to minimize."""
    if pump_mapper is not None:
        pump = pump_mapper(pump)

    pump = np.round(pump, 0)
    if pump.sum() < pump_min_size:
        return 1e10
    pump_with_opt_modes = pump_overlapps[modes_to_optimise][:, pump == 1].sum(axis=1)
    pump_without_opt_modes = sorted(
        pump_overlapps[~modes_to_optimise][:, pump == 1].sum(axis=1), reverse=True
    )

    if mode == "diff":
        return np.mean(pump_without_opt_modes[:n_modes]) - np.min(pump_with_opt_modes)
    if mode == "diff2":
        return np.max(pump_without_opt_modes) - np.min(pump_with_opt_modes)
    if mode == "ratio":
        # return np.mean(pump_without_opt_modes[:n_modes]) / np.min(pump_with_opt_modes)
        return np.max(pump_without_opt_modes) / np.min(pump_with_opt_modes)
    raise Exception("Optimisation mode not understood")


def _optimise_diff_evolution(seed, costf=None, bounds=None, disp=False, maxiter=1000, popsize=5):
    """Wrapper of differnetial evolution algorithm to launch multiple seeds."""
    return optimize.differential_evolution(
        func=costf,
        bounds=bounds,
        maxiter=maxiter,
        disp=disp,
        popsize=popsize,
        workers=1,
        seed=seed,
        recombination=0.8,
        mutation=[0.5, 1.5],
        strategy="randtobest1bin",
    )


def _overlap_matrix_element(graph, mode):
    """Compute the overlapp between a mode and each inner edges of the graph."""
    return list(
        -q_value(mode)
        * compute_overlapping_single_edges(mode, graph)
        * np.imag(gamma(to_complex(mode), graph.graph["params"]))
    )


def compute_pump_overlapping_matrix(graph, modes_df):
    """Compute the matrix of pump overlapp with each edge."""
    with multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool:
        overlapp_iter = pool.imap(partial(_overlap_matrix_element, graph), modes_df["passive"])
        pump_overlapps = np.empty([len(modes_df["passive"]), len(graph.edges)])
        for mode_id, overlapp in tqdm(enumerate(overlapp_iter), total=len(pump_overlapps)):
            pump_overlapps[mode_id] = overlapp
    return pump_overlapps


def optimize_pump(  # pylint: disable=too-many-locals
    modes_df,
    graph,
    lasing_modes_id,
    pump_min_frac=0.5,
    maxiter=500,
    popsize=5,
    seed=42,
    n_seeds=24,
    disp=False,
    use_modes=False,
):
    """Optimise the pump for lasing a set of modes.

    Args:
        modes_df (dataframe): modes dataframe
        graph (networkx): quantum raph
        lasing_modes_id (list): list of modes to optimise the pump for lasing first
        pump_min_frac (float): minimum fraction of edges in the pump
        maxiter (int): maximum number of iterations (for scipy.optimize.differential_evolution)
        popsize (int): size of population (for scipy.optimize.differential_evolution)
        seed (int): seed for random number generator
        n_seeds (int): number of run with different seends in parallel
        disp (bool): if True, display the optimisation iterations
        use_modes (bool): if True, use passive mode profiles to design pump

    Returns:
        optimal_pump, pump_overlapps, costs: best pump, overlapping matrix, all costs from seeds
    """
    np.random.seed(seed)

    if "pump" not in graph.graph["params"]:
        graph.graph["params"]["pump"] = np.ones(len(graph.edges))

    pump_overlapps = compute_pump_overlapping_matrix(graph, modes_df)

    mode_mask = np.array(len(pump_overlapps) * [False])
    lasing_modes_id = np.array(lasing_modes_id)
    mode_mask[lasing_modes_id] = True
    pump_min_size = int(pump_min_frac * len(np.where(graph.graph["params"]["inner"])[0]))

    mean_edge_modes = None
    if use_modes:
        mean_edge_modes = []
        for mode in modes_df["passive"]:
            _m = mean_mode_on_edges(mode, graph)
            mean_edge_modes.append(_m / _m.max())

        mean_edge_modes = np.array(mean_edge_modes)

    if use_modes:
        _map = partial(pump_mapping, mean_edge_modes=mean_edge_modes)

    _costf = partial(
        pump_cost,
        modes_to_optimise=mode_mask,
        pump_min_size=pump_min_size,
        pump_overlapps=pump_overlapps,
        pump_mapper=_map if use_modes else None,
    )

    bounds = len(mean_edge_modes) * [(-10, 10)] if use_modes else len(graph.edges) * [(0, 1)]
    # we don't pump the outer edges by restricting the bounds
    for i, _ in enumerate(bounds):
        if graph.graph["params"]["inner"][i] == 0:
            bounds[i] = (0.0, 0.0)

    _optimizer = partial(
        _optimise_diff_evolution,
        costf=_costf,
        bounds=bounds,
        disp=disp,
        maxiter=maxiter,
        popsize=popsize,
    )

    with multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool:
        results = list(
            tqdm(
                pool.imap(_optimizer, np.random.randint(0, 100000, n_seeds)),
                total=n_seeds,
            )
        )

    costs = [result.fun for result in results]
    optimal_pump = results[np.argmin(costs)].x
    final_cost = _costf(optimal_pump)

    if use_modes:
        optimal_pump = _map(optimal_pump)
    else:
        optimal_pump = np.round(optimal_pump, 0)

    L.info("Final cost is: %s", final_cost)
    if final_cost > 0:
        L.info("This pump may not provide single lasing!")

    return optimal_pump, pump_overlapps, costs, final_cost


def make_threshold_pump(graph, mode, target=0.3):
    """Create a pump profile using edges with most electric field on a mode.

    Args:
        target (float): target surface area to cover with the pump
    """
    edge_solution = mean_mode_on_edges(mode, graph)
    inner = np.array([graph[edge[0]][edge[1]]["inner"] for edge in graph.edges], dtype=int)
    tot_L = get_total_inner_length(graph)

    def surf(frac):
        pump_edges = np.where(edge_solution < frac * max(edge_solution), 0, 1)
        pump = inner * pump_edges
        return abs(
            sum(graph[u][v]["length"] for i, (u, v) in enumerate(graph.edges) if pump[u] == 1)
            / tot_L
            - target
        )

    x = np.linspace(0, 1, 100)
    frac = x[np.argmin([surf(_x) for _x in x])]
    pump_edges = np.where(edge_solution < frac * max(edge_solution), 0, 1)
    return (inner * pump_edges).tolist()
