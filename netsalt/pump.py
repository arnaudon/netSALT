"""Pump optimisation module."""
import logging
import multiprocessing
from functools import partial

import pulp
import numpy as np
from scipy import optimize
from tqdm import tqdm

from netsalt.modes import compute_overlapping_single_edges, mean_mode_on_edges
from netsalt.physics import gamma, q_value
from netsalt.utils import to_complex

L = logging.getLogger(__name__)


def pump_cost(pump, modes_to_optimise, pump_overlapps, pump_min_size=None, epsilon=0):
    """Cost function to minimize."""
    if not isinstance(modes_to_optimise, (list, np.ndarray)):
        modes_to_optimise = [modes_to_optimise]
    if len(modes_to_optimise) < len(pump_overlapps):
        mode_mask = np.array(len(pump_overlapps) * [False])
        mode_mask[np.array(modes_to_optimise)] = True
    else:
        mode_mask = modes_to_optimise

    if pump_min_size is not None and pump.sum() < pump_min_size:
        return 1e10

    pump = np.round(pump, 0)
    pump_without_opt_modes = pump_overlapps[~mode_mask].dot(pump)
    pump_with_opt_modes = pump_overlapps[mode_mask].dot(pump)
    return (epsilon + np.max(pump_without_opt_modes)) / np.min(pump_with_opt_modes)


def _optimise_diff_evolution(seed, costf=None, bounds=None, disp=False, maxiter=1000, popsize=5):
    """Wrapper of differential evolution algorithm to launch multiple seeds."""
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
    """Compute the overlap between a mode and each inner edges of the graph."""
    return list(
        -q_value(mode)
        * compute_overlapping_single_edges(mode, graph)
        * np.imag(gamma(to_complex(mode), graph.graph["params"]))
    )


def compute_pump_overlapping_matrix(graph, modes_df):
    """Compute the matrix of pump overlap with each edge."""
    pump_overlapps = np.empty([len(modes_df["passive"]), len(graph.edges)])
    with multiprocessing.Pool(graph.graph["params"]["n_workers"]) as pool:
        for mode_id, overlap in tqdm(
            enumerate(pool.imap(partial(_overlap_matrix_element, graph), modes_df["passive"])),
            total=len(pump_overlapps),
        ):
            pump_overlapps[mode_id] = overlap
    return pump_overlapps


def optimize_pump_diff_evolution(  # pylint: disable=too-many-locals
    modes_df,
    graph,
    lasing_modes_id,
    pump_min_frac=0.0,
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
        use_modes (bool): if True, use passive mode profiles to design pump (experimental)

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

    _costf = partial(
        pump_cost,
        modes_to_optimise=mode_mask,
        pump_min_size=pump_min_size,
        pump_overlapps=pump_overlapps,
    )

    bounds = len(graph.edges) * [(0, 1)]
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
    optimal_pump = np.round(results[np.argmin(costs)].x, 0)
    """
    # find threshold to binarize
    cs = []
    ts = np.linspace(0.00, 1.0, 100)
    for t in ts:
        p = optimal_pump.copy()
        p[optimal_pump < t] = 0
        p[optimal_pump > t] = 1
        c = _costf(p)
        cs.append(c if not np.isnan(c) else 1e10)
    t = ts[np.argmin(cs)]
    optimal_pump[optimal_pump < t] = 0
    optimal_pump[optimal_pump > t] = 1
    """
    final_cost = _costf(optimal_pump)

    L.info("Final cost is: %s", final_cost)
    if final_cost > 0:
        L.info("This pump may not provide single lasing!")

    return optimal_pump, pump_overlapps, costs, final_cost


def optimize_pump(
    modes_df, graph, lasing_modes_id, eps_min=5., eps_max=10, eps_n=10, cost_diff_min=1e-4
):
    """Optimizes a pump profile using linear programming with regularisation.

    Args:
        modes_df (dataframe): data with modes
        lasing_modes_id (list): list of mode to optimise
        eps_min (float): minimal value of regularisation (setting 0 may lead to very wrong pumps)
        eps_max (float): maximum value of regularisation (large values are not usually needed)
        eps_n (int): number of regularisations tried  with np.linespace(eps_min, eps_max, eps_n).
        cost_diff_min (float): min value of difference in cost below which we discard an edge
    """
    pump_overlapps = compute_pump_overlapping_matrix(graph, modes_df)
    mode_mask = np.array(len(pump_overlapps) * [False])
    lasing_modes_id = np.array(lasing_modes_id)
    mode_mask[lasing_modes_id] = True
    _costf = partial(pump_cost, modes_to_optimise=mode_mask, pump_overlapps=pump_overlapps)
    over_opt = pump_overlapps[mode_mask]
    over_others = pump_overlapps[~mode_mask]
    n_edges = len(graph.edges)

    def get_LP_prob(epsilon):
        """Setup the linear problem using PuLP."""
        Ys = [pulp.LpVariable(f"Y_{i}", 0, None) for i in range(n_edges)]
        m = pulp.LpVariable("m", 0, None)
        t = pulp.LpVariable("t", 0, None)

        prob = pulp.LpProblem("pump optimisation", pulp.LpMinimize)
        prob += m + epsilon * t

        for i, over in enumerate(over_opt):
            prob += pulp.lpSum([o * Y for o, Y in zip(over, Ys)]) == 1, f"constant_{i}"
        for i, over in enumerate(over_others):
            prob += pulp.lpSum([o * Y for o, Y in zip(over, Ys)]) <= m, f"maximum_{i}"

        inner = np.array([graph[edge[0]][edge[1]]["inner"] for edge in graph.edges], dtype=bool)

        for i, Y in enumerate(Ys):
            if inner[i]:
                prob += Y <= t, f"bound_{i}"
            else:
                prob += Y == 0
        return prob, m, t, Ys

    prob, m, t, Ys = get_LP_prob(0)

    def _opt(epsilon, prob, m, t, Ys):
        """Solve LP problem with epsilon regularisation."""
        prob += m + epsilon * t

        prob.solve()

        optimal_pump = []
        for Y in Ys:
            v = pulp.value(Y)
            if v is None:
                v = 0
            optimal_pump.append(v)
        pump = np.array(optimal_pump)
        pump[pump > 0] = 1.0

        c0 = _costf(pump, epsilon=epsilon)
        ids = []
        for i in range(len(pump)):
            if pump[i] == 1:
                _pump = pump.copy()
                _pump[i] = 0
                if _costf(_pump, epsilon=epsilon) - c0 < cost_diff_min:
                    ids.append(i)
        pump[ids] = 0

        final_cost = _costf(pump, epsilon=epsilon)
        return final_cost, pump

    epsilons = np.linspace(eps_min, eps_max, eps_n)
    cs = []
    ps = []
    for epsilon in epsilons:
        c, p = _opt(epsilon, prob, m, t, Ys)
        cs.append(c)
        ps.append(p)

    optimal_pump = ps[np.argmin(cs)]
    final_cost = min(cs)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.plot(epsilons, cs)
    plt.axvline(epsilons[np.argmin(cs)], c="k")
    plt.xlabel("epsilon")
    plt.ylabel("cost")
    plt.savefig(f"cost_opt_{lasing_modes_id[0]}.pdf")

    return optimal_pump, pump_overlapps, [final_cost], final_cost


def make_threshold_pump(graph, lasing_modes_id, modes_df, pump_min_size=None):
    """Create a pump profile using edges with most electric field on a mode to optimise cost."""
    if len(lasing_modes_id) > 1:
        raise Exception("Threshold pump is only for single mode at the moment.")

    edge_solution = mean_mode_on_edges(modes_df["passive"][lasing_modes_id[0]], graph)
    inner = np.array([graph[edge[0]][edge[1]]["inner"] for edge in graph.edges], dtype=int)

    pump_overlapps = compute_pump_overlapping_matrix(graph, modes_df)
    mode_mask = np.array(len(pump_overlapps) * [False])
    lasing_modes_id = np.array(lasing_modes_id)
    mode_mask[lasing_modes_id] = True

    def cost(frac):
        pump = inner * np.where(edge_solution < frac * max(edge_solution), 0, 1)
        return pump_cost(pump, mode_mask, pump_overlapps, pump_min_size=pump_min_size)

    # just a brute force search here seems ok
    fracs = np.linspace(0, 1, 1000)
    frac = fracs[np.argmin([cost(frac) for frac in fracs])]
    print(cost(frac), frac)
    pump_edges = inner * np.where(edge_solution < frac * max(edge_solution), 0, 1)
    return pump_edges.tolist()
