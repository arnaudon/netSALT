"""Pump optimisation module."""
import logging
import multiprocessing
from functools import partial

import numpy as np
from scipy import optimize
from tqdm import tqdm

from .modes import compute_overlapping_single_edges, mean_mode_on_edges
from .physics import gamma, q_value
from .utils import to_complex

L = logging.getLogger(__name__)


def pump_cost(pump, modes_to_optimise, pump_overlapps, pump_min_size=None):
    """Cost function to minimize."""
    if pump_min_size is not None and pump.sum() < pump_min_size:
        return 1e10
    pump_with_opt_modes = pump_overlapps[modes_to_optimise].dot(pump)
    pump_without_opt_modes = sorted(pump_overlapps[~modes_to_optimise].dot(pump), reverse=True)
    return np.max(pump_without_opt_modes) / np.min(pump_with_opt_modes)


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


def optimize_pump_old(  # pylint: disable=too-many-locals
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
    optimal_pump = results[np.argmin(costs)].x
    print(optimal_pump)
    # find threshold to binarize
    cs = []
    ts = np.linspace(0.00, 1.0, 1000)
    for t in ts:
        p = optimal_pump.copy()
        p[optimal_pump < t] = 0
        p[optimal_pump > t] = 1
        c = _costf(p)
        cs.append(c if not np.isnan(c) else 1e10)
    t = ts[np.argmin(cs)]
    optimal_pump[optimal_pump < t] = 0
    optimal_pump[optimal_pump > t] = 1
    final_cost = _costf(optimal_pump)
    print(final_cost)

    L.info("Final cost is: %s", final_cost)
    if final_cost > 0:
        L.info("This pump may not provide single lasing!")

    return optimal_pump, pump_overlapps, costs, final_cost

import pulp

from pulp import *
def optimize_pump(
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

    pump_overlapps = compute_pump_overlapping_matrix(graph, modes_df)
    mode_mask = np.array(len(pump_overlapps) * [False])
    lasing_modes_id = np.array(lasing_modes_id)
    mode_mask[lasing_modes_id] = True
    _costf = partial(
        pump_cost,
        modes_to_optimise=mode_mask,
        pump_min_size=pump_min_size,
        pump_overlapps=pump_overlapps,
    )
    over_opt = pump_overlapps[mode_mask]
    over_others = pump_overlapps[~mode_mask]
    n_edges = len(graph.edges)

    Ys = list(LpVariable.dict('Y', [f'{i}' for i in range(n_edges)], 0, None).values())
    print(Ys)
    m = LpVariable('m', 0, None)
    t = LpVariable('t', 0, None)
    prob = LpProblem("pump optimisation",LpMinimize)
    prob += m

    for i, over in enumerate(over_opt):
        prob += lpSum([over[i]*Ys[i] for i in range(len(Ys))]) == 1, f"constant_{i}"
    for i, over in enumerate(over_others):
        prob += lpSum([over[i]*Ys[i] for i in range(len(Ys))]) <= m, f"maximum_{i}"
    for i, Y in enumerate(Ys):
        prob += Y<=t, f"bound_{i}"

    print(prob)
    prob.solve()
    print("Status:", LpStatus[prob.status])

    # The optimised objective function value is printed to the screen
    print("Cost  = ", value(prob.objective))
    optimal_pump = np.array([value(Y) for Y in Ys])

    cs = []
    ts = np.linspace(0.00, 1.0, 1000)
    for t in ts:
        p = optimal_pump.copy()
        p[optimal_pump < t] = 0
        p[optimal_pump > t] = 1
        c = _costf(p)
        cs.append(c if not np.isnan(c) else 1e10)
    t = ts[np.argmin(cs)]
    optimal_pump[optimal_pump < t] = 0
    optimal_pump[optimal_pump > t] = 1
    final_cost = _costf(optimal_pump)
    print(final_cost, optimal_pump)
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
def  ljk():

    # Creates a list of all the supply nodes
    Warehouses = ["A", "B"]

    # Creates a dictionary for the number of units of supply for each supply node
    supply = {"A": 1000,
            "B": 4000}

    # Creates a list of all demand nodes
    Bars = ["1", "2", "3", "4", "5"]

    # Creates a dictionary for the number of units of demand for each demand node
    demand = {"1":500,
            "2":900,
            "3":1800,
            "4":200,
            "5":700,}

    # Creates a list of costs of each transportation path
    costs = [   #Bars
            #1 2 3 4 5
            [2,4,5,2,1],#A   Warehouses
            [3,1,3,2,3] #B
            ]

    # The cost data is made into a dictionary
    costs = makeDict([Warehouses,Bars],costs,0)
    print(costs['A']['1'])

    # Creates the 'prob' variable to contain the problem data
    prob = LpProblem("Beer Distribution Problem",LpMinimize)

    # Creates a list of tuples containing all the possible routes for transport
    Routes = [(w,b) for w in Warehouses for b in Bars]

    # A dictionary called 'Vars' is created to contain the referenced variables(the routes)
    vars = LpVariable.dicts("Route",(Warehouses,Bars),0,None,LpInteger)

    # The objective function is added to 'prob' first
    prob += lpSum([vars[w][b]*costs[w][b] for (w,b) in Routes]), "Sum_of_Transporting_Costs"

    # The supply maximum constraints are added to prob for each supply node (warehouse)
    for w in Warehouses:
        prob += lpSum([vars[w][b] for b in Bars])<=supply[w], "Sum_of_Products_out_of_Warehouse_%s"%w

    # The demand minimum constraints are added to prob for each demand node (bar)
    for b in Bars:
        prob += lpSum([vars[w][b] for w in Warehouses])>=demand[b], "Sum_of_Products_into_Bar%s"%b

    print(prob)
    # The problem data is written to an .lp file
    prob.writeLP("BeerDistributionProblem.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    # The optimised objective function value is printed to the screen
    print("Total Cost of Transportation = ", value(prob.objective))

