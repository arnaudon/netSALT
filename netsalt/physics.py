"""All physics-related functions."""
import logging
from functools import partial

import numpy as np

from .utils import from_complex

L = logging.getLogger(__name__)


def gamma(freq, params):
    r"""Gamma function.

    The gamma function is

    .. math::

        \gamma(k) = \frac{\gamma_\perp}{ \mathrm{real}(k) - k_a + j\gamma_\perp}

    Args:
        freq (float): frequency
        params (dict): parameters, must include 'gamma_perp' and 'k_a'
    """
    return params["gamma_perp"] / (np.real(freq) - params["k_a"] + 1.0j * params["gamma_perp"])


def set_dispersion_relation(graph, dispersion_relation, params):
    """Set the dispersion relation on the graph.

    Args:
        graph (networkx graph): current graph
        dispersion_relation (function): dispersion relation function
        params (dict): parameters
    """
    graph.graph["dispersion_relation"] = partial(dispersion_relation, params=params)


def dispersion_relation_linear(freq, params=None):
    r"""Linear dispersion relation with wavespeed.

    The dispersion relation is

    .. math::

        \omega(k) = \frac{k}{c}

    Args:
        freq (float): frequency
        params (dict): parameters, must include wavespeed 'c'
    """
    if not params or "c" not in params:
        raise Exception("Please correct provide dispersion parameters")
    return freq / params["c"]


def dispersion_relation_dielectric(freq, params=None):
    """Linear dispersion relation with dielectric constant.

    Args:
        freq (float): frequency
        params (dict): parameters, must include 'gamma_perp' and 'k_a'
    """
    if not params:
        raise Exception("Please provide dispersion parameters")
    return freq * np.sqrt(params["dielectric_constant"])


def dispersion_relation_pump(freq, params=None):
    r"""Dispersion relation with dielectric constant and pump.

    If a pump is given in params

    .. math::

        \omega(k) = k \sqrt{\epsilon + \gamma(k) D_0 \delta_\mathrm{pump}}

    otherwise

    .. math::

        \omega(k) = k \sqrt{\epsilon}

    Args:
        freq (float): frequency
        params (dict): parameters, must include the dielectric_constant in params,
            if pump is in params, it must include D0 and necessary parameter
            for the computation of :math:`gamma`
    """
    if not params:
        raise Exception("Please provide dispersion parameters")

    if "pump" not in params or "D0" not in params:
        return freq * np.sqrt(params["dielectric_constant"])

    return freq * np.sqrt(
        params["dielectric_constant"] + gamma(freq, params) * params["D0"] * params["pump"]
    )


def set_dielectric_constant(graph, params, custom_values=None):
    """Set dielectric constant in params, from dielectric constant or refraction index.

    Args:
        graph (networkx graph): current graph
        params (dict): parameters
        custom_values (list): custum edge values for dielectric constant
    """

    if "dielectric_params" in params and "refraction_params" in params:
        L.info(
            "Dielectric_params and refraction_params are provided, \
            so we will only use dielectric_params"
        )

    if "dielectric_params" not in params:
        if "refraction_params" not in params:
            raise Exception("Please provide dielectric_params or refraction_params!")
        params["dielectric_params"] = {}
        params["dielectric_params"]["method"] = params["refraction_params"]["method"]
        params["dielectric_params"]["inner_value"] = (
            params["refraction_params"]["inner_value"] ** 2
            - params["refraction_params"]["loss"] ** 2
        )
        params["dielectric_params"]["loss"] = (
            2.0 * params["refraction_params"]["inner_value"] * params["refraction_params"]["loss"]
        )
        params["dielectric_params"]["outer_value"] = params["refraction_params"]["outer_value"] ** 2

    if params["dielectric_params"]["method"] == "uniform":
        for u, v in graph.edges:
            if graph[u][v]["inner"]:
                graph[u][v]["dielectric_constant"] = (
                    params["dielectric_params"]["inner_value"]
                    + 1.0j * params["dielectric_params"]["loss"]
                )
            else:
                graph[u][v]["dielectric_constant"] = params["dielectric_params"]["outer_value"]

    if params["dielectric_params"]["method"] == "random":
        for u, v in graph.edges:
            graph[u][v]["dielectric_constant"] = np.random.normal(
                params["dielectric_params"]["mean"],
                params["dielectric_params"]["std"],
                1,
            )

    if params["dielectric_params"]["method"] == "custom":
        for ei, e in enumerate(graph.edges):
            graph[e[0]][e[1]]["dielectric_constant"] = custom_values[ei]

    update_params_dielectric_constant(graph, params)


def update_params_dielectric_constant(graph, params):
    """Update the dielectric constant values in the params dictionary.

    Args:
        graph (networkx graph): current graph
        params (dict): parameters, must include 'gamma_perp' and 'k_a'
    """
    params["dielectric_constant"] = [graph[u][v]["dielectric_constant"] for u, v in graph.edges]


def q_value(mode):
    r"""Compute the :math:`\mathcal Q` value of a mode.

    It is defined as

    .. math::

        \mathcal Q = \frac{\mathrm{Real} k}{2 \mathrm{Im}(k)}

    Args:
        mode (complex): complex values mode
    """
    mode = from_complex(mode)
    return 0.5 * mode[0] / mode[1]
