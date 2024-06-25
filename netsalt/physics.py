"""All physics-related functions."""
import logging

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
    if "gamma_perp" not in params:
        return -1.0j
    return params["gamma_perp"] / (np.real(freq) - params["k_a"] + 1.0j * params["gamma_perp"])


def set_dispersion_relation(graph, dispersion_relation):
    """Set the dispersion relation on the graph.

    Args:
        graph (networkx graph): current graph
        dispersion_relation (function): dispersion relation function
        params (dict): parameters
    """
    graph.graph["dispersion_relation"] = dispersion_relation


def dispersion_relation_linear(freq, params=None):
    r"""Linear dispersion relation with wavespeed.

    The dispersion relation is

    .. math::

        k(\omega) = \frac{\omega}{c}

    Args:
        freq (float): frequency
        params (dict): parameters, must include wavespeed 'c'
    """
    if not params or "c" not in params:
        raise Exception("Please correct provide dispersion parameters")
    return freq / np.array(params["c"])


def dispersion_relation_resistance(freq, params=None):
    r"""Linear dispersion relation with wavespeed.

    The dispersion relation is

    .. math::

        k(\omega) = \sqrt{\frac{\omega^2}{c^2} + i R C \omega}

    Args:
        freq (float): frequency
        params (dict): parameters, must include wavespeed 'c', compliance C and edge resistances R
    """
    if not params or "c" not in params:
        raise Exception("Please correct provide dispersion parameters")
    return np.sqrt(
        (freq / params["c"]) ** 2 + 1.0j * freq * params.get("C", 1.0) * params.get("R", 0.0)
    )


def dispersion_relation_dielectric(freq, params=None):
    """Linear dispersion relation with dielectric constant.

    Args:
        freq (float): frequency
        params (dict): parameters, must include 'gamma_perp' and 'k_a'
    """
    if not params:
        raise Exception("Please provide dispersion parameters")
    return freq * np.array(np.sqrt(params["dielectric_constant"])) / params.get("c", 1.0)


def dispersion_relation_pump(freq, params=None):
    r"""Dispersion relation with dielectric constant and pump.

    If a pump is given in params

    .. math::

        k(\omega) = \omega \sqrt{\epsilon + \gamma(\omega) D_0 \delta_\mathrm{pump}}

    otherwise

    .. math::

        k(\omega) = \omega \sqrt{\epsilon}

    Args:
        freq (float): frequency
        params (dict): parameters, must include the dielectric_constant in params,
            if pump is in params, it must include D0 and necessary parameter
            for the computation of :math:`gamma`
    """
    if not params:
        raise Exception("Please provide dispersion parameters")

    if "pump" not in params or "D0" not in params:
        return freq * np.array(np.sqrt(params["dielectric_constant"])) / params.get("c", 1.0)

    return freq * np.sqrt(
        np.array(params["dielectric_constant"]) / params.get("c", 1.0)
        + gamma(freq, params) * params["D0"] * params["pump"]
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
            graph[e[0]][e[1]]["dielectric_constant"] = (
                custom_values["constant"][ei] + 1.0j * custom_values["loss"][ei]
            )

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
