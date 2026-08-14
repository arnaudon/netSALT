"""Input/output for quantum graphs and results.

Graph I/O defaults to JSON (node-link format) rather than pickle. Pickle
deserialisation executes arbitrary code in the file, which is a denial-of-
service and code-execution sink when .pkl/.gpickle files are received from
other researchers or downloaded. JSON is data-only and round-trips the
graph structure, ``NetSaltParams``, node positions, edge dielectric
constants, and the registered dispersion relation.

Pickle remains supported for backward compatibility with existing
artefacts but emits a :class:`DeprecationWarning` and will be removed.
"""

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from . import physics
from .params import NetSaltParams

_PICKLE_SUFFIXES = {".pkl", ".gpickle"}

# Registry of dispersion relations that can round-trip through JSON. New
# dispersion relations added to physics.py should be registered here.
_DISPERSION_REGISTRY = {
    fn.__qualname__: fn
    for fn in (
        physics.dispersion_relation_linear,
        physics.dispersion_relation_resistance,
        physics.dispersion_relation_dielectric,
        physics.dispersion_relation_pump,
    )
}


class _GraphJSONEncoder(json.JSONEncoder):
    """Encode numpy arrays, complex numbers, NetSaltParams, and registered
    callables so :func:`nx.node_link_data` output becomes JSON-serialisable."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            if np.iscomplexobj(obj):
                return {
                    "__complex_array__": True,
                    "real": obj.real.tolist(),
                    "imag": obj.imag.tolist(),
                }
            return obj.tolist()
        if isinstance(obj, complex | np.complexfloating):
            return {"__complex__": True, "real": float(obj.real), "imag": float(obj.imag)}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, NetSaltParams):
            return {"__netsalt_params__": True, "data": obj.to_dict()}
        if callable(obj) and getattr(obj, "__qualname__", None) in _DISPERSION_REGISTRY:
            return {"__callable__": obj.__qualname__}
        return super().default(obj)


def _decode(obj: Any) -> Any:
    """Reverse the transformations done by :class:`_GraphJSONEncoder`."""
    if isinstance(obj, dict):
        if obj.get("__complex__"):
            return complex(obj["real"], obj["imag"])
        if obj.get("__complex_array__"):
            return np.asarray(obj["real"]) + 1j * np.asarray(obj["imag"])
        if obj.get("__netsalt_params__"):
            return NetSaltParams.from_dict(_decode(obj["data"]))
        if obj.get("__callable__"):
            name = obj["__callable__"]
            if name in _DISPERSION_REGISTRY:
                return _DISPERSION_REGISTRY[name]
            raise ValueError(f"Unknown callable referenced in graph payload: {name!r}")
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(x) for x in obj]
    return obj


def save_graph(graph, filename: str = "graph.json") -> None:
    """Save a quantum graph to disk.

    JSON (``.json``) is the default and only safe format. Passing a
    ``.pkl``/``.gpickle`` filename still works but emits a
    :class:`DeprecationWarning`.
    """
    path = Path(filename)
    if path.suffix in _PICKLE_SUFFIXES:
        warnings.warn(
            "Pickle graph I/O is unsafe on untrusted files. Save as .json "
            "instead — save_graph(graph, 'graph.json').",
            DeprecationWarning,
            stacklevel=2,
        )
        with open(path, "wb") as pickle_file:
            pickle.dump(graph, pickle_file)
        return

    payload = nx.node_link_data(graph, edges="edges")
    with open(path, "w") as json_file:
        json.dump(payload, json_file, cls=_GraphJSONEncoder)


def load_graph(filename: str = "graph.json", *, allow_pickle: bool = False, as_class: bool = False):
    """Load a quantum graph from disk.

    Args:
        filename: graph file. ``.json`` (node-link) is the safe default;
            ``.pkl``/``.gpickle`` require an explicit ``allow_pickle=True``
            because unpickling executes arbitrary code in the source file.
        allow_pickle: explicit opt-in for pickle-format files. Only enable
            for files you produced yourself or fully trust.
        as_class: if True, return a
            :class:`~netsalt.quantum_graph.QuantumGraph` instead of a plain
            ``networkx.Graph``. Defaults to False so existing callers are
            unaffected. Only applies to the JSON path; the pickle path returns
            whatever type was pickled.
    """
    path = Path(filename)
    if path.suffix in _PICKLE_SUFFIXES:
        if not allow_pickle:
            raise ValueError(
                f"{path} is a pickle file, which is unsafe to load from an "
                "untrusted source. Pass allow_pickle=True to opt in, or "
                "re-save the graph as .json via save_graph."
            )
        warnings.warn(
            "Loading a pickle graph executes arbitrary code. Prefer .json.",
            DeprecationWarning,
            stacklevel=2,
        )
        with open(path, "rb") as pickle_file:
            return pickle.load(pickle_file)

    with open(path) as json_file:
        payload = json.load(json_file)
    payload = _decode(payload)
    graph = nx.node_link_graph(payload, edges="edges")
    # Positions stored as lists; the compute path expects numpy arrays.
    for node in graph.nodes:
        pos = graph.nodes[node].get("position")
        if isinstance(pos, list):
            graph.nodes[node]["position"] = np.asarray(pos, dtype=float)
    # Edge-level complex arrays, and graph-level aggregates like "lengths",
    # are expected to be ndarrays by the compute path.
    for key in ("lengths", "ks"):
        value = graph.graph.get(key)
        if isinstance(value, list):
            graph.graph[key] = np.asarray(value)
    params = graph.graph.get("params")
    if params is not None and not isinstance(params, NetSaltParams):
        graph.graph["params"] = NetSaltParams.from_dict(params)
    if as_class:
        from .quantum_graph import QuantumGraph  # local import avoids an import cycle

        return QuantumGraph(graph)
    return graph


#: ``modes_df.attrs`` keys holding a dataframe rather than a scalar.
_MODES_ATTR_FRAME_KEYS = ("salt_diagnostics",)


def _attrs_path(filename: str) -> Path:
    """Sidecar carrying ``modes_df.attrs`` next to the HDF5 file.

    A sidecar rather than a second HDF5 key: ``modal_intensities.h5`` and
    friends have always held exactly one dataset, and ``pd.read_hdf(path)``
    without a key raises as soon as there are two. Adding a key would break
    every existing reader (it broke the functional test's own comparator), so
    the HDF5 shape is left alone.

    The attrs are stored at all because pandas does not persist
    ``DataFrame.attrs``: without this the solver diagnostics were present on a
    fresh run and silently absent on a cached one, and the pipeline reloads
    every step it has already computed. For a solver whose contract is "report,
    do not correct", losing the report on the second run is the failure that
    matters.
    """
    path = Path(filename)
    return path.with_name(path.stem + "_attrs.json")


def save_modes(modes_df, filename: str = "results.h5") -> None:
    """Save modes dataframe into hdf5 (fixed format) and a CSV sidecar.

    ``format="fixed"`` and ``mode="w"`` are pinned explicitly so the behaviour
    doesn't silently change across pandas releases. The modes dataframe holds
    complex-valued columns, which pytables stores as pickled objects and
    warns about on every write; that warning is harmless here, so it's
    suppressed locally.

    ``modes_df.attrs`` goes to a JSON sidecar; see :func:`_attrs_path`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        modes_df.to_hdf(filename, key="modes", format="fixed", mode="w")
    modes_df.to_csv(Path(filename).with_suffix(".csv"))

    attrs = dict(getattr(modes_df, "attrs", {}) or {})
    if not attrs:
        return
    payload: dict[str, Any] = {"scalars": {}, "frames": {}}
    for key, value in attrs.items():
        if key in _MODES_ATTR_FRAME_KEYS and isinstance(value, pd.DataFrame):
            frame = value.copy()
            tuple_columns = [
                column
                for column in frame.columns
                if frame[column].map(lambda v: isinstance(v, tuple)).any()
            ]
            for column in tuple_columns:
                frame[column] = frame[column].map(lambda v: list(v) if isinstance(v, tuple) else v)
            payload["frames"][key] = {
                "data": json.loads(frame.to_json(orient="split")),
                "tuple_columns": tuple_columns,
            }
        else:
            payload["scalars"][key] = value
    _attrs_path(filename).write_text(json.dumps(payload, default=str))


def load_modes(filename: str = "results.h5"):
    """Return modes dataframe from hdf5, with ``attrs`` restored from the sidecar.

    A file written before the sidecar existed simply has none, and loads exactly
    as it always did.
    """
    modes_df = pd.read_hdf(filename, "modes")
    path = _attrs_path(filename)
    if not path.exists():
        return modes_df

    payload = json.loads(path.read_text())
    for key, value in payload.get("scalars", {}).items():
        # json turns dict keys into strings; salt_unit_scale is keyed by mode id.
        if isinstance(value, dict):
            value = {
                (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): v
                for k, v in value.items()
            }
        modes_df.attrs[key] = value
    for key, stored in payload.get("frames", {}).items():
        frame = pd.DataFrame(**{k: v for k, v in stored["data"].items() if k != "dtype"})
        for column in stored.get("tuple_columns", []):
            if column in frame.columns:
                frame[column] = frame[column].map(lambda v: tuple(v) if isinstance(v, list) else v)
        modes_df.attrs[key] = frame
    return modes_df


def save_qualities(qualities, filename: str = "results.h5") -> None:
    """Save qualities in the results hdf5 file.

    Uses ``mode="a"`` so this key is appended to whatever ``save_modes`` wrote,
    rather than overwriting the ``modes`` key silently.
    """
    pd.DataFrame(data=qualities, index=None, columns=None).to_hdf(
        filename, key="qualities", format="fixed", mode="a"
    )


def load_qualities(filename: str = "results.h5"):
    """Load qualities from the results hdf5 file."""
    return pd.read_hdf(filename, "qualities").to_numpy()
