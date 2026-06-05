The quantum graph module
=========================

This module builds a quantum graph from a ``networkx`` graph (edge lengths,
pumps, dielectric) and constructs the Laplacian / weight / incidence matrices.

The functions here all take a graph as their first argument. For an object-style
API, :class:`~netsalt.quantum_graph.QuantumGraph` is a thin ``networkx.Graph``
subclass whose methods (``laplacian``, ``mode_quality``, ``scan_frequencies``,
``with_pump``, …) delegate to those functions — see
:meth:`~netsalt.quantum_graph.QuantumGraph.from_networkx` to build one.

Per-pump and per-mode work is done on throwaway copies via
:func:`~netsalt.quantum_graph.graph_with_params` (and its
:func:`~netsalt.quantum_graph.graph_with_pump` shortcut) rather than by mutating
the shared ``graph.graph["params"]`` in place.

.. automodule:: netsalt.quantum_graph
   :members:
