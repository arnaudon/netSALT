..  -*- coding: utf-8 -*-

.. _contents:

*netSALT*: simulating lasing networks
========================================

*netSALT* is a python package to solve the SALT equation on networkx. It builds on the theory of quantum graphs to include nonlinear modal interactions.


Installation
************

We recommend `uv <https://docs.astral.sh/uv/>`_ for managing the environment.
From a checkout::

       $ git clone git@github.com:arnaudon/netSALT.git
       $ cd netSALT
       $ uv venv
       $ uv pip install -e .

or from PyPI::

       $ uv pip install netsalt

Plain ``pip`` works too (``pip install -e .`` or ``pip install netsalt``).


Usage
*****

netSALT can be used directly as a library or through a plain-Python pipeline
driven by YAML config files.

**Object API.** :class:`netsalt.QuantumGraph` is a thin ``networkx.Graph``
subclass that carries the quantum-graph state and exposes the build/solve
helpers as methods::

       import networkx as nx
       import numpy as np
       from netsalt import QuantumGraph
       from netsalt.physics import dispersion_relation_dielectric

       g = nx.path_graph(20)
       positions = np.array([[float(i), 0.0] for i in range(20)])
       qg = QuantumGraph.from_networkx(
           g,
           params={
               "open_model": "open",
               "c": 1.0,
               "dielectric_params": {
                   "method": "uniform",
                   "inner_value": 4.0,
                   "loss": 0.0,
                   "outer_value": 1.0,
               },
           },
           positions=positions,
       )
       qg.set_dispersion_relation(dispersion_relation_dielectric)
       qg.set_dielectric_constant()

       L = qg.laplacian(2.0 + 0.0j)        # quantum Laplacian L(k)
       quality = qg.mode_quality([2.0, 0.0])

Each method delegates to the matching module-level function, so the procedural
API (``construct_laplacian``, ``mode_quality``, …) keeps working unchanged.

**Pipeline.** For a full passive/lasing/controllability run, drive the
pipeline from a YAML config::

       $ python -m netsalt lasing config.yaml   # passive | lasing | controllability

Each step caches to disk and is skipped when its output already exists (pass
``--force`` to recompute). See the ``examples`` folder for ready-to-run configs.

Citing
******

If you use *netSALT* in your research, please cite:

  D. Saxena, A. Arnaudon, O. Cipolato, M. Gaio, A. Quentel, S. Yaliraki,
  D. Pisignano, A. Camposeo, M. Barahona, R. Sapienza,
  "Sensitivity and spectral control of network lasers",
  *Nat. Commun.* **13**, 6573 (2022).
  https://doi.org/10.1038/s41467-022-34073-3

A preprint is available on arXiv: https://arxiv.org/abs/2203.16974

Credits
*******

Original author:

- Alexis Arnaudon, GitHub: `arnaudon <https://github.com/arnaudon>`_

Contributors:

- Dhruv Saxena

Physics background
******************

.. toctree::
    :maxdepth: 2

    theory
    lasing

Code documentation
******************

Documentation of the code.

.. toctree::
    :maxdepth: 3

    modes
    contour
    physics
    quantum_graph
    algorithm
    pump
    pipeline
    config_loader
    plotting
    io
    params
    utils

Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
