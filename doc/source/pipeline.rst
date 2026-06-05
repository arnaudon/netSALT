The pipeline module
=========================

Plain-Python pipeline driving full passive / lasing / controllability runs from
a YAML config. Each ``step_*`` function is a single cached compute or plot step:
if its output file already exists and ``params["force"]`` is falsy, the cached
file is loaded and returned; otherwise the step runs and saves its output.

Three top-level entry points are the production interface:
:func:`~netsalt.pipeline.compute_passive_modes`,
:func:`~netsalt.pipeline.compute_lasing_modes`, and
:func:`~netsalt.pipeline.compute_controllability`. Configuration is a
:class:`~netsalt.params.NetSaltParams` instance built from a YAML file via
:func:`netsalt.config_loader.load_config`. The CLI dispatcher
(``python -m netsalt {passive|lasing|controllability} <config.yaml> [--force]``)
wraps these.

.. automodule:: netsalt.pipeline
   :members:
