The config_loader module
=========================

YAML config loader for the pipeline. A config file is a flat YAML mapping of
parameter names (matching :class:`~netsalt.params.NetSaltParams` fields plus any
extras the pipeline reads). It may optionally include a ``defaults:`` key whose
value is a path (relative to the file) to another config to inherit from;
inheritance is a single-pass shallow merge where child keys override the base.
:func:`~netsalt.config_loader.load_config` returns a validated
:class:`~netsalt.params.NetSaltParams`.

.. automodule:: netsalt.config_loader
   :members:
