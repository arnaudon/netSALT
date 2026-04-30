"""Load YAML config files into :class:`NetSaltParams`.

A config file is a flat YAML mapping of parameter names (matching
``NetSaltParams`` fields plus any extras the pipeline reads). It may
optionally include a ``defaults:`` key whose value is a path (relative
to this file) to another config file to inherit from. Inheritance is a
single-pass shallow merge: keys in the child override keys in the base.
"""

from pathlib import Path
from typing import Any

import yaml

from .params import NetSaltParams


def _load_raw(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")
    if "defaults" in raw:
        base_rel = raw.pop("defaults")
        base_path = (path.parent / base_rel).resolve()
        base = _load_raw(base_path)
        base.update(raw)
        return base
    return raw


def load_config(path: str | Path) -> NetSaltParams:
    """Load a YAML config file into a validated :class:`NetSaltParams`.

    A ``defaults: <relative-path>`` key inside the file pulls in another
    config first, then the current file's keys override.
    """
    return NetSaltParams.from_dict(_load_raw(Path(path)))
