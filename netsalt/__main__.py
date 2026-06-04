"""Run a netSALT pipeline from the command line.

Usage:
    python -m netsalt {passive|lasing|controllability} <config.yaml>

The config is loaded into a :class:`NetSaltParams` (with optional
``defaults: <relative-path>`` inheritance) and dispatched to the matching
entry point in :mod:`netsalt.pipeline`.
"""

import argparse
import sys

from .config_loader import load_config
from .pipeline import (
    compute_controllability,
    compute_lasing_modes,
    compute_passive_modes,
)

_ENTRYPOINTS = {
    "passive": compute_passive_modes,
    "lasing": compute_lasing_modes,
    "controllability": compute_controllability,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m netsalt", description=__doc__)
    parser.add_argument("workflow", choices=sorted(_ENTRYPOINTS))
    parser.add_argument("config", help="Path to the YAML config file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute every step even if its output file already exists.",
    )
    args = parser.parse_args(argv)

    params = load_config(args.config)
    if args.force:
        params["force"] = True

    _ENTRYPOINTS[args.workflow](params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
