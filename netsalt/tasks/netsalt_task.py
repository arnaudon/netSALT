"""Generic tasks base don luigi.Task."""
from pathlib import Path

import luigi
import numpy as np
import yaml

from netsalt.io import load_graph

from .config import ModeSearchConfig, PumpConfig


def ensure_dir(file_path):
    """Create directory to save file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


class NetSaltTask(luigi.Task):
    """Add the capability to rerun the task.
    Existing Remote/Local targets will be removed before running.
    """

    rerun = luigi.BoolParameter(
        default=False,
        significant=False,
    )

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        if self.rerun is True:
            targets = luigi.task.flatten(self.output())
            for target in targets:
                if target.exists() and isinstance(target, luigi.target.FileSystemTarget):
                    target.fs.remove(target.path, recursive=True)

    def get_graph(self, graph_path):  # pylint: disable=no-self-use
        """To ensure we get all parameters."""
        qg = load_graph(graph_path)
        config = ModeSearchConfig()
        qg.graph["params"].update(
            {
                "n_workers": config.n_workers,
                "k_n": config.k_n,
                "k_min": config.k_min,
                "k_max": config.k_max,
                "alpha_n": config.alpha_n,
                "alpha_min": config.alpha_min,
                "alpha_max": config.alpha_max,
                "quality_threshold": config.quality_threshold,
                "max_steps": config.max_steps,
                "max_tries_reduction": config.max_tries_reduction,
                "reduction_factor": config.reduction_factor,
                "search_stepsize": config.search_stepsize,
            }
        )
        return qg

    def get_graph_with_pump(self, graph_path):
        """To ensure we get all parameters, needs pump entry in requires."""
        qg = self.get_graph(graph_path)
        qg.graph["params"].update(
            {
                "D0_max": PumpConfig().D0_max,
                "D0_steps": PumpConfig().D0_steps,
                "pump": np.array(yaml.full_load(self.input()["pump"].open())),
            }
        )
        return qg

    def add_lasing_modes_id(self, filename):
        """Add lasing modes ids."""
        # pylint: disable=no-member,disable=not-an-iterable
        if self.lasing_modes_id is None:
            return filename
        filename = Path(filename)
        ext = filename.suffix
        return (
            "_".join([str(filename.with_suffix(""))] + [str(_id) for _id in self.lasing_modes_id])
            + ext
        )


class NetSaltWrapperTask(NetSaltTask, luigi.WrapperTask):
    """Wrapper netsalt task"""
