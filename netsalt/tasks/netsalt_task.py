"""Generic tasks base don luigi.Task."""
import shutil
from hashlib import sha256
from pathlib import Path

import luigi
import numpy as np
import yaml

from netsalt.io import load_graph

from .config import ModeSearchConfig, PumpConfig


def ensure_dir(file_path):
    """Create directory to save file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


class HashedTask(luigi.Task):
    """This class is inspired by https://github.com/gorlins/salted/, and add a default hashed
    output(self) function, using the parameter target_path.

    This class has a new attribute called self.task_hash, identifying this class w.r.t its required
    tasks (which is not the  case for task_id).

    One can use custom output function and append hash to path with get_hashed_path
    """

    target_path = luigi.Parameter(default="default_target_path_PLEASE_UPDATE")

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        self.with_hash = True
        self.task_hash = self.get_full_id()
        self.hashed_target_path = None
        self.set_hashed_target_path()

    def unset_hash(self):
        """Hack to bypass the use of hashed targets."""
        self.hashed_target_path = None

    def set_hashed_target_path(self):
        """Modify the target filename to append the task_hash."""
        if hasattr(self, "lasing_modes_id"):
            self.target_path = (
                str(Path(self.target_path).with_suffix(""))
                + "_"
                + "_".join(
                    [str(i) for i in self.lasing_modes_id]  # pylint: disable=no-member
                )
                + str(Path(self.target_path).suffix)
            )

        if self.with_hash:
            hashed_target_path = Path("hashed") / self.target_path
            self.hashed_target_path = Path(
                f"{hashed_target_path.with_suffix('')}_{self.task_hash}{hashed_target_path.suffix}"
            )

    def get_full_id(self):
        """Get the full id of a task, including required tasks and significant parameters."""
        msg = ",".join(
            [req.get_full_id() for req in luigi.task.flatten(self.requires())]
        )
        msg += ",".join(
            [self.__class__.__name__]
            + [
                f"{param_name}={repr(self.param_kwargs[param_name])}"
                for param_name, param in sorted(self.get_params())
                if param.significant
            ]
        )
        return sha256(msg.encode()).hexdigest()

    def output(self):
        """Overloading of the output class to include hash in filenames by default."""
        ensure_dir(self.target_path)
        if self.hashed_target_path is not None:
            ensure_dir(self.hashed_target_path)
            return luigi.LocalTarget(self.hashed_target_path)
        return luigi.LocalTarget(self.target_path)

    def on_success(self):
        """Create symling to localtarget file to be readable by humans."""
        if self.hashed_target_path is not None:
            if Path(self.hashed_target_path).is_dir():
                if Path(self.target_path).exists():
                    shutil.rmtree(self.target_path)
                shutil.copytree(self.hashed_target_path, self.target_path)
            else:
                shutil.copyfile(self.hashed_target_path, self.target_path)


class NetSaltTask(luigi.Task):
    """Add the capability to rerun the task.
    Existing Remote/Local targets will be removed before running.
    """

    rerun = luigi.BoolParameter(
        config_path={"section": "DEFAULT", "name": "rerun"},
        default=False,
        significant=False,
    )

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        if self.rerun is True:
            targets = luigi.task.flatten(self.output())
            for target in targets:
                if target.exists() and isinstance(
                    target, luigi.target.FileSystemTarget
                ):
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
