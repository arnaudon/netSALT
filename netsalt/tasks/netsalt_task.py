"""Generic tasks base don luigi.Task."""
import shutil
from hashlib import sha256
from pathlib import Path

import luigi


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

        self.with_hash = False  # True
        self.task_hash = self.get_full_id()
        self.hashed_target_path = None
        self.set_hashed_target_path()

    def unset_hash(self):
        """Hack to bypass the use of hashed targets."""
        self.hashed_target_path = None

    def set_hashed_target_path(self):
        """Modify the target filename to append the task_hash."""
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
        if self.hashed_target_path is not None:
            ensure_dir(self.hashed_target_path)
            return luigi.LocalTarget(self.hashed_target_path)
        ensure_dir(self.target_path)
        return luigi.LocalTarget(self.target_path)

    def on_success(self):
        """Create symling to localtarget file to be readable by humans."""
        if self.hashed_target_path is not None:
            shutil.copy(self.hashed_target_path, self.target_path)


class NetSaltTask(HashedTask):
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
