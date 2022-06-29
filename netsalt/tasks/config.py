"""Configuration classes for luigi tasks.."""
import luigi


class ModeSearchConfig(luigi.Config):
    """Configuration of mode search algorithm."""

    n_workers = luigi.IntParameter(default=1)

    k_n = luigi.IntParameter(default=100)
    k_min = luigi.FloatParameter(default=10.0)
    k_max = luigi.FloatParameter(default=12.0)
    alpha_n = luigi.IntParameter(default=100)
    alpha_min = luigi.FloatParameter(default=0.0)
    alpha_max = luigi.FloatParameter(default=0.1)

    quality_threshold = luigi.FloatParameter(default=1e-3)
    max_steps = luigi.IntParameter(default=1000)
    max_tries_reduction = luigi.IntParameter(default=50)
    reduction_factor = luigi.FloatParameter(default=1.0)
    search_stepsize = luigi.FloatParameter(default=0.001)
    quality_method = luigi.ChoiceParameter(
        default="eigenvalue", choices=["eigenvalue", "singularvalue", "determinant"]
    )


class PumpConfig(luigi.Config):
    """Configuration for pump steps."""

    D0_max = luigi.FloatParameter(default=0.05)
    D0_steps = luigi.IntParameter(default=10)
