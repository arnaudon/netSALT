"""Tasks related to lasing."""
import pickle
import pandas as pd
import luigi
import numpy as np
import yaml

from netsalt.io import load_modes, save_modes
from netsalt.modes import (
    compute_modal_intensities,
    compute_mode_competition_matrix,
    find_threshold_lasing_modes,
    pump_trajectories,
)
from netsalt.pump import make_threshold_pump
from .netsalt_task import NetSaltTask
from .passive import CreateQuantumGraph, FindPassiveModes
from .pump import OptimizePump


class CreatePumpProfile(NetSaltTask):
    """Create a pump profile."""

    lasing_modes_id = luigi.ListParameter()
    mode = luigi.ChoiceParameter(
        default="uniform", choices=["uniform", "optimized", "threshold", "custom"]
    )
    custom_pump_path = luigi.Parameter(default="pump_profile.yaml")
    pump_profile_path = luigi.Parameter(default="out/pump_profile.yaml")
    threshold_target = luigi.FloatParameter(default=0.3)

    def requires(self):
        """ """
        if self.mode == "uniform":
            return {"graph": CreateQuantumGraph()}
        if self.mode == "optimized":
            return {"optimize": OptimizePump(lasing_modes_id=self.lasing_modes_id)}
        if self.mode == "threshold":
            return {"modes": FindPassiveModes(), "graph": CreateQuantumGraph()}
        return None

    def run(self):
        """ """
        if self.mode == "uniform":
            qg = self.get_graph(self.input()["graph"].path)
            pump = np.zeros(len(qg.edges()))
            for i, (u, v) in enumerate(qg.edges()):
                if qg[u][v]["inner"]:
                    pump[i] = 1
            pump = pump.tolist()

        elif self.mode == "optimized":
            with open(self.input()["optimize"].path, "rb") as pkl:
                results = pickle.load(pkl)
            pump = results["optimal_pump"].tolist()

        elif self.mode == "threshold":
            qg = self.get_graph(self.input()["graph"].path)
            modes_df = load_modes(self.input()["modes"].path)
            pump = make_threshold_pump(qg, self.lasing_modes_id, modes_df)

        elif self.mode == "custom":
            with open(self.custom_pump_path, "r") as yml:
                pump = yaml.safe_load(yml)
        else:
            raise Exception("Mode not understood")

        with open(self.output().path, "w") as yml:
            yaml.safe_dump(pump, yml)

    def output(self):
        """ """
        return luigi.LocalTarget(self.add_lasing_modes_id(self.pump_profile_path))


class ComputeModeTrajectories(NetSaltTask):
    """Compute mode trajectories from passive modes."""

    lasing_modes_id = luigi.ListParameter()
    modes_trajectories_path = luigi.Parameter(default="out/mode_trajectories.h5")
    skip = luigi.BoolParameter(default=False)

    def requires(self):
        """ """

        return {
            "graph": CreateQuantumGraph(),
            "modes": FindPassiveModes(),
            "pump": CreatePumpProfile(lasing_modes_id=self.lasing_modes_id),
        }

    def run(self):
        """ """
        modes_df = load_modes(self.input()["modes"].path)
        qg = self.get_graph_with_pump(self.input()["graph"].path)
        if not self.skip:
            modes_df = pump_trajectories(modes_df, qg, return_approx=True)
        save_modes(modes_df, filename=self.output().path)

    def output(self):
        """ """
        return luigi.LocalTarget(self.add_lasing_modes_id(self.modes_trajectories_path))


class FindThresholdModes(NetSaltTask):
    """Find the lasing thresholds and associated modes."""

    lasing_modes_id = luigi.ListParameter()
    threshold_modes_path = luigi.Parameter(default="out/lasing_thresholds_modes.h5")

    def requires(self):
        """ """
        return {
            "graph": CreateQuantumGraph(),
            "modes": ComputeModeTrajectories(lasing_modes_id=self.lasing_modes_id),
            "pump": CreatePumpProfile(lasing_modes_id=self.lasing_modes_id),
        }

    def run(self):
        """ """
        qg = self.get_graph_with_pump(self.input()["graph"].path)
        modes_df = load_modes(self.input()["modes"].path)
        modes_df = find_threshold_lasing_modes(modes_df, qg)
        save_modes(modes_df, filename=self.output().path)

    def output(self):
        """ """
        return luigi.LocalTarget(self.add_lasing_modes_id(self.threshold_modes_path))


class ComputeModeCompetitionMatrix(NetSaltTask):
    """Compute the mode competition matrix."""

    lasing_modes_id = luigi.ListParameter()
    competition_matrix_path = luigi.Parameter(default="out/mode_competition_matrix.h5")

    def requires(self):
        """ """
        return {
            "graph": CreateQuantumGraph(),
            "modes": FindThresholdModes(lasing_modes_id=self.lasing_modes_id),
            "pump": CreatePumpProfile(lasing_modes_id=self.lasing_modes_id),
        }

    def run(self):
        """ """
        qg = self.get_graph_with_pump(self.input()["graph"].path)
        modes_df = load_modes(self.input()["modes"].path)
        mode_competition_matrix = compute_mode_competition_matrix(qg, modes_df)
        pd.DataFrame(data=mode_competition_matrix, index=None, columns=None).to_hdf(
            self.output().path, key="mode_competition_matrix"
        )

    def output(self):
        """ """
        return luigi.LocalTarget(self.add_lasing_modes_id(self.competition_matrix_path))


class ComputeModalIntensities(NetSaltTask):
    """Compute modal intensities as a function of pump strength."""

    lasing_modes_id = luigi.ListParameter()
    D0_max = luigi.FloatParameter(default=0.1)
    modal_intensities_path = luigi.Parameter(default="out/modal_intensities.h5")

    def requires(self):
        """ """
        return {
            "modes": FindThresholdModes(lasing_modes_id=self.lasing_modes_id),
            "competition_matrix": ComputeModeCompetitionMatrix(
                lasing_modes_id=self.lasing_modes_id
            ),
        }

    def run(self):
        """ """
        modes_df = load_modes(self.input()["modes"].path)
        mode_competition_matrix = pd.read_hdf(
            self.input()["competition_matrix"].path, "mode_competition_matrix"
        ).to_numpy()
        modes_df = compute_modal_intensities(modes_df, self.D0_max, mode_competition_matrix)

        save_modes(modes_df, filename=self.output().path)

    def output(self):
        """ """
        return luigi.LocalTarget(self.add_lasing_modes_id(self.modal_intensities_path))
