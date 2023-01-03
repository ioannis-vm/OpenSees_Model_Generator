"""
Model Generator for OpenSees ~ design library
"""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import pandas as pd
from ..solver import Analysis
from .basic_forces import basic_forces
from ..model import Model
from ..solver import ModalResponseSpectrumAnalysis

nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class LoadCombination:
    """
    Load combinations
    stuff in the first list are added.
    stuff inside the sub-list are enveloped
    """

    mdl: Model
    combo: dict[str, list[tuple[float, Analysis, str]]] = field(
        default_factory=dict
    )

    def envelope_basic_forces(self, elm, num_points):
        """
        Calculates the envelope of the basic forces for the
        given load combination
        """
        df_min = pd.DataFrame(
            np.full((num_points, 6), np.inf),
            columns=["nx", "qy", "qz", "tx", "mz", "my"],
        )
        df_max = pd.DataFrame(
            np.full((num_points, 6), -np.inf),
            columns=["nx", "qy", "qz", "tx", "mz", "my"],
        )
        for component_to_envelope in self.combo.values():
            df_tot = pd.DataFrame(
                np.full((num_points, 6), 0.00),
                columns=["nx", "qy", "qz", "tx", "mz", "my"],
            )
            for component_to_add in component_to_envelope:
                factor, anl, case_name_str = component_to_add
                res = basic_forces(anl, case_name_str, 0, elm, num_points)
                assert isinstance(res, pd.DataFrame)
                dframe = res * factor
                df_tot += dframe
            df_min[df_min > df_tot] = df_tot
            df_max[df_max < df_tot] = df_tot

        return df_min, df_max

    def envelope_node_displacement(self, node):
        """
        Calculates the enveloped node displacement
        """
        disp_min: nparr = np.full(6, np.inf)
        disp_max: nparr = np.full(6, -np.inf)
        for component_to_envelope in self.combo.values():
            disp_tot = np.full(6, 0.00)
            for component_to_add in component_to_envelope:
                factor, anl, case_name_str = component_to_add
                if isinstance(anl, ModalResponseSpectrumAnalysis):
                    disp: nparr = np.array(
                        anl.combined_node_disp(node.uid) * factor
                    )
                else:
                    disp = np.array(
                        anl.results[case_name_str].node_displacements[
                            node.uid
                        ][0]
                    )
                disp_tot += disp
            disp_min[disp_min > disp_tot] = disp_tot[disp_min > disp_tot]
            disp_max[disp_max < disp_tot] = disp_tot[disp_max < disp_tot]

        return disp_min, disp_max

    def envelope_node_displacement_diff(self, node_i, node_j):
        """
        Calculates the enveloped displacement difference between
        two nodes
        """
        disp_min = np.full(6, np.inf)
        disp_max = np.full(6, -np.inf)
        for component_to_envelope in self.combo.values():
            disp_tot = np.full(6, 0.00)
            for component_to_add in component_to_envelope:
                factor, anl, case_name_str = component_to_add
                if isinstance(anl, ModalResponseSpectrumAnalysis):
                    disp: nparr = np.array(
                        anl.combined_node_disp_diff(node_i.uid, node_j.uid)
                        * factor
                    )
                else:
                    disp_i: nparr = np.array(
                        anl.results[case_name_str].node_displacements[
                            node_i.uid
                        ][0]
                    )
                    disp_j: nparr = np.array(
                        anl.results[case_name_str].node_displacements[
                            node_j.uid
                        ][0]
                    )
                    disp = disp_i - disp_j
                disp_tot += disp
            disp_min[disp_min > disp_tot] = disp_tot[disp_min > disp_tot]
            disp_max[disp_max < disp_tot] = disp_tot[disp_max < disp_tot]

        return disp_min, disp_max
