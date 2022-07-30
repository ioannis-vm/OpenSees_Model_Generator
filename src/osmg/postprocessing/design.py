"""
Model Generator for OpenSees ~ design library
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
from ..solver import Analysis
from .basic_forces import basic_forces
from ..model import Model
from ..solver import ModalResponseSpectrumAnalysis
import numpy as np
import pandas as pd


case_name = str
component_name = str

@dataclass(repr=False)
class LoadCombination:
    """
    Load combinations
    stuff in the first list are added.
    stuff inside the sub-list are enveloped
    """
    mdl: Model
    combo: dict[list[component_name, list[tuple[float, Analysis, case_name]]]] = \
        field(default_factory=dict)

    def envelope_basic_forces(self, elm, num_points):
        df_min = pd.DataFrame(
            np.full((num_points, 6), np.inf),
            columns=['nx', 'qy', 'qz', 'tx', 'mz', 'my'])
        df_max = pd.DataFrame(
            np.full((num_points, 6), -np.inf),
            columns=['nx', 'qy', 'qz', 'tx', 'mz', 'my'])
        for component_to_envelope in self.combo.values():
            df_tot = pd.DataFrame(
                np.full((num_points, 6), 0.00),
                columns=['nx', 'qy', 'qz', 'tx', 'mz', 'my'])
            for component_to_add in component_to_envelope:
                factor, anl, case_name_str = component_to_add
                df = basic_forces(
                    anl, case_name_str, 0, elm, num_points) * factor
                df_tot += df
            df_min[df_min > df_tot] = df_tot
            df_max[df_max < df_tot] = df_tot

        return df_min, df_max

    def envelope_node_displacement(self, node):
        disp_min = np.full(6, np.inf)
        disp_max = np.full(6, -np.inf)
        for component_to_envelope in self.combo.values():
            disp_tot = np.full(6, 0.00)
            for component_to_add in component_to_envelope:
                factor, anl, case_name_str = component_to_add
                if isinstance(anl, ModalResponseSpectrumAnalysis):
                    disp = np.array(anl.combined_node_disp(node.uid) * factor)
                else:
                    disp = np.array(anl.results[case_name_str].node_displacements.registry[node.uid][0])
                disp_tot += disp
            disp_min[disp_min > disp_tot] = disp[disp_min > disp_tot]
            disp_max[disp_max < disp_tot] = disp[disp_max < disp_tot]

        return disp_min, disp_max

    def envelope_node_displacement_diff(self, node_i, node_j):
        disp_min = np.full(6, np.inf)
        disp_max = np.full(6, -np.inf)
        for component_to_envelope in self.combo.values():
            disp_tot = np.full(6, 0.00)
            for component_to_add in component_to_envelope:
                factor, anl, case_name_str = component_to_add
                if isinstance(anl, ModalResponseSpectrumAnalysis):
                    disp = np.array(anl.combined_node_disp_diff(node_i.uid, node_j.uid) * factor)
                else:
                    disp_i = np.array(anl.results[case_name_str].node_displacements.registry[node_i.uid][0])
                    disp_j = np.array(anl.results[case_name_str].node_displacements.registry[node_j.uid][0])
                    disp = disp_i - disp_j
                disp_tot += disp
            disp_min[disp_min > disp_tot] = disp[disp_min > disp_tot]
            disp_max[disp_max < disp_tot] = disp[disp_max < disp_tot]

        return disp_tot, disp_tot

