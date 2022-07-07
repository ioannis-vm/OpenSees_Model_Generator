"""
Model Generator for OpenSees ~ generic
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
from typing import TYPE_CHECKING
from typing import Optional
from dataclasses import dataclass
import numpy as np
from ..line import Line
from ..load_case import LoadCase
if TYPE_CHECKING:
    from ..component_assembly import ComponentAssembly
    from ..ops.node import Node
    from ..model import Model


@dataclass(repr=False)
class ElmQuerry:
    """
    Used by all component generators
    """
    model: Model

    def search_connectivity(self, nodes: list[Node]) -> Optional[ComponentAssembly]:
        """
        find component assembly based on connectivity
        """
        uids = [node.uid for node in nodes]
        uids.sort()
        uids_tuple = (*uids,)
        if uids_tuple in self.model.component_connectivity:
            return self.model.component_connectivity[uids_tuple]

    def search_node_lvl(self, x: float, y: float, lvl: int):
        """
        Looks if a node exists at the given location.
        """
        lvls = self.model.levels
        level = lvls.registry[lvl]
        # check to see if node exists
        node = level.nodes.search_xy(x, y)
        return node

    def retrieve_component(self, x, y, lvl):
        """
        """
        level = self.model.levels.registry[lvl]
        for component in level.components.registry.values():
            if len(component.external_nodes.registry) != 2:
                continue
            # consider only horizontal components
            n_i = list(component.external_nodes.registry.values())[0]
            n_j = list(component.external_nodes.registry.values())[1]
            if n_i.coords[2] != n_j.coords[2]:
                continue
            line_elems = []
            line_elems.extend(component.elastic_beamcolumn_elements
                              .registry.values())
            line_elems.extend(component.force_beamcolumn_elements
                              .registry.values())
            for elm in line_elems:
                p_i = (np.array(elm.eleNodes[0].coords)
                       + elm.geomtransf.offset_i)
                p_j = (np.array(elm.eleNodes[1].coords)
                       + elm.geomtransf.offset_j)
                line = Line('', p_i[0:2], p_j[0:2])
                try:
                   line.intersects_pt(np.array((x, y)))
                except:
                    import pdb
                    pdb.set_trace()
                    from ..graphics.preprocessing_3D import show
                    show(component.parent_model)
                if line.intersects_pt(np.array((x, y))):
                    return component

@dataclass
class LoadCaseQuerry:
    model: Model
    loadcase: LoadCase

    def level_masses(self):
        mdl = self.model
        num_lvls = len(mdl.levels.registry)
        distr = np.zeros(num_lvls)
        for key, lvl in mdl.levels.registry.items():
            for component in  lvl.components.registry.values():
                for node in component.external_nodes.registry.values():
                    mass = self.loadcase.node_mass.registry[node.uid]
                    distr[key] += mass.total()[0]
                for node in component.internal_nodes.registry.values():
                    mass = self.loadcase.node_mass.registry[node.uid]
                    distr[key] += mass.total()[0]
        return distr
