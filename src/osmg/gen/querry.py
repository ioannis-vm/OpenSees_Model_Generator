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
import numpy.typing as npt
from ..line import Line
from ..load_case import LoadCase
from .. import common
if TYPE_CHECKING:
    from ..component_assembly import ComponentAssembly
    from ..ops.node import Node
    from ..model import Model


nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class ElmQuerry:
    """
    Used by all component generators
    """
    model: Model

    def search_connectivity(
            self,
            nodes: list[Node]) -> Optional[ComponentAssembly]:
        """
        find component assembly based on connectivity
        """
        res = None
        uids = [node.uid for node in nodes]
        uids.sort()
        uids_tuple = (*uids,)
        conn_dict = self.model.component_connectivity()
        if uids_tuple in conn_dict:
            res = conn_dict[uids_tuple]
        return res

    def search_node_lvl(self,
                        x: float, y: float,
                        lvl: int,
                        z: Optional[float] = None
                        ) -> Optional[Node]:
        """
        Looks if a node exists at the given location.
        """
        lvls = self.model.levels
        level = lvls.registry[lvl]
        res = None
        # check to see if node exists
        if z:
            candidate_pt: nparr = np.array([x, y, z])
        else:
            candidate_pt = np.array(
                [x, y, level.elevation])
        for other_node in level.nodes.registry.values():
            other_pt: nparr = np.array(other_node.coords)
            if np.linalg.norm(candidate_pt - other_pt) < common.EPSILON:
                res = other_node
                break
        return res

    def retrieve_components_from_nodes(
            self,
            nodes: list[Node],
            lvl_uid: Optional[int] = None) -> dict[int, ComponentAssembly]:
        retrieved_components = {}
        if lvl_uid:
            level = self.model.levels.registry[lvl_uid]
            candidate_components = level.components.registry.values()
        else:
            candidate_components = self.model.list_of_components()
        given_node_uids = [n.uid for n in nodes]
        for component in candidate_components:
            accept = False
            external_nodes = component.external_nodes.registry.values()
            for node in external_nodes:
                if node.uid in given_node_uids:
                    accept = True
                    continue
            if accept:
                retrieved_components[component.uid] = component
        return retrieved_components


    def retrieve_component_2node_horizontal(self, x, y, lvl):
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
            line_elems.extend(component.disp_beamcolumn_elements
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
