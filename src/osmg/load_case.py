"""
Model Generator for OpenSees ~ load case
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

# pylint: disable=W1512

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from . import transformations
from . import collections
from .preprocessing.tributary_area_analysis import TributaryAreaAnaysis
from .preprocessing.rigid_diaphragm import RDAnalyzer

if TYPE_CHECKING:
    from .model import Model
    from .ops.element import ElasticBeamColumn
    from .ops.node import Node

nparr = npt.NDArray[np.float64]


@dataclass
class PointLoadMass:
    """
    Point load/mass object. Global coordinate system.
    Attributes:
        val (nparr)
    """
    val: nparr = field(
        default_factory=lambda: np.zeros(shape=6))

    def add(self, load: nparr):
        """
        Adds some quantity to the existing quantity
        """
        self.val += load

    def __repr__(self):
        res = ''
        res += 'Point Load (or mass) object\n'
        res += 'Components: (global system)\n'
        res += f'val: {self.val}\n'
        return res


@dataclass(repr=False)
class LineElementUDL:
    """
    Line element uniformly distributed load object.
    """
    parent_load_case: LoadCase
    parent_line_element: ElasticBeamColumn
    val: nparr = field(
        default_factory=lambda: np.zeros(shape=3))

    def __repr__(self):
        res = ''
        res += 'LineElementUDL object\n'
        res += f'parent_line_element.uid: {self.parent_line_element.uid}\n'
        res += 'Components:\n'
        res += f'  val: {self.val}\n'
        return res

    def add_glob(self, udl: nparr):
        """
        Adds a uniformly distributed load
        to the existing udl
        The load is defined
        with respect to the global coordinate system
        of the building, and it is converted to the
        local coordinate system prior to adding it.
        Args:
            udl (nparr): Array of size 3 containing
                components of the uniformly distributed load that is
                applied to the clear length of the element, acting on
                the global x, y, and z directions, in the direction of
                the global axes.
        """
        # STOP! if the element has the Corotational transformation, we
        # can't apply a UDL on it. We need to lump the provided UDL to
        # its external nodes.  Since the Corotational transformation
        # also does not support rigid end offests, that lumping
        # process is always valid without requiring any special
        # transformation.
        elm = self.parent_line_element
        if elm.geomtransf.transf_type == 'Corotational':
            elm_len = elm.clear_length()
            force = udl * elm_len / 2.00
            node_i_uid = elm.nodes[0].uid
            node_j_uid = elm.nodes[1].uid
            lcase = self.parent_load_case
            lcase.node_loads[node_i_uid].add(
                np.concatenate((force, np.zeros(3))))
            lcase.node_loads[node_j_uid].add(
                np.concatenate((force, np.zeros(3))))
        else:
            transf_mat = transformations.transformation_matrix(
                self.parent_line_element.geomtransf.x_axis,
                self.parent_line_element.geomtransf.y_axis,
                self.parent_line_element.geomtransf.z_axis)
            udl_local = transf_mat @ udl
            self.val += udl_local

    def to_global(self):
        """
        Returns the quantity expressed in the global coordinate system
        """
        udl = self.val
        transf_mat = transformations.transformation_matrix(
            self.parent_line_element.geomtransf.x_axis,
            self.parent_line_element.geomtransf.y_axis,
            self.parent_line_element.geomtransf.z_axis)
        return transf_mat.T @ udl


@dataclass(repr=False)
class LoadCase:
    """
    Load Case object.
    Load cases contain information related to the specified loads,
    mass, parent nodes and rigid diaphragm constraints, etc.
    Analysis objects can use multiple load cases.
    Load combination objects utilize load cases as well.
    """
    name: str
    parent_model: Model
    node_loads: collections.Collection[int, PointLoadMass] = field(init=False)
    node_mass: collections.Collection[int, PointLoadMass] = field(init=False)
    line_element_udl: collections.Collection[
        int, LineElementUDL] = field(init=False)
    tributary_area_analysis: collections.Collection[
        int, TributaryAreaAnaysis] = field(init=False)
    parent_nodes: dict[int, Node] = field(default_factory=dict)

    def __post_init__(self):
        self.node_loads = collections.Collection(self)
        self.node_mass = collections.Collection(self)
        self.line_element_udl = collections.Collection(self)
        self.tributary_area_analysis = \
            collections.Collection(self)
        # initialize loads and mass for each node and element
        for node in self.parent_model.list_of_all_nodes():
            self.node_loads[node.uid] = PointLoadMass()
            self.node_mass[node.uid] = PointLoadMass()
        for line_element in (self.parent_model
                             .list_of_beamcolumn_elements()):
            self.line_element_udl[line_element.uid] = \
                LineElementUDL(self, line_element)
        # initialize tributary area analysis for each level
        for lvlkey, lvl in self.parent_model.levels.items():
            self.tributary_area_analysis[lvlkey] = \
                TributaryAreaAnaysis(self, lvl)

    def rigid_diaphragms(self, level_uids: list[int], gather_mass=False):
        """
        Processes the geometry of the given levels and applies rigid
        diaphragm constraints
        """
        for lvl_uid in level_uids:
            lvl = self.parent_model.levels[lvl_uid]
            rda = RDAnalyzer(self, lvl)
            rda.run(gather_mass)

    def __repr__(self):
        res = ''
        res += 'LoadCase object\n'
        res += f'name: {self.name}\n'
        res += f'parent_model: {self.parent_model.name}\n'
        return res
