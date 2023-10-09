"""
Defines :obj:`~osmg.load_case.LoadCase` objects.
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

# pylint: disable=W1512

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing import Union
from typing import Optional
import numpy as np
import numpy.typing as npt
import pandas as pd
from . import transformations
from . import obj_collections
from .preprocessing.tributary_area_analysis import TributaryAreaAnaysis
from .preprocessing.rigid_diaphragm import RDAnalyzer
from .ops import element

if TYPE_CHECKING:
    from .model import Model
    from .ops.node import Node

nparr = npt.NDArray[np.float64]


@dataclass
class PointLoadMass:
    """
    Point load/mass object. Global coordinate system.

    Attributes:
        val: Value for each DOF.

    """

    val: nparr = field(default_factory=lambda: np.zeros(shape=6))

    def add(self, load: nparr) -> None:
        """
        Adds some quantity to the existing quantity.

        Example:
            >>> from osmg.load_case import PointLoadMass
            >>> load = np.array([1., 2., 3., 4., 5., 6.])
            >>> point_load = PointLoadMass()
            >>> point_load.add(load)
            >>> point_load.val
            array([1., 2., 3., 4., 5., 6.])
            >>> point_load.add(load)
            >>> point_load.val
            array([ 2.,  4.,  6.,  8., 10., 12.])

        """

        self.val += load

    def __repr__(self):
        res = ""
        res += "Point Load (or mass) object\n"
        res += "Components: (global system)\n"
        res += f"val: {self.val}\n"
        return res


@dataclass(repr=False)
class LineElementUDL:
    """
    Line element uniformly distributed load object.

    """

    parent_load_case: LoadCase
    parent_line_element: Union[
        element.ElasticBeamColumn, element.DispBeamColumn]
    val: nparr = field(default_factory=lambda: np.zeros(shape=3))

    def __repr__(self):
        res = ""
        res += "LineElementUDL object\n"
        res += f"parent_line_element.uid: {self.parent_line_element.uid}\n"
        res += "Components:\n"
        res += f"  val: {self.val}\n"
        return res

    def add_glob(self, udl: nparr) -> None:
        """
        Adds a uniformly distributed load to the existing udl The load
        is defined with respect to the global coordinate system of the
        building, and it is converted to the local coordinate system
        prior to adding it.

        Arguments:
          udl: Array of size 3 containing
            components of the uniformly distributed load that is
            applied to the clear length of the element, acting on
            the global x, y, and z directions, in the direction of
            the global axes.

        Returns:
          None

        """

        # STOP! if the element has the Corotational transformation, we
        # can't apply a UDL on it. We need to lump the provided UDL to
        # its external nodes.  Since the Corotational transformation
        # also does not support rigid end offests, that lumping
        # process is always valid without requiring any special
        # transformation.
        elm = self.parent_line_element
        assert isinstance(
            elm, (element.ElasticBeamColumn, element.DispBeamColumn))
        if elm.geomtransf.transf_type == "Corotational":
            elm_len = elm.clear_length()
            force = udl * elm_len / 2.00
            node_i_uid = elm.nodes[0].uid
            node_j_uid = elm.nodes[1].uid
            lcase = self.parent_load_case
            lcase.node_loads[node_i_uid].add(
                np.concatenate((force, np.zeros(3)))
            )
            lcase.node_loads[node_j_uid].add(
                np.concatenate((force, np.zeros(3)))
            )
        else:
            transf_mat = transformations.transformation_matrix(
                self.parent_line_element.geomtransf.x_axis,
                self.parent_line_element.geomtransf.y_axis,
                self.parent_line_element.geomtransf.z_axis,
            )
            udl_local = transf_mat @ udl
            self.val += udl_local

    def to_global(self) -> nparr:
        """
        Returns the quantity expressed in the global coordinate
        system.

        """
        udl = self.val
        transf_mat = transformations.transformation_matrix(
            self.parent_line_element.geomtransf.x_axis,
            self.parent_line_element.geomtransf.y_axis,
            self.parent_line_element.geomtransf.z_axis,
        )
        return transf_mat.T @ udl


@dataclass(repr=False)
class LoadCase:
    """
    Load Case object.
    Load cases contain information related to the specified loads,
    mass, parent nodes and rigid diaphragm constraints, etc.
    Analysis objects can use multiple load cases.
    Load combination objects utilize load cases as well.

    Attributes:
      name: A string representing the name of the load case.
      parent_model: A Model object representing the parent model of
        the load case.
      node_loads: A collection of PointLoadMass objects, indexed by
        the node id.
      node_mass: A collection of PointLoadMass objects, indexed by the
        node id.
      line_element_udl: A collection of LineElementUDL objects,
        indexed by the element id.
      tributary_area_analysis: A collection of TributaryAreaAnaysis
        objects, indexed by the level id.
      parent_nodes: A dictionary of Node objects, indexed by the node
        id.

    """

    name: str
    parent_model: Model
    node_loads: obj_collections.Collection[int, PointLoadMass] = field(
        init=False
    )
    node_mass: obj_collections.Collection[int, PointLoadMass] = field(
        init=False
    )
    line_element_udl: obj_collections.Collection[int, LineElementUDL] = field(
        init=False
    )
    tributary_area_analysis: obj_collections.Collection[
        int, TributaryAreaAnaysis
    ] = field(init=False)
    parent_nodes: dict[int, Node] = field(default_factory=dict)
    equaldof: Optional[int] = field(default=None)

    def __post_init__(self):
        self.node_loads = obj_collections.Collection(self)
        self.node_mass = obj_collections.Collection(self)
        self.line_element_udl = obj_collections.Collection(self)
        self.tributary_area_analysis = obj_collections.Collection(self)
        # initialize loads and mass for each node and element
        for node in self.parent_model.list_of_all_nodes():
            self.node_loads[node.uid] = PointLoadMass()
            self.node_mass[node.uid] = PointLoadMass()
        for elm in self.parent_model.list_of_elements():
            # only proceed for certain elements
            if not isinstance(elm, (
                    element.ElasticBeamColumn, element.DispBeamColumn)):
                continue
            self.line_element_udl[elm.uid] = LineElementUDL(
                self, elm
            )
        # initialize tributary area analysis for each level
        for lvlkey, lvl in self.parent_model.levels.items():
            self.tributary_area_analysis[lvlkey] = TributaryAreaAnaysis(
                self, lvl
            )

    def rigid_diaphragms(
            self, level_uids: list[int],
            gather_mass: bool = False) -> None:
        """
        Processes the geometry of the given levels and applies rigid
        diaphragm constraints.

        """

        for lvl_uid in level_uids:
            lvl = self.parent_model.levels[lvl_uid]
            rda = RDAnalyzer(self, lvl)
            rda.run(gather_mass)

    def number_of_free_dofs(self):
        """
        Calculates the number of free DOFS of the model, considering
        all (potentially) defined constraints, restraints and parent
        nodes.

        """

        mdl = self.parent_model
        all_nodes = mdl.dict_of_all_nodes()
        # parent_nodes = {
        #     node.uid: node
        #     for node in self.parent_nodes.values()}
        # all_nodes.update(parent_nodes)
        free_dofs = pd.DataFrame(
            np.ones((len(all_nodes), 6), dtype=int),
            index=all_nodes.keys(),
            columns=[1, 2, 3, 4, 5, 6],
        ).sort_index(axis="index")

        # consider the restraints
        def restraints(row, all_nodes):
            uid = row.name
            node = all_nodes[uid]
            restraints = node.restraint
            row[restraints] = 0

        free_dofs.apply(restraints, axis=1, args=(all_nodes,))
        # consider the constraints
        num_diaphragms = 0
        for uid in self.parent_nodes:
            num_diaphragms += 1
            lvl = mdl.levels[uid]
            constrained_nodes = [
                n for n in lvl.nodes.values() if n.coords[2] == lvl.elevation
            ]
            for constrained_node in constrained_nodes:
                free_dofs.loc[constrained_node.uid, :] = (0, 0, 1, 1, 1, 0)
        return int(free_dofs.to_numpy().sum() + num_diaphragms * 3)

    def __repr__(self):
        res = ""
        res += "LoadCase object\n"
        res += f"name: {self.name}\n"
        res += f"parent_model: {self.parent_model.name}\n"
        return res
