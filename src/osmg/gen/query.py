"""
Query objects.

Objects used to retrieve other objects, or information related to
them, based on certain criteria.

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

# pylint: disable=inconsistent-return-statements

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import numpy.typing as npt

from osmg import common
from osmg.line import Line
from osmg.ops import element

if TYPE_CHECKING:
    from osmg.component_assembly import ComponentAssembly
    from osmg.load_case import LoadCase
    from osmg.model import Model
    from osmg.ops.node import Node


nparr = npt.NDArray[np.float64]


@dataclass(repr=False)
class ElmQuery:
    """
    Retrieves nodes and component assemblies.

    Attributes:
    ----------
      model: Model to be searched.

    """

    model: Model

    def search_connectivity(self, nodes: list[Node]) -> ComponentAssembly | None:
        """
        Find component assembly objects.

        Finds component assembly objects based on the nodes the are
        connected to.

        Arguments:
          nodes: List containing nodes. If a component assembly is
            connected to those nodes, it is returned, otherwise the
            method returns None. If the component assembly is
            partially connected to the nodes (e.g. connected to them,
            but also to other nodes), it is not returned.

        Returns:
          The connectivity data.
        """
        uids = [node.uid for node in nodes]
        uids.sort()
        uids_tuple = (*uids,)
        conn_dict = self.model.component_connectivity()
        return conn_dict.get(uids_tuple)

    def search_node_lvl(
        self,
        x_loc: float,
        y_loc: float,
        lvl: int,
        z_loc: float | None = None,
        *,
        internal: bool = False,
    ) -> Node | None:
        """
        Look if a primary node exists at the specified location.

        Arguments:
          x_loc: x-coordinate.
          y_loc: y-coordinate.
          lvl: Key of the level to be searched.
          z_loc: z-coordinate.
          internal: Whether to include internal nodes in the search.

        Returns:
          The node if found.
        """
        lvls = self.model.levels
        level = lvls[lvl]
        res = None
        # check to see if node exists
        if z_loc:
            candidate_pt: nparr = np.array([x_loc, y_loc, z_loc])
            ndims = 3
        else:
            candidate_pt = np.array([x_loc, y_loc])
            ndims = 2
        nodes = level.nodes
        if internal:
            for comp in level.components.values():
                nodes.update(comp.internal_nodes)
        for other_node in nodes.values():
            other_pt: nparr = np.array(other_node.coords[:ndims])
            if np.linalg.norm(candidate_pt - other_pt) < common.EPSILON:
                res = other_node
                break
        return res

    def retrieve_components_from_nodes(
        self, nodes: list[Node], lvl_uid: int | None = None
    ) -> dict[int, ComponentAssembly]:
        """
        Retrieve component assemblies.

        Retrieves component assemblies if at least one of their
        external nodes matches the given list of nodes.

        Arguments:
          nodes: List of primary nodes.
          lvl_uid: ID of the level to be searched. If None is
            specified, the method searches all levels.

        Returns:
          The retrieved components.
        """
        retrieved_components = {}
        if lvl_uid:
            level = self.model.levels[lvl_uid]
            candidate_components = level.components.values()
        else:
            candidate_components = self.model.list_of_components()
        given_node_uids = [n.uid for n in nodes]
        for component in candidate_components:
            accept = False
            external_nodes = component.external_nodes.values()
            for node in external_nodes:
                if node.uid in given_node_uids:
                    accept = True
                    continue
            if accept:
                retrieved_components[component.uid] = component
        return retrieved_components

    def retrieve_component_from_nodes(
        self, nodes: list[Node], lvl_uid: int | None = None
    ) -> ComponentAssembly | None:
        """
        Retrieve a single component assembly.

        Retrieves a single component assembly if all of its external
        nodes match the given list of nodes.

        Arguments:
          nodes: List of primary nodes.
          lvl_uid: Key of the level to be searched. If None is
            specified, the method searches all levels.

        Returns:
          The retrieved component.
        """
        retrieved_component = None
        if lvl_uid:
            level = self.model.levels[lvl_uid]
            candidate_components = level.components.values()
        else:
            candidate_components = self.model.list_of_components()
        given_node_uids = [n.uid for n in nodes]
        for component in candidate_components:
            reject = False
            external_nodes = component.external_nodes.values()
            for node in external_nodes:
                if node.uid not in given_node_uids:
                    reject = True
                    continue
            if not reject:
                retrieved_component = component
        return retrieved_component

    def retrieve_component(  # noqa: C901
        self, x_loc: float, y_loc: float, lvl: int
    ) -> ComponentAssembly | None:
        """
        Retrieve a component assembly.

        Retrieves a component assembly of a level if any of its
        line elements passes through the specified point.

        Arguments:
          x_loc: x-coordinate
          y_loc: y-coordinate
          lvl: Key of the level to be searched.

        Returns:
        -------
          The first element found, None otherwise.

        """
        level = self.model.levels[lvl]
        for component in level.components.values():
            if len(component.external_nodes) != 2:  # noqa: PLR2004
                continue
            line_elems: list[
                element.TrussBar | element.ElasticBeamColumn | element.DispBeamColumn
            ] = []
            for elm in component.elements.values():
                if isinstance(elm, element.TrussBar):
                    line_elems.append(elm)
                if isinstance(elm, element.ElasticBeamColumn):
                    line_elems.append(elm)
                if isinstance(elm, element.DispBeamColumn):
                    line_elems.append(elm)

            for elm in line_elems:
                if isinstance(elm, element.TrussBar):
                    p_i = np.array(elm.nodes[0].coords)
                    p_j = np.array(elm.nodes[1].coords)
                else:
                    p_i = np.array(elm.nodes[0].coords) + elm.geomtransf.offset_i
                    p_j = np.array(elm.nodes[1].coords) + elm.geomtransf.offset_j
                if np.linalg.norm(p_i[0:2] - p_j[0:2]) < common.EPSILON:
                    if (
                        np.linalg.norm(np.array((x_loc, y_loc)) - p_i[0:2])
                        < common.EPSILON
                    ):
                        return component
                else:
                    line = Line('', p_i[0:2], p_j[0:2])
                    line.intersects_pt(np.array((x_loc, y_loc)))
                    if line.intersects_pt(np.array((x_loc, y_loc))):
                        return component
        return None


@dataclass
class LoadCaseQuery:
    """
    Retrieve information associated with load cases.

    Attributes:
    ----------
      model: Model object to be searched.
      loadcase: LoadCase object to be searched.

    """

    model: Model
    loadcase: LoadCase

    def level_masses(self) -> nparr:
        """Return the total mass of each level."""
        mdl = self.model
        num_lvls = len(mdl.levels)
        distr = np.zeros(num_lvls)
        for key, lvl in mdl.levels.items():
            for node in lvl.nodes.values():
                mass = self.loadcase.node_mass[node.uid]
                distr[key] += mass.val[0]

            for component in lvl.components.values():
                for node in component.internal_nodes.values():
                    mass = self.loadcase.node_mass[node.uid]
                    distr[key] += mass.val[0]
        for uid, node in self.loadcase.parent_nodes.items():
            distr[uid] += self.loadcase.node_mass[node.uid].val[0]
        return distr
