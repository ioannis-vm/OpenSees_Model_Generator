"""Defines the RDAnalyzer object."""

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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from osmg import common, load_case
from osmg.ops.node import Node

if TYPE_CHECKING:
    from osmg.level import Level
    from osmg.load_case import LoadCase


@dataclass(repr=False)
class RDAnalyzer:
    """
    Rigid Diaphragm Analyzer object. Used to apply rigid diaphragm
    constraints and optionally gather mass from the nodes of the level
    to the primary nodes

    """

    parent_loadcase: LoadCase
    parent_level: Level

    def run(self, gather_mass: bool) -> None:
        """
        Applies rigid diaphragm constraints and optionally gathers
        mass from the nodes of the level to the primary nodes

        """
        lvl = self.parent_level
        loadcase = self.parent_loadcase
        # gather all nodes of the level
        nodes: list[Node] = []
        nodes.extend(lvl.nodes.values())
        for component in lvl.components.values():
            nodes.extend(component.internal_nodes.values())

        # determine center of mass
        masses = []
        coords = []
        for node in nodes:
            mass = loadcase.node_mass[node.uid].val[0]
            masses.append(mass)
            coords.append(node.coords[0:2])
        total_mass = sum(masses)
        if np.abs(total_mass) <= common.EPSILON:
            msg = "Can't generate parent node without defined mass."
            raise ValueError(msg)
        coords_np = np.array(coords) * np.column_stack((masses, masses))
        center = np.sum(coords_np / total_mass, axis=0)
        parent_node = Node(
            lvl.parent_model.uid_generator.new('node'),
            [*center, lvl.elevation],
        )
        parent_node.restraint = [False, False, True, True, True, False]
        loadcase.parent_nodes[lvl.uid] = parent_node
        loadcase.node_loads[parent_node.uid] = load_case.PointLoadMass()
        loadcase.node_mass[parent_node.uid] = load_case.PointLoadMass()
        if gather_mass:
            # gather all mass from the nodes of the level to the
            # parent node
            level_nodes: list[Node] = []
            level_nodes.extend(lvl.nodes.values())
            for comp in lvl.components.values():
                level_nodes.extend(comp.internal_nodes.values())
            for node in level_nodes:
                mass = loadcase.node_mass[node.uid].val[0]
                loadcase.node_mass[node.uid].val = np.zeros(6)
                dist2 = (
                    np.linalg.norm(
                        np.array(node.coords[0:2])
                        - np.array(parent_node.coords[0:2])
                    )
                    ** 2
                )
                loadcase.node_mass[parent_node.uid].val += np.array(
                    (mass, mass, 0.00, 0.00, 0.00, mass * dist2)
                )
