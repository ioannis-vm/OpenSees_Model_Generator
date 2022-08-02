"""
Model Generator for OpenSees ~ node generator
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
from dataclasses import dataclass
from ..ops.node import Node
if TYPE_CHECKING:
    from .model import Model


@dataclass(repr=False)
class NodeGenerator:
    """
    Introduces node objects to a model.
    """
    model: Model

    def add_node_lvl(self, x_loc: float, y_loc: float, lvl: int) -> Node:
        """
        Adds a node at the specified coordinates and level
        """
        lvls = self.model.levels
        level = lvls[lvl]
        node = Node(
            uid=self.model.uid_generator.new('node'),
            coords=[x_loc, y_loc, level.elevation])
        level.nodes.add(node)
        return node

    def add_node_lvl_xyz(self, x_loc: float, y_loc: float,
                         z_loc: float, lvl: int) -> Node:
        """
        Adds a node at the specified coordinates and level
        having a custo elevation
        """
        lvls = self.model.levels
        level = lvls[lvl]
        node = Node(
            uid=self.model.uid_generator.new('node'),
            coords=[x_loc, y_loc, z_loc])
        level.nodes.add(node)
        return node

    def add_node_active(self, x_loc, y_loc):
        """
        Adds a node at the specified coordinates to all active levels.
        """
        lvls = self.model.levels
        assert lvls.active, 'No active levels.'
        for key in lvls.active:
            self.add_node_lvl(x_loc, y_loc, key)
