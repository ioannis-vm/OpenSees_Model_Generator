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
from typing import Optional
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

    def add_node_lvl(self, x: float, y: float, lvl: int) -> Node:
        """

        """
        lvls = self.model.levels
        level = lvls.registry[lvl]
        node = Node(
            uid=self.model.uid_generator.new('node'),
            coords=[x, y, level.elevation])
        level.nodes.add(node)
        return node

    def add_node_active(self, x, y):
        """

        """
        lvls = self.model.levels
        assert lvls.active, 'No active levels.'
        for key in lvls.active:
            self.add_node_lvl(x, y, key)

