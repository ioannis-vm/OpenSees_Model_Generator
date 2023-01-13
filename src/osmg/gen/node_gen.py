"""
Objects that introduce nodes to a model.

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
from typing import TYPE_CHECKING
from dataclasses import dataclass
from ..ops.node import Node

if TYPE_CHECKING:
    from .model import Model


@dataclass(repr=False)
class NodeGenerator:
    """
    NodeGenerator objects introduce node objects to a model.

    Attributes:
      model: Model to act upon.

    """

    model: Model

    def add_node_lvl(self, x_loc: float, y_loc: float, lvl: int) -> Node:
        """
        Adds a node at the specified coordinates and level.

        Arguments:
          x_loc: x coordinate of the node.
          y_loc: y coordinate of the node.
          lvl: Level at which the node should be added.

        Returns:
          The node object that was added.

        Example:
          >>> from osmg.model import Model
          >>> from osmg.gen.node_gen import NodeGenerator
          >>> model = Model('test_model')
          >>> model.add_level(0, 0.00)
          >>> generator = NodeGenerator(model)
          >>> generator.add_node_lvl(2.00, 3.00, 0)
          Node object
            uid: 0
            coords: [2.0, 3.0, 0.0]
            restraint: [False, False, False, False, False, False]
          <BLANKLINE>

        """
        lvls = self.model.levels
        level = lvls[lvl]
        node = Node(
            uid=self.model.uid_generator.new("node"),
            coords=[x_loc, y_loc, level.elevation],
        )
        level.nodes.add(node)
        return node

    def add_node_lvl_xyz(
        self, x_loc: float, y_loc: float, z_loc: float, lvl: int
    ) -> Node:
        """
        Adds a node at the specified coordinates and level having a
        custom elevation.

        Arguments:
            x_loc: x coordinate of the node.
            y_loc: y coordinate of the node.
            z_loc: Elevation of the node.
            lvl: Level uid at which the node should be added.

        Returns:
            The node object that was added.

        Example:
            >>> from osmg.model import Model
            >>> from osmg.gen.node_gen import NodeGenerator
            >>> model = Model('test_model')
            >>> model.add_level(0, 0.00)
            >>> generator = NodeGenerator(model)
            >>> generator.add_node_lvl_xyz(2.00, 3.00, 4.00, 0)
            Node object
              uid: 0
              coords: [2.0, 3.0, 4.0]
              restraint: [False, False, False, False, False, False]
            <BLANKLINE>

        """
        lvls = self.model.levels
        level = lvls[lvl]
        node = Node(
            uid=self.model.uid_generator.new("node"),
            coords=[x_loc, y_loc, z_loc],
        )
        level.nodes.add(node)
        return node

    def add_node_active(self, x_loc, y_loc):
        """
        Adds a node[/s] at the specified coordinates to all active levels.

        Arguments:
            x_loc: x coordinate of the node[/s].
            y_loc: y coordinate of the node[/s].

        Example:
            >>> from osmg.model import Model
            >>> from osmg.gen.node_gen import NodeGenerator
            >>> model = Model('test_model')
            >>> model.add_level(0, 0.00)
            >>> model.add_level(1, 1.00)
            >>> model.add_level(2, 2.00)
            >>> model.levels.set_active([1, 2])
            >>> generator = NodeGenerator(model)
            >>> generator.add_node_active(2.00, 3.00)
            >>> model.levels[0].nodes.__srepr__()
            '[Collection of 0 items]'
            >>> model.levels[1].nodes.__srepr__()
            '[Collection of 1 items]'
            >>> model.levels[2].nodes.__srepr__()
            '[Collection of 1 items]'

        """
        lvls = self.model.levels
        assert lvls.active, "No active levels."
        for key in lvls.active:
            self.add_node_lvl(x_loc, y_loc, key)
