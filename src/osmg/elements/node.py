"""Defines :obj:`~osmg.elements.node.Node` objects."""

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
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Self

from osmg.graphics.visibility import NodeVisibility

if TYPE_CHECKING:
    from osmg.creators.uid import UIDGenerator

@dataclass
@total_ordering
class Node:
    """
    OpenSees node.

    https://openseespydoc.readthedocs.io/en/latest/src/node.html?highlight=node

    Attributes:
    ----------
        uid_generator: Unique ID generator object.
        coords: List of node coordinates.
        uid: Unique ID of the node, assigned using the generator object.
        restraint: List of boolean values identifying whether the
          corresponding DOF is restrained.

    """

    coords: list[float]
    uid_generator: UIDGenerator
    uid: list[bool] = field(init=False)
    restraint: list[bool] = field(init=False)
    visibility: NodeVisibility = field(default_factory=NodeVisibility)

    def __post_init__(self) -> None:
        """Post-initialization."""
        self.restraint = [False] * 6
        self.uid = self.uid_generator.new(self)
        

    def __le__(self, other: Self) -> bool:
        """
        Less or equal determination rule.

        Returns:
          The outcome of the less or equal operation.
        """
        return self.uid <= other.uid

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
        res = ''
        res += 'Node object\n'
        res += f'  uid: {self.uid}\n'
        res += f'  coords: {self.coords}\n'
        res += f'  restraint: {self.restraint}\n'
        return res
