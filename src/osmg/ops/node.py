"""
Defines :obj:`~osmg.ops.node.Node` objects.
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

from dataclasses import dataclass, field
from functools import total_ordering
from ..graphics.visibility import NodeVisibility


@dataclass
@total_ordering
class Node:
    """
    OpenSees node
    https://openseespydoc.readthedocs.io/en/latest/src/node.html?highlight=node

    Attributes:
        uid: Unique ID of the node.
        coords: List of node coordinates.
        restraint: List of boolean values identifying whether the
          corresponding DOF is restrained.

    """

    uid: int
    coords: list[float]
    restraint: list[bool] = field(init=False)
    visibility: NodeVisibility = field(default_factory=NodeVisibility)

    def __post_init__(self):
        self.restraint = [False] * 6

    def __le__(self, other):
        return self.uid < other.uid

    def __repr__(self):
        res = ""
        res += "Node object\n"
        res += f"  uid: {self.uid}\n"
        res += f"  coords: {self.coords}\n"
        res += f"  restraint: {self.restraint}\n"
        return res
