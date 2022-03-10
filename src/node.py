"""
Model builder for OpenSees ~ Node
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSees_Model_Builder

from dataclasses import dataclass, field
from functools import total_ordering
from itertools import count
import numpy as np
from utility import common

node_ids = count(0)


@dataclass
@total_ordering
class Node:
    """
    Node object.
    Attributes:
        uniq_id (int): unique identifier
        coords (np.ndarray): Coordinates of the location of the node
        restraint_type (str): Can be either "free", "pinned", or "fixed".
                       It can also be "parent" or "internal", but
                       this is only specified for nodes that are made
                       automatically.
        mass (np.ndarray): Mass with respect to the global coordinate system
                           (shape = 3 for all nodes except parent nodes, where
                            inertia terms are also present, and shape = 6).
        load (np.ndarray): Load with respect to the global coordinate system.
        load_fl (np.ndarray): similar to load, coming from the floors.
        tributary_area: This attribute holds the results of tributary area
                        analysis done inside the `preprocess` method of the
                        `building` objects, and is used to store the floor
                        area that corresponds to that node (if beams  with
                        offsets are connected to it)
        column_above, column_below, beams, braces_above, braces_below, ...:
            Pointers to connected elements (only for primary nodes)
    """

    coords: np.ndarray
    restraint_type: str = field(default="free")
    mass: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=3), repr=False)
    load: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=6), repr=False)
    load_fl: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=6), repr=False)
    tributary_area: float = field(default=0.00, repr=False)
    column_above: 'LineElementSequence' = field(
        default=None, repr=False)
    column_below: 'LineElementSequence' = field(
        default=None, repr=False)
    beams: list['LineElementSequence'] = field(
        default_factory=list, repr=False)

    def __post_init__(self):
        self.uniq_id = next(node_ids)

    def __eq__(self, other):
        """
        For nodes, a fudge factor is used to
        assess equality.
        """
        p0 = np.array(self.coords)
        p1 = np.array(other.coords)
        return np.linalg.norm(p0 - p1) < common.EPSILON

    def __le__(self, other):
        d_self = self.coords[1] * common.ALPHA + self.coords[0]
        d_other = other.coords[1] * common.ALPHA + other.coords[0]
        return d_self <= d_other

    def load_total(self):
        """
        Returns the total load applied on the node,
        by summing up the floor's contribution to the
        generic component.
        """
        return self.load + self.load_fl


@dataclass
class Nodes:
    """
    This class is a collector for the nodes, and provides
    methods that perform operations using nodes.
    """

    node_list: list[Node] = field(default_factory=list)

    def add(self, node: Node):
        """
        Add a node in the nodes collection,
        if it does not already exist
        """
        if node not in self.node_list:
            self.node_list.append(node)
        else:
            raise ValueError('Node already exists: '
                             + repr(node))
        self.node_list.sort()

    def remove(self, node: Node):
        """
        Remove a node from the nodes collection,
        if it was there.
        """
        if node in self.node_list:
            self.node_list.remove(node)
        self.node_list.sort()

    def clear(self):
        """
        Removes all nodes
        """
        self.node_list = []

    def __repr__(self):
        out = str(len(self.node_list)) + " nodes\n"
        for node in self.node_list:
            out += repr(node) + "\n"
        return out

