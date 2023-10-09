"""
Collection of objects which, as a group, represent some part of a
structure.
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
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from .ops import element
from . import obj_collections


nparr = npt.NDArray[np.float64]


@dataclass
class ComponentAssembly:
    """
    A component assembly represents some part of a structure and holds
    various lower-level elements such as nodes and beamcolumn
    elements.

    Attributes:
      uid: Unique identifier of the component assembly
      parent_collection: The collection of
        elements to which the component assembly belongs.
      component_purpose: The functionality of the component assembly
      external_nodes: the external nodes to which the
        component assembly is connected.
        these nodes should exist as part of a level.
      internal_nodes: internal nodes that are
        required for the connectivity of the elements of the component
        assembly.
        these nodes only exist as part of the component assembly.
      elements:
        Collection containing the elements that are part of the
        component assembly.

    """

    uid: int
    parent_collection: obj_collections.Collection[int, ComponentAssembly]
    component_purpose: str
    external_nodes: obj_collections.NodeCollection = field(init=False)
    internal_nodes: obj_collections.NodeCollection = field(init=False)
    elements: (
        obj_collections.CollectionWithConnectivity[int, element.Element]
    ) = field(init=False)

    def __post_init__(self):
        self.external_nodes = obj_collections.NodeCollection(self)
        self.internal_nodes = obj_collections.NodeCollection(self)
        self.elements = (
            obj_collections.CollectionWithConnectivity(self)
        )

    def __srepr__(self):
        """
        Short version of repr
        """
        return f"Component assembly, uid: {self.uid}"

    def __repr__(self):
        res = ""
        res += "Component assembly object\n"
        res += f"uid: {self.uid}\n"
        res += f"component_purpose: {self.component_purpose}\n"
        res += "External Nodes\n"
        for node in self.external_nodes.values():
            res += f"  {node.uid}, {node.coords}"
        return res

    def dict_of_elements(self):
        """
        Returns a dictionary of all element objects in the model.
        The keys are the uids of the objects.

        """

        res = {}
        for elm in self.elements.values():
            res[elm.uid] = elm
        return res

    def list_of_elements(self):
        """
        Returns a list of all element objects in the model.

        """

        return list(self.dict_of_elements().values())

    def element_connectivity(self):
        """
        Returns the connectivity of all elements. Elements are
        connected to external nodes. Each component assembly can be
        represented by a tuple of node uids of its connected nodes in
        ascending order. This method returns a dictionary having these
        tuples as keys, and the associated components as values.

        """

        res = {}
        elms = self.list_of_elements()
        for elm in elms:
            uids = [nd.uid for nd in elm.nodes]
            uids.sort()
            uids_tuple = (*uids,)
            res[uids_tuple] = elm
        return res
