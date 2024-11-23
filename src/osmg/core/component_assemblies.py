"""
Component assemblies.

Collections of objects which, as a group, represent some part of a
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
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from osmg.core import osmg_collections
from osmg.core.uid_object import UIDObject

if TYPE_CHECKING:
    from osmg.elements import element

nparr = npt.NDArray[np.float64]


@dataclass
class ComponentAssembly(UIDObject):
    """
    Component assembly object.

    A component assembly represents some part of a structure and holds
    various lower-level elements such as nodes and beam-column
    elements.

    Attributes:
        uid: Unique identifier of the component assembly.
        tags: List of tags, used to filter component assemblies if
            needed.
        external_nodes: The external nodes to which the component
            assembly is connected.
        internal_nodes: Internal nodes that are required for the
            connectivity of the elements within the component
            assembly. These nodes only exist as part of the component
            assembly.
        elements: Collection containing the elements that are part of
            the component assembly.
    """

    tags: set[str]
    external_nodes: osmg_collections.NodeCollection = field(
        default_factory=osmg_collections.NodeCollection
    )
    internal_nodes: osmg_collections.NodeCollection = field(
        default_factory=osmg_collections.NodeCollection
    )
    elements: osmg_collections.CollectionWithConnectivity[int, element.Element] = (
        field(default_factory=osmg_collections.CollectionWithConnectivity)
    )

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
        res = ''
        res += 'Component assembly object\n'
        res += f'uid: {self.uid}\n'
        res += 'External Nodes\n'
        for node in self.external_nodes.values():
            res += f'  {node.uid}, {node.coordinates}'
        return res

    def element_connectivity(
        self,
    ) -> dict[tuple[int, ...], element.Element]:
        """
        Element connectivity.

        Returns:
          The connectivity of all elements. Elements are connected to
          external nodes. Each component assembly can be represented
          by a tuple of node uids of its connected nodes in ascending
          order. This method returns a dictionary having these tuples
          as keys, and the associated components as values.
        """
        res = {}
        elms = self.elements.values()
        for elm in elms:
            uids = [x.uid for x in elm.nodes]
            uids.sort()
            uids_tuple = (*uids,)
            res[uids_tuple] = elm
        return res
