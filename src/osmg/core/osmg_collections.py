"""Collections are designated containers of objects of a particular type."""

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
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import numpy.typing as npt

from osmg.core import common
from osmg.elements import element, node

if TYPE_CHECKING:
    from osmg.elements.node import Node

nparr = npt.NDArray[np.float64]

# pylint: disable=invalid-name
TV = TypeVar('TV')
# pylint: enable=invalid-name


@dataclass(repr=False)
class Collection(dict[int, TV]):
    """Collection of objects."""

    def add(self, obj: TV) -> None:
        """
        Add an object to the collection.

        The object needs to have a unique id attribute, `uid`.

        Arguments:
          obj: Object to be added.

        Raises:
          ValueError: If the object does not have a uid attribute.
          ValueError: If the object already exists.
        """
        if not hasattr(obj, 'uid'):
            msg = 'Object does not have a uid attribute'
            raise ValueError(msg)
        if obj.uid in self:
            msg = f'uid {obj.uid} already exists'
            raise ValueError(msg)
        self[obj.uid] = obj

    def retrieve_by_attr(self, attr: str, val: object) -> TV:
        """
        Retrieve object by attribute value.

        Retrieve an object from the collection based on an attribute
        value. If more than one instances satisfy the criterion, the
        function returns the first occurrence.

        Arguments:
          attr: The name of the attribute.
          val: The value that the attribute should have.

        Returns:
          The retrieved object.

        Raises:
          ValueError: If the object is not found.
        """
        res = None
        for thing in self.values():
            found = False
            if hasattr(thing, attr):
                other_val = getattr(thing, attr)
                if other_val == val:
                    res = thing
                    found = True
                    break
            if found:
                break
        if res is None:
            msg = (
                f'Item having the value "{val}" '
                f'in the attribute "{attr}" not found in collection.'
            )
            raise ValueError(msg)
        return res

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
        res = ''
        res += 'Collection Object\n'
        res += f'ID: {id(self)}\n'
        res += f'Number of objects: {len(self)}\n'
        return res


@dataclass(repr=False)
class NodeCollection(Collection[int, node.Node]):
    """
    Node collection.

    Attributes:
      named_contents: Used to assign names to nodes for easy retrieval.

    """

    named_contents: dict[str, node.Node] = field(default_factory=dict)

    def add(self, node: Node, name: str) -> None:
        """
        Add a named Node to the collection.

        The Node needs to have a unique id attribute, `uid`.

        Arguments:
          node: Node to be added.

        Raises:
          ValueError: If the Node does not have a uid attribute.
          ValueError: If the Node already exists.
        """
        super().add(node)
        self.named_contents[name] = node

    def search_xy(self, coords: tuple[float, ...]) -> Node | None:
        """
        Obtain the node that occupies a given point if it exists.

        Returns:
          The node.
        """
        candidate_pt: nparr = np.array(coords)
        for other_node in self.values():
            other_pt: nparr = np.array(other_node.coords)
            if np.linalg.norm(candidate_pt - other_pt) < common.EPSILON:
                return other_node
        return None


@dataclass(repr=False)
class CollectionWithConnectivity(Collection[int, TV]):
    """
    Collection of elements for which connectivity matters.

    Attributes:
    ----------
        parent: Object to which the Collection belongs.

    """

    named_contents: dict[str, element.ElasticBeamColumn] = field(
        default_factory=dict
    )

    def add(self, obj: object) -> None:
        """
        Add an element to the collection.

        The method also checks to see if an object having the same
        connectivity exists in the collection, and raises an error if
        it does.

        Arguments:
          obj: Object to be added.
        """
        uids = [x.uid for x in obj.nodes]
        uids.sort()
        uids_tuple = (*uids,)  # noqa: F841
        # TODO(JVM): Fix this region.
        super().add(obj)
