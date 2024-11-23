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
from typing import TYPE_CHECKING, Generic, TypeVar

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
class NamedCollection(Collection[int, TV], Generic[TV]):
    """
    A collection that allows assigning and retrieving objects by name.

    Attributes:
    ----------
    named_contents : dict[str, TV]
        A mapping of names to objects for easy retrieval.
    """

    named_contents: dict[str, TV] = field(default_factory=dict)

    def add(self, obj: TV, name: str | None = None) -> None:
        """
        Add a named object to the collection.

        The Node needs to have a unique id attribute, `uid`.
        Parameters
        ----------
        obj : TV
            Object to be added.
        name : str
            The name to assign to the object.

        """
        super().add(obj)
        if name:
            self.named_contents[name] = obj

    def retrieve_by_name(self, name: str) -> TV:
        """
        Retrieve an object by its name.

        Parameters
        ----------
        name : str
            The name of the object to retrieve.

        Returns:
        -------
        TV
            The object associated with the given name.

        Raises:
        ------
        KeyError
            If no object is found with the given name.
        """
        if name not in self.named_contents:
            raise KeyError(f"No object found with the name '{name}'.")
        return self.named_contents[name]

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
        -------
        str
            The string representation of the object.
        """
        res = ''
        res += 'NamedCollection Object\n'
        res += f'ID: {id(self)}\n'
        res += f'Number of objects: {len(self)}\n'
        res += f'Named objects: {list(self.named_contents.keys())}\n'
        return res


@dataclass(repr=False)
class CollectionWithConnectivity(Collection[int, TV]):
    """Collection of elements for which connectivity matters."""

    def add(self, obj: object) -> None:
        """
        Add an element to the collection.

        Arguments:
          obj: Object to be added.
        """
        super().add(obj)


@dataclass(repr=False)
class NodeCollection(NamedCollection[node.Node]):
    """
    A collection specifically for managing named Nodes.

    Methods:
    -------
    search_by_coordinates:
        Search for a node at the specified coordinates.
    """

    def search_by_coordinates(
        self, coordinates: tuple[float, ...], radius=common.EPSILON
    ) -> Node | None:
        """
        Obtain the node that occupies a given point if it exists.

        Parameters
        ----------
        coordinates : tuple[float, ...]
            The coordinates to search at.

        Returns:
        -------
        Node | None
            The node at the given coordinates, or None if no node is found.
        """
        candidate_pt: nparr = np.array(coordinates)
        for other_node in self.values():
            other_pt: nparr = np.array(other_node.coordinates)
            if np.linalg.norm(candidate_pt - other_pt) < radius:
                return other_node
        return None
