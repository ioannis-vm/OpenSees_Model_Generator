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
from typing import TYPE_CHECKING, Any, List, TypeVar

import numpy as np
import numpy.typing as npt

from . import common
from .ops import element, node

if TYPE_CHECKING:
    from osmg.ops.node import Node

nparr = npt.NDArray[np.float64]

# pylint: disable=invalid-name
TK = TypeVar('TK')
TV = TypeVar('TV')
# pylint: enable=invalid-name


@dataclass(repr=False)
class Collection(dict[TK, TV]):
    """
    Collection of objects.

    Attributes:
    ----------
        parent: Object to which the Collection belongs.

    Example:
        >>> # obj_collections require parent objects to which they belong
        >>> parent = 52
        >>> my_collection = Collection(parent=parent)
        >>> my_collection.parent
        52
        >>> # now `my_collection` knows that it belongs to the object `52`.
        >>>
        >>>
        >>>
        >>> # Collection.add method
        >>>
        >>> from osmg.ops.node import Node
        >>> my_collection = Collection(parent=None)
        >>> len(my_collection)
        0
        >>> # something to add:
        >>> new_node = Node(uid=0, coords=[0.00, 0.00])
        >>> # add the node to the collection
        >>> my_collection.add(new_node)
        >>> # it has been added:
        >>> len(my_collection)
        1
        >>> # and it can be accessed by its uid:
        >>> id(new_node) == id(my_collection[0])
        True
        >>> # adding something without a uid fails:
        >>> my_collection.add(42)
        Traceback (most recent call last):
            ...
        KeyError: 'Object does not have a uid attribute'
        >>> # adding an object with the same uid fails:
        >>> my_collection.add(new_node)
        Traceback (most recent call last):
            ...
        KeyError: 'uid 0 already exists'
        >>>
        >>>
        >>>
        >>> # obj_collections.retrieve_by_attr method:
        >>>
        >>> from osmg.ops.section import Section
        >>> sec_collection = Collection(parent=None)
        >>> sec_collection.add(Section(name='sec_1', uid=0))
        >>> sec_collection.add(Section(name='sec_2', uid=1))
        >>> sec_collection.retrieve_by_attr('name', 'sec_1')
        Section(name='sec_1', uid=0)
        >>> # we 'll use this later:
        >>>
        >>>
        >>>
        >>> # obj_collections.__srepr__ method:
        >>>
        >>> sec_collection.__srepr__()
        '[Collection of 2 items]'

    """

    parent: Any = field(repr=False)

    def add(self, obj: object) -> None:
        """
        Add an object to the collection.

        The object needs to have a unique id attribute, `uid`.

        Arguments:
          obj: Object to be added.

        Raises:
          KeyError: If the object does not have a uid attribute.
          KeyError: If the object already exists.
        """
        if not hasattr(obj, 'uid'):
            msg = 'Object does not have a uid attribute'
            raise KeyError(msg)
        if obj.uid in self:
            msg = f'uid {obj.uid} already exists'
            raise KeyError(msg)
        self[obj.uid] = obj

    def retrieve_by_attr(self, attr: str, val: object) -> object:
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
        res += f'Parent object: {self.parent}\n'
        res += f'Registry size: {len(self)}\n'
        return res


@dataclass(repr=False)
class CollectionActive(Collection[TK, TV]):
    """
    Collection with support for currently active objects.

    Attributes:
    ----------
      active: List of unique IDs that correspond to the active
        objects.

    Example:
        >>> from osmg.level import Level
        >>> my_collection = CollectionActive(parent=None)
        >>> my_collection.add(Level(
        ...     parent_model=None,
        ...     uid=0,
        ...     elevation=0.00))
        >>> my_collection.add(Level(
        ...     parent_model=None,
        ...     uid=1,
        ...     elevation=1.00))
        >>>
        >>>
        >>>
        >>> # CollectionActive.set_active method:
        >>>
        >>> # set them both as active
        >>> my_collection.set_active([0, 1])
        >>> my_collection.active
        [0, 1]
        >>> # if uid is not present, it fails
        >>> my_collection.set_active([2])
        Traceback (most recent call last):
            ....
        KeyError: 'uid 2 not present in collection.'

    """

    active: list[TK] = field(default_factory=list)

    def set_active(self, uids: list[TK]) -> None:
        """
        Set the active objects.

        Arguments:
            uids: uids of the objects to set as active

        Raises:
          KeyError: IF the uid is not present in the collection.
        """
        for uid in uids:
            if uid not in self:
                msg = f'uid {uid} not present in collection.'
                raise KeyError(msg)
        self.active = uids

    def set_active_all(self) -> None:
        """
        Set the active objects.

        Arguments:
            uids: uids of the objects to set as active

        """
        self.active = []
        for key in self:
            self.active.append(key)


@dataclass(repr=False)
class NodeCollection(Collection[int, node.Node]):
    """
    Node collection.

    Attributes:
    ----------
        parent: Object to which the Collection belongs.

    Example:
        >>> from osmg.ops.node import Node
        >>> from osmg.level import Level
        >>> level = Level(parent_model=None, uid=0, elevation=0.00)
        >>>
        >>> my_collection = NodeCollection(parent=level)
        >>>
        >>> n_1 = Node(uid=0, coords=[0.00, 0.00, 0.00])
        >>> n_2 = Node(uid=1, coords=[1.00, 1.00, 0.00])
        >>> my_collection.add(n_1)
        >>> my_collection.add(n_2)
        >>>
        >>>
        >>>
        >>> # NodeCollection.search_xy method:
        >>>
        >>> retrieved_node = my_collection.search_xy(1.00, 1.00)
        >>> id(retrieved_node) == id(n_2)  # should be the same object
        True

    """

    named_contents: dict[str, node.Node] = field(default_factory=dict)

    def search_xy(self, x_coord: float, y_coord: float) -> Node | None:
        """
        Obtain the node that occupies a given point if it exists.

        Returns:
          The node.
        """
        candidate_pt: nparr = np.array([x_coord, y_coord, self.parent.elevation])
        for other_node in self.values():
            other_pt: nparr = np.array(other_node.coords)
            if np.linalg.norm(candidate_pt - other_pt) < common.EPSILON:
                return other_node
        # indent the following line once to the right, and you'll be
        # spending a couple of nights trying to figure out why there
        # are so many duplicate nodes
        return None


@dataclass(repr=False)
class CollectionWithConnectivity(Collection[TK, TV]):
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
        uids = [nd.uid for nd in obj.nodes]
        uids.sort()
        uids_tuple = (*uids,)  # noqa: F841
        # TODO(JVM): Fix this region.
        super().add(obj)
