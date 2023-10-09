"""
Collections are designated containers of objects of a particular type.

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
from typing import Any
from typing import List
from typing import TypeVar
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from . import common
from .ops import node
from .ops import element


nparr = npt.NDArray[np.float64]

# pylint: disable=invalid-name
TK = TypeVar("TK")
TV = TypeVar("TV")
# pylint: enable=invalid-name


@dataclass(repr=False)
class Collection(dict[TK, TV]):
    """
    Collection of objects.

    Attributes:
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

    def add(self, obj: Any) -> None:
        """
        Add an object to the collection. The object needs to have
        a unique id attribute, `uid`.

        Arguments:
          obj: Object to be added.

        """
        if not hasattr(obj, "uid"):
            raise KeyError("Object does not have a uid attribute")
        if obj.uid in self:
            raise KeyError(f"uid {obj.uid} already exists")
        self[obj.uid] = obj

    def retrieve_by_attr(self, attr: str, val: Any) -> Any:
        """
        Retrieve an object from the collection based on an attribute
        value. If more than one instances satisfy the criterion, the
        function returns the first occurrence.

        Arguments:
          attr: The name of the attribute.
          val: The value that the attribute should have.

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
            raise ValueError(
                f'Item having the value "{val}" '
                f'in the attribute "{attr}" not found in collection.')
        return res

    def __srepr__(self):
        """
        Concise version of `repr`.

        """
        return f"[Collection of {len(self)} items]"

    def __repr__(self):
        res = ""
        res += "Collection Object\n"
        res += f"ID: {id(self)}\n"
        res += f"Parent object: {self.parent}\n"
        res += f"Registry size: {len(self)}\n"
        return res


@dataclass(repr=False)
class CollectionActive(Collection[TK, TV]):
    """
    Collection with support for currently active objects.

    Attributes:
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

    active: List[TK] = field(default_factory=list)

    def set_active(self, uids: List[TK]) -> None:
        """
        Sets the active objects.

        Arguments:
            uids: uids of the objects to set as active

        """
        for uid in uids:
            if uid not in self:
                raise KeyError(f"uid {uid} not present in collection.")
        self.active = uids

    def set_active_all(self):
        """
        Sets the active objects.

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

    def search_xy(self, x_coord, y_coord):
        """
        Returns the node that occupies a given point if it exists
        """

        candidate_pt: nparr = np.array(
            [x_coord, y_coord, self.parent.elevation]
        )
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
    Collection of elements for which it is important to consider their
    connectivity.

    Attributes:
        parent: Object to which the Collection belongs.

    """

    named_contents: dict[str, element.ElasticBeamColumn] = field(
        default_factory=dict
    )

    def add(self, obj):
        """
        Adds an element to the collection.
        The method also checks to see if an object having the same
        connectivity exists in the collection, and raises an error if
        it does.

        Arguments:
          obj: Object to be added.

        """
        uids = [nd.uid for nd in obj.nodes]
        uids.sort()
        uids_tuple = (*uids,)
        if uids_tuple in obj.parent_component.element_connectivity():
            raise ValueError("This should never happen!")
        super().add(obj)
