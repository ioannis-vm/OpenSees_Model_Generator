"""
Model Generator for OpenSees ~ collections
Collections are designated containers of objects of a particular type.
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from __future__ import annotations
from typing import Any
from typing import TypeVar
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from . import common
from .ops import node
from .ops import element
from .ops.element import ZeroLength


nparr = npt.NDArray[np.float64]

TK = TypeVar('TK')
TV = TypeVar('TV')


@dataclass
class Collection(dict[TK, TV]):
    """
    Collection of objects.
    Attributes:
        parent (Any)
    """
    parent: Any = field(repr=False)

    def add(self, obj):
        """
        Add an object to the collection
        """
        if not hasattr(obj, 'uid'):
            raise KeyError('Object does not have a uid attribute')
        if obj.uid in self:
            raise KeyError('Object uid already exists')
        self[obj.uid] = obj

    def retrieve_by_attr(self, attr: Any, val: Any) -> Any:
        res = None
        for thing in self.values():
            if hasattr(thing, attr):
                other_val = getattr(thing, attr)
                if other_val == val:
                    res = thing
        return res

    def __srepr__(self):
        """
        Short version of repr
        """
        return f'[Collection of {len(self)} items]'

    def __repr__(self):
        res = ''
        res += 'Collection Object\n'
        res += f'ID: {id(self)}\n'
        res += f'Parent object: {self.parent}\n'
        res += f'Registry size: {len(self)}\n'
        return res


@dataclass(repr=False)
class CollectionActive(Collection[TK, TV]):
    """
    Collection with support for currently active objects
    """
    active: list[TK] = field(default_factory=list)

    def set_active(self, uids: list[TK]):
        """
        Sets the active objects.
        Args:
            uids (list[int]): uids of the objects to set as active
        """
        for uid in uids:
            assert uid in self
        self.active = uids

    def set_active_all(self):
        """
        Sets the active objects.
        Args:
            uids (list[int]): uids of the objects to set as active
        """
        self.active = []
        for key in self:
            self.active.append(key)


@dataclass(repr=False)
class NodeCollection(Collection[int, node.Node]):
    """
    Node collection.
    Attributes:
        parent (Any)
    """
    named_contents: dict[str, node.Node] = field(default_factory=dict)

    def search_xy(self, x_coord, y_coord):
        """
        Returns the node that occupies a given point if it exists
        """

        candidate_pt: nparr = np.array(
            [x_coord, y_coord, self.parent.elevation])
        for other_node in self.values():
            other_pt: nparr = np.array(other_node.coords)
            if np.linalg.norm(candidate_pt - other_pt) < common.EPSILON:
                return other_node
        # indent the following line once to the right, and you'll be
        # spending a couple of nights trying to figure out why there
        # are so many duplicate nodes
        return None


@dataclass(repr=False)
class elasticBeamColumnCollection(Collection[int, element.elasticBeamColumn]):
    """
    Elastic beam column element collection.
    Attributes:
        parent (Any)
    """
    named_contents: dict[
        str, element.elasticBeamColumn] = field(
            default_factory=dict)

    def add(self, elm):
        # update component assembly connectivity
        uids = [nd.uid for nd in elm.eleNodes]
        uids.sort()
        uids_tuple = (*uids,)
        if uids_tuple in elm.parent_component.element_connectivity():
            raise ValueError('This should never happen!')
        super().add(elm)


@dataclass(repr=False)
class dispBeamColumnCollection(Collection[int, element.dispBeamColumn]):
    """
    Force-based beam column element collection.
    Attributes:
        parent (Any)
    """
    named_contents: dict[
        str, element.dispBeamColumn] = field(
            default_factory=dict)

    def add(self, elm):
        # update component assembly connectivity
        uids = [nd.uid for nd in elm.eleNodes]
        uids.sort()
        uids_tuple = (*uids,)
        if uids_tuple in elm.parent_component.element_connectivity():
            raise ValueError('This should never happen!')
        super().add(elm)


@dataclass(repr=False)
class zerolengthCollection(Collection[int, ZeroLength]):
    named_contents: dict[str, ZeroLength] = field(default_factory=dict)

    def add(self, elm):
        # update component assembly connectivity
        uids = [nd.uid for nd in elm.eleNodes]
        uids.sort()
        uids_tuple = (*uids,)
        if uids_tuple in elm.parent_component.element_connectivity():
            raise ValueError('This should never happen!')
        super().add(elm)
