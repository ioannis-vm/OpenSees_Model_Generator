"""
Model Generator for OpenSees ~ collections
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
from typing import TYPE_CHECKING
from typing import Any
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from . import common
if TYPE_CHECKING:
    from .level import Level
    from .ops.node import Node
    from .component_assembly import ComponentAssembly
    from .ops.element import LineElementUDL
    from .ops.element import elasticBeamColumn
    from .ops.element import dispBeamColumn
    from .physical_material import PhysicalMaterial
    from .ops.section import Section
    from .ops.uniaxialMaterial import uniaxialMaterial
    from .mesh import Mesh


nparr = npt.NDArray[np.float64]


@dataclass
class Collection:
    """
    Collection of objects.
    Attributes:
        parent (Any)
        registry (dict[int, Any])
    """
    parent: Any = field(repr=False)
    registry: dict[int, Any] = field(default_factory=dict)

    def add(self, obj):
        """
        Add an object to the collection
        """
        if not hasattr(obj, 'uid'):
            raise KeyError('Object does not have a uid attribute')
        if obj.uid in self.registry:
            raise KeyError('Object uid already exists')
        self.registry[obj.uid] = obj

    def retrieve_by_attr(self, attr: Any, val: Any) -> Any:
        res = None
        for thing in self.registry.values():
            if hasattr(thing, attr):
                other_val = getattr(thing, attr)
                if other_val == val:
                    res = thing
        return res

    def __repr__(self):
        return f'[Collection of {len(self.registry)} items]'

    def __str__(self):
        return f'[Collection of {len(self.registry)} items]'


@dataclass(repr=False)
class CollectionActive(Collection):
    """
    Collection with support for currently active objects
    """
    active: list[int] = field(default_factory=list)

    def set_active(self, uids: list[int]):
        """
        Sets the active objects.
        Args:
            uids (list[int]): uids of the objects to set as active
        """
        for uid in uids:
            assert uid in self.registry
        self.active = uids

    def set_active_all(self):
        """
        Sets the active objects.
        Args:
            uids (list[int]): uids of the objects to set as active
        """
        self.active = []
        for key in self.registry:
            self.active.append(key)


@dataclass(repr=False)
class LevelCollection(CollectionActive):
    """
    Level collection.
    Attributes:
        parent (Any)
        registry (dict[int, Level])
        active (dict[int])
    """
    registry: dict[int, Level] = field(default_factory=dict)


@dataclass(repr=False)
class ComponentCollection(Collection):
    """
    Component collection.
    Attributes:
        parent (Any)
        registry (dict[int, Level])
    """
    registry: dict[int, ComponentAssembly] = field(default_factory=dict)

@dataclass(repr=False)
class SectionCollection(Collection):
    """
    Section collection.
    Attributes:
        parent (Any)
        registry (dict[int, Section])
    """
    registry: dict[int, Section] = field(default_factory=dict)


@dataclass(repr=False)
class UniaxialMaterialCollection(Collection):
    """
    Uniaxial Material collection.
    Attributes:
        parent (Any)
        registry (dict[str, PhysicalMaterial])
    """
    registry: dict[str, uniaxialMaterial] = field(default_factory=dict)


@dataclass(repr=False)
class PhysicalMaterialCollection(Collection):
    """
    Physical Material collection.
    Attributes:
        parent (Any)
        registry (dict[str, PhysicalMaterial])
    """
    registry: dict[str, PhysicalMaterial] = field(default_factory=dict)


@dataclass(repr=False)
class NodeCollection(Collection):
    """
    Node collection.
    Attributes:
        parent (Any)
        registry (dict[int, Node])
    """
    registry: dict[int, Node] = field(default_factory=dict)

    def search_xy(self, x_coord, y_coord):
        """
        Returns the node that occupies a given point if it exists
        """

        candidate_pt: nparr = np.array(
            [x_coord, y_coord, self.parent.elevation])
        for other_node in self.registry.values():
            other_pt: nparr = np.array(other_node.coords)
            if np.linalg.norm(candidate_pt - other_pt) < common.EPSILON:
                return other_node
        # indent the following line once to the right, and you'll be
        # spending a couple of nights trying to figure out why there
        # are so many duplicate nodes...
        return None


@dataclass(repr=False)
class elasticBeamColumnCollection(Collection):
    """
    Elastic beam column element collection.
    Attributes:
        parent (Any)
        registry (dict[int, elasticBeamColumn])
    """
    registry: dict[int, elasticBeamColumn] = field(default_factory=dict)

    def add(self, elm):
        # update component assembly connectivity
        uids = [nd.uid for nd in elm.eleNodes]
        uids.sort()
        uids_tuple = (*uids,)
        if uids_tuple in elm.parent_component.element_connectivity():
            raise ValueError('This should never happen!')
        elm.parent_component.element_connectivity[uids_tuple] = elm
        super().add(elm)


@dataclass(repr=False)
class dispBeamColumnCollection(Collection):
    """
    Force-based beam column element collection.
    Attributes:
        parent (Any)
        registry (dict[int, elasticBeamColumn])
    """
    registry: dict[int, dispBeamColumn] = field(default_factory=dict)

    def add(self, elm):
        # update component assembly connectivity
        uids = [nd.uid for nd in elm.eleNodes]
        uids.sort()
        uids_tuple = (*uids,)
        if uids_tuple in elm.parent_component.element_connectivity():
            raise ValueError('This should never happen!')
        elm.parent_component.element_connectivity()[uids_tuple] = elm
        super().add(elm)


@dataclass(repr=False)
class LineElementUDLCollection(Collection):
    """
    Line element UDL collection.
    Attributes:
        parent (Any)
        registry (dict[int, elasticBeamColumn])
    """
    registry: dict[int, LineElementUDL] = field(default_factory=dict)


@dataclass(repr=False)
class NodePointLoadCollection(Collection):
    """
    Node load collection.
    Attributes:
        parent (Any)
        registry (dict[int, elasticBeamColumn])
    """
    registry: dict[int, PointLoad] = field(default_factory=dict)


@dataclass(repr=False)
class NodeMassCollection(Collection):
    """
    Node mass collection.
    Attributes:
        parent (Any)
        registry (dict[int, elasticBeamColumn])
    """
    registry: dict[int, PointMass] = field(default_factory=dict)


