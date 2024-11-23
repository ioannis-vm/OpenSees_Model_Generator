"""Defines Model objects."""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

# pylint: disable=W1512

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from osmg.core.gridsystem import GridSystem, GridSystem2D
from osmg.core.osmg_collections import CollectionWithConnectivity, NodeCollection
from osmg.creators.uid import UIDGenerator

nparr = npt.NDArray[np.float64]

if TYPE_CHECKING:
    from osmg.core.component_assemblies import ComponentAssembly


@dataclass(repr=False)
class Model:
    """
    Base Model object.

    A general representation of a structural model.

    Attributes:
        name (str): Name of the model.
        grid_system (GridSystem): Grid system of the model.
        uid_generator (UIDGenerator): Object for generating unique IDs.
    """

    name: str
    uid_generator: UIDGenerator = field(default_factory=UIDGenerator)
    nodes: NodeCollection = field(default_factory=NodeCollection)
    components: CollectionWithConnectivity[ComponentAssembly] = field(
        default_factory=CollectionWithConnectivity
    )

    def bounding_box(self, padding: float) -> tuple[nparr, nparr]:
        """Obtain the axis-aligned bouding box of the building."""
        p_min = np.full(3, np.inf)
        p_max = np.full(3, -np.inf)
        for node in self.list_of_primary_nodes():
            point: nparr = np.array(node.coords)
            p_min = np.minimum(p_min, point)
            p_max = np.maximum(p_max, point)
        p_min -= np.full(3, padding)
        p_max += np.full(3, padding)
        # type hints gone mad  >.<   ...
        return p_min, p_max  # type:ignore

    def reference_length(self) -> float:
        """
        Obtain the largest bounding box dimension.

        (used in graphics)
        """
        p_min, p_max = self.bounding_box(padding=0.00)
        ref_len = np.max(p_max - p_min)
        return ref_len

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f'~~~ Model Object: {self.name} ~~~'


@dataclass(repr=False)
class Model2D(Model):
    """
    2D Model object.

    A 2D model representation.

    Attributes:
        name (str): Name of the model.
        grid_system (GridSystem2D): Grid system for the 2D model.
    """

    ndf: Literal[2, 3] = field(default=3)
    grid_system: GridSystem2D = field(default_factory=GridSystem2D)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f'~~~ 2D Model Object: {self.name} ~~~'


@dataclass(repr=False)
class Model3D(Model):
    """
    3D Model object.

    A 3D model representation.

    Attributes:
        name (str): Name of the model.
        grid_system (GridSystem): Grid system for the 3D model.
    """

    ndf: Literal[3, 6] = field(default=6)
    grid_system: GridSystem = field(default_factory=GridSystem)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f'~~~ 3D Model Object: {self.name} ~~~'
