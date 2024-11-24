"""Defines Model objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from osmg.core.gridsystem import GridSystem, GridSystem2D
from osmg.core.osmg_collections import ComponentAssemblyCollection, NodeCollection
from osmg.creators.uid import UIDGenerator

nparr = npt.NDArray[np.float64]


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
    components: ComponentAssemblyCollection = field(
        default_factory=ComponentAssemblyCollection
    )

    def bounding_box(self, padding: float) -> tuple[nparr, nparr]:
        """
        Obtain the axis-aligned bounding box of the building.

        Returns:
          Bounding box.
        """
        p_min = np.full(3, np.inf)
        p_max = np.full(3, -np.inf)
        for node in list(self.nodes.values()):
            point: nparr = np.array(node.coordinates)
            p_min = np.minimum(p_min, point)
            p_max = np.maximum(p_max, point)
        p_min -= np.full(3, padding)
        p_max += np.full(3, padding)
        return p_min, p_max

    def reference_length(self) -> float:
        """
        Obtain the largest bounding box dimension.

        (used in graphics)

        Returns:
          The largest dimension.
        """
        p_min, p_max = self.bounding_box(padding=0.00)
        return float(np.max(p_max - p_min))

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
