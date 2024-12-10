"""Defines Model objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from osmg.core.common import NDM
from osmg.core.gridsystem import GridSystem, GridSystem2D
from osmg.core.osmg_collections import ComponentAssemblyCollection, NodeCollection
from osmg.creators.uid import UIDGenerator

if TYPE_CHECKING:
    from osmg.model_objects.node import Node

from osmg.core.common import numpy_array


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
    dimensionality: Literal['2D Truss', '2D Frame', '3D Truss', '3D Frame']
    uid_generator: UIDGenerator = field(default_factory=UIDGenerator)
    nodes: NodeCollection = field(default_factory=NodeCollection)
    components: ComponentAssemblyCollection = field(
        default_factory=ComponentAssemblyCollection
    )

    def bounding_box(self, padding: float) -> tuple[numpy_array, numpy_array]:
        """
        Obtain the axis-aligned bounding box of the building.

        Returns:
          Bounding box.
        """
        num_dimensions = NDM[self.dimensionality]
        p_min = np.full(num_dimensions, np.inf)
        p_max = np.full(num_dimensions, -np.inf)
        for node in list(self.nodes.values()):
            point: numpy_array = np.array(node.coordinates)
            p_min = np.minimum(p_min, point)
            p_max = np.maximum(p_max, point)
        p_min -= np.full(num_dimensions, padding)
        p_max += np.full(num_dimensions, padding)
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

    def get_all_nodes(
        self, ignore_by_tag: set[str] | None = None
    ) -> dict[int, Node]:
        """
        Get all nodes in the model.

        Params:
          ignore_by_tag: Set of tags of components to ignore.

        Returns:
          A dictionary with the nodes. Keys are their UIDs.
        """
        # primary nodes
        primary_nodes = self.nodes
        # internal nodes (of component assemblies)
        internal_nodes: dict[int, Node] = {}
        components = self.components.values()
        for component in components:
            if ignore_by_tag and component.tags & ignore_by_tag:
                continue
            internal_nodes.update(component.internal_nodes)

        all_nodes: dict[int, Node] = {}
        all_nodes.update(primary_nodes)
        all_nodes.update(internal_nodes)
        return all_nodes

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

    dimensionality: Literal['2D Truss', '2D Frame']
    grid_system: GridSystem2D = field(default_factory=GridSystem2D)

    def __post_init__(self) -> None:
        """
        Post-initialization.

        Raises:
          ValueError: If the `dimensionality` assignment is invalid.
        """
        if self.dimensionality not in {'2D Truss', '2D Frame'}:
            msg = f'Dimensionality `{self.dimensionality}` is not compatible with a `Model2D` object.'
            raise ValueError(msg)

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

    dimensionality: Literal['3D Truss', '3D Frame']
    grid_system: GridSystem = field(default_factory=GridSystem)

    def __post_init__(self) -> None:
        """
        Post-initialization.

        Raises:
          ValueError: If the `dimensionality` assignment is invalid.
        """
        if self.dimensionality not in {'3D Truss', '3D Frame'}:
            msg = f'Dimensionality `{self.dimensionality}` is not compatible with a `Model3D` object.'
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f'~~~ 3D Model Object: {self.name} ~~~'
