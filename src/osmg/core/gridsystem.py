"""Grid system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from osmg.geometry.line import Line

if TYPE_CHECKING:
    from osmg.core.common import numpy_array

T = TypeVar('T')


@dataclass
class BaseGridSystem(Generic[T]):
    """Base class for grid systems.

    Attributes:
        grids: A dictionary of grids, parameterized by the type `T`.
        levels: A dictionary of levels, each represented by a float.
    """

    grids: dict[str, T] = field(default_factory=dict)
    levels: dict[str, float] = field(default_factory=dict)

    def add_level(self, name: str, elevation: float) -> None:
        """Add a level to the dictionary.

        Args:
            name: The name of the level.
            elevation: The elevation of the level.
        """
        self.levels[name] = elevation

    def get_level_elevation(self, level_name: str) -> float:
        """Retrieve the elevation of a level by its name.

        Args:
            level_name: The name of the level.

        Returns:
            The elevation of the level.

        Raises:
            ValueError: If the specified level does not exist.
        """
        if level_name not in self.levels:
            msg = f"Level '{level_name}' does not exist."
            raise ValueError(msg)
        return self.levels[level_name]


@dataclass
class GridSystem(BaseGridSystem[Line]):
    """Grid system using dictionaries for grids and levels.

    Attributes:
        grids: A dictionary of `Line` objects representing the grids.
        levels: A dictionary of level elevations.
    """

    def add_grid(self, name: str, start: numpy_array, end: numpy_array) -> None:
        """Add a grid to the dictionary.

        Args:
            name: The name of the grid.
            start: Starting coordinates of the grid line.
            end: Ending coordinates of the grid line.
        """
        self.grids[name] = Line(tag=name, start=start, end=end)

    def get_intersection_coordinates(
        self, grid_name_1: str, grid_name_2: str
    ) -> numpy_array | None:
        """Find the intersection of two grids.

        Args:
            grid_name_1: The name of the first grid.
            grid_name_2: The name of the second grid.

        Returns:
            The coordinates of the intersection point if it exists.

        Raises:
            ValueError: If either of the specified grids does not exist.
        """
        grid_1 = self.grids.get(grid_name_1)
        grid_2 = self.grids.get(grid_name_2)

        if grid_1 is None or grid_2 is None:
            msg = f"Grids '{grid_name_1}' and '{grid_name_2}' must exist."
            raise ValueError(msg)

        return grid_1.intersect(grid_2)


@dataclass
class GridSystem2D(BaseGridSystem[float]):
    """Grid system for 2D models using dictionaries for grids and levels.

    Attributes:
        grids: A dictionary of grid coordinates as floats.
        levels: A dictionary of level elevations.
    """

    def add_grid(self, name: str, location: float) -> None:
        """Add a grid to the dictionary.

        Args:
            name: The name of the grid.
            location: The x or y coordinate of the grid line.
        """
        self.grids[name] = location

    def get_grid_location(self, grid_name: str) -> float:
        """Retrieve the location of a grid by its name.

        Args:
            grid_name: The name of the grid.

        Returns:
            The location of the grid.

        Raises:
            ValueError: If the specified grid does not exist.
        """
        if grid_name not in self.grids:
            msg = f"Grid '{grid_name}' does not exist."
            raise ValueError(msg)
        return self.grids[grid_name]
