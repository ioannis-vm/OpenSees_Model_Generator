"""Grid system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from osmg.geometry.line import Line

if TYPE_CHECKING:
    from osmg.core.common import numpy_array


T = TypeVar('T')


class LevelWrapper:
    """A wrapper for a level that allows chained navigation.

    Attributes:
        system: The grid system that this level belongs to.
        level_name: The name of the level.
    """

    def __init__(self, system: BaseGridSystem, level_name: str) -> None:  # type: ignore
        """
        Initialize a LevelWrapper.

        Args:
            system: The grid system containing the levels.
            level_name: The name of the level to wrap.
        """
        self.system = system
        self.level_name = level_name

    def previous(self) -> LevelWrapper:
        """
        Navigate to the previous level in a circular fashion.

        Returns:
            A LevelWrapper for the previous level.
        """
        previous_level = self.system.get_previous_level(self.level_name)
        return LevelWrapper(self.system, previous_level)

    def next(self) -> LevelWrapper:
        """
        Navigate to the next level in a circular fashion.

        Returns:
            A LevelWrapper for the next level.
        """
        next_level = self.system.get_next_level(self.level_name)
        return LevelWrapper(self.system, next_level)

    def elevation(self) -> float:
        """
        Get the elevation of this level.

        Returns:
            The elevation of the level as a float.
        """
        return self.system.levels[self.level_name]

    def __repr__(self) -> str:
        """
        Represent the LevelWrapper as a string.

        Returns:
            A string representation of the LevelWrapper including the
            level name and elevation.
        """
        return f"LevelWrapper(level_name='{self.level_name}', elevation={self.elevation()})"


class GridWrapper(Generic[T]):
    """A wrapper for a grid that allows chained navigation."""

    def __init__(self, system: BaseGridSystem, grid_name: str) -> None:  # type: ignore
        """
        Initialize a GridWrapper.

        Args:
            system: The grid system containing the grids.
            grid_name: The name of the grid to wrap.
        """
        self.system = system
        self.grid_name: str = grid_name

    def previous(self) -> GridWrapper[T]:
        """
        Navigate to the previous grid in a circular fashion.

        Returns:
            A GridWrapper for the previous grid.
        """
        previous_grid = self.system.get_previous_grid(self.grid_name)
        return GridWrapper(self.system, previous_grid)

    def next(self) -> GridWrapper[T]:
        """
        Navigate to the next grid in a circular fashion.

        Returns:
            A GridWrapper for the next grid.
        """
        next_grid = self.system.get_next_grid(self.grid_name)
        return GridWrapper(self.system, next_grid)

    def data(self) -> T:
        """
        Get the data associated with this grid.

        Returns:
            The data of the grid.
        """
        return self.system.grids[self.grid_name]  # type: ignore

    def __repr__(self) -> str:
        """
        Represent the GridWrapper as a string.

        Returns:
            A string representation of the GridWrapper including the grid name and its data.
        """
        return f"GridWrapper(grid_name='{self.grid_name}', data={self.data()})"


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
        """
        Add a level to the dictionary.

        Args:
            name: The name of the level.
            elevation: The elevation of the level.
        """
        self.levels[name] = elevation

    def get_level(self, level_name: str) -> LevelWrapper:
        """
        Retrieve a LevelWrapper for a given level.

        Args:
            level_name: The name of the level to retrieve.

        Returns:
            A LevelWrapper object for the specified level.

        Raises:
            ValueError: If the specified level does not exist.
        """
        if level_name not in self.levels:
            msg = f"Level '{level_name}' does not exist."
            raise ValueError(msg)
        return LevelWrapper(self, level_name)

    def get_previous_level(self, level_name: str) -> str:
        """
        Get the name of the previous level in a circular fashion.

        Args:
            level_name: The name of the current level.

        Returns:
            The name of the previous level.

        Raises:
            ValueError: If the specified level does not exist.
        """
        level_names = list(self.levels.keys())
        if level_name not in level_names:
            msg = f'Level `{level_name}` does not exist.'
            raise ValueError(msg)
        index = level_names.index(level_name)
        return level_names[index - 1]

    def get_next_level(self, level_name: str) -> str:
        """
        Get the name of the next level in a circular fashion.

        Args:
            level_name: The name of the current level.

        Returns:
            The name of the next level.

        Raises:
            ValueError: If the specified level does not exist.
        """
        level_names = list(self.levels.keys())
        if level_name not in level_names:
            msg = f'Level `{level_name}` does not exist.'
            raise ValueError(msg)
        index = level_names.index(level_name)
        return level_names[(index + 1) % len(level_names)]

    def add_grid(self, name: str, data: T) -> None:
        """
        Add a grid to the dictionary.

        Args:
            name: The name of the grid.
            data: The data associated with the grid.
        """
        self.grids[name] = data

    def get_grid(self, grid_name: str) -> GridWrapper[T]:
        """
        Retrieve a GridWrapper for a given grid.

        Args:
            grid_name: The name of the grid to retrieve.

        Returns:
            A GridWrapper object for the specified grid.

        Raises:
            ValueError: If the specified grid does not exist.
        """
        if grid_name not in self.grids:
            msg = f"Grid '{grid_name}' does not exist."
            raise ValueError(msg)
        return GridWrapper(self, grid_name)

    def get_previous_grid(self, grid_name: str) -> str:
        """
        Get the name of the previous grid in a circular fashion.

        Args:
            grid_name: The name of the current grid.

        Returns:
            The name of the previous grid.

        Raises:
            ValueError: If the specified grid does not exist.
        """
        grid_names = list(self.grids.keys())
        if grid_name not in grid_names:
            msg = f'Grid `{grid_name}` does not exist.'
            raise ValueError(msg)
        index = grid_names.index(grid_name)
        return grid_names[index - 1]

    def get_next_grid(self, grid_name: str) -> str:
        """
        Get the name of the next grid in a circular fashion.

        Args:
            grid_name: The name of the current grid.

        Returns:
            The name of the next grid.

        Raises:
            ValueError: If the specified grid does not exist.
        """
        grid_names = list(self.grids.keys())
        if grid_name not in grid_names:
            msg = f'Grid `{grid_name}` does not exist.'
            raise ValueError(msg)
        index = grid_names.index(grid_name)
        return grid_names[(index + 1) % len(grid_names)]


@dataclass
class GridSystem(BaseGridSystem[Line]):
    """
    Grid system using dictionaries for grids and levels.

    Attributes:
        grids: A dictionary of `Line` objects representing the grids.
        levels: A dictionary of level elevations.
    """

    def add_grid(self, name: str, start: numpy_array, end: numpy_array) -> None:  # type: ignore
        """
        Add a grid to the dictionary.

        Args:
            name: The name of the grid.
            start: Starting coordinates of the grid line.
            end: Ending coordinates of the grid line.
        """
        self.grids[name] = Line(tag=name, start=start, end=end)

    def get_intersection_coordinates(
        self, grid_name_1: str, grid_name_2: str
    ) -> numpy_array | None:
        """
        Find the intersection of two grids.

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
    """
    Grid system for 2D models using dictionaries for grids and levels.

    Attributes:
        grids: A dictionary of grid coordinates as floats.
        levels: A dictionary of level elevations.
    """

    def add_grid(self, name: str, location: float) -> None:
        """
        Add a grid to the dictionary.

        Args:
            name: The name of the grid.
            location: The x or y coordinate of the grid line.
        """
        self.grids[name] = location

    def get_grid_location(self, grid_name: str) -> float:
        """
        Retrieve the location of a grid by its name.

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
