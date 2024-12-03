"""Unit tests for gridsystem."""

import numpy as np
import pytest

from osmg.core.gridsystem import BaseGridSystem, GridSystem, GridSystem2D
from osmg.geometry.line import Line


class TestBaseGridSystem:
    """Tests for the BaseGridSystem class."""

    def test_add_and_retrieve_level(self) -> None:
        """Test adding and retrieving a level."""
        grid_system = BaseGridSystem[str]()
        grid_system.add_level('Ground Floor', 0.0)
        assert grid_system.get_level('Ground Floor').elevation() == 0.0  # type: ignore

    def test_retrieve_nonexistent_level(self) -> None:
        """Test retrieving a nonexistent level raises ValueError."""
        grid_system = BaseGridSystem[str]()
        with pytest.raises(ValueError, match="Level 'Basement' does not exist."):
            grid_system.get_level('Basement').elevation()  # type: ignore


class TestGridSystem:
    """Tests for the GridSystem class."""

    def test_add_and_retrieve_grid(self) -> None:
        """Test adding and retrieving a grid."""
        grid_system = GridSystem()
        start = np.array([0.0, 0.0])
        end = np.array([10.0, 0.0])
        grid_system.add_grid('Grid A', start, end)
        assert isinstance(grid_system.grids['Grid A'], Line)
        assert np.array_equal(grid_system.grids['Grid A'].start, start)
        assert np.array_equal(grid_system.grids['Grid A'].end, end)

    def test_find_intersection(self) -> None:
        """Test finding the intersection of two grids."""
        grid_system = GridSystem()
        grid_system.add_grid('Grid A', np.array([0.0, 0.0]), np.array([10.0, 0.0]))
        grid_system.add_grid('Grid B', np.array([5.0, -5.0]), np.array([5.0, 5.0]))
        intersection = grid_system.get_intersection_coordinates('Grid A', 'Grid B')
        assert intersection is not None
        assert np.array_equal(intersection, np.array([5.0, 0.0]))

    def test_intersection_with_nonexistent_grid(self) -> None:
        """Test finding the intersection with a nonexistent grid raises ValueError."""
        grid_system = GridSystem()
        grid_system.add_grid('Grid A', np.array([0.0, 0.0]), np.array([10.0, 0.0]))
        with pytest.raises(
            ValueError, match="Grids 'Grid A' and 'Grid C' must exist."
        ):
            grid_system.get_intersection_coordinates('Grid A', 'Grid C')


class TestGridSystem2D:
    """Tests for the GridSystem2D class."""

    def test_add_and_retrieve_grid(self) -> None:
        """Test adding and retrieving a grid."""
        grid_system = GridSystem2D()
        grid_system.add_grid('Grid X', 10.0)
        assert grid_system.get_grid_location('Grid X') == 10.0

    def test_retrieve_nonexistent_grid(self) -> None:
        """Test retrieving a nonexistent grid raises ValueError."""
        grid_system = GridSystem2D()
        with pytest.raises(ValueError, match="Grid 'Grid Y' does not exist."):
            grid_system.get_grid_location('Grid Y')

    def test_add_and_retrieve_level(self) -> None:
        """Test adding and retrieving a level."""
        grid_system = GridSystem2D()
        grid_system.add_level('Ground Floor', 0.0)
        assert grid_system.get_level('Ground Floor').elevation() == 0.0  # type: ignore

    def test_retrieve_nonexistent_level(self) -> None:
        """Test retrieving a nonexistent level raises ValueError."""
        grid_system = GridSystem2D()
        with pytest.raises(ValueError, match="Level 'Roof' does not exist."):
            grid_system.get_level('Roof').elevation()  # type: ignore
