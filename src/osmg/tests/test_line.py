"""Unit tests for the Line class."""

import numpy as np
import pytest

from osmg.geometry.line import Line


class TestLine:
    """Tests for the Line class."""

    def test_length(self) -> None:
        """Test the length of a line."""
        line = Line('l1', np.array([0, 0]), np.array([2, 2]))
        assert line.length() == pytest.approx(2.8284271247461903)

    def test_direction(self) -> None:
        """Test the direction vector of a line."""
        line = Line('l1', np.array([0, 0]), np.array([2, 2]))
        np.testing.assert_allclose(
            line.direction(), np.array([0.70710678, 0.70710678])
        )

    def test_intersect(self) -> None:
        """Test the intersection of two lines."""
        line1 = Line('l1', np.array([0, 0]), np.array([2, 2]))
        line2 = Line('l2', np.array([1, 0]), np.array([1, 3]))
        np.testing.assert_allclose(line1.intersect(line2), np.array([1.0, 1.0]))  # type: ignore

    def test_intersect_no_intersection(self) -> None:
        """Test when two lines do not intersect."""
        line1 = Line('l1', np.array([0, 0]), np.array([1, 1]))
        line2 = Line('l2', np.array([2, 2]), np.array([3, 3]))
        assert line1.intersect(line2) is None

    def test_intersects_pt(self) -> None:
        """Test whether a point lies on the line."""
        line = Line('l1', np.array([0, 0]), np.array([1, 1]))
        assert line.intersects_pt(np.array([0.5, 0.5])) is True
        assert line.intersects_pt(np.array([0, 0])) is True
        assert line.intersects_pt(np.array([1, 1])) is True
        assert line.intersects_pt(np.array([2, 2])) is False

    def test_intersects_pt_zero_length(self) -> None:
        """Test whether intersects_pt raises an error for zero-length line."""
        line = Line('l1', np.array([0, 0]), np.array([0, 0]))
        with pytest.raises(ValueError, match='Line has zero length.'):
            line.intersects_pt(np.array([0, 0]))

    def test_point_distance(self) -> None:
        """Test the distance from a point to the line."""
        line = Line('l1', np.array([1, 1]), np.array([3, 3]))
        assert line.point_distance(np.array([4, 2])) == pytest.approx(
            1.4142135623730951
        )
        assert line.point_distance(np.array([2, 2])) == pytest.approx(0.0)
        assert line.point_distance(np.array([0, 0])) is None
        assert line.point_distance(np.array([4, 4])) is None

    def test_project(self) -> None:
        """Test the projection of a point onto the line."""
        line = Line('test', np.array([0, 0]), np.array([10, 0]))
        np.testing.assert_allclose(
            line.project(np.array([5, 0])),  # type: ignore
            np.array([5.0, 0.0]),  # type: ignore
        )
        np.testing.assert_allclose(
            line.project(np.array([5, 5])),  # type: ignore
            np.array([5.0, 0.0]),  # type: ignore
        )
        assert line.project(np.array([-5, 5])) is None
        assert line.project(np.array([15, 5])) is None
