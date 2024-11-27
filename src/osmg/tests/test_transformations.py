"""Unit tests for coordinate transformation operations."""

import numpy as np
import pytest

from osmg.geometry.transformations import (
    local_axes_from_points_and_angle,
    offset_transformation_3d,
    rotation_matrix_2d,
    rotation_matrix_3d,
    transformation_matrix,
)


class TestRotationMatrix2D:
    """Tests for the `rotation_matrix_2d` function."""

    def test_rotation_matrix_2d(self) -> None:
        """Test the 2D rotation matrix."""
        result = rotation_matrix_2d(np.pi / 2)
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        assert np.allclose(result, expected)

    def test_rotation_matrix_2d_invalid_input(self) -> None:
        """Test that a TypeError is raised for invalid input."""
        with pytest.raises(TypeError, match='ang parameter should be a float.'):
            rotation_matrix_2d('90')  # type: ignore  # Invalid input type


class TestRotationMatrix3D:
    """Tests for the `rotation_matrix_3d` function."""

    def test_rotation_matrix_3d(self) -> None:
        """Test the 3D rotation matrix."""
        axis = np.array([1, 0, 0])
        theta = np.pi / 2
        result = rotation_matrix_3d(axis, theta)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        assert np.allclose(result, expected)

    def test_rotation_matrix_3d_invalid_axis(self) -> None:
        """Test that the function handles invalid axis input."""
        axis = np.array([0, 0])  # Invalid axis (not 3D)
        theta = np.pi / 2
        with pytest.raises(ValueError, match='not enough values to unpack'):
            rotation_matrix_3d(axis, theta)


class TestTransformationMatrix:
    """Tests for the `transformation_matrix` function."""

    def test_transformation_matrix_identity(self) -> None:
        """Test the transformation matrix for identity axes."""
        vec_x = np.array([1.0, 0.0, 0.0])
        vec_y = np.array([0.0, 1.0, 0.0])
        vec_z = np.array([0.0, 0.0, 1.0])
        result = transformation_matrix(vec_x, vec_y, vec_z)
        expected = np.eye(3)
        assert np.allclose(result, expected)

    def test_transformation_matrix_permuted_axes(self) -> None:
        """Test the transformation matrix with permuted axes."""
        vec_x = np.array([1.0, 0.0, 0.0])
        vec_y = np.array([0.0, 0.0, 1.0])
        vec_z = np.array([0.0, 1.0, 0.0])
        result = transformation_matrix(vec_x, vec_y, vec_z)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        assert np.allclose(result, expected)


class TestLocalAxesFromPointsAndAngle:
    """Tests for the `local_axes_from_points_and_angle` function."""

    def test_local_axes_horizontal(self) -> None:
        """Test local axes for a horizontal element."""
        point_i = np.array([0.0, 0.0, 0.0])
        point_j = np.array([1.0, 0.0, 0.0])
        ang = 0.0
        result = local_axes_from_points_and_angle(point_i, point_j, ang)
        expected_x = np.array([1.0, 0.0, 0.0])
        expected_y = np.array([0.0, 0.0, 1.0])
        expected_z = np.array([0.0, -1.0, 0.0])
        assert np.allclose(result[0], expected_x)  # type: ignore
        assert np.allclose(result[1], expected_y)  # type: ignore
        assert np.allclose(result[2], expected_z)  # type: ignore

    def test_local_axes_vertical(self) -> None:
        """Test local axes for a vertical element."""
        point_i = np.array([0.0, 0.0, 0.0])
        point_j = np.array([0.0, 0.0, 1.0])
        ang = np.pi / 4
        result = local_axes_from_points_and_angle(point_j, point_i, ang)
        expected_x = np.array([0.0, 0.0, -1.0])
        expected_y = np.array([-np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0])
        expected_z = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0])
        assert np.allclose(result[0], expected_x)  # type: ignore
        assert np.allclose(result[1], expected_y)  # type: ignore
        assert np.allclose(result[2], expected_z)  # type: ignore

    def test_local_axes_vertical_upside_down(self) -> None:
        """Test that a ValueError is raised for an upside-down vertical element."""
        point_i = np.array([0.0, 0.0, 0.0])
        point_j = np.array([0.0, 0.0, 1.0])
        ang = 0.0
        with pytest.raises(ValueError, match='Vertical element defined upside down'):
            local_axes_from_points_and_angle(point_i, point_j, ang)


class TestOffsetTransformation:
    """Tests for the `offset_transformation` function."""

    def test_offset_transformation(self) -> None:
        """Test the offset transformation calculation."""
        offset = np.array([1.0, 0.0, 0.0])
        u_vec = np.array([0.01, -0.02, 0.005])
        r_vec = np.array([0.0002, -0.0003, 0.0001])
        result = offset_transformation_3d(offset, u_vec, r_vec)
        expected = np.array([0.01, -0.0199, 0.0053])
        assert np.allclose(result, expected)  # type: ignore
