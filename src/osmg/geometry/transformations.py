"""Coordinate transformation operations."""

from __future__ import annotations

import numpy as np

from osmg.core import common
from osmg.core.common import THREE_DIMENSIONAL, TWO_DIMENSIONAL, numpy_array


def rotation_matrix_2d(ang: float) -> numpy_array:
    """
    Obtain a 2D transformation matrix.

    Parameters
    ----------
        ang: Angle in radians to rotate the matrix by.

    Returns:
    -------
        A 2x2 transformation matrix.

    Example:
        >>> rotation_matrix_2d(np.pi / 2)
        array([[ 6.123234e-17, -1.000000e+00],
               [ 1.000000e+00,  6.123234e-17]])

    Raises:
    ------
        TypeError: If `ang` is not a float.

    """
    if not isinstance(ang, float):
        msg = 'ang parameter should be a float.'
        raise TypeError(msg)
    return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])


def rotation_matrix_3d(axis: numpy_array, theta: float) -> numpy_array:
    """
    3D rotation matrix.

    Returns the rotation matrix associated with counterclockwise
    rotation about the given axis by theta radians.

    Parameters
    ----------
        axis: 3D vector representing the axis of rotation.
        theta: Angle of rotation in radians.

    Returns:
    -------
        3x3 transformation matrix representing the rotation.

    Example:
        >>> # this is how to run that function:
        >>> res = rotation_matrix_3d(np.array([1, 0, 0]), np.pi/2)
        >>> # this is the expected result:
        >>> expected_res = np.array(
        ...     [[ 1.00000000e+00,  0.00000000e+00, -0.00000000e+00],
        ...      [-0.00000000e+00,  2.22044605e-16, -1.00000000e+00],
        ...      [ 0.00000000e+00,  1.00000000e+00,  2.22044605e-16]])
        >>> assert np.allclose(res, expected_res)

    """
    v_a = np.cos(theta / 2.0)
    v_b, v_c, v_d = -axis * np.sin(theta / 2.0)
    v_aa, v_bb, v_cc, v_dd = v_a * v_a, v_b * v_b, v_c * v_c, v_d * v_d
    v_bc, v_ad, v_ac, v_ab, v_bd, v_cd = (
        v_b * v_c,
        v_a * v_d,
        v_a * v_c,
        v_a * v_b,
        v_b * v_d,
        v_c * v_d,
    )
    return np.array(
        [
            [v_aa + v_bb - v_cc - v_dd, 2 * (v_bc + v_ad), 2 * (v_bd - v_ac)],
            [2 * (v_bc - v_ad), v_aa + v_cc - v_bb - v_dd, 2 * (v_cd + v_ab)],
            [2 * (v_bd + v_ac), 2 * (v_cd - v_ab), v_aa + v_dd - v_bb - v_cc],
        ]
    )


def transformation_matrix(
    vec_x: numpy_array, vec_y: numpy_array, vec_z: numpy_array
) -> numpy_array:
    """
    Obtain a transformation matrix.

    Returns a transformation matrix that transforms points from
    the coordinate system in which the x, y and z axes are expressed,
    to the local coordinate system defined by them.

    Arguments:
        vec_x: Local x axis expressed in the global system
        vec_y: (similar)
        vec_z: (similar)

    Returns:
    -------
        global to local transformation matrix.

    Note: For orthogonal axes, transpose to obtain the inverse transform.

    Example:
        >>> # this is how to run that function:
        >>> res = transformation_matrix(
        ...     np.array([1., 0., 0.]),
        ...     np.array([0., 1., 0.]),
        ...     np.array([0., 0., 1.]))
        >>> expected_result = np.array((
        ...     [[1., 0., 0.],
        ...      [0., 1., 0.],
        ...      [0., 0., 1.]]))
        >>> assert np.allclose(res, expected_result)
        >>> res = transformation_matrix(
        ...     np.array([1., 0., 0.]),
        ...     np.array([0., 0., 1.]),
        ...     np.array([0., 1., 0.]))
        >>> expected_result = np.array((
        ...     [[1., 0., 0.],
        ...      [0., 0., 1.],
        ...      [0., 1., 0.]]))
        >>> assert np.allclose(res, expected_result)

    """
    tr_global_to_local: numpy_array = np.vstack((vec_x, vec_y, vec_z))
    return tr_global_to_local


def transformation_matrix_2d(vec_x: numpy_array, vec_y: numpy_array) -> numpy_array:
    """
    Obtain a transformation matrix in 2D.

    Returns a transformation matrix that transforms points from
    the coordinate system in which the x and y axes are expressed,
    to the local coordinate system defined by them.

    Arguments:
        vec_x: Local x axis expressed in the global system
        vec_y: (similar)

    Returns:
    -------
        global to local transformation matrix.
    """
    tr_global_to_local: numpy_array = np.vstack((vec_x, vec_y))
    return tr_global_to_local


def local_axes_from_points_and_angle(
    point_i: numpy_array, point_j: numpy_array, ang: float
) -> (
    tuple[numpy_array, numpy_array, numpy_array]
    | tuple[numpy_array, None, numpy_array]
):
    """
    Calculate local axes from two points and a rotation angle.

    This function computes the local coordinate system for a linear element
    defined by a start and end point, optionally applying a rotation angle
    around the local x-axis. It supports both 2D and 3D coordinate systems.
    In 2D, the y-axis is omitted, and the local coordinate system is defined
    in the XZ plane, with the Y axis assumed to point out of the screen. In 3D,
    the full local coordinate system is calculated.

    Args:
        point_i (numpy_array): The start point of the element.
        point_j (numpy_array): The end point of the element.
        ang (float): Rotation angle in radians around the local x-axis.

    Returns:
        tuple[numpy_array, numpy_array | None, numpy_array]: A tuple containing:
            - The local x-axis (numpy_array).
            - The local y-axis (numpy_array) or None (in 2D).
            - The local z-axis (numpy_array).

    Raises:
        ValueError: If the input points do not have the same dimension.
        ValueError: If a vertical element in 3D is defined upside down.
        ValueError: If the coordinates are not 2D or 3D.

    Example:
        For 3D:
            >>> point_i = np.array([0, 0, 0])
            >>> point_j = np.array([1, 0, 0])
            >>> ang = 0
            >>> local_axes_from_points_and_angle(point_i, point_j, ang)
            (array([1., 0., 0.]), array([0., 0., 1.]), array([ 0., -1.,  0.]))

        For 2D:
            >>> point_i = np.array([0, 0])
            >>> point_j = np.array([1, 0])
            >>> ang = 0
            >>> local_axes_from_points_and_angle(point_i, point_j, ang)
            (array([1., 0.]), None, array([ 0., -1.]))
    """
    if point_i.shape != point_j.shape:
        msg = 'Start and end points must have the same dimension.'
        raise ValueError(msg)

    # Determine 2D or 3D
    dim = point_i.shape[0]
    if dim == TWO_DIMENSIONAL:
        assert ang == 0.00, 'Angle should be 0.00 in 2D cases.'
        return _local_axes_2d(point_i, point_j)
    if dim == THREE_DIMENSIONAL:
        return _local_axes_3d(point_i, point_j, ang)
    msg = 'Only 2D or 3D coordinates are supported.'
    raise ValueError(msg)


def _local_axes_2d(
    point_i: numpy_array, point_j: numpy_array
) -> tuple[numpy_array, None, numpy_array]:
    """
    Compute local axes for a 2D linear element.

    In the 2D case, the local coordinate system is defined in the XZ plane,
    with the Y axis pointing out of the screen. This function calculates the
    local x-axis based on the direction of the line and derives the z-axis
    by rotating the x-axis 90 degrees counterclockwise.

    Args:
        point_i (numpy_array): The start point of the element.
        point_j (numpy_array): The end point of the element.
        ang (float): Rotation angle (not used in the 2D case).

    Returns:
        tuple[numpy_array, None, numpy_array]: A tuple containing:
            - The local x-axis (numpy_array).
            - None for the y-axis (2D case).
            - The local z-axis (numpy_array).

    Example:
        >>> point_i = np.array([0, 0])
        >>> point_j = np.array([1, 0])
        >>> _local_axes_2d(point_i, point_j, 0)
        (array([1., 0.]), None, array([ 0., -1.]))
    """
    # x-axis
    x_axis = point_j - point_i
    x_axis /= np.linalg.norm(x_axis)

    # z-axis: Rotate x-axis by 90 degrees
    z_axis = np.array([-x_axis[1], x_axis[0]])
    z_axis /= np.linalg.norm(z_axis)

    # No y-axis in 2D
    return x_axis, None, z_axis


def _local_axes_3d(
    point_i: numpy_array, point_j: numpy_array, ang: float
) -> tuple[numpy_array, numpy_array, numpy_array]:
    """
    Compute local axes for a 3D linear element.

    In the 3D case, the local coordinate system consists of the x-axis,
    y-axis, and z-axis. The function accounts for special cases such as
    vertical elements and applies the specified rotation angle around
    the x-axis.

    Args:
        point_i (numpy_array): The start point of the element.
        point_j (numpy_array): The end point of the element.
        ang (float): Rotation angle in radians around the local x-axis.

    Returns:
        tuple[numpy_array, numpy_array, numpy_array]: A tuple containing:
            - The local x-axis (numpy_array).
            - The local y-axis (numpy_array).
            - The local z-axis (numpy_array).

    Raises:
        ValueError: If the element is vertical and defined upside down.

    Example:
        >>> point_i = np.array([0, 0, 0])
        >>> point_j = np.array([1, 0, 0])
        >>> ang = np.pi / 4
        >>> _local_axes_3d(point_i, point_j, ang)
        (array([1., 0., 0.]), array([0., 0., 1.]), array([ 0., -1.,  0.]))
    """
    # x-axis
    x_axis = point_j - point_i
    x_axis /= np.linalg.norm(x_axis)

    # Check if the element is vertical
    diff = np.abs(np.linalg.norm(x_axis - np.array([0.0, 0.0, -1.0])))
    if diff < common.EPSILON:
        # Vertical case
        z_axis = np.array([np.cos(ang), np.sin(ang), 0.0])
        y_axis = np.cross(z_axis, x_axis)
    else:
        # Non-vertical case
        diff = np.abs(np.linalg.norm(x_axis - np.array([0.0, 0.0, 1.0])))
        if diff < common.EPSILON:
            msg = 'Vertical element defined upside down.'
            raise ValueError(msg)

        up_direction = np.array([0.0, 0.0, 1.0])
        # Orthogonalize y-axis with respect to x-axis
        y_axis = up_direction - np.dot(up_direction, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis)
        y_axis = np.dot(rotation_matrix_3d(x_axis, ang), y_axis)

        # z-axis
        z_axis = np.cross(x_axis, y_axis)

    return x_axis, y_axis, z_axis


def offset_transformation_2d(
    offset: numpy_array, u_vec: numpy_array, r_angle: float
) -> numpy_array:
    """
    Offset transformation.

    Calculate the displacement at the end of a rigid offset by
    specifying the displacement and rotation of the other end.

    A rigid offset connects two nodes and transmits forces between
    them, but does not allow any relative displacement or rotation
    between the nodes.

    Args:
        offset:
          Vector pointing from the node of the rigid offset where the
          displacement is known to the node where we want to obtain
          the displacement. The vector should be given in the global
          coordinate system.
        u_vec:
          Displacement of the node where the displacement is known,
          given in the global coordinate system.
        r_angle:
          Rotation of the node where the displacement is known, given
          as a vector of the form [rx, ry, rz] representing the
          rotation around the x, y, and z axes, respectively.

    Returns:
    -------
        Displacement at the other end of the rigid offset,
        given in the global coordinate system.

    """
    result: numpy_array = u_vec + np.array((-offset[1], offset[0])) * r_angle
    return result


def offset_transformation_3d(
    offset: numpy_array, u_vec: numpy_array, r_vec: numpy_array
) -> numpy_array:
    """
    Offset transformation.

    Calculate the displacement at the end of a rigid offset by
    specifying the displacement and rotation of the other end.

    A rigid offset connects two nodes and transmits forces between
    them, but does not allow any relative displacement or rotation
    between the nodes.

    Arguments:
        offset:
          Vector pointing from the node of the rigid offset where the
          displacement is known to the node where we want to obtain
          the displacement. The vector should be given in the global
          coordinate system.
        u_vec:
          Displacement of the node where the displacement is known,
          given in the global coordinate system.
        r_vec:
          Rotation of the node where the displacement is known, given
          as a vector of the form [rx, ry, rz] representing the
          rotation around the x, y, and z axes, respectively.

    Returns:
    -------
        Displacement at the other end of the rigid offset,
        given in the global coordinate system.

    Example:
        Calculate the displacement of the end of a rigid offset with a
        length of 1 meter, given a displacement of [4, 5, 6] and a
        rotation of [7, 8, 9] at the other end:

        >>> offset_transformation(np.array([1., 0., 0.]),
        ...     np.array([0.01, -0.02, 0.005]),
        ...     np.array([0.0002, -0.0003, 0.0001]))
        array([ 0.01  , -0.0199,  0.0053])

    """
    t_rigid: numpy_array = np.array(
        [
            [0.00, +offset[2], -offset[1]],
            [-offset[2], 0.00, +offset[0]],
            [+offset[1], -offset[0], 0.00],
        ]
    )
    return u_vec + t_rigid @ r_vec
