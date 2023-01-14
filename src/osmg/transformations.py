"""
Coordinate transformation operations.

"""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

import numpy as np
import numpy.typing as npt
from . import common

nparr = npt.NDArray[np.float64]


def rotation_matrix_2d(ang: float) -> nparr:
    """
    Returns a 2D transformation matrix.

    Parameters:
        ang: Angle in radians to rotate the matrix by.

    Returns:
        A 2x2 transformation matrix.

    Example:
        >>> rotation_matrix_2d(np.pi / 2)
        array([[ 6.123234e-17, -1.000000e+00],
               [ 1.000000e+00,  6.123234e-17]])

    Raises:
        TypeError: If `ang` is not a float.

    """

    if not isinstance(ang, float):
        raise TypeError("ang parameter should be a float.")
    return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])


def rotation_matrix_3d(axis: nparr, theta: float) -> nparr:
    """
    Returns the rotation matrix associated with counterclockwise
    rotation about the given axis by theta radians.

    Parameters:
        axis: 3D vector representing the axis of rotation.
        theta: Angle of rotation in radians.

    Returns:
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


def transformation_matrix(vec_x: nparr, vec_y: nparr, vec_z: nparr) -> nparr:
    """
    Returns a transformation matrix that transforms points from
    the coordinate system in which the x, y and z axes are expressed,
    to the local coordinate system defined by them.

    Arguments:
        vec_x: Local x axis expressed in the global system
        vec_y: (similar)
        vec_z: (similar)

    Returns:
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

    tr_global_to_local: nparr = np.vstack((vec_x, vec_y, vec_z))
    return tr_global_to_local


def local_axes_from_points_and_angle(
    point_i: nparr, point_j: nparr, ang: float
) -> tuple[nparr, nparr, nparr]:
    """
    Given a start point, and end point, and an angle,
    obtain the local coordinate system of a linear element.

    Arguments:
        point_i: Start point
        point_j: End point
        ang: Parameter that controls the rotation of the section
          around the x-axis. Counterclockwise rotation is
          posotive. 0.00 corresponds to vertical elements whose local
          z axis coincides with the local x axis, and horizontal
          elements whose local z axis is horizontal.

    Returns:
        Local coordinate system
        vectors. The first element is the local x axis, the second
        element is the local y axis, and the third element is the
        local z axis.

    Raises:
        ValueError: If the start point and end point define a vertical element
            that is defined upside down (i.e., with the start point at a lower
            height than the end point).

    Note:
        For vertical elements, the local x axis will be the vector connecting
        the start and end points, and the local z axis will be perpendicular
        to the local x axis and lying on the plane defined by the global xy
        plane and the local x axis. For horizontal elements, the local z axis
        will be parallel to the global z axis.

    Example:
        >>> point_i = np.array([0, 0, 0])
        >>> point_j = np.array([1, 0, 0])
        >>> ang = 0
        >>> local_axes_from_points_and_angle(point_i, point_j, ang)
        (array([1., 0., 0.]), array([0., 0., 1.]), array([ 0., -1.,  0.]))

    """

    # x-axis
    x_axis = point_j - point_i
    x_axis = x_axis / np.linalg.norm(x_axis)
    # y and z axes
    diff = np.abs(np.linalg.norm(x_axis - np.array([0.00, 0.00, -1.00])))
    if diff < common.EPSILON:
        # vertical case
        z_axis: nparr = np.array([np.cos(ang), np.sin(ang), 0.0])
        y_axis: nparr = np.cross(z_axis, x_axis)
    else:
        # not vertical case.
        # check if the element is upside down
        diff = np.abs(np.linalg.norm(x_axis - np.array([0.00, 0.00, 1.00])))
        if diff < common.EPSILON:
            raise ValueError("Vertical element defined upside down")
        up_direction: nparr = np.array([0.0, 0.0, 1.0])
        # orthogonalize with respect to x-axis
        y_axis = up_direction - np.dot(up_direction, x_axis) * x_axis
        # ..and normalize
        y_axis = y_axis / np.linalg.norm(y_axis)
        y_axis = np.dot(rotation_matrix_3d(x_axis, ang), y_axis)
        # z-axis
        z_axis = np.cross(x_axis, y_axis)

    return x_axis, y_axis, z_axis  # type: ignore


def offset_transformation(offset: nparr, u_vec: nparr, r_vec: nparr) -> nparr:
    """
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

    t_rigid: nparr = np.array(
        [
            [0.00, +offset[2], -offset[1]],
            [-offset[2], 0.00, +offset[0]],
            [+offset[1], -offset[0], 0.00],
        ]
    )
    return u_vec + t_rigid @ r_vec
