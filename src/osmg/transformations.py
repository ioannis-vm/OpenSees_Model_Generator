"""
Performs coordinate transformations
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

import numpy as np
import numpy.typing as npt
from . import common

nparr = npt.NDArray[np.float64]


def rotation_matrix_2d(ang: float) -> nparr:
    """
    Return a 2D transformation matrix
    """
    return np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])


def rotation_matrix_3d(
        axis: nparr,
        theta: float) -> nparr:
    """
    Return the rotation matrix associated with counterclockwise
    rotation about the given axis by theta radians.
    Courtesy of
    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def transformation_matrix(vec_x: nparr,
                          vec_y: nparr,
                          vec_z: nparr
                          ) -> nparr:
    """
    Returns a transformation matrix that transforms points from
    the coordinate system in which the x, y and z axes are expressed,
    to the local coordinate system defined by them.
    Args:
        vec_x (nparr): Local x axis expressed in the global system
        vec_y (nparr): (similar)
        vec_z (nparr): (similar)
    Returns:
        (nparr): global to local transformation matrix.
    Note: For orthogonal axes, transpose to obtain the inverse transform.
    """
    tr_global_to_local: nparr = \
        np.vstack((vec_x, vec_y, vec_z))
    return tr_global_to_local


def local_axes_from_points_and_angle(point_i: nparr,
                                     point_j: nparr,
                                     ang: float) -> \
        tuple[nparr,
              nparr,
              nparr]:
    """
    Given a start point, and end point, and an angle,
    obtain the local coordinate system of a linear element.
    Args:
        point_i (nparr): Start point
        point_j (nparr): End point
        and (float): Parameter that controls the rotation of the
            section around the x-axis. Counterclockwise rotation is
            posotive. 0.00 corresponds to:
              vertical elements whose local z axis coincides with
                                the local x axis
              horizontal elements whose local z axis is horizontal.
    """
    # x
    x_axis = point_j - point_i
    x_axis = x_axis / np.linalg.norm(x_axis)
    # y and z
    diff = np.abs(
        np.linalg.norm(x_axis - np.array([0.00, 0.00, -1.00])))
    if diff < common.EPSILON:
        # vertical case
        z_axis: nparr = np.array([np.cos(ang), np.sin(ang), 0.0])
        y_axis: nparr = np.cross(z_axis, x_axis)
    else:
        # note: all vertical elements must have node i higher than node j
        # we raise an error otherwise.
        # check:
        diff = np.abs(
            np.linalg.norm(x_axis - np.array([0.00, 0.00, 1.00])))
        if diff < common.EPSILON:
            raise ValueError("Vertical element defined upside down")
        # not vertical case
        up_direction: nparr = np.array([0.0, 0.0, 1.0])
        # orthogonalize with respect to x_vec
        y_axis = \
            up_direction - np.dot(up_direction, x_axis) * x_axis
        # ..and normalize
        y_axis = y_axis / np.linalg.norm(y_axis)
        y_axis = np.dot(rotation_matrix_3d(
            x_axis, ang), y_axis)
        # determine z axis from the cross-product
        z_axis = np.cross(x_axis, y_axis)

    return x_axis, y_axis, z_axis  # type: ignore


def offset_transformation(offset: nparr,
                          u: nparr,
                          r: nparr) -> nparr:
    """
    Obtain the displacement of the end of a rigid offeset
    by specifying the displacement and rotation of the
    other end.
    Args:
        offset: Vector pointing from the node of the rigid
                offset of which the displacement is known
                to the node where we want to obtain the
                displacement.
    Returns:
        Displacements at the other end.
    """
    t_rigid: nparr = np.array([[0.00, +offset[2], -offset[1]],
                               [-offset[2], 0.00, +offset[0]],
                               [+offset[1], -offset[0], 0.00]])
    return u + t_rigid @ r
