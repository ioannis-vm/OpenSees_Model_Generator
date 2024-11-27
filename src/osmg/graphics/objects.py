"""Core plotting objects."""

from __future__ import annotations

from math import fabs

import numpy as np
import plotly.graph_objects as go

from osmg.core.common import EPSILON
from osmg.geometry.transformations import (
    local_axes_from_points_and_angle,
    transformation_matrix,
)


def arrow(
    total_length: float = 1.00,
    head_length: float = 0.45,
    head_width: float = 0.30,
    base_width: float = 0.05,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[int, int, int], ...],
]:
    """
    Define the vertices and edges of an arrow.

    The tip of the arrow is at the axes origin, and it's pointing
    upward.

    Returns:
      Tuple containing the vertices and faces of the mesh in the form
      of tuples.
    """
    tl = total_length
    hl = head_length
    hhw = head_width / 2.00
    bhw = base_width / 2.00

    vertices = (
        (0.0, 0.0, 0.0),
        (-hhw, -hhw, -hl),
        (+hhw, -hhw, -hl),
        (+hhw, +hhw, -hl),
        (-hhw, +hhw, -hl),
        (-bhw, -bhw, -hl),
        (+bhw, -bhw, -hl),
        (+bhw, +bhw, -hl),
        (-bhw, +bhw, -hl),
        (-bhw, -bhw, -tl),
        (+bhw, -bhw, -tl),
        (+bhw, +bhw, -tl),
        (-bhw, +bhw, -tl),
    )
    faces = (
        # top part
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 1),
        # base
        (5, 9, 10),
        (5, 10, 6),
        (6, 10, 11),
        (6, 11, 7),
        (7, 11, 12),
        (7, 12, 8),
        (8, 12, 9),
        (8, 9, 5),
    )
    return vertices, faces


def positioned_arrow(
    start_location: tuple[float, float, float],
    end_location: tuple[float, float, float],
    head_length: float = 0.45,
    head_width: float = 0.30,
    base_width: float = 0.05,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[int, int, int], ...],
]:
    """
    Define vertices and faces for a positioned arrow.

    Returns:
      Vertices and faces.

    Raises:
      ValueError: If the start and end locations are coinciding.
    """
    start_vec = np.array(start_location)
    end_vec = np.array(end_location)

    arrow_length = float(np.linalg.norm(end_vec - start_vec))

    # Check for the case where no rotation is required.
    if (
        fabs(start_location[0] - end_location[0]) < EPSILON
        and fabs(start_location[1] - end_location[1]) < EPSILON
    ):
        if fabs(start_location[2] - end_location[2]) < EPSILON:
            msg = 'Start and end locations should not be the same.'
            raise ValueError(msg)
        if end_location[2] > start_location[2]:
            vertices, faces = arrow(
                -arrow_length, -head_length, head_width, base_width
            )
        else:
            vertices, faces = arrow(
                arrow_length, head_length, head_width, base_width
            )
        # translation
        vertices = tuple(np.array(vertices) + start_vec)
    else:
        vertices, faces = arrow(arrow_length, head_length, head_width, base_width)
        # Rotation and translation required.
        x_axis, y_axis, z_axis = local_axes_from_points_and_angle(
            end_vec, start_vec, ang=0.00
        )
        assert y_axis is not None
        orient_to_x_axis = np.array(
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
        )
        transformation_mat = transformation_matrix(x_axis, y_axis, z_axis).T
        vertices = tuple(
            (transformation_mat @ orient_to_x_axis @ np.array(vertices).T).T
            + start_vec
        )

    return vertices, faces


def main() -> None:
    """Use for testing."""
    # Define the vertices of the arrow

    vertices, faces = positioned_arrow(
        start_location=(0.00, 0.00, 0.00),
        end_location=(0.00, 0.00, 1.00),
        head_length=0.2,
        head_width=0.2,
        base_width=0.05,
    )

    x, y, z = zip(*vertices)
    i, j, k = zip(*faces)

    # Create a 3D mesh
    mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5, color='lightblue')

    # Setup the layout of the scene
    layout = go.Layout(
        scene={
            'xaxis': {
                'nticks': 4,
                'range': [-1, 2],
            },
            'yaxis': {
                'nticks': 4,
                'range': [-1, 2],
            },
            'zaxis': {
                'nticks': 4,
                'range': [-1, 2],
            },
        }
    )

    # Create a figure and add the mesh
    fig = go.Figure(data=[mesh], layout=layout)

    # Show the plot
    fig.show()


if __name__ == '__main__':
    main()
