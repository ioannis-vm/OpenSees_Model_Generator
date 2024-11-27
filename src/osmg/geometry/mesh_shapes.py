"""Generates meshes for preconfigured sections."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from osmg.geometry.mesh import (
    Edge,
    Mesh,
    Vertex,
    define_halfedges,
    obtain_closed_loops,
    orient_loops,
    sanity_checks,
)

numpy_array = npt.NDArray[np.float64]


def generate(edges: list[Edge]) -> Mesh:
    """
    Generate a mesh from the given edges.

    Returns:
      The generated mesh.
    """
    halfedges = define_halfedges(edges)
    loops = obtain_closed_loops(halfedges)
    _, internal, trivial = orient_loops(loops)
    sanity_checks(internal, trivial)
    return Mesh(internal[0])


def define_edges(vertices: list[Vertex]) -> list[Edge]:
    """
    Define edges from an ordered list of vertices.

    Returns:
      The defined edges.
    """
    n_v = len(vertices)
    edges = []
    for i in range(n_v - 1):
        v_i = vertices[i]
        v_j = vertices[i + 1]
        edges.append(Edge(v_i, v_j))
    v_i = vertices[-1]
    v_j = vertices[0]
    edges.append(Edge(v_i, v_j))
    return edges


def w_mesh(
    sec_b: float,
    sec_h: float,
    sec_tw: float,
    sec_tf: float,
    target_area: float,
) -> Mesh:
    """
    W-section mesh.

    Defines a loop of counterclockwise halfedges that form the shape
    of the W section with the specified parameters.  The origin
    coincides with the centroid.

    Arguments:
      sec_b: total width
      sec_h: total height
      sec_tw: web thickness
      sec_tf: flange thickness
      target_area: AISC database area to determine fillets, because
        trying to do that using `T` doesn't work.

    Returns:
      The generated mesh.
    """
    area_diff = target_area - (sec_b * sec_tf * 2.0 + (sec_h - 2 * sec_tf) * sec_tw)
    if area_diff < 0:
        # This happens for W14X426
        area_diff = 1e-4

    dist = np.sqrt(area_diff / (2.0**2 - np.pi)) * 0.9565
    # note: 0.9565 is a correction factor to account for
    # the fact that we approximate the arcs with
    # four line segments, thus putting more material in there
    k = (sec_b - 2.0 * dist - sec_tw) / 2.0
    vertices = [
        Vertex((sec_b / 2.0, sec_h / 2.0)),
        Vertex((-sec_b / 2.0, sec_h / 2.0)),
        Vertex((-sec_b / 2.0, sec_h / 2.0 - sec_tf)),
        Vertex((-sec_b / 2.0 + k, sec_h / 2.0 - sec_tf)),
        Vertex(
            (
                -sec_b / 2.0 + k + dist * np.cos(3.0 * np.pi / 8.0),
                sec_h / 2.0 - sec_tf - dist + dist * np.sin(3.0 * np.pi / 8.0),
            )
        ),
        Vertex(
            (
                -sec_b / 2.0 + k + dist * np.cos(1.0 * np.pi / 4.0),
                sec_h / 2.0 - sec_tf - dist + dist * np.sin(1.0 * np.pi / 4.0),
            )
        ),
        Vertex(
            (
                -sec_b / 2.0 + k + dist * np.cos(1.0 * np.pi / 8.0),
                sec_h / 2.0 - sec_tf - dist + dist * np.sin(1.0 * np.pi / 8.0),
            )
        ),
        Vertex((-sec_b / 2.0 + k + dist, sec_h / 2.0 - sec_tf - dist)),
        Vertex((-sec_b / 2.0 + k + dist, -sec_h / 2.0 + sec_tf + dist)),
        Vertex(
            (
                -sec_b / 2.0 + k + dist * np.cos(1.0 * np.pi / 8.0),
                -sec_h / 2.0 + sec_tf + dist - dist * np.sin(1.0 * np.pi / 8.0),
            )
        ),
        Vertex(
            (
                -sec_b / 2.0 + k + dist * np.cos(1.0 * np.pi / 4.0),
                -sec_h / 2.0 + sec_tf + dist - dist * np.sin(1.0 * np.pi / 4.0),
            )
        ),
        Vertex(
            (
                -sec_b / 2.0 + k + dist * np.cos(3.0 * np.pi / 8.0),
                -sec_h / 2.0 + sec_tf + dist - dist * np.sin(3.0 * np.pi / 8.0),
            )
        ),
        Vertex((-sec_b / 2.0 + k, -sec_h / 2.0 + sec_tf)),
        Vertex((-sec_b / 2.0, -(sec_h / 2.0 - sec_tf))),
        Vertex((-sec_b / 2.0, -sec_h / 2.0)),
        Vertex((sec_b / 2.0, -sec_h / 2.0)),
        Vertex((sec_b / 2.0, -(sec_h / 2 - sec_tf))),
        Vertex((+sec_b / 2.0 - k, -sec_h / 2.0 + sec_tf)),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(3.0 * np.pi / 8.0),
                -sec_h / 2.0 + sec_tf + dist - dist * np.sin(3.0 * np.pi / 8.0),
            )
        ),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(1.0 * np.pi / 4.0),
                -sec_h / 2.0 + sec_tf + dist - dist * np.sin(1.0 * np.pi / 4.0),
            )
        ),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(1.0 * np.pi / 8.0),
                -sec_h / 2.0 + sec_tf + dist - dist * np.sin(1.0 * np.pi / 8.0),
            )
        ),
        Vertex((+sec_b / 2.0 - k - dist, -sec_h / 2.0 + sec_tf + dist)),
        Vertex((+sec_b / 2.0 - k - dist, +sec_h / 2.0 - sec_tf - dist)),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(1.0 * np.pi / 8.0),
                +sec_h / 2.0 - sec_tf - dist + dist * np.sin(1.0 * np.pi / 8.0),
            )
        ),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(1.0 * np.pi / 4.0),
                +sec_h / 2.0 - sec_tf - dist + dist * np.sin(1.0 * np.pi / 4.0),
            )
        ),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(3.0 * np.pi / 8.0),
                +sec_h / 2.0 - sec_tf - dist + dist * np.sin(3.0 * np.pi / 8.0),
            )
        ),
        Vertex((+sec_b / 2.0 - k, sec_h / 2.0 - sec_tf)),
        Vertex((sec_b / 2.0, sec_h / 2.0 - sec_tf)),
    ]
    edges = define_edges(vertices)
    return generate(edges)


def rect_mesh(dim_b: float, dim_h: float) -> Mesh:
    """
    Rectangular mesh.

    Defines a loop of counterclockwise halfedges
    that form the shape of the rectangular section with
    the specified parameters.
    The origin coincides with the centroid.

    Arguments:
        dim_b: total width
        dim_h: total height

    Returns:
      The generated mesh.
    """
    vertices = [
        Vertex((dim_b / 2.0, dim_h / 2.0)),
        Vertex((-dim_b / 2.0, dim_h / 2.0)),
        Vertex((-dim_b / 2.0, -dim_h / 2.0)),
        Vertex((dim_b / 2.0, -dim_h / 2.0)),
    ]
    edges = define_edges(vertices)
    return generate(edges)


def circ_mesh(dim_d: float) -> Mesh:
    """
    Circular mesh.

    Defines a loop of counterclockwise halfedges
    that form the shape of the circular section with
    the specified parameters.
    The origin coincides with the centroid.

    Arguments:
        dim_d: total diameter

    Returns:
      The generated mesh.
    """
    radius = dim_d / 2.0
    num_vertices = 32  # Number of vertices on the circumference

    angle_increment = 2 * np.pi / num_vertices

    vertices = []
    for i in range(num_vertices):
        angle = i * angle_increment
        vertices.append(Vertex((radius * np.cos(angle), radius * np.sin(angle))))

    edges = define_edges(vertices)
    return generate(edges)


def generic_snap_points(mesh: Mesh) -> dict[str, numpy_array]:
    """
    Generate generic snap points for a section object.

    Returns:
      The snap points.
    """
    bbox = mesh.bounding_box()
    z_min, y_min, z_max, y_max = bbox.flatten()
    snap_points: dict[str, numpy_array] = {}
    snap_points['centroid'] = -np.array([0.0, 0.0])
    snap_points['top_center'] = -np.array([0.0, y_max])
    snap_points['top_left'] = -np.array([z_min, y_max])
    snap_points['top_right'] = -np.array([z_max, y_max])
    snap_points['center_left'] = -np.array([z_min, 0.0])
    snap_points['center_right'] = -np.array([z_max, 0.0])
    snap_points['bottom_center'] = -np.array([0.0, y_min])
    snap_points['bottom_left'] = -np.array([z_min, y_min])
    snap_points['bottom_right'] = -np.array([z_max, y_min])
    return snap_points
