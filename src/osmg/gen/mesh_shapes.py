"""
Generates meshes for preconfigured sections.

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
from ..mesh import Mesh
from ..mesh import Vertex
from ..mesh import Edge
from ..mesh import orient_loops
from ..mesh import obtain_closed_loops
from ..mesh import sanity_checks
from ..mesh import define_halfedges

nparr = npt.NDArray[np.float64]


def generate(edges):
    """
    Generates halfedges from the given edges.

    """

    halfedges = define_halfedges(edges)
    loops = obtain_closed_loops(halfedges)
    _, internal, trivial = orient_loops(loops)
    sanity_checks(internal, trivial)
    return Mesh(internal[0])


def define_edges(vertices):
    """
    Defines edges from an ordered list of vertices.

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


def w_mesh(sec_b, sec_h, sec_tw, sec_tf, target_area=None):
    """
    Defines a loop of counterclockwise halfedges that form the shape
    of the W section with the specified parameters.  The origin
    coincides with the centroid.

    Arguments:
      b: total width
      h: total height
      tw: web thickness
      tf: flange thickness
      target_area: AISC database area to determine fillets, because
        trying to do that using `T` doesn't work.

    """
    area_diff = target_area - (
        sec_b * sec_tf * 2.0 + (sec_h - 2 * sec_tf) * sec_tw
    )
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
                -sec_h / 2.0
                + sec_tf
                + dist
                - dist * np.sin(1.0 * np.pi / 8.0),
            )
        ),
        Vertex(
            (
                -sec_b / 2.0 + k + dist * np.cos(1.0 * np.pi / 4.0),
                -sec_h / 2.0
                + sec_tf
                + dist
                - dist * np.sin(1.0 * np.pi / 4.0),
            )
        ),
        Vertex(
            (
                -sec_b / 2.0 + k + dist * np.cos(3.0 * np.pi / 8.0),
                -sec_h / 2.0
                + sec_tf
                + dist
                - dist * np.sin(3.0 * np.pi / 8.0),
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
                -sec_h / 2.0
                + sec_tf
                + dist
                - dist * np.sin(3.0 * np.pi / 8.0),
            )
        ),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(1.0 * np.pi / 4.0),
                -sec_h / 2.0
                + sec_tf
                + dist
                - dist * np.sin(1.0 * np.pi / 4.0),
            )
        ),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(1.0 * np.pi / 8.0),
                -sec_h / 2.0
                + sec_tf
                + dist
                - dist * np.sin(1.0 * np.pi / 8.0),
            )
        ),
        Vertex((+sec_b / 2.0 - k - dist, -sec_h / 2.0 + sec_tf + dist)),
        Vertex((+sec_b / 2.0 - k - dist, +sec_h / 2.0 - sec_tf - dist)),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(1.0 * np.pi / 8.0),
                +sec_h / 2.0
                - sec_tf
                - dist
                + dist * np.sin(1.0 * np.pi / 8.0),
            )
        ),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(1.0 * np.pi / 4.0),
                +sec_h / 2.0
                - sec_tf
                - dist
                + dist * np.sin(1.0 * np.pi / 4.0),
            )
        ),
        Vertex(
            (
                +sec_b / 2.0 - k - dist * np.cos(3.0 * np.pi / 8.0),
                +sec_h / 2.0
                - sec_tf
                - dist
                + dist * np.sin(3.0 * np.pi / 8.0),
            )
        ),
        Vertex((+sec_b / 2.0 - k, sec_h / 2.0 - sec_tf)),
        Vertex((sec_b / 2.0, sec_h / 2.0 - sec_tf)),
    ]
    edges = define_edges(vertices)
    return generate(edges)


def rect_mesh(dim_b, dim_h):
    """
    Defines a loop of counterclockwise halfedges
    that form the shape of the rectangular section with
    the specified parameters.
    The origin coincides with the centroid.

    Arguments:
        b: total width
        h: total height

    """

    vertices = [
        Vertex((dim_b / 2.0, dim_h / 2.0)),
        Vertex((-dim_b / 2.0, dim_h / 2.0)),
        Vertex((-dim_b / 2.0, -dim_h / 2.0)),
        Vertex((dim_b / 2.0, -dim_h / 2.0)),
    ]
    edges = define_edges(vertices)
    return generate(edges)


def circ_mesh(dim_d):
    """
    Defines a loop of counterclockwise halfedges
    that form the shape of the circular section with
    the specified parameters.
    The origin coincides with the centroid.

    Arguments:
        d: total diameter

    """
    radius = dim_d / 2.0
    num_vertices = 32  # Number of vertices on the circumference

    angle_increment = 2 * np.pi / num_vertices

    vertices = []
    for i in range(num_vertices):
        angle = i * angle_increment
        vertices.append(Vertex((
            radius * np.cos(angle),
            radius * np.sin(angle)
        )))

    edges = define_edges(vertices)
    return generate(edges)


def generic_snap_points(mesh: Mesh) -> dict[str, nparr]:
    """
    Generates generic snap poitns for a section object.

    """

    bbox = mesh.bounding_box()
    z_min, y_min, z_max, y_max = bbox.flatten()
    snap_points: dict[str, nparr] = {}
    snap_points["centroid"] = -np.array([0.0, 0.0])
    snap_points["top_center"] = -np.array([0.0, y_max])
    snap_points["top_left"] = -np.array([z_min, y_max])
    snap_points["top_right"] = -np.array([z_max, y_max])
    snap_points["center_left"] = -np.array([z_min, 0.0])
    snap_points["center_right"] = -np.array([z_max, 0.0])
    snap_points["bottom_center"] = -np.array([0.0, y_min])
    snap_points["bottom_left"] = -np.array([z_min, y_min])
    snap_points["bottom_right"] = -np.array([z_max, y_min])
    return snap_points
