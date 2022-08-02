"""
Generates meshes for preconfigured sections
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
from ..mesh import Mesh
from ..mesh import Vertex
from ..mesh import Edge
from ..mesh import orient_loops
from ..mesh import obtain_closed_loops
from ..mesh import sanity_checks
from ..mesh import define_halfedges
from ..import common

nparr = npt.NDArray[np.float64]


def generate(edges):
    """
    Generates halfedges from the given edges
    """
    halfedges = define_halfedges(edges)
    loops = obtain_closed_loops(halfedges)
    _, internal, trivial = orient_loops(loops)
    sanity_checks(internal, trivial)
    return Mesh(internal[0])


def define_edges(vertices):
    """
    Defines edges from an ordered list of vertices
    """
    n_v = len(vertices)
    edges = []
    for i in range(n_v - 1):
        v_i = vertices[i]
        v_j = vertices[i+1]
        edges.append(Edge(v_i, v_j))
    v_i = vertices[-1]
    v_j = vertices[0]
    edges.append(Edge(v_i, v_j))
    return edges


def w_mesh(sec_b, sec_h, sec_tw, sec_tf, target_area=None):
    """
    Defines a loop of counterclockwise halfedges
    that form the shape of the W section with
    the specified parameters.
    The origin coincides with the centroid.
    Input:
        b: total width
        h: total height
        tw: web thickness
        tf: flange thickness
        target_area: AISC database area
            to determine fillets,
            because trying to do that using
            `T` doesn't work.
    """
    area_diff = target_area - (sec_b*sec_tf*2.+(sec_h-2*sec_tf)*sec_tw)
    if area_diff < 0:
        # This happens for W14X426
        area_diff = 1e-4

    dist = np.sqrt(area_diff/(2.**2-np.pi)) * 0.9565
    # note: 0.9565 is a correction factor to account for
    # the fact that we approximate the arcs with
    # four line segments, thus putting more material in there
    k = (sec_b - 2. * dist - sec_tw) / 2.
    vertices = [
        Vertex((sec_b/2., sec_h/2.)),
        Vertex((-sec_b/2., sec_h/2.)),
        Vertex((-sec_b/2., sec_h/2.-sec_tf)),
        Vertex((-sec_b/2.+k, sec_h/2.-sec_tf)),
        Vertex((-sec_b/2.+k+dist * np.cos(3.*np.pi/8.),
                sec_h/2.-sec_tf-dist + dist*np.sin(3.*np.pi/8.))),
        Vertex((-sec_b/2.+k+dist*np.cos(1.*np.pi/4.),
                sec_h/2.-sec_tf-dist+dist*np.sin(1.*np.pi/4.))),
        Vertex((-sec_b/2.+k+dist*np.cos(1.*np.pi/8.),
                sec_h/2.-sec_tf-dist+dist*np.sin(1.*np.pi/8.))),

        Vertex((-sec_b/2.+k+dist, sec_h/2.-sec_tf-dist)),
        Vertex((-sec_b/2.+k+dist, -sec_h/2.+sec_tf+dist)),
        Vertex((-sec_b/2.+k+dist*np.cos(1.*np.pi/8.),
                -sec_h/2.+sec_tf+dist-dist*np.sin(1.*np.pi/8.))),
        Vertex((-sec_b/2.+k+dist*np.cos(1.*np.pi/4.),
                -sec_h/2.+sec_tf+dist-dist*np.sin(1.*np.pi/4.))),
        Vertex((-sec_b/2.+k+dist*np.cos(3.*np.pi/8.),
                -sec_h/2.+sec_tf+dist-dist*np.sin(3.*np.pi/8.))),

        Vertex((-sec_b/2.+k, -sec_h/2.+sec_tf)),
        Vertex((-sec_b/2., -(sec_h/2.-sec_tf))),
        Vertex((-sec_b/2., -sec_h/2.)),
        Vertex((sec_b/2., -sec_h/2.)),
        Vertex((sec_b/2., -(sec_h/2-sec_tf))),
        Vertex((+sec_b/2.-k, -sec_h/2.+sec_tf)),
        Vertex((+sec_b/2.-k-dist*np.cos(3.*np.pi/8.),
                -sec_h/2.+sec_tf+dist-dist*np.sin(3.*np.pi/8.))),
        Vertex((+sec_b/2.-k-dist*np.cos(1.*np.pi/4.),
                -sec_h/2.+sec_tf+dist-dist*np.sin(1.*np.pi/4.))),
        Vertex((+sec_b/2.-k-dist*np.cos(1.*np.pi/8.),
                -sec_h/2.+sec_tf+dist-dist*np.sin(1.*np.pi/8.))),

        Vertex((+sec_b/2.-k-dist, -sec_h/2.+sec_tf+dist)),
        Vertex((+sec_b/2.-k-dist, +sec_h/2.-sec_tf-dist)),
        Vertex((+sec_b/2.-k-dist*np.cos(1.*np.pi/8.),
                +sec_h/2.-sec_tf-dist+dist*np.sin(1.*np.pi/8.))),
        Vertex((+sec_b/2.-k-dist*np.cos(1.*np.pi/4.),
                +sec_h/2.-sec_tf-dist+dist*np.sin(1.*np.pi/4.))),
        Vertex((+sec_b/2.-k-dist*np.cos(3.*np.pi/8.),
                +sec_h/2.-sec_tf-dist+dist*np.sin(3.*np.pi/8.))),

        Vertex((+sec_b/2.-k, sec_h/2.-sec_tf)),
        Vertex((sec_b/2., sec_h/2.-sec_tf))
    ]
    edges = define_edges(vertices)
    return generate(edges)


def hss_rect_mesh(sec_ht: float, sec_b: float, sec_t: float):
    """
    Defines a loop of counterclockwise halfedges
    that form the shape of the rectangular HSS
    with the specified parameters.
    The origin coincides with the centroid.
    Input:
        ht (float): Overall depth
        b (float): Overall width
        t (float): Wall thickness
    """
    dim_a = sec_b / 2.
    dim_c = sec_ht / 2.
    dim_u = dim_a - sec_t
    dim_v = dim_c - sec_t
    dim_e = common.EPSILON
    vertices = [
        Vertex((+dim_e, -dim_c)),
        Vertex((+dim_a, -dim_c)),
        Vertex((+dim_a, +dim_c)),
        Vertex((-dim_a, +dim_c)),
        Vertex((-dim_a, -dim_c)),
        Vertex((-dim_e, -dim_c)),
        Vertex((-dim_e, -dim_v)),
        Vertex((-dim_u, -dim_v)),
        Vertex((-dim_u, +dim_v)),
        Vertex((+dim_u, +dim_v)),
        Vertex((+dim_u, -dim_v)),
        Vertex((+dim_e, -dim_v)),
    ]
    edges = define_edges(vertices)
    return generate(edges)


def hss_circ_mesh(o_d: float, t_des: float, n_pts: int):
    """
    Defines a loop of counterclockwise halfedges
    that form the shape of the circular
    HSS with the specified parameters.
    The origin coincides with the centroid.
    Input:
        o_d (float): Outside diameter
        t_des (float): Design wall thickness
        n_pts (int) Number of points to approximate
              a circle.
    """
    dim_e = common.EPSILON
    t_param = np.linspace(0., 2.*np.pi, n_pts)
    pts_normalized: nparr = np.column_stack(
        (np.sin(t_param), -np.cos(t_param)))
    pts_outer = pts_normalized * o_d
    pts_outer[0, 0] += dim_e
    pts_outer[-1, 0] -= dim_e
    pts_inner: nparr = np.flip(pts_normalized * (o_d - t_des), axis=0)
    pts_inner[0, 0] -= dim_e
    pts_inner[-1, 0] += dim_e
    pts_all: nparr = np.concatenate((pts_outer, pts_inner))
    vertices = []
    for point in pts_all:
        vertices.append(Vertex((point[0], point[1])))
    edges = define_edges(vertices)
    return generate(edges)


def rect_mesh(dim_b, dim_h):
    """
    Defines a loop of counterclockwise halfedges
    that form the shape of the rectangular section with
    the specified parameters.
    The origin coincides with the centroid.
    Input:
        b: total width
        h: total height
    """
    vertices = [
        Vertex((dim_b/2., dim_h/2.)),
        Vertex((-dim_b/2., dim_h/2.)),
        Vertex((-dim_b/2., -dim_h/2.)),
        Vertex((dim_b/2., -dim_h/2.))
    ]
    edges = define_edges(vertices)
    return generate(edges)


def generic_snap_points(mesh: Mesh) -> dict[str, nparr]:
    """
    Generates generic snap poitns
    for a section object.
    """
    bbox = mesh.bounding_box()
    z_min, y_min, z_max, y_max = bbox.flatten()
    snap_points: dict[str, nparr] = {}
    snap_points['centroid'] = - np.array([0., 0.])
    snap_points['top_center'] = - np.array([0., y_max])
    snap_points['top_left'] = - np.array([z_min, y_max])
    snap_points['top_right'] = - np.array([z_max, y_max])
    snap_points['center_left'] = - np.array([z_min, 0.])
    snap_points['center_right'] = - np.array([z_max, 0.])
    snap_points['bottom_center'] = - np.array([0., y_min])
    snap_points['bottom_left'] = - np.array([z_min, y_min])
    snap_points['bottom_right'] = - np.array([z_max, y_min])
    return snap_points
