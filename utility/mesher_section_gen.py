from utility import mesher, common
import numpy as np


def generate(edges):
    halfedges = mesher.define_halfedges(edges)
    loops = mesher.obtain_closed_loops(halfedges)
    external, _, trivial = mesher.orient_loops(loops)
    mesher.sanity_checks(external, trivial)
    return mesher.Mesh(external[0])


def define_edges(vertices):
    n_v = len(vertices)
    edges = []
    for i in range(n_v - 1):
        vi = vertices[i]
        vj = vertices[i+1]
        edges.append(mesher.Edge(vi, vj))
    vi = vertices[-1]
    vj = vertices[0]
    edges.append(mesher.Edge(vi, vj))
    return edges


def w_mesh(b, h, tw, tf):
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
    """
    vertices = [
        mesher.Vertex((b/2., h/2.)),
        mesher.Vertex((-b/2., h/2.)),
        mesher.Vertex((-b/2., h/2.-tf)),
        mesher.Vertex((-tw/2., h/2.-tf)),
        mesher.Vertex((-tw/2., -(h/2.-tf))),
        mesher.Vertex((-b/2., -(h/2.-tf))),
        mesher.Vertex((-b/2., -h/2.)),
        mesher.Vertex((b/2., -h/2.)),
        mesher.Vertex((b/2., -(h/2-tf))),
        mesher.Vertex((tw/2., -(h/2-tf))),
        mesher.Vertex((tw/2., h/2.-tf)),
        mesher.Vertex((b/2., h/2.-tf))
    ]
    edges = define_edges(vertices)
    return generate(edges)


def HSS_rect_mesh(ht: float, b: float, t: float):
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
    a = b / 2.
    c = ht / 2.
    u = a - t
    v = c - t
    e = common.EPSILON
    vertices = [
        mesher.Vertex((+e, -c)),
        mesher.Vertex((+a, -c)),
        mesher.Vertex((+a, +c)),
        mesher.Vertex((-a, +c)),
        mesher.Vertex((-a, -c)),
        mesher.Vertex((-e, -c)),
        mesher.Vertex((-e, -v)),
        mesher.Vertex((-u, -v)),
        mesher.Vertex((-u, +v)),
        mesher.Vertex((+u, +v)),
        mesher.Vertex((+u, -v)),
        mesher.Vertex((+e, -v)),
    ]
    edges = define_edges(vertices)
    return generate(edges)


def HSS_circ_mesh(od: float, tdes: float, n_pts: int):
    """
    Defines a loop of counterclockwise halfedges
    that form the shape of the circular
    HSS with the specified parameters.
    The origin coincides with the centroid.
    Input:
        od (float): Outside diameter
        tdes (float): Design wall thickness
        n_pts (int) Number of points to approximate
              a circle.
    """
    e = common.EPSILON
    t_param = np.linspace(0., 2.*np.pi, n_pts)
    pts_normalized = np.column_stack(
        (np.sin(t_param), -np.cos(t_param)))
    pts_outer = pts_normalized * od
    pts_outer[0, 0] += e
    pts_outer[-1, 0] -= e
    pts_inner = np.flip(pts_normalized * (od - tdes), axis=0)
    pts_inner[0, 0] -= e
    pts_inner[-1, 0] += e
    pts_all = np.concatenate((pts_outer, pts_inner))
    vertices = []
    for point in pts_all:
        vertices.append(mesher.Vertex(tuple(point)))
    edges = define_edges(vertices)
    return generate(edges)


def rect_mesh(b, h):
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
        mesher.Vertex((b/2., h/2.)),
        mesher.Vertex((-b/2., h/2.)),
        mesher.Vertex((-b/2., -h/2.)),
        mesher.Vertex((b/2., -h/2.))
    ]
    edges = define_edges(vertices)
    return generate(edges)
