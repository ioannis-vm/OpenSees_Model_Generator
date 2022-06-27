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

import mesher
import common
import numpy as np


def generate(edges):
    halfedges = mesher.define_halfedges(edges)
    loops = mesher.obtain_closed_loops(halfedges)
    external, internal, trivial = mesher.orient_loops(loops)
    mesher.sanity_checks(internal, trivial)
    return mesher.Mesh(internal[0])


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


def w_mesh(b, h, tw, tf, target_area=None):
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
    area_diff = target_area - (b*tf*2.+(h-2*tf)*tw)
    if area_diff < 0:
        # This happens for W14X426
        area_diff = 1e-4
        
    r = np.sqrt(area_diff/(2.**2-np.pi)) * 0.9565
    # note: 0.9565 is a correction factor to account for
    # the fact that we approximate the arcs with
    # four line segments, thus putting more material in there
    k = (b - 2. * r - tw) / 2.
    vertices = [
        mesher.Vertex((b/2., h/2.)),
        mesher.Vertex((-b/2., h/2.)),
        mesher.Vertex((-b/2., h/2.-tf)),
        mesher.Vertex((-b/2.+k, h/2.-tf)),
        mesher.Vertex((-b/2.+k+r * np.cos(3.*np.pi/8.),
                       h/2.-tf-r + r*np.sin(3.*np.pi/8.))),
        mesher.Vertex((-b/2.+k+r*np.cos(1.*np.pi/4.),
                       h/2.-tf-r+r*np.sin(1.*np.pi/4.))),
        mesher.Vertex((-b/2.+k+r*np.cos(1.*np.pi/8.),
                       h/2.-tf-r+r*np.sin(1.*np.pi/8.))),

        mesher.Vertex((-b/2.+k+r, h/2.-tf-r)),
        mesher.Vertex((-b/2.+k+r, -h/2.+tf+r)),
        mesher.Vertex((-b/2.+k+r*np.cos(1.*np.pi/8.),
                       -h/2.+tf+r-r*np.sin(1.*np.pi/8.))),
        mesher.Vertex((-b/2.+k+r*np.cos(1.*np.pi/4.),
                       -h/2.+tf+r-r*np.sin(1.*np.pi/4.))),
        mesher.Vertex((-b/2.+k+r*np.cos(3.*np.pi/8.),
                       -h/2.+tf+r-r*np.sin(3.*np.pi/8.))),

        mesher.Vertex((-b/2.+k, -h/2.+tf)),
        mesher.Vertex((-b/2., -(h/2.-tf))),
        mesher.Vertex((-b/2., -h/2.)),
        mesher.Vertex((b/2., -h/2.)),
        mesher.Vertex((b/2., -(h/2-tf))),
        mesher.Vertex((+b/2.-k, -h/2.+tf)),
        mesher.Vertex((+b/2.-k-r*np.cos(3.*np.pi/8.),
                       -h/2.+tf+r-r*np.sin(3.*np.pi/8.))),
        mesher.Vertex((+b/2.-k-r*np.cos(1.*np.pi/4.),
                       -h/2.+tf+r-r*np.sin(1.*np.pi/4.))),
        mesher.Vertex((+b/2.-k-r*np.cos(1.*np.pi/8.),
                       -h/2.+tf+r-r*np.sin(1.*np.pi/8.))),

        mesher.Vertex((+b/2.-k-r, -h/2.+tf+r)),
        mesher.Vertex((+b/2.-k-r, +h/2.-tf-r)),
        mesher.Vertex((+b/2.-k-r*np.cos(1.*np.pi/8.),
                       +h/2.-tf-r+r*np.sin(1.*np.pi/8.))),
        mesher.Vertex((+b/2.-k-r*np.cos(1.*np.pi/4.),
                       +h/2.-tf-r+r*np.sin(1.*np.pi/4.))),
        mesher.Vertex((+b/2.-k-r*np.cos(3.*np.pi/8.),
                       +h/2.-tf-r+r*np.sin(3.*np.pi/8.))),

        mesher.Vertex((+b/2.-k, h/2.-tf)),
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
        vertices.append(mesher.Vertex((point[0], point[1])))
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


def generic_snap_points(mesh: mesher.Mesh) -> dict:
    """
    Generates generic snap poitns
    for a section object.
    """
    bbox = mesh.bounding_box()
    z_min, y_min, z_max, y_max = bbox.flatten()
    snap_points = {}
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
