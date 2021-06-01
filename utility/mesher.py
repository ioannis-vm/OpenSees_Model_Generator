"""
TODO Module docstring
"""

from __future__ import annotations
from typing import List
from itertools import count
import pandas as pd
import numpy as np
import skgeom as sg
import matplotlib.pyplot as plt


class Vertex:
    """
    2D Vertex.
    It knows all the edges connected to it.
    Each instance has an automatically generated unique id.
    """
    _ids = count(0)

    def __init__(self, coords: tuple[float, float]):
        self.coords = coords
        self.edges = []
        self.nid = next(self._ids)

    def __eq__(self, other):
        return self.nid == other.nid

    def __repr__(self):
        return str(self.nid)


class Edge:
    """
    2D oriented Edge.
    Connected to two vertices v_i and v_j.
    Has two halfedges, h_i and h_j.
    Each instance has an automatically generated unique id.
    """

    _ids = count(0)

    def __init__(self, v_i: Vertex, v_j: Vertex):
        self.v_i = v_i
        self.v_j = v_j
        self.nid = next(self._ids)
        self.h_i = None
        self.h_j = None
        if self not in self.v_i.edges:
            self.v_i.edges.append(self)
        if self not in self.v_j.edges:
            self.v_j.edges.append(self)

    def __repr__(self):
        return str(self.nid)

    def define_halfedge(self, vertex: Vertex):
        """
        For the current edge instance and given one of its vertices,
        we want the halfedge that points to the direction
        away from the given vertex.
        We create it if it does not exist.
        """
        if vertex == self.v_i:
            if not self.h_i:
                halfedge = Halfedge(self.v_i, self)
                self.h_i = halfedge
            else:
                halfedge = self.h_i
        elif vertex == self.v_j:
            if not self.h_j:
                halfedge = Halfedge(self.v_j, self)
                self.h_j = halfedge
            else:
                halfedge = self.h_j
        else:
            raise ValueError(
                "The edge is not connected to the given vertex.")
        return halfedge

    def other_vertex(self, vertex):
        """
        We have an edge instance.
        It has two vertices. This method returns
        the other vertex provided one of the vertices.
        """
        if self.v_i == vertex:
            v_other = self.v_j
        elif self.v_j == vertex:
            v_other = self.v_i
        else:
            raise ValueError("The edge is not connected to the given vertex")
        return v_other


class Halfedge:
    """
    Halfedge object.
    Every edge has two halfedges.
    A halfedge has a direction, pointing from one
    of the corresponding edge's vertices to the other.
    The `vertex` attribute corresponds to the
    edge's vertex that the halfedge originates from.
    Halfedges have a `next` attribute that
    points to the next halfedge, forming closed
    loops, or sequences, that use here to retrieve
    the faces from the given edges and vertices,
    which is the purpose of this module.
    """

    _ids = count(0)

    def __init__(self, vertex: Vertex, edge: Edge):
        self.vertex = vertex
        self.edge = edge
        self.nid = self.nid = next(self._ids)
        self.nxt = None

    def __repr__(self):
        return str(self.nid)

    def direction(self):
        """
        Calculates the angular direction of the halfedge
        using the arctan2 function
        """
        drct = (np.array(self.edge.other_vertex(self.vertex).coords) -
                np.array(self.vertex.coords))
        drct = drct / np.linalg.norm(drct)
        return np.arctan2(drct[1], drct[0])


class Mesh:
    """
    A container that holds a list of unique halfedges.
    Vertices and edges can be retrieved from those.
    The mesh is assumed to be flat (2D).
    """

    def __init__(self, halfedges: List[Halfedge]):
        self.halfedges = halfedges

    def __repr__(self):
        num = len(self.halfedges)
        return("Mesh object containing " + str(num) + " halfedges.")

    def geometric_properties(self):
        coords = np.array([h.vertex.coords for h in self.halfedges])
        return geometric_properties(coords)

    def bounding_box(self):
        coords = np.array([h.vertex.coords for h in self.halfedges])
        xmin = min(coords[:, 0])
        xmax = max(coords[:, 0])
        ymin = min(coords[:, 1])
        ymax = max(coords[:, 1])
        return(np.array([[xmin, ymin], [xmax, ymax]]))


def define_halfedges(edges):
    """
    Use the specified edges and the `traverse` function
    to define the halfedges of the mesh.
    """

    def traverse(h_start, v_start, v_current, e_current, h_current):
        """
        This is the heart of the algorithm; a recursive procedure
        that allows traversing through the entire datastructure
        of vertices and edges to define all the halfedges and their
        next halfedge.
        All the defined halfedges form a number of closed sequences.
        """

        def ang_reduce(ang):
            while ang < 0:
                ang += 2.*np.pi
            while ang >= 2.*np.pi:
                ang -= 2.*np.pi
            return ang

        v_next = e_current.other_vertex(v_current)
        if v_next == v_start:
            h_current.nxt = h_start
        else:
            halfedges = []  # [h.e for h in halfedges]
            for edge in v_next.edges:
                halfedges.append(edge.define_halfedge(v_next))

            angles = np.full(len(halfedges), 0.00)
            for i, h_other in enumerate(halfedges):
                if h_other.edge == h_current.edge:
                    angles[i] = 1000.
                else:
                    angles[i] = ang_reduce(
                        (h_current.direction() - np.pi) - h_other.direction())

            h_current.nxt = halfedges[np.argmin(angles)]

            v_current = h_current.nxt.vertex
            e_current = h_current.nxt.edge
            h_current = h_current.nxt

            traverse(h_start, v_start, v_current, e_current, h_current)

            del halfedges[np.argmin(angles)]

            halfedges_without_next = []
            for halfedge in halfedges:
                if not halfedge.nxt:
                    halfedges_without_next.append(halfedge)
            for halfedge in halfedges_without_next:
                h_start = halfedge
                v_start = halfedge.vertex
                v_current = halfedge.vertex
                e_current = halfedge.edge
                h_current = halfedge

                traverse(h_start, v_start, v_current, e_current, h_current)

    e_start = edges[0]
    v_start = e_start.v_i

    h_start = e_start.define_halfedge(v_start)

    v_current = v_start
    e_current = e_start
    h_current = h_start

    traverse(h_start, v_start, v_current, e_current, h_current)

    # collect the retrieved halfedges in a list
    # (add unique halfedges)
    halfedges = []
    for edge in edges:
        if edge.h_i not in halfedges:
            halfedges.append(edge.h_i)
        if edge.h_j not in halfedges:
            halfedges.append(edge.h_j)

    return halfedges


def geometric_properties(coords):
    """
    Given an array of coordinates
    that represents a simple closed polygon,
    calculate the centroid and rotational moment of inertia
    around the centroid.
    Note: even though the polygon is closed, the last point should
    not be repeated in the input matrix.
    """

    # repeat the first row at the end to close the shape
    coords = np.vstack((coords, coords[0, :]))
    # # reverse the order to make it counterclockwise
    # coords = np.flip(coords, axis=0)
    x = coords[:, 0]

    y = coords[:, 1]
    area = np.sum(x * np.roll(y, -1) -
                  np.roll(x, -1) * y) / 2.00
    x_cent = (np.sum((x + np.roll(x, -1)) *
                     (x*np.roll(y, -1) -
                      np.roll(x, -1)*y)))/(6.0*area)
    y_cent = (np.sum((y + np.roll(y, -1)) *
                     (x*np.roll(y, -1) -
                      np.roll(x, -1)*y)))/(6.0*area)
    centroid = np.array((x_cent, y_cent))
    coords_centered = coords - centroid
    x = coords_centered[:, 0]
    y = coords_centered[:, 1]
    alpha = x * np.roll(y, -1) - \
        np.roll(x, -1) * y
    # planar moment of inertia wrt horizontal axis
    ixx = np.sum((y**2 + y * np.roll(y, -1) +
                  np.roll(y, -1)**2)*alpha)/12.00
    # planar moment of inertia wrt vertical axis
    iyy = np.sum((x**2 + x * np.roll(x, -1) +
                  np.roll(x, -1)**2)*alpha)/12.00

    ixy = np.sum((x*np.roll(y, -1)
                  + 2.0*x*y
                  + 2.0*np.roll(x, -1) * np.roll(y, -1)
                  + np.roll(x, -1) * y)*alpha)/24.
    # polar (torsional) moment of inertia
    ir = ixx + iyy
    # mass moment of inertia wrt in-plane rotation
    ir_mass = (ixx + iyy) / area

    inertia = {'ixx': ixx, 'iyy': iyy,
               'ixy': ixy, 'ir': ir, 'ir_mass': ir_mass}

    return {'area': area, 'centroid': centroid, 'inertia': inertia}


def obtain_closed_loops(halfedges):
    """
    Given a list of the unique halfedges of a mesh,
    determine all the faces of the mesh, represented
    as lists of halfedge sequences (closed loops).
    """
    def is_in_some_loop(halfedge, loops):
        for loop in loops:
            if halfedge in loop:
                return True
        return False

    loops = []
    for halfedge in halfedges:
        if loops:
            if is_in_some_loop(halfedge, loops):
                continue
        loop = [halfedge]
        nxt = halfedge.nxt
        while(nxt != halfedge):
            loop.append(nxt)
            nxt = nxt.nxt
        loops.append(loop)
    # remove the largest loop (that corresponds to the exterior halfedges)
    loop_areas = [geometric_properties(
        np.array([h.vertex.coords for h in loop]))['area']
        for loop in loops]
    # TODO. CAUTION: This assumes there is only one exterior loop
    # Maybe it should separate all negative areas instead
    outer = min(loop_areas)
    index = loop_areas.index(outer)
    external_loop = loops[index]
    del loops[index]
    del loop_areas[index]
    return external_loop, loops, loop_areas


def print_halfedge_results(halfedges):

    results = {
        'halfedge': [],
        'vertex': [],
        'edge': [],
        'next': [],
    }

    for h in halfedges:
        results['halfedge'].append(h)
        results['vertex'].append(h.vertex)
        results['edge'].append(h.edge)
        results['next'].append(h.nxt)

    df = pd.DataFrame(results)
    print(df)


def plot_mesh(halfedge_loop):
    n = len(halfedge_loop)
    coords = np.full((n+1, 2), 0.00)
    for i, h in enumerate(halfedge_loop):
        coords[i, :] = h.vertex.coords
    coords[-1, :] = coords[0, :]
    fig = plt.figure()
    plt.plot(coords[:, 0], coords[:, 1])
    plt.scatter(coords[:, 0], coords[:, 1])
    fig.show()


if __name__ == "__main()__":
    pass
