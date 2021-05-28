"""
TODO Module docstring
"""

from __future__ import annotations
import pandas as pd
from itertools import count
import numpy as np


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


def ang_reduce(ang):
    while ang < 0:
        ang += 2.*np.pi
    while ang >= 2.*np.pi:
        ang -= 2.*np.pi
    return ang


def traverse(h_start, v_start, v_current, e_current, h_current):
    """
    This is the heart of the algorithm; a recursive procedure
    that allows traversing through the entire datastructure
    of vertices and edges to define all the halfedges and their
    next halfedge.
    All the defined halfedges form a number of closed sequences.
    TODO not exactly... For each closed sequence, one of its halfedges is stored
    in a list, to enable access to the sequences once the algorithm
    has finished.
    """
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


def define_halfedges(edges):
    e_start = edges[0]
    v_start = e_start.v_i

    h_start = e_start.define_halfedge(v_start)

    v_current = v_start
    e_current = e_start
    h_current = h_start

    traverse(h_start, v_start, v_current, e_current, h_current)


if __name__ == "__main()__":
    pass
