"""
TODO Module docstring
"""

from __future__ import annotations
from itertools import count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skgeom as sg


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

    halfedges = []
    for edge in edges:
        if edge.h_i not in halfedges:
            halfedges.append(edge.h_i)
        if edge.h_j not in halfedges:
            halfedges.append(edge.h_j)

    return halfedges


def accumulate_areas(areas, index, value):
    """
    Using a dictionary (`areas`) to accumulate the areas
    associated with each vertex.
    `index` corresponds to the unique id of the vertex.
    `value` is the value to be added at the givn index.
    """
    try:
        # assuming the key already exists
        areas[index] = areas[index] + value
    except KeyError:
        # create it if it doesn't already exist
        areas[index] = value


def is_in_some_miniloop(halfedge, loops):
    for loop in loops:
        for other_halfedge in loop:
            if (other_halfedge.vertex.point ==
                halfedge.vertex.point and
                    other_halfedge.next.vertex.point ==
                    halfedge.next.vertex.point):
                return True
    return False


def area_from_vertices(vertices):
    return sg.Polygon(vertices).area()


def is_in_some_loop(halfedge, loops):
    for loop in loops:
        if halfedge in loop:
            return True
    return False


def draw_skeleton(fig, polygon, skeleton, show_time=False):

    coord_array = polygon.coords
    coord_array = np.vstack((coord_array, coord_array[0, :]))

    plt.plot(coord_array[:, 0], coord_array[:, 1], 'black', lw=2.0)

    for h in skeleton.halfedges:
        if h.is_bisector:
            p1 = h.vertex.point
            p2 = h.opposite.vertex.point
            plt.plot([p1.x(), p2.x()], [p1.y(), p2.y()], 'r-', lw=1.5)

    if show_time:
        for v in skeleton.vertices:
            plt.gcf().gca().add_artist(plt.Circle(
                (v.point.x(), v.point.y()),
                v.time, color='grey', fill=False, lw=0.20))


def obtain_closed_loops(halfedges):
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
    loop_areas = [area_from_vertices([h.vertex.coords
                                      for h in loop])
                  for loop in loops]
    outer = min(loop_areas)
    index = loop_areas.index(outer)
    del loops[index]
    del loop_areas[index]
    return loops, loop_areas


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


def plot_tributary_areas(loops):
    fig = plt.figure()

    for loop in loops:
        poly = sg.Polygon([h.vertex.coords for h in loop])
        skel = sg.skeleton.create_interior_straight_skeleton(poly)
        draw_skeleton(fig, poly, skel, show_time=True)
    fig.show()


def calculate_tributary_areas_from_loops(loops):

    # accumulate area part
    areas = {}

    for loop in loops:

        poly = sg.Polygon([h.vertex.coords for h in loop])
        skel = sg.skeleton.create_interior_straight_skeleton(poly)

        miniloops = []
        for halfedge in skel.halfedges:
            if miniloops:
                if is_in_some_miniloop(halfedge, miniloops):
                    continue
            miniloop = [halfedge]
            nxt = halfedge.next
            while(nxt.vertex.point != halfedge.vertex.point):
                miniloop.append(nxt)
                nxt = nxt.next
            miniloops.append(miniloop)

        miniloop_areas = [area_from_vertices(
            [h.vertex.point for h in miniloop]) for miniloop in miniloops]
        outer = min(miniloop_areas)
        index = miniloop_areas.index(outer)
        del miniloops[index]
        del miniloop_areas[index]

        for i, miniloop in enumerate(miniloops):
            area = miniloop_areas[i]
            loop_edges = [h.edge for h in loop]
            for halfedge in miniloop:
                for edge in loop_edges:
                    v_i = sg.Point2(*edge.v_i.coords)
                    v_j = sg.Point2(*edge.v_j.coords)
                    pt_1 = halfedge.vertex.point
                    pt_2 = halfedge.next.vertex.point
                    if ((pt_1 == v_i and pt_2 == v_j) or (pt_1 == v_j and pt_2 == v_i)):
                        accumulate_areas(areas, edge.nid, area)
    return areas


def tributary_areas(edges, show_figure=False, print_halfedges=False):
    halfedges = define_halfedges(edges)
    if print_halfedges:
        print_halfedge_results(halfedges)
    loops, loop_areas = obtain_closed_loops(halfedges)
    if show_figure:
        plot_tributary_areas(loops)
    areas = calculate_tributary_areas_from_loops(loops)
    return areas


if __name__ == "__main()__":
    pass
