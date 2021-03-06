"""
Enables functionality by utilizing the `halfedge`
data structure.
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ / 
# / /_/ / / / / / / /_/ /_/  
# \____/_/ /_/ /_/\__, (_)   
#                /____/      
#                            
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from __future__ import annotations
from itertools import count
from descartes.patch import PolygonPatch
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as shapely_Polygon
from utility import common


class Vertex:
    """
    2D Vertex.
    It knows all the edges connected to it.
    It knows all the halfedges leaving from it.
    Each instance has an automatically generated unique id.
    """
    _ids = count(0)

    def __init__(self, coords: tuple[float, float]):
        self.coords = coords
        self.edges = []
        self.halfedges = []
        self.uid = next(self._ids)

    def __eq__(self, other):
        return self.uid == other.uid

    def __repr__(self):
        return str(self.uid)


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
        self.uid = next(self._ids)
        self.h_i = None
        self.h_j = None
        if self not in self.v_i.edges:
            self.v_i.edges.append(self)
        if self not in self.v_j.edges:
            self.v_j.edges.append(self)

    def __repr__(self):
        return str(self.uid)

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
                raise ValueError('Halfedge h_i already defined')
        elif vertex == self.v_j:
            if not self.h_j:
                halfedge = Halfedge(self.v_j, self)
                self.h_j = halfedge
            else:
                raise ValueError('Halfedge h_j already defined')
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
        self.uid = self.uid = next(self._ids)
        self.nxt = None

    def __repr__(self):
        return str(self.uid)

    def __lt__(self, other):
        return self.uid < other.uid

    def direction(self):
        """
        Calculates the angular direction of the halfedge
        using the arctan2 function
        """
        drct = (np.array(self.edge.other_vertex(self.vertex).coords) -
                np.array(self.vertex.coords))
        norm = np.linalg.norm(drct)
        drct /= norm
        return np.arctan2(drct[1], drct[0])


class Mesh:
    """
    A container that holds a list of unique halfedges.
    Vertices and edges can be retrieved from those.
    The mesh is assumed to be flat (2D).
    """

    def __init__(self, halfedges: list[Halfedge]):
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


############################################
# Geometric Properties of Polygonal Shapes #
############################################


def polygon_area(coords: np.ndarray) -> float:
    """
    Calculates the area of a polygon.
    Args:
        coords: A matrix whose columns represent
                the coordinates and the rows
                represent the points of the polygon.
                The first point should not be repeated
                at the end, as this is done
                automatically.
    Returns:
        area (float): The area of the polygon.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    return np.sum(x * np.roll(y, -1) -
                  np.roll(x, -1) * y) / 2.00


def polygon_centroid(coords: np.ndarray) -> np.ndarray:
    """
    Calculates the centroid of a polygon.
    Args:
        coords: A matrix whose columns represent
                the coordinates and the rows
                represent the points of the polygon.
                The first point should not be repeated
                at the end, as this is done
                automatically.
    Returns:
        centroid (np.ndarray): The centroid of
                 the polygon.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    area = polygon_area(coords)
    x_cent = (np.sum((x + np.roll(x, -1)) *
                     (x*np.roll(y, -1) -
                      np.roll(x, -1)*y)))/(6.0*area)
    y_cent = (np.sum((y + np.roll(y, -1)) *
                     (x*np.roll(y, -1) -
                      np.roll(x, -1)*y)))/(6.0*area)
    return np.array((x_cent, y_cent))


def polygon_inertia(coords):
    """
    Calculates the moments of inertia of a polygon.
    Args:
        coords: A matrix whose columns represent
                the coordinates and the rows
                represent the points of the polygon.
                The first point should not be repeated
                at the end, as this is done
                automatically.
    Returns:
        dictionary, containing:
        'ixx': (float) - Moment of inertia around
                         the x axis
        'iyy': (float) - Moment of inertia around
                         the y axis
        'ixy': (float) - Product of inertia
        'ir': (float)  - Polar moment of inertia
        'ir_mass': (float) - Mass moment of inertia
        # TODO
        # The terms might not be pedantically accurate
    """
    x = coords[:, 0]
    y = coords[:, 1]
    area = polygon_area(coords)
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

    return {'ixx': ixx, 'iyy': iyy,
            'ixy': ixy, 'ir': ir, 'ir_mass': ir_mass}


def geometric_properties(coords):
    """
    Aggregates the results of the previous functions.
    """

    # repeat the first row at the end to close the shape
    coords = np.vstack((coords, coords[0, :]))
    area = polygon_area(coords)
    centroid = polygon_centroid(coords)
    coords_centered = coords - centroid
    inertia = polygon_inertia(coords_centered)

    return {'area': area, 'centroid': centroid, 'inertia': inertia}


##################################
# Defining halfedges given edges #
##################################


def define_halfedges(edges: list[Edge]) -> list[Halfedge]:
    """
    Given a list of edges, defines all the halfedges and
    associates them with their `next`.
    See note:
        https://notability.com/n/0wlJ17mt81uuVWAYVoFfV3
    To understand how it works.
    Description:
        Each halfedge stores information about its edge, vertex
        and and next halfedge. Contrary to convention, we don't
        store the twin (opposite) halfedge here, seince we don't
        need it anywhere.
    Args:
        edges (list[Edge]): List of Edge objects
    Returns:
        halfedges (list[Halfedge]): List of Halfedge objects
    """

    def ang_reduce(ang):
        while ang < 0:
            ang += 2.*np.pi
        while ang >= 2.*np.pi:
            ang -= 2.*np.pi
        return ang

    all_halfedges = []
    for edge in edges:
        v_i = edge.v_i
        v_j = edge.v_j
        h_i = edge.define_halfedge(v_i)
        h_j = edge.define_halfedge(v_j)
        all_halfedges.append(h_i)
        all_halfedges.append(h_j)
        v_i.halfedges.append(h_i)
        v_j.halfedges.append(h_j)

    # at this point we have defined all halfedges, but
    # none of them knows its `next`.
    # We now assign that attribute to all halfedges

    for h in all_halfedges:
        # We are looking for `h`'s `next`
        # determine the vertex that it starts from
        v_from = h.vertex
        # determine the vertex that it points to
        v_to = h.edge.other_vertex(v_from)
        # get a list of all halfedges leaving that vertex
        candidates_for_next = v_to.halfedges
        # determine which of all these halfedges will be the next
        angles = np.full(len(candidates_for_next), 0.00)
        for i, h_other in enumerate(candidates_for_next):
            if h_other.edge == h.edge:
                angles[i] = 1000.
                # otherwise we would assign its conjugate as next
            else:
                angles[i] = ang_reduce(
                    (h.direction() - np.pi) - h_other.direction())
        h.nxt = candidates_for_next[np.argmin(angles)]

    return all_halfedges

    # # debug
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal')
    # for edge in edges:
    #     p1 = edge.v_i.coords
    #     p2 = edge.v_j.coords
    #     coords = np.row_stack((p1, p2))
    #     ax.plot(coords[:, 0], coords[:, 1])
    # for h in halfedges:
    #     if h.nxt:
    #         h_nxt = h.nxt
    #         e = h.edge
    #         if h_nxt.edge:
    #             e_nxt = h_nxt.edge
    #             p1 = (np.array(e.v_i.coords) + np.array(e.v_j.coords))/2.
    #             p2 = (np.array(e_nxt.v_i.coords) + np.array(e_nxt.v_j.coords))/2.
    #             dx = p2 - p1
    #             ax.arrow(*p1, *dx)
    # plt.show()


def obtain_closed_loops(halfedges):
    """
    Given a list of halfedges,
    this function uses their `next` attribute to
    group them into sequences of closed loops
    (ordered lists of halfedges of which the
    `next` halfedge of the last list element
    points to the first halfedge in the list, and
    the `next` halfedge of any list element
    points to the next halfedge in the list.
    Args:
        halfedges (list[Halfedge]):
                  list of halfedges
    Returns:
        loops (list[list[Halfedge]]) with the
              aforementioned property.
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
    return loops


def orient_loops(loops):
    """
    Separates loops to internal (counterclockwise)
    and external (clockwise). Also gathers trivial
    loops, i.e. halfedge sequences that define polygons
    that have no area (e.g. h1 -> h2 -> h1).
    Args:
        loops (list[list[Halfedge]]) (see `obtain_closed_loops`)
    Returns:
        external_loops (list[list[Halfedge]])
        internal_loops (list[list[Halfedge]])
        trivial_loops (list[list[Halfedge]])
    """
    internal_loops = []
    external_loops = []
    trivial_loops = []
    loop_areas = [polygon_area(
        np.array([h.vertex.coords for h in loop]))
        for loop in loops]
    for i, area in enumerate(loop_areas):
        if area > common.EPSILON:
            internal_loops.append(loops[i])
        elif area < -common.EPSILON:
            external_loops.append(loops[i])
        else:
            trivial_loops.append(loops[i])
    return external_loops, internal_loops, trivial_loops


#######################################
# Breaking a shape into little pieces #
#######################################


def subdivide_polygon(halfedges, n_x=10, n_y=25, plot=False):
    """
    Used to define the fibers of fiber sections.
    Args:
        halfedges (list[Halfedge]): Sequence of halfedges
                  that defines the shape of a section.
        n_x (int): Number of spatial partitions in the x direction
        n_y (int): Number of spatial partitions in the y direction
        plot (bool): Plots the resulting polygons for debugging 
    Returns:
        pieces (list[shapely_Polygon]): shapely_Polygon
               objects that represent single fibers.
    """
    section_polygon = shapely_Polygon([h.vertex.coords for h in halfedges])
    x_min, y_min, x_max, y_max = section_polygon.bounds
    x_array = np.linspace(x_min, x_max, num=n_x, endpoint=True)
    y_array = np.linspace(y_min, y_max, num=n_y, endpoint=True)
    pieces = []
    for i in range(len(x_array)-1):
        for j in range(len(y_array)-1):
            tile = shapely_Polygon([(x_array[i], y_array[j]),
                                    (x_array[i+1], y_array[j]),
                                    (x_array[i+1], y_array[j+1]),
                                    (x_array[i], y_array[j+1])])
            subregion = section_polygon.intersection(tile)
            if subregion.area != 0.0:
                pieces.append(subregion)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        patch = PolygonPatch(section_polygon, alpha=0.5, zorder=2)
        ax.add_patch(patch)
        for subregion in pieces:
            patch = PolygonPatch(subregion, alpha=0.5, zorder=2)
            ax.add_patch(patch)
        for subregion in pieces:
            ax.scatter(subregion.centroid.x, subregion.centroid.y)
        ax.margins(0.10)
        plt.show()
    return pieces


#############
# Debugging #
#############


def print_halfedge_results(halfedges):
    """
    Prints the ids of the defined halfedges
    and their vertex, edge and next, for
    debugging.
    """
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

    print(results)


def plot_loop(halfedge_loop):
    """
    Plots the vertices/edges of a list of halfedges.
    """
    n = len(halfedge_loop)
    coords = np.full((n+1, 2), 0.00)
    for i, h in enumerate(halfedge_loop):
        coords[i, :] = h.vertex.coords
    coords[-1, :] = coords[0, :]
    fig = plt.figure()
    plt.plot(coords[:, 0], coords[:, 1])
    plt.scatter(coords[:, 0], coords[:, 1])
    fig.show()



def plot_edges(edges):
    """
    Plots the given edges.
    """
    fig = plt.figure()
    for i, e in enumerate(edges):
        coords = np.full((2, 2), 0.00)
        coords[0, :] = e.v_i.coords
        coords[1, :] = e.v_j.coords
        plt.plot(coords[:, 0], coords[:, 1])
    fig.show()


def sanity_checks(external, trivial):
    """
    Perform some checks to make sure
    assumptions are not violated.
    """
    #   We expect no trivial loops
    if trivial:
        print("Warning: Found trivial loop")
        for i, trv in enumerate(trivial):
            for h in trv:
                print(h.vertex.coords)
            plot_loop(trv)
    #   We expect a single external loop
    if len(external) > 1:
        print("Warning: Found multiple external loops")
        for i, ext in enumerate(external):
            print(i+1)
            for h in ext:
                print(h.vertex.coords)
            plot_loop(ext)


if __name__ == "__main()__":
    pass
