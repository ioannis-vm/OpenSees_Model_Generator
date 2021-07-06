"""
Enables functionality by utilizing the `halfedge`
data structure.
"""

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
    Each instance has an automatically generated unique id.
    """
    _ids = count(0)

    def __init__(self, coords: tuple[float, float]):
        self.coords = coords
        self.edges = []
        self.uniq_id = next(self._ids)

    def __eq__(self, other):
        return self.uniq_id == other.uniq_id

    def __repr__(self):
        return str(self.uniq_id)


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
        self.uniq_id = next(self._ids)
        self.h_i = None
        self.h_j = None
        if self not in self.v_i.edges:
            self.v_i.edges.append(self)
        if self not in self.v_j.edges:
            self.v_j.edges.append(self)

    def __repr__(self):
        return str(self.uniq_id)

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
        self.uniq_id = self.uniq_id = next(self._ids)
        self.nxt = None

    def __repr__(self):
        return str(self.uniq_id)

    def __lt__(self, other):
        return self.uniq_id < other.uniq_id

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
    associates them with their `next` using a recursive approach.
    See note:
        https://notability.com/n/0wlJ17mt81uuVWAYVoFfV3
    To understand how it works, it is advised to draw a single
    example and follow the execution of the code while reading
    the comments.
    Description:
        Each halfedge stores information about its edge, vertex
        and and next halfedge. Contrary to convention, we don't
        store the twin (opposite) halfedge.
    Args:
        edges (list[Edge]): List of Edge objects
    Returns:
        halfedges (list[Halfedge]): List of Halfedge objects
    """

    # This function is called inside the current function.
    # It is advised to skip it for now, and read the comments
    # in order to understand the logic.
    def traverse(h_start, v_start, v_current, e_current, h_current):
        """
        This is the recursive procedure
        that allows traversing through the entire datastructure
        of vertices and edges to define all the halfedges.
        Args:
            h_start: The halfedge that we started from.
            v_start: The vertex taht we started from.
            v_current: The current vertex.
            e_current: The current edge.
            h_current: The current halfedge.
        """
        def ang_reduce(ang):
            while ang < 0:
                ang += 2.*np.pi
            while ang >= 2.*np.pi:
                ang -= 2.*np.pi
            return ang
        # Use the current edge and vertex to obtain the
        # next vertex
        v_next = e_current.other_vertex(v_current)
        # If we end up on the vertex that we started from,
        if v_next == v_start:
            # Then the `next` of our current halfedge is the
            # one we started from
            h_current.nxt = h_start
        # Otherwise,
        else:
            # Define the halfedges of all edges that are
            # connected to the vertex `v_next`
            halfedges = []
            for edge in v_next.edges:
                halfedges.append(edge.define_halfedge(v_next))
            # Get the angles from the current halfedge to the
            # newly defined halfedges.
            # We will use those angles to obtain and assign
            # the `next` of the current halfedge.
            # The `next` is the one that forms the smallest
            # angle with the current.
            angles = np.full(len(halfedges), 0.00)
            for i, h_other in enumerate(halfedges):
                if h_other.edge == h_current.edge:
                    # This will be zero, but we are only
                    # interested in the case where the
                    # other halfedge is not the current halfedge.
                    angles[i] = 1000.
                else:
                    angles[i] = ang_reduce(
                        (h_current.direction() - np.pi) - h_other.direction())
            h_current.nxt = halfedges[np.argmin(angles)]
            # We have defined halfedges, and assigned the `next` to our
            # current halfedge. Now we repeat the cycle using that `next`
            # ass our current halfedge.
            v_current = h_current.nxt.vertex
            e_current = h_current.nxt.edge
            h_current = h_current.nxt
            traverse(h_start, v_start, v_current, e_current, h_current)
            # At this point, we have returned from a recusrion step.
            # this can only happen if the `next` of the current halfedge
            # was found to be the initial halfedge, meaning that a loop
            # was formed. Therefore, we need to search for a halfedge
            # that has been defined in the previous steps that doesn't
            # have a `next` assigned to it.
            del halfedges[np.argmin(angles)]
            halfedges_without_next = []
            for halfedge in halfedges:
                if not halfedge.nxt:
                    halfedges_without_next.append(halfedge)
            # If the list is empty, we are done.
            # If the list is not empty, the following code will run.
            for halfedge in halfedges_without_next:
                # Consider that halfedge to be the start of the
                # loop, and initiate the recursive procedure.
                h_start = halfedge
                v_start = halfedge.vertex
                v_current = halfedge.vertex
                e_current = halfedge.edge
                h_current = halfedge
                traverse(h_start, v_start, v_current, e_current, h_current)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Pick an initial edge. Could be any.
    e_start = edges[0]
    # Obtain one of its vertices
    v_start = e_start.v_i
    # Define the first halfedge
    h_start = e_start.define_halfedge(v_start)
    # Store the current v, e and h and start recursion
    v_current = v_start
    e_current = e_start
    h_current = h_start
    # Recursion!
    traverse(h_start, v_start, v_current, e_current, h_current)

    # At this point, all halfedges have been defined and
    #     pointed to their next.
    # Collect the generated halfedges in a list
    halfedges = []  # We will put them here
    for edge in edges:
        halfedges.append(edge.h_i)
        halfedges.append(edge.h_j)
    halfedges.sort()
    return halfedges


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
