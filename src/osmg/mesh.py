"""
Defines objects used in mesing operations.
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

from __future__ import annotations
from typing import Optional
from typing import Any
from itertools import count
from descartes.patch import PolygonPatch  # type: ignore
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as shapely_Polygon  # type: ignore
from . import common

nparr = npt.NDArray[np.float64]


class Vertex:
    """
    2D Vertex.
    It knows all the edges connected to it.
    It knows all the halfedges leaving from it.
    Each instance has an automatically generated unique id.

    Attributes:
        coords: Coordinates of the vertex.
        edges: List of edges connected to the vertex.
        halfedges: List of halfedges leaving from the vertex.
        uid: Unique identifier of the vertex.

    Example:
        >>> from osmg.mesh import Vertex
        >>> v = Vertex((0.0, 0.0))

    """

    _ids = count(0)

    def __init__(self, coords: tuple[float, float]):
        """
        Initializes a new instance of the `Vertex` class.

        Arguments:
            coords: Coordinates of the vertex.

        """

        self.coords = coords
        self.edges: list[Edge] = []
        self.halfedges: list[Halfedge] = []
        self.uid: int = next(self._ids)

    def __eq__(self, other):
        """
        Check for equality based on the uid of the vertex.

        Arguments:
            other: Other vertex to compare with.

        Returns:
            bool: `True` if the two vertices are equal, `False`
                  otherwise.

        Example:
            >>> from osmg.mesh import Vertex
            >>> v1 = Vertex((0, 0))
            >>> v2 = Vertex((1, 1))
            >>> v3 = Vertex((0, 0))
            >>> v1 == v2
            False
            >>> v1 == v3
            False
            >>> v1 == v1
            True

        """

        return self.uid == other.uid

    def __repr__(self):
        """
        Returns a string representation of the vertex.

        Returns:
            String representation of the vertex.

        """

        return f"(V{self.uid} @ {self.coords}) "


class Edge:
    """
    2D oriented Edge. Connected to two vertices `v_i` and `v_j`.  Has
    two halfedges, `h_i` and `h_j`.  Each instance has an
    automatically generated unique id.

    """

    _ids = count(0)

    def __init__(self, v_i: Vertex, v_j: Vertex):
        """
        Initializes a new edge with the given vertices.
        If the vertices don't already have an edge connecting them,
        this edge is added to their list of edges.

        """

        self.v_i = v_i
        self.v_j = v_j
        self.uid = next(self._ids)
        self.h_i: Optional[Halfedge] = None
        self.h_j: Optional[Halfedge] = None
        if self not in self.v_i.edges:
            self.v_i.edges.append(self)
        if self not in self.v_j.edges:
            self.v_j.edges.append(self)

    def __repr__(self):
        """
        Returns a string representation of this edge, in the form
        `(E{self.uid} @ V{self.v_i.uid}, V{self.v_j.uid})`

        """

        return f"(E{self.uid} @ V{self.v_i.uid}, V{self.v_j.uid}) "

    def define_halfedge(self, vertex: Vertex) -> Halfedge:
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
                raise ValueError("Halfedge h_i already defined")
        elif vertex == self.v_j:
            if not self.h_j:
                halfedge = Halfedge(self.v_j, self)
                self.h_j = halfedge
            else:
                raise ValueError("Halfedge h_j already defined")
        else:
            raise ValueError("The edge is not connected to the given vertex.")
        return halfedge

    def other_vertex(self, vertex):
        """
        Returns the vertex of this edge that is not the given vertex.
        If the given vertex is not connected to this edge, a ValueError
        is raised.

        Example:
            >>> from osmg.mesh import Vertex
            >>> v1 = Vertex((0, 0))
            >>> v2 = Vertex((1, 0))
            >>> v3 = Vertex((2, 0))
            >>> e = Edge(v2, v3)
            >>> e.other_vertex(v2).coords
            (2, 0)
            >>> e.other_vertex(v3).coords
            (1, 0)
            >>> e.other_vertex(v1)
            Traceback (most recent call last):
                ...
            ValueError: The edge is not connected to the given vertex

        """

        if self.v_i == vertex:
            v_other = self.v_j
        elif self.v_j == vertex:
            v_other = self.v_i
        else:
            raise ValueError("The edge is not connected to the given vertex")
        return v_other

    def overlaps_or_crosses(self, other: Edge) -> bool:
        """
        Returns True if this edge overlaps or crosses another edge.
        Edges are allowed to share one vertex (returns False), but not
        both (returns True).

        Arguments:
            other : Edge
                The other edge to check for overlap or cross with this
                edge.

        Returns: bool
            True if this edge overlaps or crosses the other edge,
            False otherwise.

        Example:
            >>> from osmg.mesh import Vertex, Edge
            >>> v1 = Vertex((0, 0))
            >>> v2 = Vertex((0, 1))
            >>> v3 = Vertex((1, 1))
            >>> v4 = Vertex((1, 0))
            >>> e1 = Edge(v1, v2)
            >>> e2 = Edge(v3, v4)
            >>> e1.overlaps_or_crosses(e2)
            False
            >>> e2.overlaps_or_crosses(e1)
            False
            >>> e3 = Edge(v1, v3)
            >>> e4 = Edge(v2, v4)
            >>> e3.overlaps_or_crosses(e4)
            True
            >>> e4.overlaps_or_crosses(e3)
            True
            >>> e5 = Edge(v1, v4)
            >>> e6 = Edge(v2, v3)
            >>> e5.overlaps_or_crosses(e6)
            False
            >>> e6.overlaps_or_crosses(e5)
            False

        """

        # location of this edge
        vec_ra: nparr = np.array(self.v_i.coords)
        # direction of this edge
        vec_da: nparr = np.array(self.v_j.coords) - np.array(self.v_i.coords)
        # location of other edge
        vec_rb: nparr = np.array(other.v_i.coords)
        # direction of other edge
        vec_db: nparr = np.array(other.v_j.coords) - np.array(other.v_i.coords)
        # verify that the edges have nonzero length
        assert not np.isclose(vec_da @ vec_da, 0.00)
        assert not np.isclose(vec_db @ vec_db, 0.00)

        mat_a: nparr = np.column_stack((vec_da, -vec_db))
        mat_b = vec_rb - vec_ra
        determinant = np.linalg.det(mat_a)

        if np.isclose(determinant, 0.00):

            # there are infinite solutions
            # or there are no solutions
            # i.e., the edges are parallel.
            # If they are parallel but nor colinear, then they don't
            # overlap and the method should return False
            # If they are colinear, then they might overlap. If they
            # do, the method should return True, otherwise False.

            # first check if they are parallel but not colinear
            # project start of other vertex onto line of this vertex
            vec_rb_diff = vec_rb - vec_ra
            vec_proj_pt = (vec_rb_diff @ vec_da) / (
                vec_da @ vec_da
            ) * vec_da + vec_ra
            vec_dist = vec_rb - vec_proj_pt
            distance = np.sqrt(vec_dist @ vec_dist)

            if not np.isclose(distance, 0.00):
                # The edges are parallel but not collinear, so they
                # can't be intersecting.
                return False

            # If the previous statement was not true, we will arrive
            # here. The edges are colinear. Depending on their
            # relative position on their common line, they might share
            # no common points, one common point, or an entire
            # segment.
            # To solve this, we define ta to be a scalar that
            # determines a point on vertex i by evaluating: vec_ra +
            # ta * vec_da.
            # ta = 0 ==> on vertex_i, ta = 1 ==> on vertex j, of this
            # edge.
            # Similarly, there exists a tb that can be used to
            # identify a point on vertex j
            # But insdtead, we determine the location of vertex i and
            # j of the other edge in terms of ta. That is:
            # ta = c_i ==> vertex i of other edge, ta = c_j ==> vertex
            # j of other edge.
            # We can then determine which of the three cases we are
            # in, based on the values of c_i and c_j.
            c_i = (vec_da @ (vec_rb - vec_ra)) / (vec_da @ vec_da)
            c_j = (vec_da @ ((vec_rb + vec_db) - vec_ra)) / (vec_da @ vec_da)
            # either they should be both < 0 (which means that the
            # other edge is "before" this edge), or they should be
            # both > 1.00 (which means that the other edge is "after"
            # this edge). Any other case corresponds to an overlap.

            # each of c_i, c_j can either be {<0.00, ==0, 00<1.00, ==1, >1.0}
            # in each case the answer will depend on what the other one is.
            # note: we need to account for floating-point precision
            # when making comparisons.
            epsilon = common.EPSILON
            if (
                (c_i < 0.00 - epsilon and np.isclose(c_j, 0.00))
                or (c_i > 1.00 + epsilon and np.isclose(c_j, 1.00))
                or (np.isclose(c_i, 1.00) and c_j > 1.00 + epsilon)
                or (np.isclose(c_i, 0.00) and c_j < 0.00 - epsilon)
            ):
                # they share one vertex without overlap
                return False
            if (c_i < 0.00 - epsilon and c_j < 0.00 - epsilon) or (
                c_i > 1.00 + epsilon and c_j > 1.00 + epsilon
            ):
                # definitely no overlap
                return False
            # in any other case, they overlap
            return True

        # Otherwise they are not parallel.
        # there is at least one solution
        sol = np.linalg.solve(mat_a, mat_b)
        # if both constants are between 0 and 1
        # the edges overlap within their length
        # otherwise, their extensions overlap, which
        # is not an issue.
        if 0.00 < sol[0] < 1.00 and 0.00 < sol[1] < 1.00:
            return True

        return False


class Halfedge:
    """
    Halfedge object. Every edge has two halfedges.  A halfedge has a
    direction, pointing from one of the corresponding edge's vertices
    to the other.  The `vertex` attribute corresponds to the edge's
    vertex that the halfedge originates from.  Halfedges have a `next`
    attribute that points to the next halfedge, forming closed loops,
    or sequences, which is the purpose of this module.

    Example:
        >>> from osmg.mesh import Vertex, Edge, Halfedge
        >>> v1 = Vertex((0, 0))
        >>> v2 = Vertex((2, 2))
        >>> edge = Edge(v1, v2)
        >>> halfedge1 = Halfedge(v1, edge)
        >>> halfedge2 = Halfedge(v2, edge)
        >>> halfedge1.nxt = halfedge2
        >>> halfedge2.nxt = halfedge1
        >>> print(halfedge1)
        (H0 from E7 to E7 next H1)
        >>> print(halfedge2)
        (H1 from E7 to E7 next H0)

    """

    _ids = count(0)

    def __init__(self, vertex: Vertex, edge: Edge):
        """
        Initializes the halfedge object.

        Arguments:
            vertex: The vertex that the halfedge originates from.
            edge: The edge that the halfedge is a part of.

        """

        self.vertex = vertex
        self.edge = edge
        self.uid: int = next(self._ids)
        self.nxt = None

    def __repr__(self):
        """
        Returns a string representation of the halfedge, in the form
        `(H0 from E0 to E0 next H1)`

        """

        if self.nxt:
            out = (
                f"(H{self.uid} from E{self.edge.uid}"
                f" to E{self.nxt.edge.uid} next H{self.nxt.uid})"
            )
        else:
            out = f"(H{self.uid}"
        return out

    def __lt__(self, other):
        """
        Comparison function used for sorting. Compares the halfedge ids.

        """

        return self.uid < other.uid

    def direction(self):
        """
        Calculates the angular direction of the halfedge
        using the arctan2 function (in radians).

        Example:
            >>> from osmg.mesh import Vertex, Edge, Halfedge
            >>> v1 = Vertex((0.0, 0.0))
            >>> v2 = Vertex((2.0, 2.0))
            >>> edge = Edge(v1, v2)
            >>> halfedge1 = Halfedge(v1, edge)
            >>> halfedge2 = Halfedge(v2, edge)
            >>> halfedge1.direction()
            0.7853981633974483
            >>> halfedge2.direction()
            -2.356194490192345

        """

        drct: nparr = np.array(
            self.edge.other_vertex(self.vertex).coords
        ) - np.array(self.vertex.coords)
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
        return f"Mesh object containing {num} halfedges."

    def geometric_properties(self):
        """
        Calculates the geometric properties of the shape defined by
        the mesh

        """

        coords: nparr = np.array([h.vertex.coords for h in self.halfedges])
        return geometric_properties(coords)

    def bounding_box(self):
        """
        Returns a bounding box of the mesh

        """

        coords: nparr = np.array([h.vertex.coords for h in self.halfedges])
        xmin = min(coords[:, 0])
        xmax = max(coords[:, 0])
        ymin = min(coords[:, 1])
        ymax = max(coords[:, 1])
        return np.array([[xmin, ymin], [xmax, ymax]])


############################################
# Geometric Properties of Polygonal Shapes #
############################################


def polygon_area(coords: nparr) -> float:
    """
    Calculates the area of a polygon.

    Arguments:
        coords: A matrix whose columns represent
                the coordinates and the rows
                represent the points of the polygon.
                The first point should not be repeated
                at the end, as this is done
                automatically.

    Returns:
        area: The area of the polygon.

    Example:
        >>> coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        >>> polygon_area(coords)
        1.0

    """

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    return float(
        np.sum(
            x_coords * np.roll(y_coords, -1) - np.roll(x_coords, -1) * y_coords
        )
        / 2.00
    )


def polygon_centroid(coords: nparr) -> nparr:
    """
    Calculates the centroid of a polygon.

    Arguments:
        coords: A matrix whose columns represent
                the coordinates and the rows
                represent the points of the polygon.
                The first point should not be repeated
                at the end, as this is done
                automatically.

    Returns:
        centroid: The centroid of the polygon.

    Example:
        >>> coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        >>> polygon_centroid(coords)
        array([0.5, 0.5])

    """

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    area = polygon_area(coords)
    x_cent = (
        np.sum(
            (x_coords + np.roll(x_coords, -1))
            * (
                x_coords * np.roll(y_coords, -1)
                - np.roll(x_coords, -1) * y_coords
            )
        )
    ) / (6.0 * area)
    y_cent = (
        np.sum(
            (y_coords + np.roll(y_coords, -1))
            * (
                x_coords * np.roll(y_coords, -1)
                - np.roll(x_coords, -1) * y_coords
            )
        )
    ) / (6.0 * area)
    return np.array((x_cent, y_cent))


def polygon_inertia(coords):
    """
    Calculates the moments of inertia of a polygon.

    Arguments:
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

    Example:
        >>> coords = np.array([[-2, -1], [-2, 1], [1, 1], [1, -1]])
        >>> res = polygon_inertia(coords)
        >>> res['ixx']
        -2.0
        >>> res['iyy']
        -6.0
        >>> res['ixy']
        0.0
        >>> res['ir']
        -8.0
        >>> res['ir_mass']
        1.3333333333333333

    """

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    area = polygon_area(coords)
    alpha = x_coords * np.roll(y_coords, -1) - np.roll(x_coords, -1) * y_coords
    # planar moment of inertia wrt horizontal axis
    ixx = (
        np.sum(
            (
                y_coords**2
                + y_coords * np.roll(y_coords, -1)
                + np.roll(y_coords, -1) ** 2
            )
            * alpha
        )
        / 12.00
    )
    # planar moment of inertia wrt vertical axis
    iyy = (
        np.sum(
            (
                x_coords**2
                + x_coords * np.roll(x_coords, -1)
                + np.roll(x_coords, -1) ** 2
            )
            * alpha
        )
        / 12.00
    )

    ixy = (
        np.sum(
            (
                x_coords * np.roll(y_coords, -1)
                + 2.0 * x_coords * y_coords
                + 2.0 * np.roll(x_coords, -1) * np.roll(y_coords, -1)
                + np.roll(x_coords, -1) * y_coords
            )
            * alpha
        )
        / 24.0
    )
    # polar (torsional) moment of inertia
    i_r = ixx + iyy
    # mass moment of inertia wrt in-plane rotation
    ir_mass = (ixx + iyy) / area

    return {"ixx": ixx, "iyy": iyy, "ixy": ixy, "ir": i_r, "ir_mass": ir_mass}


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

    return {"area": area, "centroid": centroid, "inertia": inertia}


##################################
# Defining halfedges given edges #
##################################

# auxiliary functions


def ang_reduce(ang):
    """
    Brings and angle expressed in radians in the interval [0, 2pi)

    """

    while ang < 0:
        ang += 2.0 * np.pi
    while ang >= 2.0 * np.pi:
        ang -= 2.0 * np.pi
    return ang


def define_halfedges(edges: list[Edge]) -> list[Halfedge]:
    """
    Given a list of edges, defines all the halfedges and
    associates them with their `next`.

    Note:
        See https://notability.com/n/0wlJ17mt81uuVWAYVoFfV3

        Each halfedge stores information about its edge, vertex and
        and next halfedge. Contrary to convention, we don't store the
        twin (opposite) halfedge here, seince we don't need it
        anywhere.

        This function receives a list of Edge objects as input and
        returns a list of Halfedge objects. The function first creates
        a Halfedge object for each vertex of each Edge object, using
        the `define_halfedge` method of the Edge class. These Halfedge
        objects are stored in a list called all_halfedges. For each
        Halfedge object, the function also updates the list of
        halfedges leaving the vertex that the halfedge originates
        from. For example, if we have two Halfedge objects h1 and h2,
        both originating from the same vertex v, v.halfedges will be a
        list containing h1 and h2. (This is useful because it allows
        us to easily access all the halfedges that originate from a
        particular vertex, which we need later on in the algorithm.)

        After all halfedges have been created, the function assigns
        the next attribute of each halfedge, which points to the next
        halfedge in the sequence. To do this, it loops through all
        halfedges and, for each halfedge h, it determines the vertex
        v_to that h points to, gets a list of all halfedges leaving
        v_to, and assigns the next attribute of h to the halfedge in
        that list that has the smallest angular difference with
        respect to the direction of h.

    Arguments:
        edges: List of Edge objects

    Returns:
        halfedges: List of Halfedge objects

    Example:
        >>> from osmg.mesh import Vertex, Edge, Halfedge
        >>> # define some vertices
        >>> v1 = Vertex((0.0, 0.0))
        >>> v2 = Vertex((1.0, 0.0))
        >>> v3 = Vertex((1.0, 1.0))
        >>> v4 = Vertex((0.0, 1.0))
        >>> # define some edges
        >>> e1 = Edge(v1, v2)
        >>> e2 = Edge(v2, v3)
        >>> e3 = Edge(v3, v4)
        >>> e4 = Edge(v4, v1)
        >>> # define the halfedges
        >>> halfedges = define_halfedges([e1, e2, e3, e4])
        >>> # check that the `next` attribute of each halfedge
        >>> # is correctly assigned
        >>> for h in halfedges:
        ...     assert h.nxt.vertex == h.edge.other_vertex(h.vertex)

    """

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

    for halfedge in all_halfedges:
        # We are looking for `h`'s `next`
        # determine the vertex that it starts from
        v_from = halfedge.vertex
        # determine the vertex that it points to
        v_to = halfedge.edge.other_vertex(v_from)
        # get a list of all halfedges leaving that vertex
        candidates_for_next = v_to.halfedges
        # determine which of all these halfedges will be the next
        angles = np.full(len(candidates_for_next), 0.00)
        for i, h_other in enumerate(candidates_for_next):
            if h_other.edge == halfedge.edge:
                angles[i] = 1000.0
                # otherwise we would assign its conjugate as next
            else:
                angles[i] = ang_reduce(
                    (halfedge.direction() - np.pi) - h_other.direction()
                )
        halfedge.nxt = candidates_for_next[np.argmin(angles)]

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
    #             p1 = (np.array(e.v_i.coords)
    #                   + np.array(e.v_j.coords))/2.
    #             p2 = (np.array(e_nxt.v_i.coords)
    #                   + np.array(e_nxt.v_j.coords))/2.
    #             dx = p2 - p1
    #             ax.arrow(*p1, *dx)
    # plt.show()


def obtain_closed_loops(halfedges):
    """
    Given a list of halfedges, this function uses their `next`
    attribute to group them into sequences of closed loops (ordered
    lists of halfedges of which the `next` halfedge of the last list
    element points to the first halfedge in the list, and the `next`
    halfedge of any list element points to the next halfedge in the
    list.

    Arguments:
        halfedges: list of halfedges

    Returns:
        loops with the aforementioned property.

    """

    def is_in_some_loop(halfedge, loops):
        for loop in loops:
            if halfedge in loop:
                return True
        return False

    loops: list[list[Halfedge]] = []
    for halfedge in halfedges:
        if loops:
            if is_in_some_loop(halfedge, loops):
                continue
        loop = [halfedge]
        nxt = halfedge.nxt
        while nxt != halfedge:
            loop.append(nxt)
            nxt = nxt.nxt
        loops.append(loop)
    return loops


def orient_loops(loops):
    """
    Separates loops to internal (counterclockwise) and external
    (clockwise). Also gathers trivial loops, i.e. halfedge sequences
    that define polygons that have no area (e.g. h1 -> h2 -> h1).

    Arguments:
        loops (list[list[Halfedge]]) (see `obtain_closed_loops`)

    Returns:
        external_loops (list[list[Halfedge]])
        internal_loops (list[list[Halfedge]])
        trivial_loops (list[list[Halfedge]])

    """

    internal_loops = []
    external_loops = []
    trivial_loops = []
    loop_areas = [
        polygon_area(np.array([h.vertex.coords for h in loop]))
        for loop in loops
    ]
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


def subdivide_polygon(outside, holes, n_x, n_y, plot=False):
    """
    Used to define the fibers of fiber sections.

    Arguments:
        halfedges: Sequence of halfedges that defines the shape of a
                  section.
        n_x: Number of spatial partitions in the x direction
        n_y: Number of spatial partitions in the y direction
        plot: Plots the resulting polygons for debugging

    Returns:
        pieces: shapely_Polygon objects that represent single fibers.

    """

    outside_polygon = shapely_Polygon(
        [h.vertex.coords for h in outside.halfedges]
    )
    hole_polygons = []
    for hole in holes.values():
        hole_polygons.append(
            shapely_Polygon([h.vertex.coords for h in hole.halfedges])
        )
    remaining_polygon = outside_polygon
    for hole_polygon in hole_polygons:
        remaining_polygon = remaining_polygon.difference(hole_polygon)
    x_min, y_min, x_max, y_max = outside_polygon.bounds
    x_array = np.linspace(x_min, x_max, num=n_x, endpoint=True)
    y_array = np.linspace(y_min, y_max, num=n_y, endpoint=True)
    pieces = []
    for i in range(len(x_array) - 1):
        for j in range(len(y_array) - 1):
            tile = shapely_Polygon(
                [
                    (x_array[i], y_array[j]),
                    (x_array[i + 1], y_array[j]),
                    (x_array[i + 1], y_array[j + 1]),
                    (x_array[i], y_array[j + 1]),
                ]
            )
            subregion = remaining_polygon.intersection(tile)
            if subregion.area != 0.0:
                pieces.append(subregion)
    if plot:
        fig = plt.figure()
        ax_1 = fig.add_subplot(111)
        ax_1.set_aspect("equal")
        patch = PolygonPatch(remaining_polygon, alpha=0.5, zorder=2)
        ax_1.add_patch(patch)
        for subregion in pieces:
            patch = PolygonPatch(subregion, alpha=0.5, zorder=2)
            ax_1.add_patch(patch)
        for subregion in pieces:
            ax_1.scatter(subregion.centroid.x, subregion.centroid.y)
        ax_1.margins(0.10)
        plt.show()
    return pieces


def subdivide_hss_rect(
        sec_h: float, sec_b: float,
        sec_t: float, plot: bool = False) \
        -> list[shapely_Polygon]:
    """
    Used to define the fibers of steel HSS fiber sections.

    Arguments:
      sec_h: Section height
      sec_b: Section width
      sec_t: Section thickness

    Returns:
        pieces: shapely_Polygon objects that represent single fibers.

    """

    outside_polygon = shapely_Polygon(
        np.array(
            (
                (sec_h, sec_b),
                (sec_h, -sec_b),
                (-sec_h, -sec_b),
                (-sec_h, sec_b),
            )
        )
    )
    hole_polygon = shapely_Polygon(
        np.array(
            (
                (sec_h - sec_t, sec_b - sec_t),
                (sec_h - sec_t, -sec_b + sec_t),
                (-sec_h + sec_t, -sec_b + sec_t),
                (-sec_h + sec_t, sec_b - sec_t),
            )
        )
    )
    remaining_polygon = outside_polygon.difference(hole_polygon)
    x_min, y_min, x_max, y_max = outside_polygon.bounds
    # cutting it into 8 regions
    pieces = []
    for ylow, yhigh in zip(
        (y_min, y_min + sec_t, y_max - sec_t),
        (y_min + sec_t, y_max - sec_t, y_max),
    ):
        for xlow, xhigh in zip(
            (x_min, x_min + sec_t, x_max - sec_t),
            (x_min + sec_t, x_max - sec_t, x_max),
        ):
            x_array = np.linspace(xlow, xhigh, num=5, endpoint=True)
            y_array = np.linspace(ylow, yhigh, num=5, endpoint=True)
            for i in range(len(x_array) - 1):
                for j in range(len(y_array) - 1):
                    tile = shapely_Polygon(
                        [
                            (x_array[i], y_array[j]),
                            (x_array[i + 1], y_array[j]),
                            (x_array[i + 1], y_array[j + 1]),
                            (x_array[i], y_array[j + 1]),
                        ]
                    )
                    subregion = remaining_polygon.intersection(tile)
                    if subregion.area != 0.0:
                        pieces.append(subregion)

    if plot:
        fig = plt.figure()
        ax_1 = fig.add_subplot(111)
        ax_1.set_aspect("equal")
        # patch = PolygonPatch(remaining_polygon, alpha=0.5, zorder=2)
        # ax_1.add_patch(patch)
        for subregion in pieces:
            patch = PolygonPatch(subregion, alpha=0.5, zorder=2)
            ax_1.add_patch(patch)
        for subregion in pieces:
            ax_1.scatter(subregion.centroid.x, subregion.centroid.y)
        ax_1.margins(0.10)
        plt.show()

    return pieces


def subdivide_hss_circ(
        sec_d: float, sec_t: float,
        plot: bool = False) \
        -> list[shapely_Polygon]:
    """
    Used to define the fibers of steel HSS fiber sections.

    Arguments:
      sec_d: Section diameter
      sec_t: Section thickness

    Returns:
        pieces: shapely_Polygon objects that represent single fibers.

    """

    num_subdiv_t = 3
    num_subdiv_circ = 12

    radius = sec_d/2.00

    pieces = []

    for i in range(num_subdiv_t):
        for j in range(num_subdiv_circ):
            rr_i = radius - i * sec_t/num_subdiv_t
            rr_j = radius - (i+1.0) * sec_t/num_subdiv_t
            ang_i = (2.00 * np.pi) / num_subdiv_circ * j
            ang_j = (2.00 * np.pi) / num_subdiv_circ * (j+1.00)
            pt1 = (rr_i * np.cos(ang_i), rr_i * np.sin(ang_i))
            pt2 = (rr_i * np.cos(ang_j), rr_i * np.sin(ang_j))
            pt3 = (rr_j * np.cos(ang_j), rr_j * np.sin(ang_j))
            pt4 = (rr_j * np.cos(ang_i), rr_j * np.sin(ang_i))
            pol = shapely_Polygon((pt1, pt2, pt3, pt4))
            pieces.append(pol)

    if plot:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        for piece in pieces:
            patch = Polygon(piece.exterior.coords, alpha=0.5, zorder=2)
            ax.add_patch(patch)
        for piece in pieces:
            ax.scatter(piece.centroid.x, piece.centroid.y)
        ax.margins(0.10)
        plt.show()

    return pieces


#############
# Debugging #
#############


def print_halfedge_results(halfedges):
    """
    Prints the ids of the defined halfedges and their vertex, edge and
    next, for debugging.

    """

    results: dict[str, list[Any]] = {
        "halfedge": [],
        "vertex": [],
        "edge": [],
        "next": [],
    }

    for halfedge in halfedges:
        results["halfedge"].append(halfedge)
        results["vertex"].append(halfedge.vertex)
        results["edge"].append(halfedge.edge)
        results["next"].append(halfedge.nxt)

    print(results)


def plot_loop(halfedge_loop):
    """
    Plots the vertices/edges of a list of halfedges.

    """

    num = len(halfedge_loop)
    coords = np.full((num + 1, 2), 0.00)
    for i, halfedge in enumerate(halfedge_loop):
        coords[i, :] = halfedge.vertex.coords
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
    for edge in edges:
        coords = np.full((2, 2), 0.00)
        coords[0, :] = edge.v_i.coords
        coords[1, :] = edge.v_j.coords
        plt.plot(coords[:, 0], coords[:, 1])
    fig.show()


def sanity_checks(external, trivial):
    """
    Perform some checks to make sure assumptions are not violated.

    """

    #   We expect no trivial loops
    if trivial:
        print("Warning: Found trivial loop")
        for i, trv in enumerate(trivial):
            for halfedge in trv:
                print(halfedge.vertex.coords)
            plot_loop(trv)
    #   We expect a single external loop
    if len(external) > 1:
        print("Warning: Found multiple external loops")
        for i, ext in enumerate(external):
            print(i + 1)
            for halfedge in ext:
                print(halfedge.vertex.coords)
            plot_loop(ext)


if __name__ == "__main()__":
    pass
