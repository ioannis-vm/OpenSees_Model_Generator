"""
Model Generator for OpenSees ~ grids
"""
#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ / 
# / /_/ / / / / / / /_/ /_/  
# \____/_/ /_/ /_/\__, (_)   
#                /____/      
#                            
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from dataclasses import dataclass, field
from functools import total_ordering
import numpy as np
from utility import common


def point_exists_in_list(pt: np.ndarray,
                         pts: list[np.ndarray]) -> bool:
    """
    Determines whether a given list containing points
    (represented with numpy arrays) contains a point
    that is equal (with a fudge factor) to a given point.
    Args:
        pt (np.ndarray): A numpy array to look for
        pts (list[np.ndarray]): A list to search for pt
    """
    for other in pts:
        dist = np.linalg.norm(pt - other)
        if dist < common.EPSILON:
            return True
    return False


@dataclass
@total_ordering
class GridLine:
    """
    Gridlines can be used to
    speed up the definition of the model.
    They are defined as line segments that have a
    starting and an ending point. They do not have
    to be permanent. Gridlines can be defined
    and used temporarily to define some elements,
    and later be discarded or altered in order
    to define other elements.
    Attributes:
        tag (str): Name of the gridline
        start (list(float]): X,Y coordinates of starting point
        end ~ similar to start
        start_np (np.ndarray): Numpy array of starting point
        end_np ~ similar to start
        length (float): Length of the gridline
        direction (float): Direction (angle) measured
                  using a counterclockwise convention
                  with the global X axis corresponding to 0.
    """
    tag: str
    start: list[float]
    end: list[float]
    start_np: np.ndarray = field(init=False, repr=False)
    end_np:   np.ndarray = field(init=False, repr=False)
    length: float = field(init=False, repr=False)
    direction: float = field(init=False, repr=False)

    def __post_init__(self):
        self.start_np = np.array(self.start)
        self.end_np = np.array(self.end)
        self.length = np.linalg.norm(self.end_np - self.start_np)
        self.direction = (self.end_np - self.start_np) / self.length

    def __eq__(self, other):
        return self.tag == other.tag

    def __le__(self, other):
        return self.tag <= other.tag

    def intersect(self, grd: 'GridLine'):
        """
        Obtain the intersection with
        another gridline (if it exists)

        Parameters:
            grd(GridLine): a gridline to intersect
        Returns:
            list[float]: Intersection point

        Derivation:
            If the intersection point p exists, we will have
            p = ra.origin + ra.dir * u
            p = rb.origin + rb.dir * v
            We determine u and v(if possible) and check
            if the intersection point lies on both lines.
            If it does, the lines intersect.
        """
        ra_dir = self.direction
        rb_dir = grd.direction
        mat = np.array(
            [
                [ra_dir[0], -rb_dir[0]],
                [ra_dir[1], -rb_dir[1]]
            ]
        )
        if np.abs(np.linalg.det(mat)) <= common.EPSILON:
            # The lines are parallel
            # in this case, we check if they have
            # a common starting or ending point
            # (we ignore the case of a common segment,
            #  as it has no practical use for our purposes).
            if np.linalg.norm(self.start_np - grd.start_np) <= common.EPSILON:
                return self.start_np
            elif np.linalg.norm(self.start_np - grd.end_np) <= common.EPSILON:
                return self.start_np
            elif np.linalg.norm(self.end_np - grd.start_np) <= common.EPSILON:
                return self.end_np
            elif np.linalg.norm(self.end_np - grd.end_np) <= common. EPSILON:
                return self.end_np
            else:
                return None
        # Get the origins
        ra_ori = self.start_np
        rb_ori = grd.start_np
        # System left-hand-side
        bvec = np.array(
            [
                [rb_ori[0] - ra_ori[0]],
                [rb_ori[1] - ra_ori[1]],
            ]
        )
        # Solve to get u and v in a vector
        uvvec = np.linalg.solve(mat, bvec)
        # Terminate if the intersection point
        # does not lie on both lines
        if uvvec[0] < 0 - common.EPSILON:
            return None
        if uvvec[1] < 0 - common.EPSILON:
            return None
        if uvvec[0] > self.length + common.EPSILON:
            return None
        if uvvec[1] > grd.length + common.EPSILON:
            return None
        # Otherwise the point is valid
        pt = ra_ori + ra_dir * uvvec[0]
        return np.array([pt[0], pt[1]])

    def intersects_pt(self, pt: np.ndarray) -> bool:
        """
        Check whether the given point pt
        lies on the gridline
        Parameters:
            pt (np.ndarray): a 2D point
        """

        ra = self.end_np - self.start_np
        norm2 = np.dot(ra, ra)
        rb = (pt - self.start_np)
        cross = np.linalg.norm(np.cross(ra, rb))
        dot_normalized = np.dot(ra, rb)/norm2

        if cross < common.EPSILON:
            return bool(0.00 <= dot_normalized <= 1.00)
        else:
            return False


@dataclass
class GridSystem:
    """
    This class is a collector for the gridlines, and provides
    methods that perform operations using gridlines.
    """

    grids: list[GridLine] = field(default_factory=list)

    def get(self, gridline_tag: str):
        """
        Returns the gridline with the given tag,
        or None if there is no gridline with the
        specified tag.
        """
        result = None
        for gridline in self.grids:
            if gridline.tag == gridline_tag:
                result = gridline
        return result

    def add(self, grdl: "GridLine"):
        """
        Add a gridline in the grid system,
        if it is not already in
        """
        if grdl not in self.grids:
            self.grids.append(grdl)
        else:
            raise ValueError('Gridline already exists: '
                             + repr(grdl))
        self.grids.sort()

    def remove(self, grdl: "GridLine"):
        """
        Remove a gridline from the grid system
        """
        self.grids.remove(grdl)

    def clear(self, tags: list[str]):
        """
        Removes the gridlines in the given list,
        specified by their tag.
        """
        for tag in tags:
            grdl = self.get(tag)
            self.grids.remove(grdl)

    def clear_all(self):
        """
        Removes all gridlines.
        """
        self.grids = []

    def intersection_points(self):
        """
        Returns a list of all the points
        defined by gridline intersections
        """
        pts = []  # intersection points
        for i, grd1 in enumerate(self.grids):
            for j in range(i+1, len(self.grids)):
                grd2 = self.grids[j]
                pt = grd1.intersect(grd2)
                if pt is not None:  # if an intersection point exists
                    # and is not already in the list
                    if not point_exists_in_list(pt, pts):
                        pts.append(pt)
        return pts

    def intersect(self, grd: GridLine):
        """
        Returns a list of all the points
        defined by the intersection of a given
        gridline with all the other gridlines
        in the gridsystem
        """
        pts = []  # intersection points
        for other_grd in self.grids:
            # ignore current grid
            if other_grd == grd:
                continue
            # get the intersection point, if any
            pt = grd.intersect(other_grd)
            if pt is not None:  # if there is an intersection
                # and is not already in the list
                if not point_exists_in_list(pt, pts):
                    pts.append(pt)
            # We also need to sort the list.
            # We do this by sorting the instersection points
            # by their distance from the current gridline's
            # starting point.
            distances = [np.linalg.norm(pt-grd.start_np)
                         for pt in pts]
            pts = [x for _, x in sorted(zip(distances, pts))]
        return pts

    def __repr__(self):
        out = "The building has " + \
            str(len(self.grids)) + " gridlines\n"
        for grd in self.grids:
            out += repr(grd) + "\n"
        return out


