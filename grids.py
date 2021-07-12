"""
Building Modeler for OpenSeesPy ~ Gridlines
"""

#   __                 UC Berkeley
#   \ \/\   /\/\/\     John Vouvakis Manousakis
#    \ \ \ / /    \    Dimitrios Konstantinidis
# /\_/ /\ V / /\/\ \
# \___/  \_/\/    \/   April 2021
#
# https://github.com/ioannis-vm/OpenSeesPy_Building_Modeler

from dataclasses import dataclass, field
from functools import total_ordering
import numpy as np
from utility import common


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
