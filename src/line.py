"""
Model Generator for OpenSees ~ line
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
import numpy as np
import common


@dataclass
class Line:
    """
    todo - in progress
    """
    tag: str
    start: np.ndarray = field(repr=False)
    end: np.ndarray = field(repr=False)

    def length(self):
        """
        todo - work in progress
        """
        return np.linalg.norm(self.end - self.start)

    def direction(self):
        """
        todo - work in progress
        """
        return (self.end - self.start) / self.length()

    def intersect(self, other: 'Line'):
        """
        todo - in progress
        """
        ra_dir = self.direction()
        rb_dir = other.direction()
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
            if np.linalg.norm(self.start - other.start) <= common.EPSILON:
                result = self.start
            elif np.linalg.norm(self.start - other.end) <= common.EPSILON:
                result = self.start
            elif np.linalg.norm(self.end - other.start) <= common.EPSILON:
                result = self.end
            elif np.linalg.norm(self.end - other.end) <= common. EPSILON:
                result = self.end
            else:
                result = None
        # Get the origins
        ra_ori = self.start
        rb_ori = other.start
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
            result = None
        if uvvec[1] < 0 - common.EPSILON:
            result = None
        if uvvec[0] > self.length() + common.EPSILON:
            result = None
        if uvvec[1] > other.length() + common.EPSILON:
            result = None
        # Otherwise the point is valid
        point = ra_ori + ra_dir * uvvec[0]
        result = np.array([point[0], point[1]])

        return result

    def intersects_pt(self, point: np.ndarray) -> bool:
        """
        Check whether the given point pt
        lies on the gridline
        Parameters:
            pt (np.ndarray): a 2D point
        """
        ra = self.end - self.start
        norm2 = np.dot(ra, ra)
        rb = (point - self.start)
        cross = np.linalg.norm(np.cross(ra, rb))
        dot_normalized = np.dot(ra, rb)/norm2
        if cross < common.EPSILON:
            res = bool(0.00 <= dot_normalized <= 1.00)
        else:
            res = False
        return res
