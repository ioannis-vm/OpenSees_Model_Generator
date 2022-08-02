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
from typing import Optional
import numpy as np
import numpy.typing as npt
from .import common

# pylint: disable=no-else-return

nparr = npt.NDArray[np.float64]


@dataclass
class Line:
    """
    Finite-length line segment object.
    Used internally whenever operations involving lines are reuired.
    Attributes:
      tag (str)
      start (numpy.ndarray): starting point
      end (numpy.ndarray): end point
    """
    tag: str
    start: nparr = field(repr=False)
    end: nparr = field(repr=False)

    def __repr__(self):
        res = ''
        res += 'Line object\n'
        res += f'  start: {self.start}\n'
        res += f'  end: {self.end}\n'
        return res

    def length(self):
        """
        returns the length of the line.
        """
        return np.linalg.norm(self.end - self.start)

    def direction(self):
        """
        returns a unit verctor pointing from the start to the end of
        the line.
        """
        return (self.end - self.start) / self.length()

    def intersect(self, other: 'Line'):
        """
        Calculates the intersection point of this line with another line.
        Returns None if the lines don't intersect.
        Note: 'line' is actually a finite-length line segment.
        Parameters:
          other (Line): the other line
        """
        ra_dir = self.direction()
        rb_dir = other.direction()
        mat: nparr = np.array(
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
        bvec: nparr = np.array(
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

    def intersects_pt(self, point: nparr) -> bool:
        """
        Check whether the given point pt
        lies on the line
        Parameters:
            pt (nparr): a point
        """
        r_a = self.end - self.start
        norm2 = np.dot(r_a, r_a)
        if np.abs(norm2) < 1.0e-4:
            raise ValueError
        r_b = (point - self.start)
        cross = np.linalg.norm(np.cross(r_a, r_b))
        dot_normalized = np.dot(r_a, r_b)/norm2  # type: ignore
        if cross < common.EPSILON:
            res = bool(0.00 <= dot_normalized <= 1.00)
        else:
            res = False
        return res

    def point_distance(self, point: nparr) -> Optional[float]:
        """
        Calculates the projection of a point on the line.
        If the projection falls on the line segment, it returns the
        distance from the point to the line.
        Arguments:
          point (numpy.ndarray): the point's coordinates
        """
        r_a = self.end - self.start
        r_b = point - self.start
        proj_point = (r_b @ r_a) / (r_a @ r_a) * r_a
        if self.intersects_pt(proj_point + self.start):
            res: Optional[float] = float(np.linalg.norm(r_b - proj_point))
        else:
            res = None
        return res

    def project(self, point: nparr) -> Optional[nparr]:
        """
        Calculates the projection of a point on the line.
        If the projection falls on the line segment, it returns the
        projected point.
        Arguments:
          point (numpy.ndarray): the point's coordinates
        """
        r_a = self.end - self.start
        r_b = point - self.start
        proj_point = (r_b @ r_a) / (r_a @ r_a) * r_a + self.start
        if self.intersects_pt(proj_point):
            return proj_point
        else:
            return None
