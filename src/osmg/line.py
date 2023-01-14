"""
Defines :obj:`~osmg.line.Line` objects.
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

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import numpy.typing as npt
from . import common

# pylint: disable=no-else-return

nparr = npt.NDArray[np.float64]


@dataclass
class Line:
    """
    Finite-length line segment object.
    Used internally whenever operations involving lines are reuired.

    Attributes:
      tag: line tag.
      start: starting point.
      end: end point.

    """

    tag: str
    start: nparr = field(repr=False)
    end: nparr = field(repr=False)

    def __repr__(self):
        res = ""
        res += "Line object\n"
        res += f"  start: {self.start}\n"
        res += f"  end: {self.end}\n"
        return res

    def length(self):
        """
        Returns the length of the line.

        Example:
            >>> from osmg.line import Line
            >>> l1 = Line('l1', np.array([0, 0]), np.array([2, 2]))
            >>> l1.length()
            2.8284271247461903

        """

        return np.linalg.norm(self.end - self.start)

    def direction(self):
        """
        Returns a unit vector pointing from the start to the end of
        the line.

        Example:
            >>> from osmg.line import Line
            >>> l1 = Line('l1', np.array([0, 0]), np.array([2, 2]))
            >>> l1.direction()
            array([0.70710678, 0.70710678])

        """

        return (self.end - self.start) / self.length()

    def intersect(self, other: "Line") -> nparr:
        """
        Calculates the intersection point of this line with another
        line. Returns None if the lines don't intersect.  Note: 'line'
        is actually a finite-length line segment.

        Parameters:
          other: the other line

        Example:
            >>> from osmg.line import Line
            >>> l1 = Line('l1', np.array([0, 0]), np.array([2, 2]))
            >>> l2 = Line('l2', np.array([1, 0]), np.array([1, 3]))
            >>> l1.intersect(l2)
            array([1., 1.])

        """

        ra_dir = self.direction()
        rb_dir = other.direction()
        mat: nparr = np.array(
            [[ra_dir[0], -rb_dir[0]], [ra_dir[1], -rb_dir[1]]]
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
            elif np.linalg.norm(self.end - other.end) <= common.EPSILON:
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
        Check whether the given point pt lies on the line.

        Parameters:
            point: a point

        Returns: True if the point lies on the line, False otherwise

        Example:
            >>> from osmg.line import Line
            >>> l = Line('my_line', np.array([0, 0]), np.array([1, 1]))
            >>> l.intersects_pt(np.array([0.5, 0.5]))
            True
            >>> l.intersects_pt(np.array([0, 0]))
            True
            >>> l.intersects_pt(np.array([1, 1]))
            True
            >>> l.intersects_pt(np.array([2, 2]))
            False

        """

        r_a = self.end - self.start
        norm2 = np.dot(r_a, r_a)
        if np.abs(norm2) < 1.0e-4:
            raise ValueError
        r_b = point - self.start
        cross = np.linalg.norm(np.cross(r_a, r_b))
        dot_normalized = np.dot(r_a, r_b) / norm2  # type: ignore
        if cross < common.EPSILON:
            res = bool(0.00 <= dot_normalized <= 1.00)
        else:
            res = False
        return res

    def point_distance(self, point: nparr) -> Optional[float]:
        """
        Calculate the minimum distance between the line segment and a
        point.  If the point falls on the line but is outside of the
        line segment, returns None.

        Parameters:
            point: the point

        Returns: the minimum distance

        Example:
            >>> line = Line(tag='line',
            ...             start=np.array([1, 1]),
            ...             end=np.array([3, 3]))
            >>> point = np.array([4, 2])
            >>> line.point_distance(point)
            1.4142135623730951
            >>> point = np.array([2, 2])
            >>> line.point_distance(point)
            0.0
            >>> point = np.array([0, 0])
            >>> line.point_distance(point)

            >>> point = np.array([4, 4])
            >>> line.point_distance(point)

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
        projected point, otherwise it returns None.

        Arguments:
          point: the point's coordinates

        Example:
            >>> line = Line('test', np.array([0, 0]), np.array([10, 0]))
            >>> line.project(np.array([5, 0]))
            array([5., 0.])
            >>> line.project(np.array([5, 5]))
            array([5., 0.])
            >>> line.project(np.array([-5, 5]))

            >>> line.project(np.array([15, 5]))

        """

        r_a = self.end - self.start
        r_b = point - self.start
        proj_point = (r_b @ r_a) / (r_a @ r_a) * r_a + self.start
        if self.intersects_pt(proj_point):
            return proj_point
        else:
            return None
