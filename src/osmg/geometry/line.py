"""Defines :obj:`~osmg.line.Line` objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from osmg.core import common

numpy_array = npt.NDArray[np.float64]


@dataclass
class Line:
    """
    Finite-length line segment object.

    Used internally whenever operations involving lines are required.

    Attributes:
    ----------
      tag: line tag.
      start: starting point.
      end: end point.

    """

    tag: str
    start: numpy_array = field(repr=False)
    end: numpy_array = field(repr=False)

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
        res = ''
        res += 'Line object\n'
        res += f'  start: {self.start}\n'
        res += f'  end: {self.end}\n'
        return res

    def length(self) -> float:
        """
        Obtain the length of the line.

        Example:
            >>> from osmg.line import Line
            >>> l1 = Line('l1', np.array([0, 0]), np.array([2, 2]))
            >>> l1.length() == 2.8284271247461903
            True

        Returns:
          The string representation of the object.
        """
        return float(np.linalg.norm(self.end - self.start))

    def direction(self) -> numpy_array:
        """
        Line direction.

        Returns a unit vector pointing from the start to the end of
        the line.

        Example:
            >>> from osmg.line import Line
            >>> l1 = Line('l1', np.array([0, 0]), np.array([2, 2]))
            >>> l1.direction()
            array([0.70710678, 0.70710678])

        Returns:
          The string representation of the object.
        """
        return (self.end - self.start) / self.length()

    def intersect(self, other: Line) -> numpy_array | None:
        """
        Intersection point.

        Calculates the intersection point of this line with another
        line. Returns None if the lines don't intersect.  Note: 'line'
        is actually a finite-length line segment.

        Parameters
        ----------
          other: the other line

        Example:
            >>> from osmg.line import Line
            >>> l1 = Line('l1', np.array([0, 0]), np.array([2, 2]))
            >>> l2 = Line('l2', np.array([1, 0]), np.array([1, 3]))
            >>> l1.intersect(l2)
            array([1., 1.])

        Returns:
          The intersection point if it exists.
        """
        ra_dir = self.direction()
        rb_dir = other.direction()
        mat: numpy_array = np.array(
            [[ra_dir[0], -rb_dir[0]], [ra_dir[1], -rb_dir[1]]]
        )
        if np.abs(np.linalg.det(mat)) <= common.EPSILON:
            # The lines are parallel
            # in this case, we check if they have
            # a common starting or ending point
            # (we ignore the case of a common segment,
            #  as it has no practical use for our purposes).
            if (
                np.linalg.norm(self.start - other.start) <= common.EPSILON
                or np.linalg.norm(self.start - other.end) <= common.EPSILON
            ):
                return self.start
            if (
                np.linalg.norm(self.end - other.start) <= common.EPSILON
                or np.linalg.norm(self.end - other.end) <= common.EPSILON
            ):
                return self.end
            return None
        # Get the origins
        ra_ori = self.start
        rb_ori = other.start
        # System left-hand-side
        bvec: numpy_array = np.array(
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
        if uvvec[0] > self.length() + common.EPSILON:
            return None
        if uvvec[1] > other.length() + common.EPSILON:
            return None
        # Otherwise the point is valid
        point = ra_ori + ra_dir * uvvec[0]
        return np.array([point[0], point[1]])

    def intersects_pt(self, point: numpy_array) -> bool:
        """
        Check whether the given point pt lies on the line.

        Parameters
        ----------
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

        Returns:
          Whether the given point pt lies on the line.

        Raises:
          ValueError: If the line has zero length.
        """
        r_a = self.end - self.start
        norm2 = np.dot(r_a, r_a)
        if np.abs(norm2) < common.EPSILON:
            msg = 'Line has zero length.'
            raise ValueError(msg)
        r_b = point - self.start

        r_a_3d = np.append(r_a, 0)
        r_b_3d = np.append(r_b, 0)
        cross = np.linalg.norm(np.cross(r_a_3d, r_b_3d))

        dot_normalized = np.dot(r_a, r_b) / norm2  # type: ignore
        if cross < common.EPSILON:
            res = bool(0.00 <= dot_normalized <= 1.00)
        else:
            res = False
        return res

    def point_distance(self, point: numpy_array) -> float | None:
        """
        Minimum distance.

        Calculate the minimum distance between the line segment and a
        point.  If the point falls on the line but is outside of the
        line segment, returns None.

        Parameters:
            point: the point

        Returns:
          The minimum distance.

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
            res: float | None = float(np.linalg.norm(r_b - proj_point))
        else:
            res = None
        return res

    def project(self, point: numpy_array) -> numpy_array | None:
        """
        Projection.

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

        Returns:
          The projection point if it exists.
        """
        r_a = self.end - self.start
        r_b = point - self.start
        proj_point: numpy_array = (r_b @ r_a) / (r_a @ r_a) * r_a + self.start
        if self.intersects_pt(proj_point):
            return proj_point
        return None
