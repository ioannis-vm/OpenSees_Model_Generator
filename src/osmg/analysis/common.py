"""Common low-level analysis objects."""

from __future__ import annotations


class ConcentratedValue(tuple[float, ...]):
    """Concentrated value, such as a point load or mass."""

    __slots__ = ()

    def __repr__(self) -> str:
        """
        Get a string representation.

        Returns:
          A simple string representation of the object.
        """
        return f"{self.__class__.__name__}({', '.join(map(str, self))})"


class PointLoad(ConcentratedValue):
    """Point load."""


class PointMass(ConcentratedValue):
    """Point mass."""


class UDL(tuple[float, ...]):
    """
    Beamcolumn element UDL.

    Uniformly distributed load expressed in the global coordinate
    system of the structure.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """
        Get a string representation.

        Returns:
          A simple string representation of the object.
        """
        return f"{self.__class__.__name__}({', '.join(map(str, self))})"
