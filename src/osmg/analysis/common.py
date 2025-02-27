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

    def __add__(self, other: ConcentratedValue) -> ConcentratedValue:
        """
        Add two ConcentratedValue instances element-wise.

        Args:
          other: Another ConcentratedValue instance.

        Returns:
          A new ConcentratedValue instance with the sum of corresponding elements.
        """
        if not isinstance(other, ConcentratedValue):
            return NotImplemented
        return ConcentratedValue(a + b for a, b in zip(self, other))

    def __mul__(self, scalar: float) -> ConcentratedValue:
        """
        Multiply each element of ConcentratedValue by a scalar.

        Args:
          scalar: A numeric value to multiply each element by.

        Returns:
          A new ConcentratedValue instance with each element scaled.
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return ConcentratedValue(a * scalar for a in self)


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

    def __add__(self, other: UDL) -> UDL:
        """
        Add two UDL instances element-wise.

        Args:
          other: Another UDL instance.

        Returns:
          A new UDL instance with the sum of corresponding elements.
        """
        if not isinstance(other, UDL):
            return NotImplemented
        return UDL(a + b for a, b in zip(self, other))

    def __mul__(self, scalar: float) -> UDL:
        """
        Multiply each element of UDL by a scalar.

        Args:
          scalar: A numeric value to multiply each element by.

        Returns:
          A new UDL instance with each element scaled.
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return UDL(a * scalar for a in self)
