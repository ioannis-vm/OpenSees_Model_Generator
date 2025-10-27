"""Defines OpenSees frictionModel objects."""

from __future__ import annotations

from dataclasses import dataclass

from osmg.core.uid_object import UIDObject


@dataclass
class FrictionModel(UIDObject):
    """
    OpenSees frictionModel.

    https://openseespydoc.readthedocs.io/en/latest/src/frictionModel.html

    """

    name: str

    def ops_args(self) -> list[object]:
        """Obtain the OpenSees arguments."""
        msg = 'Subclasses should implement this.'
        raise NotImplementedError(msg)


@dataclass
class Coulomb(FrictionModel):
    """
    Coulomb.

    https://openseespydoc.readthedocs.io/en/latest/src/Coulomb.html

    """

    mu: float

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return ['Coulomb', self.uid, self.mu]
