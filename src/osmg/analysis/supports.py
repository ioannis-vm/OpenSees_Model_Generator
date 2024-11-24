"""Node supports."""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(repr=False)
class FixedSupport:
    """Fixed support."""

    dof_restraints: tuple(bool, ...)


@dataclass(repr=False)
class ElasticSupport:
    """Flexible support."""

    dof_restraints: tuple(float, ...)
