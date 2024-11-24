"""Load cases."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from osmg.analysis.supports import ElasticSupport, FixedSupport
from osmg.core.common import EPSILON

if TYPE_CHECKING:
    from osmg.core.model import Model2D, Model3D


@dataclass(repr=False)
class ConcentratedValue:
    """Concentrated value, such as a point load or mass."""

    value: tuple[float, ...]


@dataclass(repr=False)
class PointLoad(ConcentratedValue):
    """Point load."""


@dataclass(repr=False)
class PointMass(ConcentratedValue):
    """Point load."""


@dataclass(repr=False)
class UDL:
    """
    Beamcolumn element UDL.

    Uniformly distributed load expressed in the global coordinate
    system of the structure.
    """

    value: tuple[float, ...]


@dataclass(repr=False)
class LoadRegistry:
    """Load registry."""

    nodal_loads: dict[int, PointLoad] = field(default_factory=dict)
    nodal_mass: dict[int, PointMass] = field(default_factory=dict)
    element_udl: dict[int, UDL] = field(default_factory=dict)


@dataclass(repr=False)
class LoadCase:
    """Load case."""

    fixed_supports: dict[int, FixedSupport] = field(default_factory=dict)
    elastic_supports: dict[int, ElasticSupport] = field(default_factory=dict)
    load_registry: LoadRegistry = field(default_factory=LoadRegistry)

    def add_supports_at_level(
        self,
        model: Model2D | Model3D,
        support: ElasticSupport | FixedSupport,
        level_tag: str,
    ) -> None:
        """
        Add the given support at the specified level.

        Determines all primary nodes that have an elevation equal to
        the specified level's elevation and assigns the specified
        support to them.

        Assumes that the last coordinate of the nodes corresponds to
        elevation.

        Raises:
          TypeError: If the provided support is not a known support
            type.
        """
        nodes = list(model.nodes.values())
        level_elevation = model.grid_system.get_level_elevation(level_tag)
        for node in nodes:
            if np.abs(node.coordinates[-1] - level_elevation) < EPSILON:
                if isinstance(support, FixedSupport):
                    self.fixed_supports[node.uid] = support
                elif isinstance(support, ElasticSupport):
                    self.elastic_supports[node.uid] = support
                else:
                    msg = f'Unsupported object type: {type(support)}'
                    raise TypeError(msg)


@dataclass(repr=False)
class DeadLoadCase(LoadCase):
    """Dead load case."""


@dataclass(repr=False)
class LiveLoadCase(LoadCase):
    """Live load case."""


@dataclass(repr=False)
class SeismicLoadCase(LoadCase):
    """Seismic load case base class."""


@dataclass(repr=False)
class SeismicELFLoadCase(SeismicLoadCase):
    """Seismic ELF load case."""


@dataclass(repr=False)
class SeismicRSLoadCase(SeismicLoadCase):
    """Seismic RS load case."""


@dataclass(repr=False)
class SeismicTransientLoadCase(SeismicLoadCase):
    """Seismic transient load case."""


@dataclass(repr=False)
class OtherLoadCase(LoadCase):
    """Other load case."""


@dataclass(repr=False)
class LoadCaseRegistry:
    """
    Load case registry with automatic instantiation.

    Automatically creates an empty load case for each attribute when a
    string key is accessed.
    """

    dead: defaultdict[str, DeadLoadCase] = field(
        default_factory=lambda: defaultdict(DeadLoadCase)
    )
    live: defaultdict[str, LiveLoadCase] = field(
        default_factory=lambda: defaultdict(LiveLoadCase)
    )
    seismic: defaultdict[str, SeismicLoadCase] = field(
        default_factory=lambda: defaultdict(SeismicLoadCase)
    )
    other: defaultdict[str, OtherLoadCase] = field(
        default_factory=lambda: defaultdict(OtherLoadCase)
    )
