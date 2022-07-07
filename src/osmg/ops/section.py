"""
Model Generator for OpenSees ~ section
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from ..mesh import Mesh
from ..mesh import polygon_area
from .. import common
if TYPE_CHECKING:
    from .element import Element
    from ..collections import MeshCollection
    from .uniaxialMaterial import uniaxialMaterial
    from ..physical_material import PhysicalMaterial

nparr = npt.NDArray[np.float64]

# pylint disable=invalid-name
# pylint: disable=too-many-instance-attributes


@dataclass
class Section:
    """

    """
    name: str
    uid: int


@dataclass
class ElasticSection(Section):
    """

    """
    E: float
    A: float
    Iy: float
    Ix: float
    G: float
    J: float
    W: float
    outside_shape: Optional[Mesh] = field(default=None, repr=False)
    snap_points: Optional[dict[str, nparr]] = field(default=None, repr=False)

    def weight_per_length(self):
        if self.name[0] == 'W':
            res = self.W * 1.15  # misc steel and connections
        else:
            res = self.W
        return res


@dataclass(repr=False)
class SectionComponent:
    """
    Part of a section object, having a single material.
    """
    outside_shape: Mesh
    holes: dict[str, Mesh]
    ops_material: uniaxialMaterial
    physical_material: PhysicalMaterial


@dataclass(repr=False)
class FiberSection(Section):
    """
    Fiber section object.
    Can consist of multiple materials.
    The primary part of the component must have the key 'main'.
    """
    outside_shape: Mesh
    section_parts: dict[str, SectionComponent]
    J: float
    snap_points: Optional[dict[str, nparr]] = field(default=None, repr=False)
    n_x: int = field(default=10)  # todo: this shoule be editable
    n_y: int = field(default=10)

    def ops_args(self):
        return['Fiber', self.uid, '-GJ',
               self.J*self.section_parts['main'].physical_material.G]

    def weight_per_length(self):
        if self.name[0] == 'W':
            mult = 1.15  # misc steel and connections
        else:
            mult = 1.00
        res = 0.00
        for part in self.section_parts.values():
            coords: nparr = np.array([h.vertex.coords for h in part.outside_shape.halfedges])
            area = polygon_area(coords)
            for hole in part.holes:
                hole_coords: nparr = np.array([h.vertex.coords for h in hole.halfedges])
                area -= polygon_area(hole_coords)
            density = part.physical_material.density
            # todo: units
            res += area * density * common.G_CONST_IMPERIAL
        return res * mult
            
