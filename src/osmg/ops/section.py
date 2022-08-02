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
from typing import Any
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from ..mesh import Mesh
from ..mesh import polygon_area
from .. import common
if TYPE_CHECKING:
    from .uniaxial_material import UniaxialMaterial
    from ..physical_material import PhysicalMaterial

nparr = npt.NDArray[np.float64]


@dataclass
class Section:
    """
    Section object.
    The axes are defined in the same way as they are
    defined in OpenSees. The colors assigned to
    the axes for plotting follow the
    AutoCAD convention.

            y(green)
            ^         x(red)
            :       .
            :     .
            :   .
           ===
            | -------> z (blue)
           ===

    """
    name: str
    uid: int


@dataclass
class ElasticSection(Section):
    """
    Elastic Section Object
    """
    e_mod: float
    area: float
    i_y: float
    i_x: float
    g_mod: float
    j_mod: float
    sec_w: float
    outside_shape: Optional[Mesh] = field(default=None, repr=False)
    snap_points: Optional[dict[str, nparr]] = field(default=None, repr=False)
    properties: Optional[dict[str, Any]] = field(default=None, repr=False)

    def weight_per_length(self):
        """
        Returns the weight per length of a section.
        For steel W sections, it adds 15% for misc. steel and connections.
        """
        if self.name[0] == 'W':
            res = self.sec_w * 1.15  # misc steel and connections
        else:
            res = self.sec_w
        return res

    def __repr__(self):
        res = ''
        res += 'ElasticSection object\n'
        res += f'name: {self.name}\n'
        res += f'uid: {self.uid}\n'
        res += 'Properties:'
        res += f'  E: {self.e_mod}\n'
        res += f'  A: {self.area}\n'
        res += f'  Iy: {self.i_y}\n'
        res += f'  Ix: {self.i_x}\n'
        res += f'  G: {self.g_mod}\n'
        res += f'  J: {self.j_mod}\n'
        res += f'  W: {self.sec_w}\n'
        if self.outside_shape:
            res += 'outside_shape: specified\n'
        else:
            res += 'outside_shape: None\n'
        if self.snap_points:
            res += 'snap_points: specified\n'
        else:
            res += 'snap_points: None\n'
        return res


@dataclass(repr=False)
class SectionComponent:
    """
    Part of a section object, having a single material.
    """
    outside_shape: Mesh
    holes: dict[str, Mesh]
    ops_material: UniaxialMaterial
    physical_material: PhysicalMaterial

    def __repr__(self):
        res = ''
        res += 'SectionComponent object\n'
        if self.outside_shape:
            res += 'outside_shape: specified\n'
        else:
            res += 'outside_shape: None\n'
        if self.holes:
            res += 'holes: exist\n'
        else:
            res += 'holes: no holes\n'
        res += f'ops_material: {self.ops_material.name}\n'
        res += f'physical_material: {self.physical_material.name}\n'
        return res


@dataclass(repr=False)
class FiberSection(Section):
    """
    Fiber section object.
    Can consist of multiple materials.
    The primary part of the component must have the key 'main'.
    """
    outside_shape: Mesh
    section_parts: dict[str, SectionComponent]
    j_mod: float
    snap_points: Optional[dict[str, nparr]] = field(default=None, repr=False)
    n_x: int = field(default=10)  # TODO: this shoule be editable
    n_y: int = field(default=10)

    def __repr__(self):
        res = ''
        res += 'FiberSection object\n'
        for part in self.section_parts:
            res += part.__repr__()
        if self.snap_points:
            res += 'snap_points: specified\n'
        else:
            res += 'snap_points: None\n'
        res += f'n_x: {self.n_x}, n_y: {self.n_y}\n'
        return res

    def ops_args(self):
        """
        Returns the arguments required to define the object in
        OpenSees
        """
        return ['Fiber', self.uid, '-GJ',
                self.j_mod*self.section_parts['main'].physical_material.g_mod]

    def weight_per_length(self):
        """
        Returns the weight per length of a section.
        For steel W sections, it adds 15% for misc. steel and connections.
        """
        if self.name[0] == 'W':
            mult = 1.15  # misc steel and connections
        else:
            mult = 1.00
        res = 0.00
        for part in self.section_parts.values():
            coords: nparr = np.array(
                [h.vertex.coords for h in part.outside_shape.halfedges])
            area = polygon_area(coords)
            for hole in part.holes:
                hole_coords: nparr = np.array(
                    [h.vertex.coords
                     for h in part.holes[hole].halfedges])
                area -= polygon_area(hole_coords)
            density = part.physical_material.density
            # TODO: units
            res += area * density * common.G_CONST_IMPERIAL
        return res * mult
