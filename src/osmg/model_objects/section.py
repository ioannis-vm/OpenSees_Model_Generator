"""Defines :obj:`~osmg.model_objects.section.Section` objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from osmg.core import common
from osmg.core.uid_object import UIDObject
from osmg.geometry import mesh
from osmg.geometry.mesh import Mesh, polygon_area

if TYPE_CHECKING:
    from shapely.geometry import Polygon as shapely_Polygon

    from osmg.core.common import numpy_array
    from osmg.model_objects.uniaxial_material import UniaxialMaterial
    from osmg.physical_material import PhysicalMaterial


@dataclass()
class Section(UIDObject):
    """
    Section object.

    The axes are defined in the same way as they are
    defined in OpenSees. The colors assigned to
    the axes for plotting follow the
    AutoCAD convention.

    .. code-block:: python

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


@dataclass
class ElasticSection(Section):
    """
    Elastic Section Object.

    Attributes:
    ----------
      e_mod: Young's modulus.
      area: Cross-sectional area.
      i_y: Moment of inertia for strong-axis bending.
      i_x: Moment of inertia for weak-axis bending.
      g_mod: Shear modulus.
      j_mod: Torsional moment of inertia.
      sec_w: Weight per unit length.
      outside_shape: Mesh defining the outside shape of the section.
      snap_points: Dictionary containing coordinates of `snap_points`
        used by component-generating methods to position components
        relative to existing ones. See
        :func:`~osmg.creators.component.beam_placement_lookup` for example.
      properties: Dictionary containing section properties.

    """

    e_mod: float
    area: float
    i_y: float
    i_x: float
    g_mod: float
    j_mod: float
    sec_w: float
    outside_shape: Mesh | None = field(default=None, repr=False)
    snap_points: dict[str, numpy_array] | None = field(default=None, repr=False)
    properties: dict[str, Any] | None = field(default=None, repr=False)

    def weight_per_length(self) -> float:
        """
        Weight per unit length.

        Returns the weight per length of a section.
        For steel W sections, it adds 15% for misc. steel and
        connections.

        Returns:
          The weight per unit length.
        """
        if self.name[0] == 'W':
            res = self.sec_w * 1.15  # misc steel and connections
        else:
            res = self.sec_w
        return res

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
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

    Arguments:
      outside_shape: Mesh defining the outside shape
      ops_material: OpenSees material
      physical_material: Physical material
      parent_section: Parent section.
        The parent section is assigned automatically by their
        parent section iteslf, at its creation time.

    """

    outside_shape: Mesh
    holes: list[Mesh]
    ops_material: UniaxialMaterial
    physical_material: PhysicalMaterial
    parent_section: FiberSection | None = field(default=None)

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
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

    def cut_into_tiny_little_pieces(self) -> list[shapely_Polygon]:
        """
        Obtain data used to define fibers in OpenSees.

        Returns:
          The data.
        """
        # if we have an AISC HSS section, we need to discretize in a
        # certain way
        assert self.parent_section
        sec_name = self.parent_section.name
        rectangular_sections = 3
        circular_sections = 2
        if 'HSS' in sec_name and len(sec_name.split('X')) == rectangular_sections:
            # rectangular HSS section!
            pieces = mesh.subdivide_hss_rect(
                self.parent_section.properties['Ht'],
                self.parent_section.properties['B'],
                self.parent_section.properties['tdes'],
            )
        elif 'HSS' in sec_name and len(sec_name.split('X')) == circular_sections:
            # circular HSS section!
            pieces = mesh.subdivide_hss_circ(
                self.parent_section.properties['OD'],
                self.parent_section.properties['tdes'],
            )

        else:
            # fallback: use the default rectangular mesh chopper
            pieces = mesh.subdivide_polygon(
                outside=self.outside_shape,
                holes=self.holes,
                n_x=self.parent_section.n_x,
                n_y=self.parent_section.n_y,
            )

        return pieces


@dataclass(repr=False)
class FiberSection(Section):
    """
    Fiber section object.

    Can consist of multiple materials.
    The primary part of the component must have the key "main".

    """

    outside_shape: Mesh
    section_parts: dict[str, SectionComponent]
    j_mod: float
    snap_points: dict[str, numpy_array]
    properties: dict[str, Any]
    n_x: int
    n_y: int

    def __post_init__(self) -> None:
        """Post-initialization."""
        for part in self.section_parts:
            self.section_parts[part].parent_section = self

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns:
          The string representation of the object.
        """
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

    def ops_args(self) -> list[object]:
        """
        Obtain the OpenSees arguments.

        Returns:
          The OpenSees arguments.
        """
        return [
            'Fiber',
            self.uid,
            '-GJ',
            self.j_mod * self.section_parts['main'].physical_material.g_mod,
        ]

    def weight_per_length(self) -> float:
        """
        Weight per unit length.

        Returns the weight per length of a section.
        For steel W sections, it adds 15% for misc. steel and connections.

        Returns:
          The weight per unit length.
        """
        if self.name[0] == 'W':
            mult = 1.15  # misc steel and connections
        else:
            mult = 1.00
        res = 0.00
        for part in self.section_parts.values():
            coordinates: numpy_array = np.array(
                [h.vertex.coordinates for h in part.outside_shape.halfedges]
            )
            area = polygon_area(coordinates)
            for hole in part.holes:
                hole_coordinates: numpy_array = np.array(
                    [h.vertex.coordinates for h in hole.halfedges]
                )
                area -= polygon_area(hole_coordinates)
            density = part.physical_material.density
            # TODO(JVM): units
            res += area * density * common.G_CONST_IMPERIAL
        return res * mult
