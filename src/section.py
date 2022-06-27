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
from typing import Union
from dataclasses import dataclass, field
from itertools import count
from collections import OrderedDict
import numpy as np
import mesher
import mesher_section_gen
if TYPE_CHECKING:
    from components import Material

section_ids = count(0)


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
    Attributes:
        id (int): unique identifier
        sec_type (str): Flag representing the type of section
                  (e.g. W -> steel W section)
        name (str): Unique name for the section
        material (Material): Material of the section
        snap_points (dict): Dictionary containing the local
                    coordinates of a set of characetristic points.
                    These points are:
                    'centroid', 'top_center', 'top_left', 'top_right',
                    'center_left', 'center_right', 'bottom_center',
                    'bottom_left', bottom_right'
        mesh (mesher.Mesh): Mesh object defining the geometry
                            of the section
        properties (dict): Dictionary with geometric properties
                           needed for structural analysis.
                           These are:
                           A, Ix, Iy, J
    """
    sec_type: str
    name: str
    material: Material = field(repr=False)
    snap_points: Optional[dict[str, np.ndarray]] = field(default=None, repr=False)
    mesh: Optional[mesher.Mesh] = field(default=None, repr=False)
    properties: Optional[dict[str, float]] = field(default=None, repr=False)

    def __post_init__(self):
        self.uid = str(next(section_ids))

    def __eq__(self, other):
        return (self.name == other.name)

    def subdivide_section(self, n_x=10, n_y=25, plot=False):
        """
        Used to define the fibers of fiber sections.
        Args:
            n_x (int): Number of spatial partitions in the x direction
            n_y (int): Number of spatial partitions in the y direction
            plot (bool): Plots the resulting polygons for debugging
        Returns:
            pieces (list[shapely_Polygon]): shapely_Polygon
                   objects that represent single fibers.
        """
        return mesher.subdivide_polygon(
            self.mesh.halfedges, n_x=n_x, n_y=n_y, plot=plot)

    def rbs(self, reduction_factor) -> 'Section':
        """
        Given a reduction factor
        expressed as a proportion of the section's width,
        the method returns a reduced section.
        Only works for W sections.
        """
        if self.sec_type != 'W':
            raise ValueError("Only W sections are supported for RBS")
        name = self.name + '_reduced'
        if not self.properties:
            raise ValueError('The section has no properties attribute')
        b = self.properties['bf']
        h = self.properties['d']
        tw = self.properties['tw']
        tf = self.properties['tf']
        b_red = b * reduction_factor
        properties = dict(self.properties)
        properties['bf'] = b_red
        c = (b - b_red) / 2.
        area = properties['A']
        new_area = float(area) - 4. * tf * c
        mesh = mesher_section_gen.w_mesh(b_red, h, tw, tf, new_area)
        section = Section(
            'W', name,
            self.material,
            None,
            mesh, properties)
        return section


@dataclass
class Sections:
    """
    This class is a collector for sections.
    """

    registry: OrderedDict[str, Section] = field(
        default_factory=OrderedDict, repr=False)
    active: Optional[Section] = field(default=None, repr=False)

    def __post_init__(self):
        """
        Add a default section for rigid links
        and a dummy section used when plotting panel zones
        """
        self.registry['rigid'] = Section(
            'utility', 'rigid', None,
            None, mesher_section_gen.rect_mesh(1.00, 1.00))
        self.registry['dummy_PZ'] = Section(
            'utility', 'dummy_PZ', None,
            None, None)

    def add(self, section: Section):
        """
        Add a section in the section collection,
        if it does not already exist
        """
        if section.name not in self.registry:
            self.registry[section.name] = section
        else:
            raise ValueError(f'Section {section.name} already defined')

    def set_active(self, name: str):
        """
        Sets the active section.
        Any elements defined while this section is active
        will have that section.
        Args:
            name (str): Name of the previously defined
                 section to set as active.
        """
        self.active = None
        if name in self.registry:
            self.active = self.registry[name]
        else:
            raise ValueError(f'Undefined section: {name}')

    ####################
    # Shape generators #
    ####################

    def generate_W(self,
                   name: str,
                   material: Material,
                   properties: dict):
        """
        Generate a W section with specified parameters
        and add it to the sections list.
        """
        b = properties['bf']
        h = properties['d']
        tw = properties['tw']
        tf = properties['tf']
        area = properties['A']
        mesh = mesher_section_gen.w_mesh(b, h, tw, tf, area)
        bbox = mesh.bounding_box()
        z_min, y_min, z_max, y_max = bbox.flatten()
        snap_points = {
            'centroid': np.array([0., 0.]),
            'top_center': np.array([0., -y_max]),
            'top_left': np.array([-z_min, -y_max]),
            'top_right': np.array([-z_max, -y_max]),
            'center_left': np.array([-z_min, 0.]),
            'center_right': np.array([-z_max, 0.]),
            'bottom_center': np.array([0., -y_min]),
            'bottom_left': np.array([-z_min, -y_min]),
            'bottom_right': np.array([-z_max, -y_min])
        }
        section = Section('W', name, material,
                          snap_points,  mesh, properties)
        self.add(section)

    def generate_HSS(self,
                     name: str,
                     material: Material,
                     properties: dict):
        """
        Generate a HSS with specified parameters
        and add it to the sections list.
        """
        # use the name to assess whether it's a rectangular
        # or circular section
        xs = name.count('X')
        if xs == 2:
            # it's a rectangular section
            ht = properties['Ht']
            b = properties['B']
            t = properties['tdes']
            mesh = mesher_section_gen.HSS_rect_mesh(ht, b, t)
            bbox = mesh.bounding_box()
            z_min, y_min, z_max, y_max = bbox.flatten()
            snap_points = {
                'centroid': np.array([0., 0.]),
                'top_center': np.array([0., -y_max]),
                'top_left': np.array([-z_min, -y_max]),
                'top_right': np.array([-z_max, -y_max]),
                'center_left': np.array([-z_min, 0.]),
                'center_right': np.array([-z_max, 0.]),
                'bottom_center': np.array([0., -y_min]),
                'bottom_left': np.array([-z_min, -y_min]),
                'bottom_right': np.array([-z_max, -y_min])
            }
            section = Section('HSS', name, material,
                              snap_points, mesh, properties)
            self.add(section)
        elif xs == 1:
            # it's a circular section
            od = properties['OD']
            tdes = properties['tdes']
            n_pts = 25
            mesh = mesher_section_gen.HSS_circ_mesh(od, tdes, n_pts)
            bbox = mesh.bounding_box()
            z_min, y_min, z_max, y_max = bbox.flatten()
            snap_points = {
                'centroid': np.array([0., 0.]),
                'top_center': np.array([0., -y_max]),
                'top_left': np.array([-z_min, -y_max]),
                'top_right': np.array([-z_max, -y_max]),
                'center_left': np.array([-z_min, 0.]),
                'center_right': np.array([-z_max, 0.]),
                'bottom_center': np.array([0., -y_min]),
                'bottom_left': np.array([-z_min, -y_min]),
                'bottom_right': np.array([-z_max, -y_min])
            }
            section = Section('HSS', name, material,
                              snap_points, mesh, properties)
            self.add(section)
        else:
            raise ValueError("This should never happen...")

    def generate_rect(self,
                      name: str,
                      material: Material,
                      properties: dict):
        """
        Generate a rectangular section with specified
        parameters and add it to the sections list.
        """
        b = properties['b']
        h = properties['h']
        mesh = mesher_section_gen.rect_mesh(b, h)
        bbox = mesh.bounding_box()
        z_min, y_min, z_max, y_max = bbox.flatten()
        snap_points = {
            'centroid': np.array([0., 0.]),
            'top_center': np.array([0., -y_max]),
            'top_left': np.array([-z_min, -y_max]),
            'top_right': np.array([-z_max, -y_max]),
            'center_left': np.array([-z_min, 0.]),
            'center_right': np.array([-z_max, 0.]),
            'bottom_center': np.array([0., -y_min]),
            'bottom_left': np.array([-z_min, -y_min]),
            'bottom_right': np.array([-z_max, -y_min])
        }
        section = Section('rect', name, material,
                          snap_points, mesh, properties)
        self.add(section)
        temp = mesh.geometric_properties()
        properties['A'] = temp['area']
        properties['Ix'] = temp['inertia']['ixx']
        properties['Iy'] = temp['inertia']['iyy']
        properties['J'] = h * b**3 *\
            (16./3. - 3.36 * b/h * (1 - b**4/(12.*h**4)))
